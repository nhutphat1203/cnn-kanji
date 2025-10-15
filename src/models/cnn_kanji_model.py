from typing import List, Tuple
from core.logger import setup_logger
import tensorflow as tf
import numpy as np
import io
import json
import os
from pathlib import Path
import os
from fastapi import UploadFile

# Lấy đường dẫn tuyệt đối đến thư mục chứa file hiện tại
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn đến các file model và label
PATH_MODEL = os.path.join(CURRENT_DIR, 'model', 'kanji_3036_best_e4.h5')
PATH_LABEL = os.path.join(CURRENT_DIR, 'model', 'label_3036.json')
PATH_COLLECTION = os.path.join(CURRENT_DIR, 'test')

logger = setup_logger()

class ImageModel:
    def __init__(self):
        logger.info(f"Initializing CNNKanjiModel with model_path: {PATH_MODEL}")
        # Load model TensorFlow
        try:
            self.model = tf.keras.models.load_model(PATH_MODEL)
            logger.info("✅ Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e
        # Load labels từ JSON
        try:
            with open(PATH_LABEL, 'r', encoding='utf-8') as f:
                self.label_list = json.load(f)
            logger.info(f"✅ Loaded {len(self.label_list)} labels from {PATH_LABEL}")
        except Exception as e:
            logger.error(f"Failed to load labels: {e}")
            raise e

    def preprocess_image(self, image_bytes: bytes, img_size=(128, 128)) -> np.ndarray:
        """
        Chuẩn hóa ảnh grayscale từ bytes để predict.
        """
        img = tf.keras.utils.load_img(io.BytesIO(image_bytes), color_mode='grayscale', target_size=img_size)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, 1)
        return img_array

    def predict(self, image_bytes: bytes, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        image_bytes: ảnh upload từ user
        top_k: số lượng dự đoán hàng đầu cần trả về
        Return: List tuple (kanji character, confidence)
        """
        nimage = self.preprocess_image(image_bytes)
        result = self.model.predict(nimage)[0]  # output shape: (num_classes,)
        
        # Map label và confidence
        predictions = [(self.label_list[i], float(result[i])) for i in range(len(result))]
        
        # Sắp xếp giảm dần theo confidence
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Lấy top_k
        return predictions[:top_k]
    
    def save_pic(self, filename: str, contents: bytes):
        """
        Lưu file ảnh vào thư mục PATH_COLLECTION, chia theo label.
        """
        try:
            label = filename.split('_')[0]
            label_dir = os.path.join(PATH_COLLECTION, label)
            os.makedirs(label_dir, exist_ok=True)

            existing_files = [f for f in os.listdir(label_dir) if f.endswith('.png')]
            next_index = len(existing_files) + 1
            new_filename = f"{label}_{next_index:04d}.png"
            file_path = os.path.join(label_dir, new_filename)

            with open(file_path, "wb") as f:
                f.write(contents)

            logger.info(f"✅ Saved image: {file_path}")
            return {"label": label, "path": file_path}

        except Exception as e:
            logger.error(f"❌ Failed to save image: {e}")
            raise e


class CNNKanjiModel:
    """
    CNN Kanji Model for base64 image prediction
    Optimized for canvas drawing recognition
    """
    
    def __init__(self, model_path: str, label_path: str):
        """
        Initialize CNN Kanji Model
        
        Args:
            model_path: Path to .h5 model file
            label_path: Path to label JSON file
        """
        self.model_path = model_path
        self.label_path = label_path
        
        logger.info(f"Initializing CNNKanjiModel with model_path: {model_path}")
        
        # Load model
        try:
            self.model = tf.keras.models.load_model(model_path)
            logger.info("✅ Model loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise e
        
        # Load labels
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                self.labels = json.load(f)
            logger.info(f"✅ Loaded {len(self.labels)} labels from {label_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load labels: {e}")
            raise e
    
    def predict(self, image: np.ndarray) -> dict:
        """
        Predict kanji character from preprocessed image
        
        Args:
            image: Preprocessed numpy array (1, 64, 64, 3) or (1, 128, 128, 1)
        
        Returns:
            Dictionary with prediction results
        """
        try:
            # Make prediction
            predictions = self.model.predict(image, verbose=0)
            
            # Get top prediction
            top_idx = np.argmax(predictions[0])
            top_confidence = float(predictions[0][top_idx])
            top_character = self.labels[top_idx] if self.labels else str(top_idx)
            
            # Get top 5 predictions
            top5_indices = np.argsort(predictions[0])[-5:][::-1]
            top5 = [
                {
                    'character': self.labels[idx] if self.labels else str(idx),
                    'confidence': float(predictions[0][idx])
                }
                for idx in top5_indices
            ]
            
            logger.info(f"✅ Prediction: {top_character} (confidence: {top_confidence:.4f})")
            
            return {
                'character': top_character,
                'confidence': top_confidence,
                'top5': top5
            }
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {str(e)}")
            raise
