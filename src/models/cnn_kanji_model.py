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
PATH_MODEL = os.path.join(CURRENT_DIR, 'model', 'kanji_100_best.h5')
PATH_LABEL = os.path.join(CURRENT_DIR, 'model', 'label.json')
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
