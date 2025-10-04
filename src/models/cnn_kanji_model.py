from typing import List, Tuple

from core.logger import setup_logger

# import tensorflow as tf

logger = setup_logger()

class ImageModel:
    def __init__(self, model_path: str):
        logger.info(f"Initializing CNNKanjiModel with model_path: {model_path}")
        
        # self.model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")

    def _preprocessor_image(self, image):
        """_summary_

        Args:
            image (_type_): image of user input that has difference shap than model's target shape 

        Returns:
            _type_: normalize image like (96, 96, 3)
        """
        normalize_img = ''
        
        return normalize_img

    def predict(self, image) -> List[Tuple[str, float]]:
        """
            image: and image of user input
            Return
                List tuple of kanji character and percent trust
        """
        nimage = self._preprocessor_image(image)
        result = self.model.predict(nimage)
        
        predictions = []
        
        return predictions
    