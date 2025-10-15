from models.cnn_kanji_model import ImageModel, CNNKanjiModel
from functools import lru_cache
import os

# Singleton instances
cnn_kanji_model = ImageModel()
_model_instance = None


def get_reg_kanji_model():
    """
    Get regular kanji model for image upload endpoint
    
    Returns:
        ImageModel instance
    """
    return cnn_kanji_model


@lru_cache()
def get_model() -> CNNKanjiModel:
    """
    Get singleton instance of CNN Kanji Model for base64 prediction
    
    Returns:
        CNNKanjiModel instance
    """
    global _model_instance
    
    if _model_instance is None:
        # Get model path from environment or use default
        model_path = os.getenv(
            'MODEL_PATH',
            os.path.join(os.path.dirname(__file__), '..', 'models', 'model', 'kanji_3036_best_e4.h5')
        )
        
        label_path = os.getenv(
            'LABEL_PATH',
            os.path.join(os.path.dirname(__file__), '..', 'models', 'model', 'label_3036.json')
        )
        
        _model_instance = CNNKanjiModel(model_path, label_path)
    
    return _model_instance