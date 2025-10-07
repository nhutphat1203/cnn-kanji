from models.cnn_kanji_model import ImageModel

cnn_kanji_model = ImageModel()


def get_reg_kanji_model():
    """_summary_
        Dependency injection
    Returns:
        _type_: a model instance
    """
    return cnn_kanji_model