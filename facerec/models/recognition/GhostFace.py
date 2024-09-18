from models import Recognitor

import keras

class GhostFaceNetModel(Recognitor):
    """
    GhostFaceNet model class
    """

    def __init__(self):
        self.model = load_model()
        self.model_name = "GhostFaceNet"
        self.input_shape = (112, 112)
        self.output_shape = 512

def load_model():
    # model = GhostFaceNetV1()
    model = keras.models.load_model('keras_checkpoints/GhostFaceNet_o2.h5', compile=False)

    return model