from abc import ABC
from typing import Any, Union, List, Tuple
import numpy as np

from sklearn.preprocessing import normalize

import keras
from tensorflow.keras.models import Model
# import tensorflow as tf

class Recognitor(ABC):

    model_name: str
    input_shape: Tuple[int, int, int]
    output_shape: int
    model: Union[Model, Any]

    def forward(self, img: np.ndarray) -> float:
        return np.float32(normalize([self.model(img).numpy()[0]])[0])