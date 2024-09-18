from abc import ABC
from typing import Any, Tuple
import numpy as np

from sklearn.preprocessing import normalize

import keras
# import tensorflow as tf

class Recognitor(ABC):
    model: Any
    model_name: str
    input_shape: Tuple[int, int]
    output_shape: int

    def forward(self, img: np.ndarray) -> float:
        return np.float32(normalize([self.model(img).numpy()[0]])[0])