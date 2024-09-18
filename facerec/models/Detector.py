from typing import List, Tuple
from abc import ABC, abstractmethod
import numpy as np

class Detector(ABC):
    @abstractmethod
    def detect_face(self, img:np.ndarray) -> Tuple[List[int], float]:
        """
        Interface for detect and align face

        Args:
            img (np.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[int]):   
        """
        pass