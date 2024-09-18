from typing import Any, List, Tuple

import numpy as np

from facerec.logging.log import Logger
from facerec.models import Detector

logger = Logger()

class MtCnnClient(Detector):
    def __init__(self):
        self.model = self.build_model()
    
    def build_model(self) -> Any:
        try:
            from mtcnn import MTCNN
        except ModuleNotFoundError as e:
            raise ImportError(
                "Please install MTCNN using 'pip install mtcnn'"
            ) from e
        return MTCNN()

    def detect_face(self, img: np.ndarray) -> Tuple[List[int], float]:
        try:
            # mtcnn expects RGB but OpenCV read BGR
            # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = img[:, :, ::-1]
            results = self.model.detect_faces(img_rgb)
            max_area = 0
            max_bbox = []

            for result in results:
                x, y, w, h = result["box"]
                confidence = result["confidence"]
                area = w * h
                if area > max_area:
                    max_area = area
                    max_bbox = [int(x), int(y), int(w), int(h)]
            if (len(max_bbox) > 0):
                return max_bbox, confidence
            else:
                return [], 0.0
        except Exception as e:
            logger.error(f"An error occurred during face detection: {str(e)}")
            return [], 0.0 
