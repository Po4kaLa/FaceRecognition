from typing import Any, List, Tuple

import numpy as np

from facerec.logging.log import Logger
from facerec.models import Detector

logger = Logger()

PATH_WEIGHTS = "./facerec/weights/yolov8n-face.pt"

class YoloDetector(Detector):
    def __init__(self):
        self.model = self.build_model()
    
    def build_model(self) -> Any:
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as e:
            raise ImportError(
                "Please install YOLO using 'pip install ultralytics'"
            ) from e
        return YOLO(PATH_WEIGHTS)
    
    def detect_face(self, img: np.ndarray) -> Tuple[List[int], float]:
        try:
            results = self.model.predict(img, verbose=False, show=False, conf=0.25)[0]
            max_area = 0
            max_bbox = []

            for result in results:
                x, y, w, h = result.boxes.xywh.tolist()[0]
                confidence = result.boxes.conf.tolist()[0]
                area = w * h
                if area > max_area:
                    max_area = area
                    max_bbox = [int(x - w / 2), int(y - h / 2), int(w), int(h)]
            if (len(max_bbox) > 0):
                return max_bbox, confidence
            else:
                return [], 0.0
        except Exception as e:
            logger.error(f"An error occurred during face detection: {str(e)}")
            return [], 0.0  