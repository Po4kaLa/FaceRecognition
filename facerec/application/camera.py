import cv2
from facerec.logging.log import Logger

logger = Logger()

class Camera:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Camera, cls).__new__(cls)
            cls._instance.capture = cv2.VideoCapture(0)
            if not cls._instance.capture.isOpened():
                logger.error(f"Camera could not be opened.")
                raise Exception("Camera could not be opened.")
        return cls._instance

    def read_frame(self):
        ret, frame = self.capture.read()
        return ret, frame

    def release(self):
        if self.capture.isOpened():
            self.capture.release()