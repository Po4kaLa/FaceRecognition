import cv2
from commons.logger import Logger

logger = Logger()

class Camera:
    def __init__(self): 
        self.capture = cv2.VideoCapture(0) 
        if not self.capture.isOpened(): 
            logger.error(f"Camera could not be opened.") 
            raise Exception("Camera could not be opened.") 
    def read_frame(self): 
        ret, frame = self.capture.read() 
        return ret, frame

    def release(self):
        if self.capture.isOpened():
            self.capture.release()