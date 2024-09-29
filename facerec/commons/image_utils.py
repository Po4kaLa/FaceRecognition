import cv2
import tensorflow as tf

from commons.logger import Logger

logger = Logger()

def read_rgb_frame(ret, frame):
    if not ret:
        logger.error("Failed to take frame from video stream.")
        return None 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return ret, frame

def normalization_frame_tensor(frame):
    try:
        frame = tf.convert_to_tensor(frame, dtype=tf.float32)
        frame = tf.expand_dims(tf.image.resize(frame, (112, 112)), 0)
        frame = (tf.cast(frame, "float32") - 127.5) * 0.0078125
        return frame
    except Exception as e:
        logger.error(f"Error during frame preprocessing: {str(e)}")
        return None

def crop_face(frame, bbox):
    try:
        x, y, w, h = bbox
        x = max(0, x)
        y = max(0, y)
        x = min(x, frame.shape[1])
        y = min(y, frame.shape[0])
        return frame[y:y+h, x:x+w]
    except Exception as e:
        logger.error(f"Error during frame preprocessing: {str(e)}")
        return None