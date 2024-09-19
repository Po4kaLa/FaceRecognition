import os
import sys
import cProfile
import pstats
import io
sys.path.append("C:/Users/Po4ka/UPPRPO/FaceRecognition/FaceRecognition/facerec/")
import cv2
import sqlite3
import numpy as np

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QLineEdit, QMessageBox, QInputDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, QThread, pyqtSignal

import keras
import tensorflow as tf

from sklearn.preprocessing import normalize

from application.camera import Camera
from db.database import FaceDatabase
from models.recognition import GhostFace
from models.detection import Yolov8
from commons import image_utils 
from commons.logger import Logger


logger = Logger()
class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.name_dist = "Who"

        self.current_frame = None

        self.conn = sqlite3.connect('facialdb.db')
        self.cursor = self.conn.cursor()

        self.camera = cv2.VideoCapture(0)
        self.face_detector = Yolov8.YoloDetector()
        self.model = keras.models.load_model('../weights/GhostFaceNet_o2.h5', compile=False)
        self.database = FaceDatabase() 

        # GUI
        self.setWindowTitle("Face Recognition App")
        self.setGeometry(100, 100, 800, 600)

        screen_resolution = app.primaryScreen().availableGeometry()
        self.setGeometry(screen_resolution.width() // 2 - 400, screen_resolution.height() // 2 - 300, 800, 600)

        self.camera_label = QLabel(self)
        self.camera_label.setGeometry(40, 60, 480, 480)

        self.add_to_db_button = QPushButton("Добавить в базу", self)
        self.add_to_db_button.setGeometry(540, 100, 200, 50)
        self.add_to_db_button.clicked.connect(self.show_input_dialog)

        self.delete_button = QPushButton("Удалить", self)
        self.delete_button.setGeometry(540, 180, 200, 50)
        self.delete_button.clicked.connect(self.initiate_delete_process)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(50)

    def get_embedding(self, frame):

        frame = tf.convert_to_tensor(frame, dtype=tf.float32)
        frame = tf.expand_dims(tf.image.resize(frame, (112, 112)), 0)
        frame = (tf.cast(frame, "float32") - 127.5) * 0.0078125
        embedding = np.float32(normalize([self.model(frame).numpy()[0]])[0])
        return embedding

    def show_input_dialog(self): 
        self.img_name_input, okPressed = QInputDialog.getText(self, "Введите имя", "Имя изображения:") 
        if self.current_frame is not None: 
            if okPressed and self.img_name_input != '': 
                self.add_to_database()
        else: 
            print("Нет доступного кадра для добавления в базу.") 
    
    
    def update_frame(self):
        ret, frame = self.camera.read()
        ret, frame = image_utils.read_rgb_frame(ret, frame)

        if ret:
            self.current_frame = frame
            bbox = self.face_detector.detect_face(self.current_frame)[0]
            if (len(bbox) > 0):
                x, y, w, h = bbox
                x = max(0, x)
                y = max(0, y)
                x = min(x, frame.shape[1])
                y = min(y, frame.shape[0])
                face_frame = frame[y:y+h, x:x+w]
                embedding = self.get_embedding(face_frame)
                name_dist = self.database.verify(self.cursor, embedding).split()[0]

                cv2.putText(frame, name_dist[0] + " " + name_dist[1], (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                #cv2.putText(frame, name_dist[0], (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.camera_label.setPixmap(pixmap)

    def add_to_database(self):
        
        def save_face(face_frame, name):
            folder_path = 'face_data'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            face_img_path = os.path.join(folder_path, f'{name}.png')
            cv2.imwrite(face_img_path, face_frame)

        if self.current_frame is not None:
            frame = self.current_frame
            bbox = self.face_detector.detect_face(self.current_frame)[0]
            if (len(bbox) > 0):
                x, y, w, h = bbox
                x = max(0, x)
                y = max(0, y)
                x = min(x, frame.shape[1])
                y = min(y, frame.shape[0])
                face_frame = frame[y:y+h, x:x+w]
                print(bbox)
                embedding = self.get_embedding(face_frame)
                self.database.save_embedding_to_database(self.conn, self.cursor, embedding, self.img_name_input)
                save_face(face_frame, self.img_name_input)
        else:
            print("Нет доступного кадра для добавления в базу данных.")
    
    def initiate_delete_process(self):
        frame = self.current_frame
        bbox = self.face_detector.detect_face(frame)[0]
        print(bbox)
        if (len(bbox) > 0):
            x, y, w, h = bbox
            x = max(0, x)
            y = max(0, y)
            x = min(x, frame.shape[1])
            y = min(y, frame.shape[0])
            face_frame = frame[y:y+h, x:x+w]
            embedding = self.get_embedding(face_frame)
            name_distance = self.database.verify(self.cursor, embedding).split()
            if name_distance[0] != "unknown":
                self.confirm_delete(name_distance[0])

    def confirm_delete(self, name):
        reply = QMessageBox.question(self, 'Подтверждение удаления', 
                                     f"Вы уверены, что хотите удалить {name}?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.database.delete_record(self.conn, self.cursor, name)
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())

