import sys
import cv2
import sqlite3
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QLineEdit, QMessageBox, QInputDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

import keras
import tensorflow as tf

from facerec.application.camera import Camera
from facerec.db.database import FaceDatabase
from facerec.models.recognition import GhostFace
from facerec.models.detection import Yolov8
from facerec.commons import image_utils 
from facerec.logging.log import Logger

logger = Logger()

class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.name_dist = "Unknown"
        self.current_frame = None

        self.camera =  Camera()

        self.face_detector = Yolov8.YoloDetector()
        self.model = GhostFace.GhostFaceNetModel()

        self.database = FaceDatabase()

        #GUI
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

    def show_input_dialog(self):
        self.img_name_input, okPressed = QInputDialog.getText(self, "Введите имя", "Имя изображения:")
        if self.current_frame is not None:
            if okPressed and self.img_name_input != '':
                self.append_database()
        else:
            print("Нет доступного кадра для добавления в базу данных.")
            
    def append_database(self):
        bbox = self.face_detector.detect_face(self.current_frame)
        if (len(bbox) > 0):
            face_frame = image_utils.crop_face(self.current_frame, bbox)
            embedding = self.model.forward(image_utils.normalization_frame_tensor(face_frame))
            self.database.save_embedding_to_database(embedding, self.img_name_input)
            FaceDatabase.save_face_image(face_frame, self.img_name_input)
    
    def update_frame(self):
        ret, frame = self.camera.read_frame()
        ret, frame = image_utils.read_rgb_frame(ret, frame)

        if ret:
            self.current_frame = frame
            bbox = self.face_detector.detect_face(self.current_frame)
            if (len(bbox) > 0):
                face_frame = image_utils.crop_face(self.current_frame, bbox)
                embedding = self.model.forward(image_utils.normalization_frame_tensor(face_frame))
                self.name_dist = FaceDatabase.verify(embedding).split()

    def initiate_delete_process(self):
        bbox = self.face_detector.detect_face(self.current_frame)
        if (len(bbox) > 0):
            face_frame = image_utils.crop_face(self.current_frame, bbox)
            embedding = self.model.forward(image_utils.normalization_frame_tensor(face_frame))
            self.name_dist = FaceDatabase.verify(embedding).split()
            if self.name_dist[0] != "unknown":
                self.confirm_delete(self.name_dist[0])

    def confirm_delete(self, name):
        reply = QMessageBox.question(self, 'Подтверждение удаления', 
                                     f"Вы уверены, что хотите удалить {name}?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            FaceDatabase.delete_record(name) 
    

    def __del__(self):
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())