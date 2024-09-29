import os
import sys

# current_dir = os.path.dirname(os.path.abspath(__file__))
# repo_path = os.path.join(current_dir, "facerec")
# sys.path.append(repo_path)
# print(repo_path)
sys.path.append("C:/Users/Po4ka/UPPRPO/FaceRecognition/FaceRecognition/facerec/")
import cProfile
import pstats
import io
import cv2
import sqlite3
import threading
import queue

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QLineEdit, QMessageBox, QInputDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, QThread, pyqtSignal

import keras
import tensorflow as tf

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

        self.camera = Camera()
        self.face_detector = Yolov8.YoloDetector()
        self.model = GhostFace.GhostFaceNetClient()
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
        self.timer.start(30)

    def show_input_dialog(self):
        self.img_name_input, okPressed = QInputDialog.getText(self, "Введите имя", "Имя изображения:")
        if self.current_frame is not None:
            if okPressed and self.img_name_input != '':

                self.append_database()
        else:
            print("Нет доступного кадра для добавления в базу.")

    def append_database(self):
        frame = self.current_frame
        try:
            bbox = self.face_detector.detect_face(self.current_frame)[0]
            if len(bbox) > 0:
                x, y, w, h = bbox
                x = max(0, x)
                y = max(0, y)
                x = min(x, frame.shape[1])
                y = min(y, frame.shape[0])

                if bbox is not None:
                    face_frame = image_utils.crop_face(self.current_frame, bbox)
                    embedding = self.model.forward(image_utils.normalization_frame_tensor(face_frame))
                    self.database.save_embedding_to_database(embedding, self.img_name_input)
                    #optional
                    self.database.save_face_image(frame, self.img_name_input)
            else:
              logger.info("Лицо не найдено в кадре.")

        except Exception as e:
            print(f"Ошибка при добавлении в базу: {e}")

    def update_frame(self):
        ret, frame = self.camera.read_frame()
        ret, frame = image_utils.read_rgb_frame(ret, frame)

        if ret:
            self.current_frame = frame
            bbox = self.face_detector.detect_face(self.current_frame)[0]
            if len(bbox) > 0:
                x, y, w, h = bbox
                x = max(0, x)
                y = max(0, y)
                x = min(x, frame.shape[1])
                y = min(y, frame.shape[0])

                if bbox is not None:
                    face_frame = image_utils.crop_face(self.current_frame, bbox)
                    embedding = self.model.forward(image_utils.normalization_frame_tensor(face_frame))
                    self.name_dist = self.database.verify(embedding).split()[0]
                else:
                    logger.error("Ошибка при извлечении вектора изображения лица.")

                cv2.putText(frame, f"{self.name_dist}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            else:
                logger.warn("No faces on frame")
        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.camera_label.setPixmap(pixmap)

    def initiate_delete_process(self):
        frame = self.current_frame  
        name_to_delete, okPressed = QInputDialog.getText(self, "Введите имя для удаления", "Имя:")
        if okPressed and name_to_delete != '':
            reply = QMessageBox.question(self, 'Подтверждение удаления', 
                                     f"Вы уверены, что хотите удалить {name_to_delete}?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                try:
                    
                    bbox = self.face_detector.detect_face(self.current_frame)[0]
                    if len(bbox) > 0:
                        x, y, w, h = bbox
                        x = max(0, x)
                        y = max(0, y)
                        x = min(x, frame.shape[1])
                        y = min(y, frame.shape[0])

                        if bbox is not None:
                            face_frame = image_utils.crop_face(self.current_frame, bbox)
                            embedding = self.model.forward(image_utils.normalization_frame_tensor(face_frame))
                            self.name_dist = self.database.verify(embedding).split()[0]
                        else:
                            logger.error("Ошибка при извлечении вектора изображения лица.")
                        if self.name_dist != "unknown":
                            self.database.delete_record(name_to_delete) 
                except Exception as e:
                    print(f"Error adding delete operation: {e}") 

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())
    window.database.close()

