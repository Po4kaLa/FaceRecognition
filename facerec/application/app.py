import os
import sys
import cProfile
import pstats
import io
sys.path.append("C:/Users/Po4ka/UPPRPO/FaceRecognition/FaceRecognition/facerec/")
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

import sqlalchemy as sa

logger = Logger()

class FaceRecognitionApp(QMainWindow):
    def __init__(self): 
        super().__init__() 

        self.name_dist = "Unknown"
        self.current_frame = None

        # self.frame_queue = queue.Queue()  
        # self.process_thread = threading.Thread(target=self.process_frames)
        # self.running = True
        # self.process_thread.start() 
        #  self.frame_queue = queue.Queue()   
        self.db_queue = queue.Queue()  # Очередь для операций с базой данных
        self.process_thread = threading.Thread(target=self.process_frames) 
        self.db_thread = threading.Thread(target=self.process_database)  # Поток для работы с БД
        self.running = True 
        self.process_thread.start()   
        self.db_thread.start()  # Запуск потока для работы с БД 

        self.camera = Camera()
        # self.camera = cv2.VideoCapture(0)

        self.face_detector = Yolov8.YoloDetector()
        self.model = GhostFace.GhostFaceNetClient()

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
            print("Нет доступного кадра для добавления в базу.") 
             
    def append_database(self): 
        try: 
            bbox = self.face_detector.detect_face(self.current_frame)[0] 
            if len(bbox) > 0: 
                face_frame = image_utils.crop_face(self.current_frame, bbox) 
                embedding = self.model.forward(image_utils.normalization_frame_tensor(face_frame)) 
                if embedding is not None:
                    self.db_queue.put((embedding, face_frame, self.img_name_input))
        
        except Exception as e:
            print(f"Ошибка при добавлении в базу: {e}")

    def process_database(self): 
        while self.running: 
            try: 
                embedding, face_frame, name = self.db_queue.get(timeout=1)
                with sqlite3.connect('your_database.db') as conn: 
                    cursor = conn.cursor()
                    cursor.execute("INSERT INTO your_table (embedding, name) VALUES (?, ?)", (embedding, name))
                    conn.commit()
            except queue.Empty: 
                continue 

    # def show_input_dialog(self):
    #     self.img_name_input, okPressed = QInputDialog.getText(self, "Введите имя", "Имя изображения:")
    #     if self.current_frame is not None:
    #         if okPressed and self.img_name_input != '':
    #             self.append_database()
    #     else:
    #         print("There is no frame available to add to the database.")
            
    # def append_database(self):
    #     try:
    #         bbox = self.face_detector.detect_face(self.current_frame)[0]
    #         if (len(bbox) > 0):
    #             face_frame = image_utils.crop_face(self.current_frame, bbox)
    #             embedding = self.model.forward(image_utils.normalization_frame_tensor(face_frame))
    #             self.database.save_embedding_to_database(embedding, self.img_name_input)
    #             self.database.save_face_image(face_frame, self.img_name_input)
    #     finally:
    #         self.append_thread_running = False

    def process_frames(self):
        while self.running:
            try:
                current_frame, bbox = self.frame_queue.get(timeout=1)  
                if bbox is not None:
                    face_frame = image_utils.crop_face(current_frame, bbox)
                    embedding = self.model.forward(image_utils.normalization_frame_tensor(face_frame))
                    if embedding is not None:
                        self.name_dist = self.database.verify(embedding).split()
                else:
                    logger.error("Error when retrieving a vector image of a face.")
            except queue.Empty:
                continue

    def update_frame(self):
        ret, frame = self.camera.read_frame()
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
                self.frame_queue.put((self.current_frame, bbox))
                cv2.putText(frame, self.name_dist[0] + " " + self.name_dist[1], (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                #cv2.putText(frame, name_dist[0], (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            else:
                logger.warn("No faces on frame")
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.camera_label.setPixmap(pixmap)

    def initiate_delete_process(self):
        bbox = self.face_detector.detect_face(self.current_frame)[0]
        if (len(bbox) > 0):
                face_frame = image_utils.crop_face(self.current_frame, bbox)
                embedding = self.model.forward(image_utils.normalization_frame_tensor(face_frame))
                if embedding is not None:
                    self.name_dist = self.database.verify(embedding).split()
                else:
                    logger.error("Error when retrieving a vector image of a face.")
                if self.name_dist[0] != "unknown":
                    self.confirm_delete(self.name_dist[0])
        else:
                logger.warn("No faces on frame")

    def confirm_delete(self, name):
        reply = QMessageBox.question(self, 'Подтверждение удаления', 
                                     f"Вы уверены, что хотите удалить {name}?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.database.delete_record(name) 
    
    def closeEvent(self, event):  
        self.running = False  
        self.process_thread.join()  
        event.accept()  
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())

