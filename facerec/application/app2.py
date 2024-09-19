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


logger = Logger()


class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.name_dist = "Who"

        self.current_frame = None

        self.frame_queue = queue.Queue()
        self.db_queue = queue.Queue()  # Очередь для операций с базой данных
        self.process_thread = threading.Thread(target=self.process_frames)
        self.db_thread = threading.Thread(target=self.process_database_queue)  # Поток для работы с БД
        self.running = True
        self.process_thread.start()
        self.db_thread.start()  # Запуск потока для обработки очереди БД

        self.camera = Camera()
        self.face_detector = Yolov8.YoloDetector()
        self.model = GhostFace.GhostFaceNetClient()
        self.database = FaceDatabase()  # Инициализация базы данных

        self.lock = threading.Lock()

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
                # Добавляем данные в очередь для базы данных
                self.append_database()
        else:
            print("Нет доступного кадра для добавления в базу.")

    def append_database(self):
        try:
            bbox = self.face_detector.detect_face(self.current_frame)[0]
            if len(bbox) > 0:
                face_frame = image_utils.crop_face(self.current_frame, bbox)
                face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
                embedding = self.model.forward(image_utils.normalization_frame_tensor(face_frame))

                if embedding is not None: 
                    with self.lock:
                        self.db_queue.put(("add", embedding, face_frame, self.img_name_input))

        except Exception as e:
            print(f"Ошибка при добавлении в базу: {e}")

    def process_frames(self):
        while self.running:
            try:
                current_frame, bbox = self.frame_queue.get()
                if bbox is not None:
                    face_frame = image_utils.crop_face(current_frame, bbox)
                    face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
                    embedding = self.model.forward(image_utils.normalization_frame_tensor(face_frame))
                    if embedding is not None:
                        self.db_queue.put(("verify", embedding,))
                else:
                    logger.error("Ошибка при извлечении вектора изображения лица.")
            except queue.Empty:
                continue

    def process_database_queue(self):
        while self.running:
            try:
                data = self.db_queue.get()
                # for i in data:
                #     print(i)
                if data[0] == "add":
                    embedding = data[1]
                    face_frame = data[2]
                    with sqlite3.connect('facialdb.db') as conn:
                        cursor = conn.cursor()
                        self.database.save_embedding_to_database(conn, cursor, embedding, self.img_name_input)
                        conn.commit()
                    with self.lock:
                        with sqlite3.connect('facialdb.db') as conn: 
                            cursor = conn.cursor()
                            self.database.save_embedding_to_database(conn, cursor, embedding, self.img_name_input)
                            conn.commit()
                    folder_path = 'face_data'
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)

                    face_img_path = os.path.join(folder_path, f'{self.img_name_input}.png')
                    cv2.imwrite(face_img_path, face_frame)

                elif data[0] == "delete":
                    name = data[1]
                    with sqlite3.connect('facialdb.db') as conn:
                        cursor = conn.cursor()
                        self.database.delete_record(conn, cursor, name) 
                elif data[0] == "verify":
                    embedding = data[1]
                    with sqlite3.connect('facialdb.db') as conn:
                        cursor = conn.cursor()
                        # print(f"Embedding Shape:")
                        self.name_dist = self.database.verify(cursor, embedding).split()[0]
                        print(self.name_dist)

            except queue.Empty:
                continue

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

                # Добавляем кадр в очередь для обработки
                self.frame_queue.put((self.current_frame, bbox))

                cv2.putText(frame, f"{self.name_dist}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            else:
                logger.warn("No faces on frame")
        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.camera_label.setPixmap(pixmap)

    def initiate_delete_process(self):
        name_to_delete, okPressed = QInputDialog.getText(self, "Введите имя для удаления", "Имя:")
        if okPressed and name_to_delete != '':
            reply = QMessageBox.question(self, 'Подтверждение удаления', 
                                     f"Вы уверены, что хотите удалить {name_to_delete}?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                try:
                    # Добавляем операцию удаления в очередь
                    self.db_queue.put(("delete", name_to_delete))
                except Exception as e:
                    print(f"Error adding delete operation: {e}")

    def closeEvent(self, event):  
        self.running = False  
        self.process_thread.join()  
        event.accept()  

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())
