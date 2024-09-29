import sqlite3
import os
import cv2
import numpy as np
import pandas as pd

from commons.logger import Logger

logger = Logger()

class FaceDatabase:
    def __init__(self):
        self.db_name = 'facialdb.db'
        self.create_table()

    def create_table(self):
        if not os.path.exists(self.db_name):
            logger.info(f"Создание новой базы данных: {self.db_name}")
            self.conn = sqlite3.connect('facialdb.db')
            self.cursor = self.conn.cursor()
            self.cursor.execute('''create table if not exists face_meta (ID INT primary key, IMG_NAME VARCHAR(10), EMBEDDING BLOB)''')
            self.conn.commit()
        else:
            self.conn = sqlite3.connect('facialdb.db')
            self.cursor = self.conn.cursor()
            logger.info(f"База данных уже существует: {self.db_name}")

    def save_embedding_to_database(self, embedding, img_name):
        if embedding is not None:
            try:
                self.cursor.execute('''SELECT MAX(ID) FROM face_meta''')
                last_id = self.cursor.fetchone()[0] 
                new_id = 1 if last_id is None else last_id + 1

                instance = {
                    "img_name": img_name,
                    "embedding": embedding
                }

                img_name = instance["img_name"]
                embeddings = instance["embedding"]

                insert_statement = "INSERT INTO face_meta (ID, IMG_NAME, EMBEDDING) VALUES (?, ?, ?)"
                insert_args = (new_id, img_name, embeddings.tobytes())
                self.cursor.execute(insert_statement, insert_args)

                self.conn.commit()
            except Exception as e:
                logger.error(f"Error adding an embedding to the database: {str(e)}")

    #save crop face (optional)
    def save_face_image(self, face_frame, name):
        folder_path = 'face_data'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            logger.info(f"Created directory: {folder_path}")
        else:
            logger.info(f"Directory already exists: {folder_path}")

        face_img_path = os.path.join(folder_path, f'{name}.png')
        cv2.imwrite(face_img_path, face_frame)

    def verify(self, target_embedding):
        
        def EuclideanDistance(row):
            source = np.array(row['embedding'])
            target = np.array(row['target'])

            distance = (source - target)
            return np.sqrt(np.sum(np.multiply(distance, distance)))

        def CosineDistance(row):
            source = np.array(row['embedding'])
            target = np.array(row['target'])

            return (np.sum(np.multiply(target, source))) / (np.sqrt(np.sum(np.multiply(target, target))) * np.sqrt(np.sum(np.multiply(source, source))))

        def GhostDistance(row):
            source = np.array(row['embedding'])
            target = np.array(row['target'])

            if source.shape != target.shape:
                raise ValueError("Shape mismatch between source and target embeddings")
            
            return np.sum(np.multiply(target, source))

        select_statement = "select img_name, embedding from face_meta"
        results = self.cursor.execute(select_statement)
        instances = []
        for result in results:
            img_name = result[0]
            embedding_bytes = result[1]
            embedding = np.frombuffer(embedding_bytes, dtype = 'float32')

            instance = []
            instance.append(img_name)
            instance.append(embedding)
            instances.append(instance)
            # print("Dataframe ", embedding)

        # print("Emb ", target_embedding[0])
        result_df = pd.DataFrame(instances, columns = ["img_name", "embedding"])
        # print(result_df.head())
        target_duplicated = np.array([target_embedding,]*result_df.shape[0])
        result_df['target'] = target_duplicated.tolist()

        result_df['distance'] = result_df.apply(GhostDistance, axis = 1)
        result_df = result_df[result_df['distance'] >= 0.25]
        result_df = result_df.sort_values(by = ['distance'], ascending=False).reset_index(drop = True)
        print(result_df.head())
        if (len(result_df) > 0):
            return result_df.iloc[0]['img_name'] + " " + str(result_df.iloc[0]['distance'])
            #return result_df.iloc[0]['img_name']
        else:
            return "Unknown"

    def delete_record(self, name):
        self.cursor.execute("SELECT * FROM face_meta WHERE IMG_NAME = ?", (str(name),))
        records = self.cursor.fetchall()
        print(f"Found records: {records}")
        try:
            self.cursor.execute("DELETE FROM face_meta WHERE IMG_NAME = ?", (str(name),))
            deleted_count = self.cursor.rowcount  # Получаем количество удаленных строк
            print(f"Deleted {deleted_count} record(s) with name: {name}")
            self.conn.commit()
            # print(name)

        except Exception as e:
                logger.error(f"Error deleting an embedding to the database: {str(e)}")

    def close(self):
        self.conn.close()
