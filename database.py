import sqlite3
import json
from flask import Flask
import numpy as np
from config import DB_PATH
import cv2 as cv

app = Flask(__name__)
database_path = DB_PATH

def init_db() :
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT,
                    name TEXT,
                    landmarks TEXT,
                    bbox TEXT,
                    embedding TEXT,
                    image BLOB
                 )''')
    conn.commit()
    conn.close()

def insert_face(student_id, name, landmarks, bbox, embedding, image):
    _, buffer = cv.imencode('.jpg', image)
    image_bytes = buffer.tobytes()

    landmarks_str = json.dumps(landmarks.tolist())
    bbox_str = json.dumps(bbox)
    embedding_str = json.dumps(embedding.tolist())

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("insert or replace into faces (student_id, name, landmarks, bbox, embedding, image)"
                   "values (?, ?, ?, ?, ?, ?)",( student_id, name, landmarks_str, bbox_str, embedding_str, image_bytes))
    conn.commit()
    conn.close()

def check_database_contents(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT COUNT(*) FROM faces")
        count = cursor.fetchone()[0]

        if count == 0:
            print("📂 Database is empty! There are no records in the table 'faces'.")
        else:
            print(f"✅ Database contains {count} records in table 'faces'.")
            # In ra một vài dòng đầu (ví dụ 5 dòng)
            cursor.execute("SELECT student_id, name, image FROM faces")
            rows = cursor.fetchall()

            for idx, row in enumerate(rows) :
                student_id, name,img_bytes = row[0], row[1], row[2]
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv.imdecode(nparr, cv.IMREAD_COLOR)
                print("🧑 Student ID: {}, Name: {}".format(student_id, name))
                cv.imshow("{} - {}".format(student_id, name), img)
                cv.waitKey(0)

            cv.destroyAllWindows()
    except sqlite3.Error as e:
        print("❌ Error while checking database:", e)
    finally:
        conn.close()