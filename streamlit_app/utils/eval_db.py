import sqlite3
import os


DB = "users.db"   # same DB as auth

def init_eval_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS human_eval (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            image_id INTEGER,
            model TEXT,
            relevance INTEGER,
            fluency INTEGER,
            descriptiveness INTEGER,
            correctness INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def insert_human_eval(username, image_id, model, relevance, fluency, descriptiveness, correctness):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
        INSERT INTO human_eval (username, image_id, model, relevance, fluency, descriptiveness, correctness)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (username, image_id, model, relevance, fluency, descriptiveness, correctness))
    conn.commit()
    conn.close()

def get_human_eval(image_id, model):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
        SELECT relevance, fluency, descriptiveness, correctness
        FROM human_eval
        WHERE image_id=? AND model=?
    """, (image_id, model))
    rows = c.fetchall()
    conn.close()
    return rows

