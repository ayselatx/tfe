import sqlite3
import json
from pathlib import Path

import numpy as np

DB_PATH = Path("data/app.db")


def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_fusion_tables():
    conn = get_conn()
    cur = conn.cursor()

    # 1. Create table if missing
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS fusion_clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            vision_model TEXT NOT NULL,
            text_model TEXT NOT NULL,
            projection TEXT NOT NULL,
            num_images INTEGER NOT NULL,
            n_clusters INTEGER NOT NULL,
            labels_json TEXT NOT NULL,
            centroids_json TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    # 2. Add mode column if missing
    cur.execute("PRAGMA table_info(fusion_clusters)")
    cols = [row[1] for row in cur.fetchall()]

    if "mode" not in cols:
        cur.execute("ALTER TABLE fusion_clusters ADD COLUMN mode TEXT")
        cur.execute("UPDATE fusion_clusters SET mode = 'images' WHERE mode IS NULL")

    conn.commit()
    conn.close()



def save_fusion_clusters(
    username,
    vision_model,
    text_model,
    projection,
    num_images,
    n_clusters,
    labels,
    centroids,
    mode,   # NEW
):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO fusion_clusters
        (username, vision_model, text_model, projection,
         num_images, n_clusters, mode, labels_json, centroids_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            username,
            vision_model,
            text_model,
            projection,
            num_images,
            n_clusters,
            mode,   # NEW
            json.dumps(list(map(int, labels))),
            json.dumps(centroids.tolist()),
        ),
    )
    conn.commit()
    conn.close()

def load_fusion_clusters(
    vision_model,
    text_model,
    projection,
    num_images,
    n_clusters,
    mode,   # NEW
):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT labels_json, centroids_json
        FROM fusion_clusters
        WHERE vision_model = ?
        AND text_model = ?
        AND projection = ?
        AND num_images = ?
        AND n_clusters = ?
        AND mode = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (vision_model, text_model, projection, num_images, n_clusters, mode),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    labels = np.array(json.loads(row[0]), dtype=int)
    centroids = np.array(json.loads(row[1]))
    return labels, centroids



def load_user_fusion_history(username):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, vision_model, text_model, projection,
               num_images, n_clusters, created_at
        FROM fusion_clusters
        WHERE username = ?
        ORDER BY created_at DESC
        """,
        (username,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows
