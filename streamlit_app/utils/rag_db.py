import sqlite3
import json
import numpy as np

def to_python_types(obj):
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_types(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj


DB_PATH = "data/rag_cache.db"


def init_rag_db():
    conn = sqlite3.connect("data/rag_cache.db")
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS rag_cache (
            image_id INTEGER,
            model_name TEXT,
            prompt TEXT,
            refined_caption TEXT,
            metrics_json TEXT,
            semantic_json TEXT,
            PRIMARY KEY (image_id, model_name, prompt)
        )
    """)

    conn.commit()
    conn.close()


def get_rag_cache(image_id, model_name, prompt):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        SELECT refined_caption, metrics_json, semantic_json
        FROM rag_cache
        WHERE image_id = ? AND model_name = ? AND prompt = ?
    """, (image_id, model_name, prompt))

    row = c.fetchone()
    conn.close()

    if row is None:
        return None

    refined, metrics_json, semantic_json = row

    return {
        "refined": refined,
        "metrics": json.loads(metrics_json),
        "semantic_diff": json.loads(semantic_json)
    }


def insert_rag_cache(image_id, model_name, prompt, result):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        INSERT OR REPLACE INTO rag_cache
        (image_id, model_name, prompt, refined_caption, metrics_json, semantic_json)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        image_id,
        model_name,
        prompt,
        result["refined"],
        json.dumps(to_python_types(result["metrics"])),
        json.dumps(to_python_types(result["semantic_diff"]))

    ))

    conn.commit()
    conn.close()
