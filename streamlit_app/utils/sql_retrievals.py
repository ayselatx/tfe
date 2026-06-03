import sqlite3
import json
from pathlib import Path

DB_PATH = Path("data/app.db")


def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_unimodal_tables():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS unimodal_retrievals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL,
            modality TEXT NOT NULL,      -- 'vision' or 'text'
            dataset TEXT NOT NULL,
            query TEXT NOT NULL,         -- image path or caption
            model TEXT NOT NULL,
            results_json TEXT NOT NULL,  -- JSON list of (item, score)
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()
    conn.close()


def save_unimodal_retrieval(user, modality, dataset, query, model, results):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO unimodal_retrievals (user, modality, dataset, query, model, results_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (user, modality, dataset, query, model, json.dumps(results)),
    )
    conn.commit()
    conn.close()


def load_cached_retrieval(modality, dataset, query, model):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT results_json
        FROM unimodal_retrievals
        WHERE modality = ? AND dataset = ? AND query = ? AND model = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (modality, dataset, query, model),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return json.loads(row[0])


def load_user_retrievals(user, modality=None):
    conn = get_conn()
    cur = conn.cursor()
    if modality:
        cur.execute(
            """
            SELECT id, modality, dataset, query, model, created_at
            FROM unimodal_retrievals
            WHERE user = ?
            AND modality = ?
            ORDER BY created_at DESC
            """,
            (user, modality),
        )
    else:
        cur.execute(
            """
            SELECT id, modality, dataset, query, model, created_at
            FROM unimodal_retrievals
            WHERE user = ?
            ORDER BY created_at DESC
            """,
            (user,),
        )
    rows = cur.fetchall()
    conn.close()
    return rows


def load_retrieval_by_id_all(rid):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT results_json
        FROM retrievals
        WHERE id = ?
    """, (rid,))
    row = cur.fetchone()
    conn.close()
    return {"results" : json.loads(row[0])}


def init_retrieval_tables():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS retrievals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL,
            retrieval_type TEXT NOT NULL,      -- unimodal, fusion, sota, multimodal, rag
            query_type TEXT NOT NULL,          -- i2t, t2i, m2t, m2i, rag
            vision_model TEXT,
            text_model TEXT,
            projection TEXT,
            fusion_operator TEXT,
            dataset TEXT NOT NULL,
            query TEXT NOT NULL,
            results_json TEXT NOT NULL,
            rag_prompt TEXT,
            rag_model TEXT,
            rag_refined_caption TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()

def save_rag_history(user, dataset, image_id, model, prompt, refined_caption):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO retrievals (
            user, retrieval_type, query_type,
            dataset, query, results_json,
            rag_prompt, rag_model, rag_refined_caption
        )
        VALUES (?, 'rag', 'rag', ?, ?, ?, ?, ?, ?)
    """, (
        user,
        dataset,
        str(image_id),
        json.dumps([]),   # no numeric results for RAG
        prompt,
        model,
        refined_caption
    ))
    conn.commit()
    conn.close()


def save_retrieval(user, retrieval_type, query_type,
                   vision_model, text_model, projection, fusion_operator,
                   dataset, query, results):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO retrievals (
            user, retrieval_type, query_type,
            vision_model, text_model, projection, fusion_operator,
            dataset, query, results_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user, retrieval_type, query_type,
        vision_model, text_model, projection, fusion_operator,
        dataset, query, json.dumps(results)
    ))
    conn.commit()
    conn.close()
    
def load_user_retrievals_all(user):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, retrieval_type, query_type,
               vision_model, text_model, projection, fusion_operator,
               dataset, query, created_at,
               rag_prompt, rag_model, rag_refined_caption
        FROM retrievals
        WHERE user = ?
        ORDER BY created_at DESC
    """, (user,))
    rows = cur.fetchall()
    conn.close()
    return rows


def load_retrieval_by_id(rid):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, user, modality, dataset, query, model, results_json, created_at
        FROM unimodal_retrievals
        WHERE id = ?
        """,
        (rid,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row[0],
        "user": row[1],
        "modality": row[2],
        "dataset": row[3],
        "query": row[4],
        "model": row[5],
        "results": json.loads(row[6]),
        "created_at": row[7],
    }

def init_custom_models_table():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS custom_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT,
            name TEXT,
            comment TEXT,
            alpha REAL,
            beta REAL,
            gamma REAL,
            w_r1 REAL,
            w_r5 REAL,
            w_r10 REAL,
            w_faith REAL,
            w_spar REAL,
            w_rank REAL,
            w_comp REAL,
            w_inf REAL,
            w_emb REAL,
            w_mem REAL,
            best_vision_model TEXT,
            best_text_model TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    
def save_custom_model(user, name, comment, weights, best_vision, best_text):
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO custom_models (
            user, name, comment,
            alpha, beta, gamma,
            w_r1, w_r5, w_r10,
            w_faith, w_spar, w_rank, w_comp,
            w_inf, w_emb, w_mem,
            best_vision_model, best_text_model
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """,
    (
        user, name, comment,
        weights["alpha"], weights["beta"], weights["gamma"],
        weights["w_r1"], weights["w_r5"], weights["w_r10"],
        weights["w_faith"], weights["w_spar"], weights["w_rank"], weights["w_comp"],
        weights["w_inf"], weights["w_emb"], weights["w_mem"],
        best_vision, best_text
    ))

    conn.commit()
    conn.close()

def load_user_custom_models(user):
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("""
        SELECT id, name, comment,
               alpha, beta, gamma,
               w_r1, w_r5, w_r10,
               w_faith, w_spar, w_rank, w_comp,
               w_inf, w_emb, w_mem,
               best_vision_model, best_text_model,
               timestamp
        FROM custom_models
        WHERE user = ?
        ORDER BY timestamp DESC
    """, (user,))
    rows = c.fetchall()
    conn.close()
    return rows

def delete_custom_model(model_id):
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("DELETE FROM custom_models WHERE id = ?", (model_id,))
    conn.commit()
    conn.close()

def update_custom_model(model_id, new_name, new_comment):
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("""
        UPDATE custom_models
        SET name = ?, comment = ?
        WHERE id = ?
    """, (new_name, new_comment, model_id))
    conn.commit()
    conn.close()

def anonymize_retrieval(rid):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE retrievals SET user = 'anonymous' WHERE id = ?", (rid,))
    conn.commit()
    conn.close()

def anonymize_unimodal_retrieval(rid):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE unimodal_retrievals SET user = 'anonymous' WHERE id = ?", (rid,))
    conn.commit()
    conn.close()
