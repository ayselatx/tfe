import streamlit as st
import hashlib
import sqlite3
import os

DB = "users.db"

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT,
            email TEXT
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password, email):
    if not username or not password:
        return False

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users VALUES (?, ?, ?)",
                  (username, hash_password(password), email))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()

def authenticate(username, password):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    return row and row[0] == hash_password(password)

def check_login_status():
    return st.session_state.get("logged_in", False)

def update_username(old_username, new_username):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    try:
        c.execute("UPDATE users SET username=? WHERE username=?", (new_username, old_username))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()

def update_email(username, new_email):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    try:
        c.execute("UPDATE users SET email=? WHERE username=?", (new_email, username))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()

def update_password(username, new_password):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    try:
        hashed = hash_password(new_password)
        c.execute("UPDATE users SET password=? WHERE username=?", (hashed, username))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()
