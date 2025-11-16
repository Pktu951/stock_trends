import sqlite3
import os

DB_FOLDER = r"C:\Users\Lukasz\Desktop\stock_project\data"
DB_FILE = os.path.join(DB_FOLDER, "stocks.db")

os.makedirs(DB_FOLDER, exist_ok=True)

def get_connection():
    conn = sqlite3.connect(DB_FILE)
    return conn

def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            rf REAL,
            lstm REAL
        )
    """)

    conn.commit()
    conn.close()