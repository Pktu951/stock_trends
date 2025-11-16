import sqlite3

def get_connection():
    conn = sqlite3.connect("stocks.db")
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