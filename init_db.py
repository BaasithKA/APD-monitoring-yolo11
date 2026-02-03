import sqlite3

DB_NAME = 'deteksi.db'
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

# Membuat tabel 'deteksi' dengan kolom 'status'
cursor.execute('''
    CREATE TABLE IF NOT EXISTS deteksi (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        detected_objects TEXT NOT NULL,
        image_path TEXT NOT NULL,
        status TEXT NOT NULL 
    )
''')

print(f"Database '{DB_NAME}' dan tabel 'deteksi' berhasil disiapkan.")
conn.commit()
conn.close()