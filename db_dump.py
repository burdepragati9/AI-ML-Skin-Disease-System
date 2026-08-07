import sqlite3
from pathlib import Path

conn = sqlite3.connect('database/app.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

cur.execute('SELECT id, doctor_id, disease, image_path, image_name, confidence, prediction_source, created_at FROM searches ORDER BY id DESC LIMIT 10')
rows = cur.fetchall()

lines = []
for r in rows:
    lines.append(str(dict(r)))

Path('db_dump.txt').write_text('\n'.join(lines), encoding='utf-8')
conn.close()
print('done')