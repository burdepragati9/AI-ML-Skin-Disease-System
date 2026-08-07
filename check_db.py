import sqlite3
c=sqlite3.connect("database/app.db")
c.row_factory=sqlite3.Row
rows=c.execute("SELECT id,disease,confidence,prediction_source,image_name,image_path FROM searches WHERE image_name LIKE '%129fa90406%' OR image_path LIKE '%129fa90406%'").fetchall()
print("Matched rows for 129fa90406:")
for r in rows:
    print(" ", dict(r))
print("Total searches:", c.execute("SELECT COUNT(*) FROM searches").fetchone()[0])
print("\nLatest 5 searches:")
for r in c.execute("SELECT id,disease,confidence,prediction_source,image_name FROM searches ORDER BY id DESC LIMIT 5").fetchall():
    print(" ", dict(r))
