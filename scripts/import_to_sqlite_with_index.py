#!/usr/bin/env python3
import sqlite3, json, os

DB_FILE   = "vector_data.db"
META_FILE = "merged_title_abstract.jsonl"
BATCH_SIZE = 10000

# Delete the existing database (make a backup first if needed)
if os.path.exists(DB_FILE):
    os.remove(DB_FILE)

conn = sqlite3.connect(DB_FILE)
cur  = conn.cursor()

# Create a table containing only vector_idx, title, and abstract
cur.execute("""
CREATE TABLE papers (
    vector_idx   INTEGER PRIMARY KEY,
    title        TEXT,
    abstract     TEXT
)
""")
conn.commit()

count = 0
batch = []
with open(META_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        rec = json.loads(line)
        batch.append((
            count,
            rec.get("title", ""),
            rec.get("abstract", "")
        ))
        count += 1
        if count % BATCH_SIZE == 0:
            cur.executemany("INSERT INTO papers VALUES (?,?,?)", batch)
            conn.commit()
            batch.clear()
            print(f"Inserted {count} rows")

if batch:
    cur.executemany("INSERT INTO papers VALUES (?,?,?)", batch)
    conn.commit()

print(f"Total inserted: {count}")
conn.close()