#!/usr/bin/env python3
import sqlite3, json, os

DB_FILE   = "vector_data_all_fields.db"
META_FILE = "merged_all_fields.jsonl"
BATCH_SIZE = 10000

# Delete the existing database (make a backup first if needed)
if os.path.exists(DB_FILE):
    os.remove(DB_FILE)

conn = sqlite3.connect(DB_FILE)
cur  = conn.cursor()

# Create a table with vector_idx, title, abstract, and update_date
cur.execute("""
CREATE TABLE papers (
    vector_idx   INTEGER PRIMARY KEY,
    metadata     TEXT
)
""")
conn.commit()

count = 0
batch = []
with open(META_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        rec = json.loads(line)
        batch.append((count, json.dumps(rec, ensure_ascii=False)))
        count += 1
        if count % BATCH_SIZE == 0:
            cur.executemany("INSERT INTO papers VALUES (?, ?)", batch)
            conn.commit()
            batch.clear()
            print(f"Inserted {count} rows")

if batch:
    cur.executemany("INSERT INTO papers VALUES (?, ?)", batch)
    conn.commit()

print(f"Total inserted: {count}")
conn.close()