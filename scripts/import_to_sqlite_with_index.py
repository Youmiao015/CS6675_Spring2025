#!/usr/bin/env python3
import sqlite3, json, os

DB_FILE   = "metadata.db"
META_FILE = "merged.jsonl"
BATCH_SIZE = 10000

# Remove old DB (backup if needed)
if os.path.exists(DB_FILE):
    os.remove(DB_FILE)

conn = sqlite3.connect(DB_FILE)
cur  = conn.cursor()

cur.execute("""
CREATE TABLE papers (
    vector_idx   INTEGER PRIMARY KEY,
    id           TEXT,
    submitter    TEXT,
    authors      TEXT,
    title        TEXT,
    comments     TEXT,
    journal_ref  TEXT,
    doi          TEXT,
    report_no    TEXT,
    categories   TEXT,
    license      TEXT,
    abstract     TEXT,
    versions     TEXT,
    update_date  TEXT,
    authors_parsed TEXT
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
            rec.get("id"),
            rec.get("submitter"),
            rec.get("authors"),
            rec.get("title"),
            rec.get("comments"),
            rec.get("journal-ref"),
            rec.get("doi"),
            rec.get("report-no"),
            rec.get("categories"),
            rec.get("license"),
            rec.get("abstract"),
            json.dumps(rec.get("versions", []), ensure_ascii=False),
            rec.get("update_date"),
            json.dumps(rec.get("authors_parsed", []), ensure_ascii=False)
        ))
        count += 1
        if count % BATCH_SIZE == 0:
            cur.executemany("INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
            conn.commit()
            batch.clear()
            print(f"Inserted {count} rows")

if batch:
    cur.executemany("INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
    conn.commit()

print(f"Total inserted: {count}")
conn.close()
