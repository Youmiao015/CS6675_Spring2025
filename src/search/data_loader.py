#!/usr/bin/env python3
import os
import json
import sqlite3
import faiss

class DataLoader:
    def __init__(self, index_path: str, db_path: str):
        self.index_path = index_path
        self.db_path = db_path
        self.conn = None

    def load_index(self):
        """Load FAISS index from disk."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        return faiss.read_index(self.index_path)

    def connect_db(self):
        """Open a SQLite connection (lazy)."""
        if self.conn is None:
            if not os.path.exists(self.db_path):
                raise FileNotFoundError(f"DB file not found: {self.db_path}")
            self.conn = sqlite3.connect(self.db_path)
        return self.conn

    # Return metadata of similar papers based on model inputs.
    # Now only returns title and abstract because the DB only stores these fields.
    def get_metadata(self, idx: int) -> dict:
        conn = self.connect_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT metadata FROM papers WHERE vector_idx = ?",
            (int(idx),)  # cast FAISS idx to Python int
        )
        
        row = cur.fetchone()
        if row is None:
            return {}
        record = json.loads(row[0])
        return record