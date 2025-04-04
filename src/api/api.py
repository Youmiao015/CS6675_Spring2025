#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from search.data_loader import DataLoader
from search.embedding_model import EmbeddingModel
from search.search_engine import SearchEngine

# Paths to pre‑built FAISS index and SQLite metadata
INDEX_FILE = "../data/faiss_index.bin"
DB_FILE    = "../data/metadata.db"

# Initialize DataLoader (no full metadata load)
loader = DataLoader(index_path=INDEX_FILE, db_path=DB_FILE)

try:
    index = loader.load_index()
except Exception as e:
    raise RuntimeError(f"Failed to load FAISS index: {e}")

# Initialize embedding model (auto‑select GPU if available)
embedding_model = EmbeddingModel(model_name="all-MiniLM-L6-v2")

# Initialize SearchEngine with DataLoader (not a full metadata list)
search_engine = SearchEngine(index=index, data_loader=loader, embedding_model=embedding_model)

app = FastAPI()

# 添加 CORS 中间件，允许所有来源的请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # 可根据需要指定具体域名，例如 ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],         # 允许所有请求方法，如 GET, POST, OPTIONS 等
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10

@app.post("/search")
def search(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    return search_engine.search(request.query, top_k=request.top_k)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)