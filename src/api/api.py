#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from search.data_loader import DataLoader
from search.embedding_model import EmbeddingModel
from search.search_engine import SearchEngine

# Paths to pre-built FAISS index and SQLite metadata (vector DB with title and abstract only)
INDEX_FILE = "../data/faiss_index_cosine.bin"
DB_FILE    = "../data/vector_data.db"

# Initialize DataLoader
loader = DataLoader(index_path=INDEX_FILE, db_path=DB_FILE)

try:
    index = loader.load_index()
except Exception as e:
    raise RuntimeError(f"Failed to load FAISS index: {e}")

# Initialize embedding model using "allenai-specter" (specialized for scientific papers)
embedding_model = EmbeddingModel(model_name="allenai-specter")

# Initialize SearchEngine with DataLoader and embedding model
search_engine = SearchEngine(index=index, data_loader=loader, embedding_model=embedding_model)

app = FastAPI()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10

@app.post("/search")
def search(request: SearchRequest):
    # Validate the query is not empty
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    return search_engine.search(request.query, top_k=request.top_k)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)