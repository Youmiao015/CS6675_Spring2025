#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from io import BytesIO

from search.data_loader import DataLoader
from search.embedding_model import EmbeddingModel
from search.search_engine import SearchEngine

# Paths to pre-built FAISS index and SQLite metadata (vector DB with title and abstract only)
INDEX_FILE = "../data/faiss_index_cosine.bin"
DB_FILE    = "../data/vector_data_all_fields.db"

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         
    allow_credentials=True,
    allow_methods=["*"],         
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10

@app.post("/search")
def search(request: SearchRequest):
    # Validate the query is not empty
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    return search_engine.search(request.query, top_k=request.top_k)

@app.post("/search/aggregate_by_year")
def aggregate_search(request: SearchRequest):
    # Validate that the input query is not empty
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Perform a search with a high top_k value to retrieve many candidate records
    search_results = search_engine.search(request.query, top_k=10000)
    
    # Initialize a dictionary for years 2007 to 2024 with zero count
    year_dict = {str(year): 0 for year in range(2007, 2025)}
    
    # Iterate over each record in the "results" list of the search results
    for record in search_results.get("results", []):
        # Skip any record that contains overall aggregated information
        if "avg_similarity" in record or "result_count" in record:
            continue
        
        # Retrieve the update_date field; expected to be a string starting with a 4-digit year
        update_date = record.get("update_date", "")
        # print("Debug - Record update_date:", update_date)
        
        # Only proceed if update_date has at least 4 characters
        if len(update_date) >= 4:
            try:
                # Extract the first 4 characters to get the year
                year = int(update_date[:4])
                # Increment count if the year is in the desired range (2007 to 2024)
                if 2007 <= year <= 2024:
                    year_dict[str(year)] += 1
            except ValueError:
                continue
    
    # Create a list of counts ordered by year from 2007 to 2024
    counts_list = [year_dict[str(year)] for year in range(2007, 2025)]
    
    # Return a JSON object containing the list of counts for each year
    return {"year_counts": counts_list}

@app.post("/search/aggregate_plot")
def aggregate_plot(request: SearchRequest):
    # Input validation
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Count papers from 2007–2024
    search_results = search_engine.search(request.query, top_k=10000)
    year_dict = {str(y): 0 for y in range(2007, 2025)}
    for rec in search_results.get("results", []):
        update_date = rec.get("update_date", "")
        if len(update_date) >= 4:
            try:
                y = int(update_date[:4])
                if 2007 <= y <= 2024:
                    year_dict[str(y)] += 1
            except ValueError:
                continue

    real_counts = np.array([year_dict[str(y)] for y in range(2007, 2025)],
                           dtype=np.int32)

    # ---------- Predict the value for 2025 ----------
    last3 = real_counts[-3:]  # Years: 2022, 2023, 2024
    if last3.sum() == 0:
        predicted_2025 = 1  # Minimum fallback value to avoid empty chart
    else:
        # Generate 3 random positive numbers and normalize as weights
        w = np.random.rand(3)
        w = w / w.sum()
        predicted_2025 = int(np.round(np.dot(last3, w)))
        predicted_2025 = max(predicted_2025, 1)  # Ensure at least 1

    # ---------- Visualization ----------
    years = list(range(2007, 2026))
    combined_counts = real_counts.tolist() + [predicted_2025]

    max_cnt = max(combined_counts)
    max_cnt = max_cnt if max_cnt > 0 else 1

    # Real values: green gradient; Predicted bar: same color, semi-transparent
    colors_real = [cm.Greens(c / max_cnt) for c in real_counts]
    base = cm.Greens(predicted_2025 / max_cnt)
    color_pred = (base[0], base[1], base[2], 0.5)
    bar_colors = colors_real + [color_pred]

    plt.figure(figsize=(10, 6))
    plt.bar(years, combined_counts, color=bar_colors)
    plt.xlabel("Year")
    plt.ylabel("Number of Papers")
    plt.title("Number of Similar Papers by Year (2007–2025)")
    plt.xticks(years, rotation=45)

    # Add question mark annotation on the 2025 bar
    plt.text(2025, predicted_2025, "?", ha="center", va="bottom",
             fontsize=14, color="red", fontweight="bold")

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")




@app.post("/search/prediction_demo")
def prediction_demo(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    search_results = search_engine.search(request.query, top_k=10000)
    year_dict = {str(year): 0 for year in range(2007, 2025)}
    for record in search_results.get("results", []):
        update_date = record.get("update_date", "")
        if len(update_date) >= 4:
            try:
                year = int(update_date[:4])
                if 2007 <= year <= 2024:
                    year_dict[str(year)] += 1
            except ValueError:
                continue

    raw_counts = np.array([year_dict[str(y)] for y in range(2007, 2025)], dtype=np.float32)
    noise = np.random.uniform(0.3, 0.7, size=raw_counts.shape)
    scaled_counts = (raw_counts * noise).astype(np.int32)

    max_count = scaled_counts.max() if scaled_counts.max() > 0 else 1
    predicted_2025 = int(max_count * 0.5)

    years = list(range(2007, 2026))
    combined_counts = scaled_counts.tolist() + [predicted_2025]

    colors = [cm.Greens(c / max_count) for c in scaled_counts]
    base_color = cm.Greens(predicted_2025 / max_count)
    predicted_color = (base_color[0], base_color[1], base_color[2], 0.5) 
    combined_colors = colors + [predicted_color]

    plt.figure(figsize=(10, 6))
    plt.bar(years, combined_counts, color=combined_colors)
    plt.xlabel("Year")
    plt.ylabel("Number of Papers")
    plt.title("Topic Prediction: Papers by Year (2007‑2025)")
    plt.xticks(years, rotation=45)

    plt.text(2025, predicted_2025, "?", ha="center", va="bottom",
             fontsize=14, color="red", fontweight="bold")

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)