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
    allow_origins=["*"],  # You may restrict this to your specific front-end domains if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        print("Debug - Record update_date:", update_date)
        
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
    # Validate input query
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Reuse the aggregation logic from the previous endpoint
    search_results = search_engine.search(request.query, top_k=10000)
    year_dict = {str(year): 0 for year in range(2007, 2025)}
    for record in search_results.get("results", []):
        update_date = record.get("update_date", "")
        print("Debug - Record update_date:", update_date)
        if len(update_date) >= 4:
            try:
                year = int(update_date[:4])
                if 2007 <= year <= 2024:
                    year_dict[str(year)] += 1
            except ValueError:
                continue

    # Create a list of counts for years 2007 to 2024
    counts_list = [year_dict[str(year)] for year in range(2007, 2025)]
    years = list(range(2007, 2025))
    counts = np.array(counts_list, dtype=np.int32)
    max_count = counts.max() if counts.max() > 0 else 1

    # Create a blue gradient: higher counts yield a darker blue.
    colors = [cm.Blues(count / max_count) for count in counts]
    
    # Create a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(years, counts, color=colors)
    plt.xlabel("Year")
    plt.ylabel("Number of Papers")
    plt.title("Number of Similar Papers by Year (2007-2024)")
    plt.xticks(years)
    
    # Save the plot to a BytesIO buffer
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    
    # Return the image as PNG content
    return Response(content=buf.getvalue(), media_type="image/png")

@app.post("/search/prediction_demo")
def prediction_demo(request: SearchRequest):
    # Validate input query
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Perform search to retrieve candidate records (top_k set high to capture many candidates)
    search_results = search_engine.search(request.query, top_k=10000)
    
    # Initialize dictionary for years 2007 to 2024 with count 0
    year_dict = {str(year): 0 for year in range(2007, 2025)}
    
    # Aggregate counts from the search results using the update_date field
    for record in search_results.get("results", []):
        update_date = record.get("update_date", "")
        # Print debug info if needed
        print("Debug - Record update_date:", update_date)
        if len(update_date) >= 4:
            try:
                year = int(update_date[:4])
                if 2007 <= year <= 2024:
                    year_dict[str(year)] += 1
            except ValueError:
                continue

    # Create a list of counts for years 2007 to 2024
    counts_list = [year_dict[str(year)] for year in range(2007, 2025)]
    
    # We'll extend the x-axis for years 2007 to 2025:
    years = list(range(2007, 2026))
    
    # For the prediction demo, assign a placeholder height for year 2025.
    # Here we use half of the maximum real count (or 1 if all are zero) as the demo value.
    real_counts = np.array(counts_list, dtype=np.int32)
    max_count = real_counts.max() if real_counts.max() > 0 else 1
    predicted_value = max_count * 0.5  # Demo predicted height for 2025
    
    # Combine the real counts with the demo predicted value for 2025
    combined_counts = counts_list + [predicted_value]
    
    # Create colors for bars: for years 2007-2024, use colors from a blue gradient;
    # for 2025, use the same base color with half transparency.
    colors = []
    for count in counts_list:
        # Normalize count against max_count for colormap scaling
        colors.append(cm.Blues(count / max_count))
    # For 2025, use the predicted_value with alpha forced to 0.5
    base_color = cm.Blues(predicted_value / max_count)
    predicted_color = (base_color[0], base_color[1], base_color[2], 0.5)
    combined_colors = colors + [predicted_color]
    
    # Create the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(years, combined_counts, color=combined_colors)
    plt.xlabel("Year")
    plt.ylabel("Number of Papers")
    plt.title("Topic Prediction: Papers by Year (2007-2025)")
    plt.xticks(years)
    
    # Annotate the prediction bar (2025) with a question mark
    plt.text(2025, predicted_value, "?", ha='center', va='bottom', fontsize=14, color='red')
    
    # Save the plot to a buffer and return as PNG image
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    
    return Response(content=buf.getvalue(), media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)