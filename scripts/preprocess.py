#!/usr/bin/env python3
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Updated DATA_FILE to the JSON file with arXiv metadata
DATA_FILE = 'arxiv_dataset/arxiv-metadata-oai-snapshot.json'
OUTPUT_INDEX = 'faiss_index.bin'
OUTPUT_META = 'papers_meta.pkl'

# Check if the data file exists
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Data file {DATA_FILE} not found. Please ensure the dataset is downloaded and placed correctly.")

print("Loading the SentenceTransformer model on GPU...")
# Load the pre-trained embedding model on GPU
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
embedding_dim = model.get_sentence_embedding_dimension()  # Get the dimension of embeddings

# Create a CPU index first, then transfer it to GPU for accelerated search
cpu_index = faiss.IndexFlatL2(embedding_dim)
gpu_res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)

# List to store paper metadata (you can store only required fields to reduce memory usage)
papers_meta = []

# Set batch size to process a few records at a time
batch_size = 10000
batch_texts = []
processed_count = 0

print("Processing data in batches...")
# Open the JSON file and process line by line (assumes JSON Lines format)
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        paper = json.loads(line)
        # Append the paper metadata (or store only selected fields)
        papers_meta.append(paper)
        # Get the abstract text for vectorization; if missing, use an empty string
        batch_texts.append(paper.get('abstract', ''))
        processed_count += 1
        
        # When batch size is reached, encode the texts and add to the Faiss index
        if processed_count % batch_size == 0:
            print(f"Processing batch {processed_count // batch_size}")
            embeddings = model.encode(batch_texts, show_progress_bar=True)
            embeddings = np.array(embeddings).astype('float32')
            gpu_index.add(embeddings)
            batch_texts = []  # Reset the batch

# Process any remaining texts in the last batch
if batch_texts:
    print("Processing final batch...")
    embeddings = model.encode(batch_texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    gpu_index.add(embeddings)

print("Finished processing all papers.")

# Transfer the GPU index back to CPU before saving
cpu_index_final = faiss.index_gpu_to_cpu(gpu_index)
faiss.write_index(cpu_index_final, OUTPUT_INDEX)
print(f"Faiss index saved to {OUTPUT_INDEX}")

# Save the metadata (papers_meta) to a pickle file for later retrieval
with open(OUTPUT_META, 'wb') as f:
    pickle.dump(papers_meta, f)
print(f"Paper metadata saved to {OUTPUT_META}")