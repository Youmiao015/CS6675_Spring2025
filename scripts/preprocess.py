#!/usr/bin/env python3
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Use the dataset filtered to include only CS papers
DATA_FILE = 'cs_papers.json' ## '../arxiv_dataset/cs_papers.json'
OUTPUT_INDEX = 'faiss_index_cosine.bin'
OUTPUT_META = 'papers_meta.pkl'

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Data file {DATA_FILE} not found. Please run filter_cs.py to generate the CS filtered dataset.")

print("Loading the SentenceTransformer model on GPU...")
# Use a model specifically designed for scientific papers
model = SentenceTransformer('allenai-specter', device='cuda')
embedding_dim = model.get_sentence_embedding_dimension()

# Use inner product (IndexFlatIP) for cosine similarity, so we need to normalize the vectors
cpu_index = faiss.IndexFlatIP(embedding_dim)
gpu_res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)

# Store paper metadata
papers_meta = []

batch_size = 10000
batch_texts = []
processed_count = 0

print("Processing data in batches...")
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        paper = json.loads(line)
        papers_meta.append(paper)
        # Concatenate title and abstract as input text
        text = paper.get('title', '') + " " + paper.get('abstract', '')
        batch_texts.append(text)
        processed_count += 1
        
        if processed_count % batch_size == 0:
            print(f"Processing batch {processed_count // batch_size}")
            embeddings = model.encode(batch_texts, show_progress_bar=True)
            embeddings = np.array(embeddings).astype('float32')
            # Normalize the embedding vectors
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
            gpu_index.add(embeddings)
            batch_texts = []

if batch_texts:
    print("Processing final batch...")
    embeddings = model.encode(batch_texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    gpu_index.add(embeddings)

print("Finished processing all papers.")

# Transfer the index from GPU back to CPU for saving
cpu_index_final = faiss.index_gpu_to_cpu(gpu_index)
faiss.write_index(cpu_index_final, OUTPUT_INDEX)
print(f"Faiss index saved to {OUTPUT_INDEX}")

# Save the paper metadata
with open(OUTPUT_META, 'wb') as f:
    pickle.dump(papers_meta, f)
print(f"Paper metadata saved to {OUTPUT_META}")