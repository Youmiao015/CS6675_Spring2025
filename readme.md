# CS6675 Spring 2025 — Paper Trend Search API

A FastAPI based service that returns semantically similar arXiv papers for any user query.

## Overview

This project implements a semantic search API over arXiv metadata using sentence embeddings and FAISS for fast approximate nearest neighbor (ANN) search.

## Project Structure

```
CS6675_Spring2025/
├── arxiv_dataset/                         # Raw arXiv metadata JSON snapshot (input source)
│   └── arxiv-metadata-oai-snapshot.json        # Original arXiv JSON export
├── data/                                  # Preprocessed output used at runtime
│   ├── merged.jsonl                            # All paper records in JSON Lines format
│   ├── metadata.db                             # SQLite database of paper metadata (indexed by vector_idx)
│   └── faiss_index.bin                         # FAISS index of paper embeddings
├── scripts/                               # Data preparation and debug utilities
│   ├── gen_requirements.py                     # Auto-generate minimal requirements.txt
│   ├── import_to_sqlite_with_index.py          # Build SQLite DB and vector_idx mapping
│   ├── merge_meta.py                           # Merge multiple metadata sources
│   ├── preprocess.py                           # Preprocess raw arXiv JSON into usable format
│   ├── test.py                                 # General test or debug script
├── src/                                   # Core service modules
│   ├── data_loader.py                          # Lazy loader: FAISS index + metadata lookup
│   ├── embedding_model.py                      # Query embedding (SentenceTransformer)
│   ├── search_engine.py                        # FAISS search + metadata retrieval
│   └── api.py                                  # FastAPI REST endpoint (/search)
└── requirements.txt                       # Minimal dependencies for deployment


```

## Setup

Install dependencies and set up a virtual environment:

```bash
sudo apt update
sudo apt install -y python3 python3-venv sqlite3

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Data Preparation

You can prepare data using one of the following approaches:

Approach 1:  
Manually download and place the required data in the `data/` and `/arxiv_dataset/` directories, following the README instructions inside those folders.

Approach 2:  
Use the preprocessing scripts provided:

```bash
cd scripts
python preprocess.py
python merge_meta.py
python import_to_sqlite_with_index.py
```

## Run the API Server

```bash
cd src
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## Example Query

You can test the search API using curl:

```bash
curl -s -X POST http://localhost:8000/search \
     -H "Content-Type: application/json" \
     -d '{"query": "machine learning", "top_k": 5}'
```

Sample Output:

```json
{
  "results": [
    {
      "id": "1703.05298",
      "title": "Neural Networks for Beginners. A fast implementation in Matlab, Torch,\n  TensorFlow",
      "abstract": "  This report provides an introduction to some Machine Learning tools within\nthe most common development environments. It mainly focuses on practical\nproblems, skipping any theoretical introduction. It is oriented to both\nstudents trying to approach Machine Learning and experts looking for new\nframeworks.\n",
      "distance": 0.7146506309509277
    },
    {
      "id": "0810.4752",
      "title": "Statistical Learning Theory: Models, Concepts, and Results",
      "abstract": "  Statistical learning theory provides the theoretical basis for many of\ntoday's machine learning algorithms. In this article we attempt to give a\ngentle, non-technical overview over the key ideas and insights of statistical\nlearning theory. We target at a broad audience, not necessarily machine\nlearning researchers. This paper can serve as a starting point for people who\nwant to get an overview on the field before diving into technical details.\n",
      "distance": 0.7291119694709778
    },
    {
      "id": "2302.05449",
      "title": "Heckerthoughts",
      "abstract": "  This manuscript is technical memoir about my work at Stanford and Microsoft\nResearch. Included are fundamental concepts central to machine learning and\nartificial intelligence, applications of these concepts, and stories behind\ntheir creation.\n",
      "distance": 0.7660136222839355
    },
    {
      "id": "2409.10304",
      "title": "Spiers Memorial Lecture: How to do impactful research in artificial\n  intelligence for chemistry and materials science",
      "abstract": "  Machine learning has been pervasively touching many fields of science.\nChemistry and materials science are no exception. While machine learning has\nbeen making a great impact, it is still not reaching its full potential or\nmaturity. In this perspective, we first outline current applications across a\ndiversity of problems in chemistry. Then, we discuss how machine learning\nresearchers view and approach problems in the field. Finally, we provide our\nconsiderations for maximizing impact when researching machine learning for\nchemistry.\n",
      "distance": 0.7806453704833984
    },
    {
      "id": "1405.1304",
      "title": "Application of Machine Learning Techniques in Aquaculture",
      "abstract": "  In this paper we present applications of different machine learning\nalgorithms in aquaculture. Machine learning algorithms learn models from\nhistorical data. In aquaculture historical data are obtained from farm\npractices, yields, and environmental data sources. Associations between these\ndifferent variables can be obtained by applying machine learning algorithms to\nhistorical data. In this paper we present applications of different machine\nlearning algorithms in aquaculture applications.\n",
      "distance": 0.8109275102615356
    }
  ],
  "aggregated_metrics": {
    "avg_distance": 0.760269820690155,
    "result_count": 5
  }
}
```

# Generate research areas with LLM
This step was performed on PACE ICE. To perform this step, first install required conda environment using `/other_requirements/generate_areas_llm_environment.yml`. Then, you can run the following two commands:
```
python scripts/process_clusters.py \
    --num_areas 3 \
    --input_file data/step5_hdbscan_umap_clusters.json \
    --output_file data/cluster_dict_with_llm_generated_areas.json
python scripts/compile_research_areas.py
```