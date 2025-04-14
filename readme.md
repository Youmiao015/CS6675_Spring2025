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
│   ├── vector_data.db                          # SQLite database of paper metadata (indexed by vector_idx)
│   └── faiss_index_cosine.bin                  # FAISS index of paper embeddings
|   └── llm_generated_areas.csv                 # CS research areas generated by Qwen2.5-7B-Instruct given clustered named entities
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
uvicorn api.api:app --reload --host 0.0.0.0 --port 8000
```

## Example Query for search API

You can test the search API using curl:

```bash
curl -s -X POST http://localhost:8000/search \
     -H "Content-Type: application/json" \
     -d '{"query": "machine learning", "top_k": 3}'
```

Sample Output:

```json
{
  "results": [
    {
      "title": "Machine Learning: When and Where the Horses Went Astray?",
      "abstract": "  Machine Learning is usually defined as a subfield of AI, which is busy with\ninformation extraction from raw data sets. Despite of its common acceptance and\nwidespread recognition, this definition is wrong and groundless. Meaningful\ninformation does not belong to the data that bear it. It belongs to the\nobservers of the data and it is a shared agreement and a convention among them.\nTherefore, this private information cannot be extracted from the data by any\nmeans. Therefore, all further attempts of Machine Learning apologists to\njustify their funny business are inappropriate.\n",
      "authors": "Emanuel Diamant",
      "update_date": "2009-11-10",
      "similarity": 0.9022916555404663
    },
    {
      "title": "Proceedings of the 29th International Conference on Machine Learning\n  (ICML-12)",
      "abstract": "  This is an index to the papers that appear in the Proceedings of the 29th\nInternational Conference on Machine Learning (ICML-12). The conference was held\nin Edinburgh, Scotland, June 27th - July 3rd, 2012.\n",
      "authors": "John Langford and Joelle Pineau (Editors)",
      "update_date": "2012-09-18",
      "similarity": 0.901700496673584
    },
    {
      "title": "Classic machine learning methods",
      "abstract": "  In this chapter, we present the main classic machine learning methods. A\nlarge part of the chapter is devoted to supervised learning techniques for\nclassification and regression, including nearest-neighbor methods, linear and\nlogistic regressions, support vector machines and tree-based algorithms. We\nalso describe the problem of overfitting as well as strategies to overcome it.\nWe finally provide a brief overview of unsupervised learning methods, namely\nfor clustering and dimensionality reduction.\n",
      "authors": "Johann Faouzi and Olivier Colliot",
      "update_date": "2023-10-19",
      "similarity": 0.8963815569877625
    }
  ],
  "aggregated_metrics": {
    "avg_similarity": 0.9001245697339376,
    "result_count": 3
  }
}
```

## Example Query for aggregate_by_year API
```bash
curl -s -X POST http://localhost:8000/search/aggregate_by_year \
     -H "Content-Type: application/json" \
     -d '{"query": "machine learning"}'
```

Sample Output:

```json
{"year_counts":[82,30,50,66,67,142,166,251,328,412,499,689,884,1124,1166,1088,1157,1374]}
```

## Example Query for search aggregate plot
```bash
curl -s -X POST http://localhost:8000/search/aggregate_by_year \
     -H "Content-Type: application/json" \
     -d '{"query": "Cloud Computing and Resource Management"}'
```

Sample Output:
![Search Aggregation Plot](/docs/images/aggregate_plot.png)


## Example Query for prediction  plot
```bash
curl -s -X POST http://localhost:8000/search/aggregate_by_year \
     -H "Content-Type: application/json" \
     -d '{"query": "Cloud Computing and Resource Management"}'
```

Sample Output:
![Search Aggregation Plot](/docs/images/prediction_demo.png)

## Frontend
```bash
cd frontend
python -m http.server 8001
```
http://localhost:8001/index.html


# Generate research areas with LLM
This step was performed on PACE ICE. To perform this step, first install required conda environment using `/other_requirements/generate_areas_llm_environment.yml`. Then, you can run the following two commands:
```
python scripts/process_clusters.py \
    --num_areas 3 \
    --input_file data/step5_hdbscan_umap_clusters.json \
    --output_file data/cluster_dict_with_llm_generated_areas.json
python scripts/compile_research_areas.py
```
