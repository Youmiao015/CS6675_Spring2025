from data_loader import DataLoader
from embedding_model import EmbeddingModel
from search_engine import SearchEngine
import pandas as pd
import json
INDEX_FILE = "../data/faiss_index.bin"
DB_FILE    = "../data/metadata.db"

loader = DataLoader(index_path=INDEX_FILE, db_path=DB_FILE)

try:
    index = loader.load_index()
except Exception as e:
    raise RuntimeError(f"Failed to load FAISS index: {e}")

embedding_model = EmbeddingModel(model_name="all-MiniLM-L6-v2")

search_engine = SearchEngine(index=index, data_loader=loader, embedding_model=embedding_model)

if __name__ == "__main__":
    import argparse

    # Set up command line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--th', type=float, default=0.5)
    args = parser.parse_args()

    # Get the threshold from command line arguments
    threshold = args.th

    with open('../topics/openalex_topics.json', 'r') as f:
        topic_list = json.load(f)

    dataset = pd.DataFrame(columns=['topic', 'paper_id', 'distance', 'updated_at'])

    results_list = []

    from tqdm import tqdm
    for topic in tqdm(topic_list, desc="Processing topics"):
        res = search_engine.search(topic, threshold=threshold)
        for result in res['results']:
            results_list.append({
                'topic': topic,
                'id': result['id'],
                'distance': result['distance'],
                'update_date': result['update_date']
            })

    output_file = 'full_dataset.csv'
    dataset = pd.DataFrame(results_list)

    dataset.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")

    dataset['year'] = pd.to_datetime(dataset['update_date']).dt.year
    
    agg_table = pd.pivot_table(
        dataset,
        values='id',
        index='topic',
        columns='year',
        aggfunc='count',
        fill_value=0
    )
    
    agg_table.to_csv('topic_year_counts.csv')
    print(f"Topic-year counts saved to topic_year_counts.csv")
