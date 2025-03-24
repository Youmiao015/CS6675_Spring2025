#!/usr/bin/env python3
import numpy as np

class SearchEngine:
    def __init__(self, index, data_loader, embedding_model):
        """
        Initialize SearchEngine with a FAISS index, a DataLoader for metadata, 
        and an embedding model for query vectorization.
        """
        self.index = index
        self.loader = data_loader
        self.model = embedding_model

    def search(self, query: str, top_k: int = 10) -> dict:
        """
        Encode the query text, run a FAISS nearest‑neighbor search, 
        then fetch and return matching records along with similarity scores.
        
        :param query: The user’s input string.
        :param top_k: Number of similar results to return.
        :return: A dictionary containing a list of result records and aggregated metrics.
        """
        # Convert query to vector
        query_vector = np.array(self.model.encode(query), dtype='float32').reshape(1, -1)

        # Retrieve nearest neighbors from FAISS index
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for distance, idx in zip(distances[0], indices[0]):
            record = self.loader.get_metadata(idx)
            if not record:
                continue
            record['distance'] = float(distance)
            results.append(record)

        avg_distance = float(np.mean(distances[0])) if results else None
        return {
            'results': results,
            'aggregated_metrics': {
                'avg_distance': avg_distance,
                'result_count': len(results)
            }
        }
