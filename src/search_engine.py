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

    def search(self, query: str, top_k: int = 10, threshold: float = None) -> dict:
        """
        Encode the query text, run a FAISS nearest‑neighbor search, 
        then fetch and return matching records along with similarity scores.
        
        :param query: The user's input string.
        :param top_k: Number of similar results to return. This is ignored if threshold is specified.
        :param threshold: Optional similarity threshold (lower is more similar). 
                         Only results with distance <= threshold will be returned.
                         When specified, all results meeting the threshold will be returned regardless of top_k.
        :return: A dictionary containing a list of result records and aggregated metrics.
        """
        # Convert query to vector
        query_vector = np.array(self.model.encode(query), dtype='float32').reshape(1, -1)

        # Set a large initial k if threshold is specified to ensure we get enough candidates
        # We need a large enough value to potentially find all matches that meet the threshold
        initial_k = 500000 if threshold is not None else top_k
        
        # Retrieve nearest neighbors from FAISS index
        distances, indices = self.index.search(query_vector, initial_k)

        results = []
        for distance, idx in zip(distances[0], indices[0]):
            # Skip results that don't meet the threshold
            if threshold is not None and distance > threshold:
                continue
                
            record = self.loader.get_metadata(idx)
            if not record:
                continue
            record['distance'] = float(distance)
            results.append(record)
            
            # Stop once we have top_k results (if no threshold is set)
            if threshold is None and len(results) >= top_k:
                break


        avg_distance = float(np.mean([r['distance'] for r in results])) if results else None
        return {
            'results': results,
            'aggregated_metrics': {
                'avg_distance': avg_distance,
                'result_count': len(results)
            }
        }
