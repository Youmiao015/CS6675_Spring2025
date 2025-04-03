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
        :param threshold: Optional similarity threshold (higher is more similar). 
                          Only results with similarity >= threshold will be returned.
                          When specified, all results meeting the threshold will be returned regardless of top_k.
        :return: A dictionary containing a list of result records and aggregated metrics.
        """
        # Convert query to vector and normalize (as done during index construction)
        query_vector = np.array(self.model.encode(query), dtype='float32').reshape(1, -1)
        norms = np.linalg.norm(query_vector, axis=1, keepdims=True)
        query_vector = query_vector / norms

        # Set a large initial k if threshold is specified to ensure we get enough candidates
        initial_k = 500000 if threshold is not None else top_k
        
        # Retrieve nearest neighbors from FAISS index
        distances, indices = self.index.search(query_vector, initial_k)

        results = []
        for similarity, idx in zip(distances[0], indices[0]):
            # Skip results that don't meet the similarity threshold
            if threshold is not None and similarity < threshold:
                continue
                
            record = self.loader.get_metadata(idx)
            if not record:
                continue
            record['similarity'] = float(similarity)
            results.append(record)
            
            # Stop once we have top_k results (if no threshold is set)
            if threshold is None and len(results) >= top_k:
                break

        avg_similarity = float(np.mean([r['similarity'] for r in results])) if results else None
        return {
            'results': results,
            'aggregated_metrics': {
                'avg_similarity': avg_similarity,
                'result_count': len(results)
            }
        }
