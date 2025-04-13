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
        # Convert the query into a vector and normalize it (same as during index construction)
        query_vector = np.array(self.model.encode(query), dtype='float32').reshape(1, -1)
        norms = np.linalg.norm(query_vector, axis=1, keepdims=True)
        query_vector = query_vector / norms

        # Use a large initial k if a threshold is specified to ensure enough candidates are retrieved
        initial_k = 500000 if threshold is not None else top_k

        # Search for nearest neighbors in the FAISS index
        distances, indices = self.index.search(query_vector, initial_k)

        results = []
        for similarity, idx in zip(distances[0], indices[0]):
            # Skip entries that don't meet the similarity threshold
            if threshold is not None and similarity < threshold:
                continue

            record = self.loader.get_metadata(idx)
            if not record:
                continue

            # Build a result record containing only the required fields
            result_item = {
                'title': record.get('title', ''),
                'abstract': record.get('abstract', ''),
                # Adjust based on your data structure: check for "author" or "authors" field
                'authors': record.get('authors', '') or record.get('author', ''),
                'update_date': record.get('update_date', '')
            }
            result_item['similarity'] = float(similarity)
            results.append(result_item)

            # Stop once top_k results have been collected (if no threshold is defined)
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
