#!/usr/bin/env python3
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        """
        Load a SentenceTransformer model on GPU if available (or CPU otherwise).
        """
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts, show_progress_bar=False):
        """
        Encode one string or a list of strings into a (n×d) numpy.float32 array.
        """
        # Normalize input to list
        inputs = [texts] if isinstance(texts, str) else texts
        embeddings = self.model.encode(inputs, show_progress_bar=show_progress_bar and len(inputs) > 1)
        return np.array(embeddings, dtype='float32')