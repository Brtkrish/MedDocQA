"""
Embedding layer.

This is the real PyTorch/HuggingFace touchpoint in the project: we load a
transformer model directly with sentence-transformers (which wraps a
PyTorch nn.Module under the hood) and run inference ourselves, rather than
calling an embeddings API. This is the piece worth speaking to in an
interview: "I loaded and ran a transformer encoder locally on PyTorch."
"""
from functools import lru_cache
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """
    Load the embedding model once and cache it. Uses GPU if available,
    otherwise falls back to CPU automatically (fine for a small corpus).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(settings.EMBEDDING_MODEL, device=device)
    return model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Encode a batch of texts into embedding vectors.
    Returns a numpy array of shape (len(texts), EMBEDDING_DIM).
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine similarity via dot product
        show_progress_bar=False,
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """Convenience wrapper for embedding a single query string."""
    return embed_texts([query])[0]
