"""One shared, locally-run embedding model (fastembed / ONNX) for retrieval and the cache."""
from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings

from . import config


@lru_cache(maxsize=1)
def get_model() -> TextEmbedding:
    return TextEmbedding(model_name=config.EMBEDDING_MODEL, cache_dir=config.MODEL_CACHE_DIR)


def embed(texts: list[str]) -> np.ndarray:
    """L2-normalised float32 embeddings, so inner product == cosine similarity."""
    vecs = np.asarray(list(get_model().embed(texts)), dtype="float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]


class LocalEmbeddings(Embeddings):
    """LangChain adapter so the FAISS vector store uses the same model."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return embed_one(text).tolist()
