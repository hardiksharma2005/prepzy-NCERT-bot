"""Retrieval over the textbook FAISS index built by scripts/ingest.py."""
from dataclasses import dataclass
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from . import config
from .embeddings import LocalEmbeddings


@dataclass
class Passage:
    chapter: str
    text: str
    score: float


@lru_cache(maxsize=1)
def get_store() -> FAISS:
    return FAISS.load_local(str(config.INDEX_DIR), LocalEmbeddings(),
                            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
                            allow_dangerous_deserialization=True)  # our own file


def retrieve(question: str, k: int = config.RETRIEVE_K) -> list[Passage]:
    results = get_store().similarity_search_with_score(question, k=k)
    return [Passage(doc.metadata["chapter"], doc.page_content, float(score))
            for doc, score in results]
