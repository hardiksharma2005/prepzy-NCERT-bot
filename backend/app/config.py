"""Runtime configuration, read from environment variables (and .env if present)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
INDEX_DIR = DATA_DIR / "textbook_index"
VOCAB_PATH = DATA_DIR / "vocab.json"
CHAPTERS_PATH = DATA_DIR / "chapters.json"
SEED_CACHE_PATH = DATA_DIR / "seed_cache.jsonl"
CACHE_DB_PATH = Path(os.getenv("CACHE_DB_PATH", DATA_DIR / "cache.sqlite3"))
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", str(ROOT / "models"))

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY", "")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "openai/gpt-oss-120b")
CONDENSE_MODEL = os.getenv("CONDENSE_MODEL", "openai/gpt-oss-20b")
# Used when the answer model is over capacity or failing.
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "qwen/qwen3.8-27b")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# Cosine similarity a cached question must reach before the safety guards even run.
CACHE_SIM_THRESHOLD = _float("CACHE_SIM_THRESHOLD", 0.86)
# Follow-ups resolved by pronoun substitution are noisier, so they need more.
CACHE_SIM_THRESHOLD_FOLLOWUP = _float("CACHE_SIM_THRESHOLD_FOLLOWUP", 0.90)
# Best textbook-chunk similarity below this means the book doesn't cover the question.
SCOPE_THRESHOLD = _float("SCOPE_THRESHOLD", 0.62)
RETRIEVE_K = int(os.getenv("RETRIEVE_K", 5))
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", 6))
