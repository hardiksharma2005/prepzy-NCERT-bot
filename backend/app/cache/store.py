"""Semantic answer cache: SQLite for persistence, an in-memory FAISS index for lookup."""
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from .guards import verify
from .text import Signature, analyse


@dataclass
class CacheEntry:
    id: int
    question: str
    answer: str
    citations: list[str]
    topic: str
    signature: Signature


@dataclass
class LookupResult:
    entry: CacheEntry | None
    score: float
    # Candidates that looked similar but were rejected, with the guard that rejected them.
    rejected: list[tuple[str, float, str]]


SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    citations TEXT NOT NULL,
    topic TEXT NOT NULL DEFAULT '',
    embedding BLOB NOT NULL,
    created_at REAL NOT NULL,
    hits INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS derived (
    source_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at REAL NOT NULL,
    hits INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (source_id, kind)
);
"""


class SemanticCache:
    def __init__(self, db_path: Path, dim: int):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()
        self._index = faiss.IndexFlatIP(dim)
        self._entries: list[CacheEntry] = []  # position i <-> FAISS row i
        self._by_id: dict[int, CacheEntry] = {}
        for row in self._db.execute(
                "SELECT id, question, answer, citations, topic, embedding FROM entries ORDER BY id"):
            self._append(CacheEntry(row[0], row[1], row[2], json.loads(row[3]), row[4],
                                    analyse(row[1])),
                         np.frombuffer(row[5], dtype="float32"))

    def __len__(self) -> int:
        return len(self._entries)

    def _append(self, entry: CacheEntry, vector: np.ndarray) -> None:
        self._index.add(vector.reshape(1, -1))
        self._entries.append(entry)
        self._by_id[entry.id] = entry

    def lookup(self, signature: Signature, vector: np.ndarray, threshold: float,
               k: int = 5, count_hit: bool = True) -> LookupResult:
        with self._lock:
            if not self._entries:
                return LookupResult(None, 0.0, [])
            scores, rows = self._index.search(vector.reshape(1, -1), min(k, len(self._entries)))
        rejected = []
        for score, row in zip(scores[0], rows[0]):
            if row < 0:
                continue
            entry = self._entries[row]
            if score < threshold:
                rejected.append((entry.question, float(score), "below similarity threshold"))
                continue
            ok, reason = verify(signature, entry.signature)
            if ok:
                if count_hit:
                    self._bump("entries", "id = ?", (entry.id,))
                return LookupResult(entry, float(score), rejected)
            rejected.append((entry.question, float(score), reason))
        return LookupResult(None, float(scores[0][0]), rejected)

    def add(self, question: str, answer: str, citations: list[str], topic: str,
            vector: np.ndarray) -> CacheEntry:
        signature = analyse(question)
        # Don't store near-duplicates of something that would already be served.
        existing = self.lookup(signature, vector, threshold=0.97, k=1,
                               count_hit=False).entry
        if existing:
            return existing
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO entries (question, answer, citations, topic, embedding, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (question, answer, json.dumps(citations), topic,
                 vector.astype("float32").tobytes(), time.time()))
            self._db.commit()
            entry = CacheEntry(cur.lastrowid, question, answer, citations, topic, signature)
            self._append(entry, vector)
        return entry

    def get(self, entry_id: int) -> CacheEntry | None:
        return self._by_id.get(entry_id)

    # --- answers derived from a cached answer ("explain it more simply") -------------

    def get_derived(self, source_id: int, kind: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT answer FROM derived WHERE source_id = ? AND kind = ?",
                                   (source_id, kind)).fetchone()
        if row:
            self._bump("derived", "source_id = ? AND kind = ?", (source_id, kind))
            return row[0]
        return None

    def add_derived(self, source_id: int, kind: str, answer: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO derived (source_id, kind, answer, created_at)"
                             " VALUES (?, ?, ?, ?)", (source_id, kind, answer, time.time()))
            self._db.commit()

    def _bump(self, table: str, where: str, args: tuple) -> None:
        with self._lock:
            self._db.execute(f"UPDATE {table} SET hits = hits + 1 WHERE {where}", args)
            self._db.commit()

    def close(self) -> None:
        self._db.close()

    def stats(self) -> dict:
        with self._lock:
            entries, hits = self._db.execute(
                "SELECT COUNT(*), COALESCE(SUM(hits), 0) FROM entries").fetchone()
            derived, dhits = self._db.execute(
                "SELECT COUNT(*), COALESCE(SUM(hits), 0) FROM derived").fetchone()
        return {"entries": entries, "entry_hits": hits,
                "derived_entries": derived, "derived_hits": dhits}

    def export(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT question, answer, citations, topic FROM entries ORDER BY id").fetchall()
        return [{"question": q, "answer": a, "citations": json.loads(c), "topic": t}
                for q, a, c, t in rows]
