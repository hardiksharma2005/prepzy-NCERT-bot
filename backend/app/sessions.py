"""In-memory conversation state. Lost on restart; the frontend recreates sessions on 404."""
import threading
import time
import uuid
from dataclasses import dataclass, field

SESSION_TTL_SECONDS = 6 * 60 * 60


@dataclass
class Session:
    id: str
    history: list[dict] = field(default_factory=list)  # {"role": "user"|"assistant", "content"}
    last_topic: str = ""               # concept of the last answered question, for "its", "it"
    last_question: str = ""            # standalone form of the last answered question
    last_entry_id: int | None = None   # cache entry behind the last answer, if any
    transforms_used: set[str] = field(default_factory=set)  # e.g. {"simplify"} for that entry
    touched: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self) -> Session:
        session = Session(id=uuid.uuid4().hex)
        with self._lock:
            self._expire()
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.touched = time.time()
            return session

    def _expire(self) -> None:
        cutoff = time.time() - SESSION_TTL_SECONDS
        for sid in [s for s, v in self._sessions.items() if v.touched < cutoff]:
            del self._sessions[sid]
