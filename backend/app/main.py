"""FastAPI backend: POST /session, POST /chat."""
import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from . import config
from .cache.store import SemanticCache
from .chat.graph import ChatPipeline
from .embeddings import embed, embed_one
from .sessions import SessionStore
from .textbook import get_store

log = logging.getLogger("prepzy")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    reply: str
    citations: list[str]
    cache_hit: bool
    latency_ms: int
    debug: dict | None = None


def load_seed(cache: SemanticCache) -> None:
    """Pre-warm an empty cache from data/seed_cache.jsonl (the disk on free hosts is wiped)."""
    if len(cache) or not config.SEED_CACHE_PATH.exists():
        return
    rows = [json.loads(line) for line in config.SEED_CACHE_PATH.read_text(encoding="utf-8")
            .splitlines() if line.strip()]
    vectors = embed([r["question"] for r in rows]) if rows else []
    for row, vec in zip(rows, vectors):
        cache.add(row["question"], row["answer"], row["citations"], row.get("topic", ""), vec)
    log.info("Seeded cache with %d entries", len(cache))


def create_app(llm=None, cache: SemanticCache | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load models and indexes up front so the first request isn't slow.
        get_store()
        embed_one("warm up")
        if cache is not None:  # (an empty cache is falsy: it has __len__)
            app.state.cache = cache
        else:
            app.state.cache = SemanticCache(config.CACHE_DB_PATH, dim=len(embed_one("x")))
        load_seed(app.state.cache)
        if llm is None:
            from .chat.llm import LLMClient
            app.state.pipeline = ChatPipeline(app.state.cache, LLMClient())
        else:
            app.state.pipeline = ChatPipeline(app.state.cache, llm)
        app.state.sessions = SessionStore()
        yield

    app = FastAPI(title="NCERT Class 10 Science Doubt Solver", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/session")
    def new_session():
        return {"session_id": app.state.sessions.create().id}

    @app.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
    def chat(req: ChatRequest, debug: bool = Query(False)):
        start = time.perf_counter()
        session = app.state.sessions.get(req.session_id)
        if session is None:
            raise HTTPException(404, "Unknown session_id; create one with POST /session")
        try:
            reply = app.state.pipeline.run(session, req.message.strip())
        except Exception as exc:
            log.exception("chat failed")
            raise HTTPException(502, f"Could not generate an answer right now: {exc}") from exc
        latency = int((time.perf_counter() - start) * 1000)
        log.info("chat hit=%s %dms trace=%s", reply.cache_hit, latency, reply.debug.get("trace"))
        return ChatResponse(reply=reply.reply, citations=reply.citations,
                            cache_hit=reply.cache_hit, latency_ms=latency,
                            debug=reply.debug if debug else None)

    @app.get("/cache/stats")
    def cache_stats():
        return app.state.cache.stats()

    return app


app = create_app()
