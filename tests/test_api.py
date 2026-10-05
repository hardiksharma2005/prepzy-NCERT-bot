"""End-to-end through FastAPI with real embeddings + textbook index and a fake LLM."""
import pytest
from fastapi.testclient import TestClient

from backend.app.cache.store import SemanticCache
from backend.app.chat.llm import AnswerOut
from backend.app.main import create_app

LIGHT = "Light – Reflection and Refraction"


class FakeLLM:
    def __init__(self):
        self.calls = 0
        self.log: list[str] = []

    def condense(self, history: str, message: str) -> str:
        self.calls += 1
        self.log.append("condense")
        if "convex" in message.lower():
            return "What is the image formed by a convex mirror?"
        return "What are the laws of refraction?"

    def answer(self, question: str, context: str) -> AnswerOut:
        self.calls += 1
        self.log.append("answer")
        if "cricket" in question.lower() or "newton" in question.lower():
            return AnswerOut(answer="", chapters=[], in_scope=False)
        topic = "refraction" if "refraction" in question.lower() else "mirror"
        return AnswerOut(answer=f"ANSWER: {question}", chapters=[LIGHT], in_scope=True, topic=topic)

    def transform(self, kind, question, answer, message, context) -> str:
        self.calls += 1
        self.log.append(f"transform:{kind}")
        return f"{kind.upper()}: {answer}"


@pytest.fixture()
def client(tmp_path):
    llm = FakeLLM()
    cache = SemanticCache(tmp_path / "cache.sqlite3", dim=384)
    with TestClient(create_app(llm=llm, cache=cache)) as c:
        c.llm = llm
        yield c


def ask(client, sid, message):
    r = client.post("/chat?debug=1", json={"session_id": sid, "message": message})
    assert r.status_code == 200, r.text
    return r.json()


def new_session(client):
    return client.post("/session").json()["session_id"]


def test_response_shape(client):
    body = ask(client, new_session(client), "What is refraction?")
    assert set(body) >= {"reply", "citations", "cache_hit", "latency_ms"}
    assert body["citations"] == [LIGHT]
    assert body["cache_hit"] is False


def test_paraphrase_hits_without_llm_and_fast(client):
    ask(client, new_session(client), "What is refraction?")
    calls = client.llm.calls
    body = ask(client, new_session(client), "What does refraction mean?")
    assert body["cache_hit"] is True
    assert client.llm.calls == calls, "a cache hit must not call the LLM"
    assert body["latency_ms"] < 500
    assert body["reply"] == "ANSWER: What is refraction?"


def test_similar_words_different_question_misses(client):
    ask(client, new_session(client), "Image by a concave mirror")
    body = ask(client, new_session(client), "Image by a convex mirror")
    assert body["cache_hit"] is False
    assert "convex" in body["reply"]


def test_different_numbers_miss(client):
    ask(client, new_session(client), "Focal length of a spherical mirror when R = 20 cm")
    body = ask(client, new_session(client), "Focal length of a spherical mirror when R = 30 cm")
    assert body["cache_hit"] is False


def test_follow_up_resolved_from_conversation_hits_cache(client):
    ask(client, new_session(client), "What are the laws of refraction?")   # cached by someone
    sid = new_session(client)
    ask(client, sid, "What is refraction?")
    calls = client.llm.calls
    body = ask(client, sid, "What about its laws?")
    assert body["cache_hit"] is True and client.llm.calls == calls


def test_follow_up_without_context_is_never_served_from_cache(client):
    ask(client, new_session(client), "What are the laws of refraction?")
    body = ask(client, new_session(client), "What about its laws?")
    assert body["cache_hit"] is False


def test_explain_more_simply_is_tied_to_previous_answer(client):
    sid = new_session(client)
    ask(client, sid, "What is refraction?")
    first = ask(client, sid, "Explain it more simply")
    assert first["cache_hit"] is False and first["reply"].startswith("SIMPLIFY")

    # Another student, same source answer, same request -> derived answer reused.
    sid2 = new_session(client)
    ask(client, sid2, "What does refraction mean?")
    calls = client.llm.calls
    second = ask(client, sid2, "Explain it more simply")
    assert second["cache_hit"] is True and client.llm.calls == calls

    # Asking again in the same conversation needs the conversation: not served from cache.
    third = ask(client, sid2, "explain it even more simply")
    assert third["cache_hit"] is False


def test_out_of_scope_is_declined_and_not_cached(client):
    sid = new_session(client)
    body = ask(client, sid, "Who won the 2011 cricket world cup?")
    assert "only help with doubts from the NCERT" in body["reply"]
    assert body["citations"] == []
    again = ask(client, new_session(client), "Who won the 2011 cricket world cup?")
    assert again["cache_hit"] is False
    assert client.app.state.cache.stats()["entries"] == 0


def test_unknown_session_is_404(client):
    r = client.post("/chat", json={"session_id": "nope", "message": "hi"})
    assert r.status_code == 404
