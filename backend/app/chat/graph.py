"""The request flow as a LangGraph state machine.

    classify ─┬─ smalltalk ───────────────────────────────────────────────▶ END
              ├─ transform ── (derived cache?) ── LLM rewrite of last answer ▶ END
              └─ lookup ──hit──────────────────────────────────────────────▶ END   (no LLM)
                    │miss
                    ▼
                 condense (follow-ups only) ── lookup again ──hit──────────▶ END
                    │miss
                    ▼
                 retrieve ──out of scope──▶ decline ───────────────────────▶ END
                    ▼
                 generate ── store in cache (if safe) ─────────────────────▶ END
"""
from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import END, StateGraph

from .. import config
from ..cache.store import SemanticCache
from ..cache.text import analyse
from ..embeddings import embed_one
from ..sessions import Session
from ..textbook import Passage, retrieve
from . import prompts
from .turns import Turn, classify


@dataclass
class Reply:
    reply: str
    citations: list[str] = field(default_factory=list)
    cache_hit: bool = False
    debug: dict = field(default_factory=dict)


class State(TypedDict, total=False):
    message: str
    session: Session
    turn: Turn
    query: str            # standalone question being answered
    passages: list[Passage]
    result: Reply
    trace: list[str]
    llm_calls_before: int


def _norm_chapter(name: str) -> str:
    return name.replace("–", "-").replace("—", "-").strip().lower()


def _context(passages: list[Passage]) -> str:
    return "\n\n".join(p.text for p in passages)


def _history_text(session: Session) -> str:
    turns = session.history[-config.HISTORY_TURNS:]
    return "\n".join(f"{m['role'].title()}: {m['content'][:600]}" for m in turns)


class ChatPipeline:
    def __init__(self, cache: SemanticCache, llm):
        self.cache = cache
        self.llm = llm
        self.graph = self._build()

    # ---- nodes ------------------------------------------------------------------------

    def _classify(self, state: State) -> State:
        turn = classify(state["message"], state["session"])
        return {"turn": turn, "query": turn.query, "trace": [f"turn={turn.kind}"]}

    def _smalltalk(self, state: State) -> State:
        return {"result": Reply(prompts.SMALLTALK_REPLY)}

    def _lookup(self, state: State) -> State:
        turn, query = state["turn"], state["query"]
        trace = state["trace"]
        threshold = (config.CACHE_SIM_THRESHOLD_FOLLOWUP if turn.kind == "followup"
                     else config.CACHE_SIM_THRESHOLD)
        result = self.cache.lookup(analyse(query), embed_one(query), threshold)
        for question, score, reason in result.rejected:
            trace.append(f"rejected '{question}' ({score:.2f}): {reason}")
        if result.entry is None:
            trace.append(f"cache miss for '{query}'")
            return {"trace": trace}
        entry = result.entry
        trace.append(f"cache hit '{entry.question}' ({result.score:.2f})")
        # A hit after an LLM condense saved the expensive answer call, but it is not
        # reported as a cache hit: an LLM was still called on this request.
        reused_after_llm = self.llm.calls > state.get("llm_calls_before", 0)
        return {"trace": trace, "result": Reply(
            entry.answer, entry.citations, cache_hit=not reused_after_llm,
            debug={"entry_id": entry.id, "topic": entry.topic, "query": entry.question})}

    def _condense(self, state: State) -> State:
        session = state["session"]
        query = self.llm.condense(_history_text(session), state["message"])
        state["trace"].append(f"condensed to '{query}'")
        return {"query": query, "turn": Turn("followup_condensed", query)}

    def _retrieve(self, state: State) -> State:
        passages = retrieve(state["query"])
        best = passages[0].score if passages else 0.0
        state["trace"].append(f"retrieval best={best:.2f} ({passages[0].chapter if passages else '-'})")
        if best < config.SCOPE_THRESHOLD:
            return {"passages": passages, "result": Reply(prompts.DECLINE)}
        return {"passages": passages}

    def _generate(self, state: State) -> State:
        passages, query = state["passages"], state["query"]
        out = self.llm.answer(query, _context(passages))
        if not out.in_scope or not out.answer.strip():
            state["trace"].append("LLM judged out of scope")
            return {"result": Reply(prompts.DECLINE)}
        retrieved = {_norm_chapter(p.chapter): p.chapter for p in passages}
        citations = list(dict.fromkeys(
            retrieved[_norm_chapter(c)] for c in out.chapters if _norm_chapter(c) in retrieved))
        citations = citations or [passages[0].chapter]
        topic = out.topic.strip() or " ".join(sorted(analyse(query).hard_terms)[:3])
        reply = Reply(out.answer.strip(), citations, debug={"topic": topic, "query": query})

        # Store only answers to a self-contained question, written without chat history.
        turn = state["turn"]
        cacheable = turn.kind in ("standalone", "followup_condensed") and turn.resolved \
            and bool(analyse(query).hard_terms)
        if cacheable:
            entry = self.cache.add(query, reply.reply, citations, topic, embed_one(query))
            reply.debug["entry_id"] = entry.id
            state["trace"].append(f"stored as cache entry {entry.id}")
        else:
            state["trace"].append("not cached (depends on conversation)")
        return {"result": reply}

    def _transform(self, state: State) -> State:
        session, turn = state["session"], state["turn"]
        if not session.last_question:
            return {"result": Reply(prompts.NO_CONTEXT_TRANSFORM)}
        source = self.cache.get(session.last_entry_id) if session.last_entry_id else None
        # Re-using a derived answer is only safe the first time this session asks for it;
        # "even simpler" after "simpler" needs the conversation, so it goes to the LLM.
        reusable = source is not None and turn.transform not in session.transforms_used
        if reusable:
            cached = self.cache.get_derived(source.id, turn.transform)
            if cached:
                state["trace"].append(f"derived cache hit ({source.id}, {turn.transform})")
                return {"result": Reply(cached, source.citations, cache_hit=True,
                                        debug={"entry_id": source.id, "topic": source.topic,
                                               "transform": turn.transform})}
        last_answer = next((m["content"] for m in reversed(session.history)
                            if m["role"] == "assistant"), "")
        question = source.question if source else session.last_question
        base_answer = source.answer if reusable else last_answer
        passages = retrieve(question)
        text = self.llm.transform(turn.transform, question, base_answer, state["message"],
                                  _context(passages))
        citations = source.citations if source else [passages[0].chapter]
        if reusable:
            self.cache.add_derived(source.id, turn.transform, text)
            state["trace"].append(f"stored derived ({source.id}, {turn.transform})")
        else:
            state["trace"].append("transform not cached (no cached source or repeated)")
        return {"result": Reply(text, citations, debug={
            "entry_id": source.id if source else None, "transform": turn.transform})}

    # ---- routing ----------------------------------------------------------------------

    @staticmethod
    def _after_classify(state: State) -> str:
        turn = state["turn"]
        if turn.kind in ("smalltalk", "transform"):
            return turn.kind
        if not turn.resolved:
            # Unresolved follow-up -> condense first. Unresolved standalone (dangling
            # pronoun, no conversation) -> answer it, but never via the cache.
            return "condense" if turn.kind == "followup" else "retrieve"
        return "lookup"

    @staticmethod
    def _after_lookup(state: State) -> str:
        if "result" in state:
            return END
        return "condense" if state["turn"].kind == "followup" else "retrieve"

    @staticmethod
    def _after_retrieve(state: State) -> str:
        return END if "result" in state else "generate"

    def _build(self):
        g = StateGraph(State)
        g.add_node("classify", self._classify)
        g.add_node("smalltalk", self._smalltalk)
        g.add_node("transform", self._transform)
        g.add_node("lookup", self._lookup)
        g.add_node("condense", self._condense)
        g.add_node("lookup_condensed", self._lookup)
        g.add_node("retrieve", self._retrieve)
        g.add_node("generate", self._generate)
        g.set_entry_point("classify")
        g.add_conditional_edges("classify", self._after_classify,
                                ["smalltalk", "transform", "lookup", "condense", "retrieve"])
        g.add_conditional_edges("lookup", self._after_lookup, ["condense", "retrieve", END])
        g.add_edge("condense", "lookup_condensed")
        g.add_conditional_edges("lookup_condensed", self._after_lookup, ["retrieve", END])
        g.add_conditional_edges("retrieve", self._after_retrieve, ["generate", END])
        for node in ("smalltalk", "transform", "generate"):
            g.add_edge(node, END)
        return g.compile()

    # ---- entry point ------------------------------------------------------------------

    def run(self, session: Session, message: str) -> Reply:
        state = self.graph.invoke({"message": message, "session": session,
                                   "llm_calls_before": self.llm.calls})
        reply: Reply = state["result"]
        reply.debug["trace"] = state.get("trace", [])
        self._update_session(session, message, state["turn"], reply)
        return reply

    @staticmethod
    def _update_session(session: Session, message: str, turn: Turn, reply: Reply) -> None:
        session.history += [{"role": "user", "content": message},
                            {"role": "assistant", "content": reply.reply}]
        if turn.kind == "smalltalk" or reply.reply == prompts.DECLINE:
            return
        if turn.kind == "transform":
            session.transforms_used.add(turn.transform)
            return
        entry_id = reply.debug.get("entry_id")
        if entry_id != session.last_entry_id:
            session.transforms_used = set()
        session.last_entry_id = entry_id
        session.last_question = reply.debug.get("query", message)
        session.last_topic = reply.debug.get("topic", "") or session.last_topic
