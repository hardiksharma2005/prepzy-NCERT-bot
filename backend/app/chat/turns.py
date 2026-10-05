"""Deterministic turn classification: decides, before any lookup, whether a message can
be answered from the cache as-is, needs the conversation to make sense, or is a request
about the previous answer ("explain it more simply")."""
import re
from dataclasses import dataclass

from ..cache.lexicon import DEMONSTRATIVES, PRONOUNS, STOPWORDS
from ..cache.text import analyse, normalize
from ..sessions import Session

SMALLTALK = re.compile(
    r"^(hi+|hello|hey|hii+|good (morning|afternoon|evening)|thanks?( you)?|thank u|ty|ok(ay)?|"
    r"cool|great|nice|bye|goodbye|who are you|what can you do|help)[\s!.?]*$")

# kind -> pattern. Matched only when the message has no new subject of its own.
TRANSFORMS = [
    ("simplify", r"\b(simpl(er|y|ify)|easier|easy (words|language)|layman|"
                 r"like i'?m (5|five|a kid)|(do ?n'?t|did ?n'?t|not) (get|understand)|confus)"),
    ("shorter", r"\b(shorter|in short|summari[sz]e|summary|briefly|one line|concise|tl;?dr)\b"),
    ("example", r"\b(examples?|real[- ]life|daily life|everyday)\b"),
    ("points", r"\b(points|bullets?|bulleted|step[- ]by[- ]step)\b"),
    ("elaborate", r"\b(elaborate|more detail|in detail|tell me more|expand|explain (it |this |that )?"
                  r"(more|further|again|better|properly))\b"),
]
_TRANSFORM_WORDS = {
    "simply", "simpler", "simplify", "easy", "easier", "word", "language", "layman", "kid",
    "five", "confused", "confusing", "get", "shorter", "summarise", "summarize", "summary",
    "line", "one", "concise", "real", "life", "real-life", "daily", "everyday", "more",
    "another", "point", "bullet", "bulleted", "step", "step-by-step", "further", "again",
    "better", "expand", "elaborate", "detail", "short", "dont", "didnt", "im", "like",
    "understood", "once", "much", "bit", "little", "now", "same", "thing",
}
_FOLLOWUP_LEAD = re.compile(r"^(what about|how about|and|also|then|but|so|what if)\b\s*")


@dataclass
class Turn:
    kind: str              # smalltalk | transform | followup | standalone
    query: str             # text to look up / answer (resolved form for follow-ups)
    resolved: bool = True  # False: follow-up we could not resolve without an LLM
    transform: str = ""


def _has_new_subject(message: str) -> bool:
    return bool(analyse(message).hard_terms - _TRANSFORM_WORDS)


def _pronoun_positions(tokens: list[str]) -> list[int]:
    hits = []
    for i, tok in enumerate(tokens):
        if tok in PRONOUNS:
            hits.append(i)
        elif tok in DEMONSTRATIVES:
            nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
            # "why does this happen" refers back; "the gas that turns lime water" does not.
            if not nxt or nxt in STOPWORDS:
                hits.append(i)
    return hits


def classify(message: str, session: Session) -> Turn:
    text = normalize(message)
    if not text or SMALLTALK.match(text):
        return Turn("smalltalk", message)

    has_context = bool(session.last_question)
    if has_context and not _has_new_subject(message):
        for kind, pattern in TRANSFORMS:
            if re.search(pattern, text):
                return Turn("transform", message, transform=kind)

    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text)
    content = [t for t in tokens if t not in STOPWORDS]
    pronouns = _pronoun_positions(tokens)
    # A pronoun after a subject in the same sentence ("why does light bend when it enters
    # glass") refers inside the sentence, not to the conversation.
    dangling = [i for i in pronouns if not any(t in content for t in tokens[:i])]
    lead = _FOLLOWUP_LEAD.match(text)

    if not has_context:
        # Nothing to refer back to; an unresolved pronoun still makes it uncacheable.
        return Turn("standalone", message, resolved=not dangling)
    if not dangling and not lead and analyse(message).hard_terms:
        return Turn("standalone", message)

    # Follow-up. Resolve deterministically when it is just "its/it/this" -> last topic.
    if dangling and session.last_topic:
        rest = _FOLLOWUP_LEAD.sub("", text) if lead else text
        resolved = re.sub(r"\b(its|it|they|them|their|these|those|this|that)\b",
                          session.last_topic, rest)
        return Turn("followup", resolved, resolved=True)
    return Turn("followup", message, resolved=False)
