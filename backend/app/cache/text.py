"""Deterministic question analysis: what a question is about, and what it asks.

Everything here is plain Python (no model calls), so it is cheap enough to run on
every cache lookup.
"""
import re
from dataclasses import dataclass, field

from .lexicon import GENERIC_TERMS, NEGATIONS, STOPWORDS, SYNONYMS

_DASHES = re.compile(r"[‐-―−]")
_CLASS_REF = re.compile(
    r"\bclass\s*(?:10|x|tenth)\b|\b(?:10th|tenth)\s+(?:class|grade|std)\b|\bchapter\s*\d+\b")
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_NUMBER = re.compile(r"(?<![a-z0-9.])(-\s*)?(\d+(?:\.\d+)?)")
_SINGULAR_S = {"lens", "series", "species", "means", "diabetes", "testes", "always", "various"}
_DIRECTION = re.compile(r"\b(from|into|towards?|away)\b")

# Ordered: the first matching intent wins, specific before general.
_INTENT_RULES = [
    ("difference", r"\b(differ\w*|distinguish\w*|compare|comparison|contrast|vs|versus)\b"),
    ("example", r"\b(examples?|instances?)\b"),
    ("uses", r"\b(uses?|applications?|advantages?|disadvantages?|importance|significance|benefits?)\b"),
    ("types", r"\b(types?|kinds?|classif\w*|categor\w*)\b"),
    ("function", r"\b(functions?|role)\b"),
    ("diagram", r"\b(draw|diagram|sketch|label+ed)\b"),
    ("why", r"\b(why|reasons?|causes?)\b"),
    ("how", r"\bhow\b(?!\s+(many|much)\b)"),
    ("quantity", r"\b(how\s+(many|much)|calculate|compute|find|determine)\b"),
    ("list", r"\b(list|name|mention|identify|enumerate)\b"),
]


def normalize(text: str) -> str:
    text = _DASHES.sub("-", text.lower())
    text = text.replace("’", "'").replace("'s", "")
    text = text.replace("'", "")
    text = _CLASS_REF.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def lemma(word: str) -> str:
    """Tiny plural stripper; enough to make 'mirrors' == 'mirror' and 'laws' == 'law'."""
    word = SYNONYMS.get(word, word)
    if word in _SINGULAR_S or len(word) <= 3 or word.endswith(("ss", "us", "is", "ics")):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("sses", "xes", "ches", "shes")) or word == "lenses":
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


def detect_intent(text: str) -> str:
    has_numbers = bool(_NUMBER.search(text))
    for name, pattern in _INTENT_RULES:
        if re.search(pattern, text):
            # "Find f when R = 20 cm" and "Focal length when R = 20 cm" ask the same thing.
            return "numerical" if has_numbers and name == "quantity" else name
    if has_numbers:
        return "numerical"
    # "What is refraction?", "Refraction?", "laws of refraction": asking what something is.
    return "what"


def extract_numbers(text: str) -> tuple[float, ...]:
    values = []
    for sign, num in _NUMBER.findall(text):
        value = float(num)
        values.append(-value if sign else value)
    return tuple(sorted(values))


@dataclass(frozen=True)
class Signature:
    """Everything the guards compare between a new question and a cached one."""

    text: str
    terms: frozenset[str]          # content words, lemmatised
    ordered_terms: tuple[str, ...]  # same, in order (used when direction matters)
    numbers: tuple[float, ...]
    intent: str
    negated: bool
    directional: bool
    hard_terms: frozenset[str] = field(default=frozenset())  # terms minus generic ones


def analyse(question: str) -> Signature:
    text = normalize(question)
    tokens = _TOKEN.findall(text)
    content = [lemma(t) for t in tokens
               if t not in STOPWORDS and not t.replace(".", "").isdigit() and t not in NEGATIONS]
    # Units glued to numbers ("20cm") leave a bare unit token; keep it as a term.
    content = [re.sub(r"^\d+(?:\.\d+)?", "", t) or t for t in content]
    terms = frozenset(content)
    return Signature(
        text=text,
        terms=terms,
        ordered_terms=tuple(dict.fromkeys(content)),
        numbers=extract_numbers(text),
        intent=detect_intent(text),
        negated=any(t in NEGATIONS for t in tokens) or "n't" in question.lower(),
        directional=bool(_DIRECTION.search(text)),
        hard_terms=frozenset(t for t in terms if t not in GENERIC_TERMS),
    )
