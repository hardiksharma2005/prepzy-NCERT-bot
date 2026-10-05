"""Safety checks a cached question must pass before its answer is reused.

Embedding similarity says "these look alike"; these guards check "these ask the same
thing". Each one is cheap, deterministic and errs towards a miss: a miss costs one LLM
call, a wrong hit teaches a student something wrong.
"""
from .lexicon import CONTRAST_PAIRS
from .text import Signature


def _contrast_conflict(a: frozenset[str], b: frozenset[str]) -> str | None:
    for x, y in CONTRAST_PAIRS:
        if (x in a and y in b and x not in b and y not in a) or \
           (y in a and x in b and y not in b and x not in a):
            return f"{x}/{y}"
    return None


def verify(query: Signature, cached: Signature) -> tuple[bool, str]:
    """Return (safe_to_reuse, reason). Reason names the first guard that failed."""
    if not query.hard_terms:
        return False, "no content terms"
    if query.numbers != cached.numbers:
        return False, f"numbers differ {query.numbers} vs {cached.numbers}"
    if query.negated != cached.negated:
        return False, "negation differs"
    if query.intent != cached.intent:
        return False, f"intent differs ({query.intent} vs {cached.intent})"
    conflict = _contrast_conflict(query.terms, cached.terms)
    if conflict:
        return False, f"contrasting terms {conflict}"
    if query.hard_terms != cached.hard_terms:
        diff = sorted(query.hard_terms ^ cached.hard_terms)
        return False, f"terms differ {diff}"
    if (query.directional or cached.directional) and \
            [t for t in query.ordered_terms if t in query.hard_terms] != \
            [t for t in cached.ordered_terms if t in cached.hard_terms]:
        return False, "direction/order differs"
    return True, "ok"
