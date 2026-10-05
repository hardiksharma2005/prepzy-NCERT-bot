"""Answer a set of common doubts once and save them as data/seed_cache.jsonl.

The deployed backend loads this file into an empty cache at startup (free hosts wipe
their disk on restart), so the very first visitor already gets cache hits.
Needs GROQ_API_KEY.  python scripts/build_seed_cache.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app import config  # noqa: E402
from backend.app.cache.store import SemanticCache  # noqa: E402
from backend.app.chat.graph import ChatPipeline  # noqa: E402
from backend.app.chat.llm import LLMClient  # noqa: E402
from backend.app.sessions import Session  # noqa: E402

QUESTIONS = [
    "What is a chemical reaction?", "What is a balanced chemical equation?",
    "What is a combination reaction?", "What is a decomposition reaction?",
    "What is a displacement reaction?", "What is rancidity?", "What is corrosion?",
    "What is an oxidation reaction?", "What is the pH scale?", "What is plaster of Paris?",
    "What is bleaching powder used for?", "Why is baking soda used in cooking?",
    "What is an indicator?", "What are the physical properties of metals?",
    "What is an alloy?", "What are ionic compounds?", "Why does carbon form covalent bonds?",
    "What is catenation?", "What are saturated and unsaturated hydrocarbons?",
    "What is a homologous series?", "What is saponification?",
    "What is photosynthesis?", "What is the function of the kidney?",
    "Why is the sky blue?", "What is double circulation?", "What is a reflex action?",
    "What are plant hormones?", "What is the role of the brain?",
    "What is binary fission?", "What is pollination?", "What is fertilisation?",
    "What are dominant and recessive traits?", "How is the sex of a child determined?",
    "What is refraction?", "What are the laws of refraction?", "What are the laws of reflection?",
    "What is the refractive index?", "What is the power of a lens?",
    "What is the mirror formula?", "What is the image formed by a concave mirror?",
    "What is the image formed by a convex mirror?", "What is myopia?",
    "What is hypermetropia?", "What is dispersion of light?", "What is the Tyndall effect?",
    "Why do stars twinkle?", "What is Ohm's law?", "What is electric current?",
    "What is resistivity?", "What is electric power?", "What is the heating effect of current?",
    "What are magnetic field lines?", "What is Fleming's left-hand rule?",
    "What is the right-hand thumb rule?", "What is a solenoid?",
    "What is a food chain?", "What is an ecosystem?", "What is ozone depletion?",
    "What are biodegradable substances?", "What is biological magnification?",
]


# Progress lives here, so a re-run resumes instead of starting over (gitignored).
BUILD_DB = config.DATA_DIR / "seed_build.sqlite3"


def ask(pipeline: ChatPipeline, question: str, attempts: int = 4):
    for attempt in range(attempts):
        try:
            return pipeline.run(Session(id="seed"), question)
        except Exception as exc:  # e.g. Groq 503 "over capacity" or a 429 rate limit
            wait = 10 * 2 ** attempt
            print(f"   ! {type(exc).__name__}: {str(exc)[:90]} - retrying in {wait}s")
            time.sleep(wait)
    return None


def main() -> None:
    cache = SemanticCache(BUILD_DB, dim=384)
    try:
        done = {row["question"] for row in cache.export()}
        pipeline = ChatPipeline(cache, LLMClient())
        failed = []
        for q in QUESTIONS:
            if q in done:
                print(f"done    {q}")
                continue
            reply = ask(pipeline, q)
            if reply is None:
                failed.append(q)
                print(f"FAILED  {q}")
            else:
                print(f"{'cached' if reply.debug.get('entry_id') else 'skip  '}  {q}")
            time.sleep(2)  # stay inside the Groq free-tier rate limit
        rows = cache.export()
    finally:
        cache.close()
    if failed:
        print(f"{len(failed)} questions failed; run the script again later to retry them.")
    config.SEED_CACHE_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} entries to {config.SEED_CACHE_PATH}")


if __name__ == "__main__":
    main()
