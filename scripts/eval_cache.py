"""Measure the cache's reuse decision on labelled question pairs.

Compares "embedding similarity alone" with "similarity + guards" (what the app uses).
    python scripts/eval_cache.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app import config  # noqa: E402
from backend.app.cache.guards import verify  # noqa: E402
from backend.app.cache.text import analyse  # noqa: E402
from backend.app.embeddings import embed  # noqa: E402

# (cached question, new question, should the cached answer be served?)
PAIRS = [
    ("What is refraction?", "What does refraction mean?", True),
    ("What is refraction?", "Define refraction", True),
    ("What is meant by refraction of light?", "What is refraction?", True),
    ("State Ohm's law", "What is Ohm's law?", True),
    ("Why is the sky blue?", "What is the reason the sky is blue?", True),
    ("Why is the sky blue?", "Why does the sky appear blue?", True),
    ("Difference between arteries and veins", "How do arteries differ from veins?", True),
    ("What is photosynthesis?", "Define photosynthesis.", True),
    ("What is the pH of pure water?", "pH of pure water?", True),
    ("What are the laws of refraction?", "Laws of refraction", True),
    ("What is a food chain?", "What is meant by a food chain?", True),
    ("What is rancidity?", "What does rancidity mean?", True),
    ("Give examples of acids", "Give some examples of acids", True),
    ("What is the function of the kidney?", "What is the role of kidneys?", True),
    ("What is myopia?", "What is short-sightedness?", True),  # synonym: expected miss today
    ("Image by a concave mirror", "Image by a convex mirror", False),
    ("Image formed by a concave lens", "Image formed by a concave mirror", False),
    ("Focal length when R = 20 cm", "Focal length when R = 30 cm", False),
    ("Object placed at 10 cm from a convex lens of focal length 15 cm",
     "Object placed at 20 cm from a convex lens of focal length 15 cm", False),
    ("What is refraction?", "Why does refraction occur?", False),
    ("What is refraction?", "What is reflection?", False),
    ("What is myopia?", "What is hypermetropia?", False),
    ("What is resistance?", "What is resistivity?", False),
    ("Resistors in series", "Resistors in parallel", False),
    ("What is aerobic respiration?", "What is anaerobic respiration?", False),
    ("Function of xylem", "Function of phloem", False),
    ("Which metals react with water?", "Which metals do not react with water?", False),
    ("Advantages of AC over DC", "Disadvantages of AC over DC", False),
    ("Light going from air to glass", "Light going from glass to air", False),
    ("What are the laws of refraction?", "What are the laws of reflection?", False),
    ("What is an exothermic reaction?", "What is an endothermic reaction?", False),
    ("What is oxidation?", "What is reduction?", False),
    ("What is a dominant trait?", "What is a recessive trait?", False),
    ("How do plants get nutrition?", "How do animals get nutrition?", False),
    ("What is the pH of a strong acid?", "What is the pH of a weak acid?", False),
    ("Why is the sky blue?", "Why is the sun reddish at sunrise?", False),
    ("Uses of bleaching powder", "What is bleaching powder?", False),
    ("What is a biodegradable substance?", "What is a non-biodegradable substance?", False),
    ("Explain the working of an electric motor", "Explain the working of an electric generator", False),
]


def main() -> None:
    vecs = embed([q for pair in PAIRS for q in pair[:2]])
    tau = config.CACHE_SIM_THRESHOLD
    rows, counts = [], {"sim": [0, 0, 0, 0], "guarded": [0, 0, 0, 0]}  # TP FP TN FN
    for i, (cached, new, label) in enumerate(PAIRS):
        sim = float(vecs[2 * i] @ vecs[2 * i + 1])
        ok, reason = verify(analyse(new), analyse(cached))
        for name, decision in (("sim", sim >= tau), ("guarded", sim >= tau and ok)):
            idx = {(True, True): 0, (True, False): 1, (False, False): 2, (False, True): 3}[
                (decision, label)]
            counts[name][idx] += 1
        verdict = "hit " if sim >= tau and ok else "miss"
        flag = "" if (sim >= tau and ok) == label else "  <-- " + ("FALSE HIT" if not label else "missed reuse")
        rows.append(f"{sim:.3f} {verdict} {'reuse' if label else 'no   '}  {cached!r} / {new!r}"
                    f"{'' if ok else f'  [{reason}]'}{flag}")
    print("\n".join(rows))
    print(f"\nthreshold = {tau}")
    for name, (tp, fp, tn, fn) in counts.items():
        label = "similarity only" if name == "sim" else "similarity + guards"
        print(f"{label:22} correct reuse {tp:2}  FALSE HITS {fp:2}  correct miss {tn:2}  missed reuse {fn:2}")


if __name__ == "__main__":
    main()
