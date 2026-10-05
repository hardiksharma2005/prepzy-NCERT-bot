"""The cache's safety guards, on the cases from the assignment and nearby traps."""
import pytest

from backend.app.cache.guards import verify
from backend.app.cache.text import analyse


def same(a: str, b: str) -> bool:
    return verify(analyse(a), analyse(b))[0]


@pytest.mark.parametrize("a,b", [
    ("What is refraction?", "What does refraction mean?"),
    ("What is refraction?", "Define refraction."),
    ("What is meant by refraction of light?", "What is refraction?"),
    ("State Ohm's law", "What is Ohm's law?"),
    ("What are the laws of refraction?", "laws of refraction"),
    ("Why is the sky blue?", "What is the reason the sky is blue?"),
    ("Difference between arteries and veins", "How do arteries differ from veins?"),
    ("Give examples of acids", "Give some examples of acid"),
    ("What is the colour of the sky?", "What is the color of the sky"),
    ("Focal length of a mirror when R = 20 cm", "Find the focal length of a mirror if R = 20cm"),
    ("What is the pH of pure water?", "pH of pure water in class 10?"),
])
def test_same_doubt_different_wording_is_reused(a, b):
    assert same(a, b), verify(analyse(a), analyse(b))[1]


@pytest.mark.parametrize("a,b", [
    ("Image by a concave mirror", "Image by a convex mirror"),
    ("Image formed by a concave lens", "Image formed by a concave mirror"),
    ("Focal length when R = 20 cm", "Focal length when R = 30 cm"),
    ("Image when object is at 10 cm", "Image when object is at -10 cm"),
    ("What is refraction?", "Why does refraction occur?"),
    ("What is refraction?", "What is reflection?"),
    ("What is myopia?", "What is hypermetropia?"),
    ("What is resistance?", "What is resistivity?"),
    ("Resistors in series", "Resistors in parallel"),
    ("What is aerobic respiration?", "What is anaerobic respiration?"),
    ("Function of xylem", "Function of phloem"),
    ("Which metals react with water?", "Which metals do not react with water?"),
    ("Advantages of AC", "Disadvantages of AC"),
    ("Uses of bleaching powder", "What is bleaching powder?"),
    ("Light going from air to glass", "Light going from glass to air"),
    ("What is the pH of a strong acid?", "What is the pH of a weak acid?"),
    ("What are the laws of refraction?", "What are the laws of reflection?"),
    ("How many chromosomes in humans?", "How many chromosomes in a gamete?"),
    ("What is it?", "What is refraction?"),
])
def test_different_question_is_not_reused(a, b):
    assert not same(a, b)
    assert not same(b, a)


def test_numbers_and_units_are_extracted():
    sig = analyse("u = -15 cm and f = 10cm, find v")
    assert sig.numbers == (-15.0, 10.0)
    assert "cm" in sig.terms
    assert sig.intent == "numerical"


def test_class_reference_is_not_a_number():
    assert analyse("Class 10 question: what is refraction").numbers == ()
