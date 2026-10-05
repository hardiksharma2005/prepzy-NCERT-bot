from backend.app.chat.turns import classify
from backend.app.sessions import Session


def fresh() -> Session:
    return Session(id="s")


def after(question: str, topic: str) -> Session:
    s = Session(id="s", last_question=question, last_topic=topic, last_entry_id=1)
    s.history = [{"role": "user", "content": question}, {"role": "assistant", "content": "..."}]
    return s


def test_standalone_question():
    turn = classify("What is refraction?", fresh())
    assert turn.kind == "standalone" and turn.resolved


def test_follow_up_with_pronoun_resolves_to_last_topic():
    turn = classify("What about its laws?", after("What is refraction?", "refraction"))
    assert turn.kind == "followup" and turn.resolved
    assert "refraction" in turn.query and "law" in turn.query


def test_elliptical_follow_up_needs_condensing():
    turn = classify("And for a convex mirror?", after("Image by a concave mirror", "concave mirror"))
    assert turn.kind == "followup" and not turn.resolved


def test_pronoun_without_history_is_uncacheable_standalone():
    turn = classify("What are its laws?", fresh())
    assert turn.kind == "standalone" and not turn.resolved


def test_pronoun_referring_inside_the_sentence_is_standalone():
    s = after("What is refraction?", "refraction")
    assert classify("Why does light bend when it enters glass?", s).kind == "standalone"
    assert classify("Name the gas that turns lime water milky", s).kind == "standalone"


def test_conversation_transforms():
    s = after("What is refraction?", "refraction")
    assert classify("Explain it more simply", s).transform == "simplify"
    assert classify("I didn't understand", s).transform == "simplify"
    assert classify("give an example", s).transform == "example"
    assert classify("Summarise that", s).transform == "shorter"
    # A new subject makes it a real question, not a transform.
    assert classify("Give examples of acids", s).kind == "standalone"


def test_smalltalk():
    assert classify("hi!", fresh()).kind == "smalltalk"
    assert classify("Thank you", fresh()).kind == "smalltalk"
