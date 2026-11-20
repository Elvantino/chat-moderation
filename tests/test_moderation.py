"""App logic tests with fake models (no torch / gradio needed)."""
import pytest

from src.moderation import escape_md, moderate, run_models, verdict_line
from src.preprocessing import LexiconMatcher
from src.registry import SPEC_BY_KEY


class FakeModel:
    def __init__(self, prob, threshold=0.5):
        self.prob, self.threshold = prob, threshold

    def predict_one(self, text):
        return int(self.prob >= self.threshold), self.prob


@pytest.fixture
def lex():
    return LexiconMatcher(["idiot*", "shit*"])


def _models(p_int8, p_base=0.1):
    return {"bert_int8": (SPEC_BY_KEY["bert_int8"], FakeModel(p_int8)),
            "baseline": (SPEC_BY_KEY["baseline"], FakeModel(p_base))}


def test_run_models_times_every_model():
    outs = run_models("hello", _models(0.2))
    assert [o.key for o in outs] == ["bert_int8", "baseline"]
    assert all(o.latency_ms >= 0 for o in outs)
    assert outs[0].verdict == "Clean"


@pytest.mark.parametrize("text, p, action", [
    ("good game", 0.10, "ALLOW"),
    ("you idiot", 0.90, "MASK"),
    ("shit, gg", 0.10, "MASK"),          # profanity masked even when classified clean
    ("you are worthless garbage", 0.80, "FLAG"),
    ("you are worthless garbage", 0.99, "BLOCK"),
])
def test_moderation_actions(lex, text, p, action):
    outs = run_models(text, _models(p))
    res = moderate(text, outs, "bert_int8", lex)
    assert res.action == action
    assert res.primary.key == "bert_int8"
    if action == "MASK":
        assert "*" in res.display_text
    if action == "BLOCK":
        assert text not in res.display_text


def test_primary_falls_back_to_first(lex):
    outs = run_models("hi", _models(0.1))
    assert moderate("hi", outs, "missing", lex).primary.key == "bert_int8"


def test_escape_md_and_verdict(lex):
    assert escape_md("f*** _x_") == "f\\*\\*\\* \\_x\\_"
    res = moderate("you idiot", run_models("you idiot", _models(0.9)), "bert_int8", lex)
    line = verdict_line(res)
    assert "Masked" in line and "p=0.90" in line and "1 term(s)" in line
