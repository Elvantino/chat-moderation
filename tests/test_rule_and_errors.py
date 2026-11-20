import numpy as np

from src.error_analysis import failures, to_markdown
from src.preprocessing import LexiconMatcher
from src.rule_baseline import LexiconRuleModel


def test_rule_baseline_predicts_from_lexicon():
    m = LexiconRuleModel(LexiconMatcher(["idiot*", "kill yourself"]))
    assert m.predict_one("you 1d10t") == (1, 1.0)
    assert m.predict_one("good game") == (0, 0.0)
    assert m.predict(["go kill yourself", "hello"]).tolist() == [1, 0]


def test_failures_picks_confident_errors():
    texts = ["a", "b", "c", "d", "x" * 500]
    y = [0, 0, 1, 1, 1]
    p = [0.9, 0.2, 0.1, 0.8, 0.05]
    r = failures(texts, y, p, threshold=0.5, top=2, max_chars=100)
    assert r["n_fp"] == 1 and r["n_fn"] == 2
    assert r["false_positives"][0]["text"] == "a"
    assert [e["text"] for e in r["false_negatives"]] == ["c"]   # long message deprioritised
    md = to_markdown({"models": {"k": {"model": "M", **r}}})
    assert "False Negatives" in md and "“a”" in md
