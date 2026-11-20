from src import config as C
from src.evaluate import latency_messages, results_to_markdown
from src.registry import SPECS, load_models


def _row(name, f1, lat):
    return {"tier": "T", "model": name, "area": "A", "macro_f1": f1, "size_mb": 1.5,
            "latency_mean_ms": lat, "latency_p95_ms": lat * 1.5}


def test_markdown_table_structure():
    md = results_to_markdown([_row("fast", 0.9, 3.0), _row("slow", 0.95, 80.0)], runs=200)
    lines = md.strip().splitlines()
    assert len(lines) == 4
    assert "Macro-F1" in lines[0] and "Size (MB)" in lines[0] and "n=200" in lines[0]
    assert "| 0.9000 |" in lines[2] and "✅" in lines[2]
    assert "❌" in lines[3]


def test_markdown_states_hardware():
    md = results_to_markdown([_row("m", 0.9, 3.0)], runs=200, hardware="Test CPU", n_test=1234)
    assert "Hardware: Test CPU" in md and "1,234-message" in md and "200 runs" in md


def test_latency_messages_deterministic():
    texts = [f"m{i}" for i in range(1000)]
    a, b = latency_messages(texts, 200), latency_messages(texts, 200)
    assert a == b and len(a) == 200 and len(set(a)) == 200
    assert len(latency_messages(texts[:50], 200)) == 50


def test_registry_covers_ladder():
    assert [s.key for s in SPECS] == ["lexicon_rule", "baseline", "neural_net",
                                      "bert_fp32", "bert_int8"]
    assert C.LATENCY_BUDGET_MS == 50.0


def test_load_models_skips_missing(monkeypatch, tmp_path):
    import src.registry as R
    fake = [R.ModelSpec("x", "X", "t", "a", "fp32", tmp_path / "nope", lambda d: None, "cmd")]
    monkeypatch.setattr(R, "SPECS", fake)
    assert R.load_models(verbose=False) == {}
    assert load_models is not None
