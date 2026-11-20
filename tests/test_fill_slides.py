"""fill_slides: tokens come only from result files; the template has every token it needs."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from fill_slides import build_tokens, fill  # noqa: E402

TEMPLATE = ROOT / "docs" / "presentation.pptx"


def _final():
    rows = [{"key": k, "model": k, "macro_f1": f, "size_mb": s, "latency_mean_ms": lat,
             "latency_p95_ms": lat * 1.5}
            for k, f, s, lat in [("lexicon_rule", 0.5, 0.004, 0.2), ("baseline", 0.8, 20.0, 1.0),
                                 ("neural_net", 0.7, 40.0, 0.8), ("bert_fp32", 0.9, 400.0, 60.0),
                                 ("bert_int8", 0.89, 170.0, 20.0)]]
    return {"n_test": 100, "runs": 200, "env": {"hardware": "CPU X"}, "rows": rows}


def test_build_tokens_derived_values():
    t, charts = build_tokens(_final(), None, None, "R", "V", None, None)
    assert t["bert_int8.OK"] == "✅" and t["bert_fp32.OK"] == "❌"
    assert t["INT8_SPEEDUP"] == "3.0" and t["INT8_F1_DELTA"] == "-0.0100"
    assert t["BEST_F1_MODEL"] == "BERT fp32" and t["N_WITHIN"] == "4"
    assert t["lexicon_rule.SIZE"] == "0.004" and t["HARDWARE"] == "CPU X"
    assert charts["chart latency"] == [0.2, 1.0, 0.8, 60.0, 20.0]
    assert "N_TRAIN" not in t          # nothing invented when data_stats.json is missing


def test_missing_results_leave_tokens_visible(tmp_path):
    pytest.importorskip("pptx")
    remaining = fill(TEMPLATE, tmp_path / "out.pptx", {}, {})
    assert "{{bert_int8.F1}}" in remaining and "{{REPO_URL}}" in remaining


def test_full_fill_leaves_nothing(tmp_path):
    pytest.importorskip("pptx")
    data = {"n_total": 10, "n_train": 8, "n_val": 1, "n_test": 1, "pct_offensive": 10.0}
    fails = {"models": {}, "slide_example": {"model": "M", "text": "t", "true": "Offensive",
                                             "pred": "Clean", "prob_offensive": 0.1}}
    t, charts = build_tokens(_final(), data, fails, "R", "V", None, "why")
    assert fill(TEMPLATE, tmp_path / "out.pptx", t, charts) == []
