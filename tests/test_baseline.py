"""Smoke test: the baseline trains, saves, reloads and beats chance on toy data."""
import random

import pandas as pd

from src import config as C
from src.baseline import BaselineModel, train_and_evaluate

CLEAN = ["good game everyone", "nice shot", "thanks for the help", "see you tomorrow",
         "that was a great match", "anyone want to queue", "gg well played", "lol nice"]
TOXIC = ["you are an idiot", "shut up loser", "f*ck you", "you stupid moron",
         "go kill yourself", "what a pathetic noob idiot", "sh1t player", "you suck idiot"]


def _toy(n, seed):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        if rng.random() < 0.3:
            rows.append((rng.choice(TOXIC) + " " + rng.choice(CLEAN), 1))
        else:
            rows.append((rng.choice(CLEAN) + " " + rng.choice(CLEAN), 0))
    return pd.DataFrame(rows, columns=[C.TEXT_COL, C.LABEL_COL])


def test_baseline_end_to_end(tmp_path):
    train, val, test = _toy(400, 0), _toy(100, 1), _toy(100, 2)
    path = tmp_path / "baseline.joblib"
    res = train_and_evaluate(train, val, test, save_path=path,
                             lexicon_path=tmp_path / "lex.txt",
                             latency_runs=100, verbose=False)
    assert res["macro_f1"] > 0.9
    assert res["size_mb"] > 0
    assert res["latency_runs"] >= 100
    assert res["latency_mean_ms"] > 0

    model = BaselineModel.load(path)
    label, p = model.predict_one("you stupid idiot")
    assert label == 1 and 0.0 <= p <= 1.0
    lex = (tmp_path / "lex.txt").read_text().split()
    assert "idiot" in lex
    assert "you" not in lex            # predictive but not offensive -> filtered by precision
