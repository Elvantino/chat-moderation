"""Find real failure examples on the test split (for the Limitations slide).

    python -m src.error_analysis                   # all available models
    python -m src.error_analysis --top 10

For each model: the most confident false positives / false negatives
(short messages preferred so they fit on a slide).  Also reports where the
primary model (INT8 BERT) disagrees with the TF-IDF baseline.

Writes models/results/failure_examples.json and failure_examples.md.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from src import config as C
from src.registry import load_models

MAX_CHARS = 160


def failures(texts, y_true, proba, threshold, top=5, max_chars=MAX_CHARS) -> dict:
    """Most confident errors; prefers messages of <= max_chars characters."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    short = np.array([len(t) <= max_chars for t in texts])

    def pick(mask, key):
        idx = np.where(mask & short)[0]
        if len(idx) == 0:
            idx = np.where(mask)[0]
        idx = idx[np.argsort(key[idx])][:top]
        return [{"text": texts[i], "true": C.LABEL_NAMES[int(y_true[i])],
                 "pred": C.LABEL_NAMES[int(pred[i])], "prob_offensive": float(proba[i])}
                for i in idx]

    fp = pick((pred == 1) & (y_true == 0), -proba)     # most confident wrong "Offensive"
    fn = pick((pred == 0) & (y_true == 1), proba)      # most confident wrong "Clean"
    return {"false_positives": fp, "false_negatives": fn,
            "n_fp": int(((pred == 1) & (y_true == 0)).sum()),
            "n_fn": int(((pred == 0) & (y_true == 1)).sum())}


def to_markdown(report: dict) -> str:
    out = ["# Failure examples (test split)\n"]
    for key, r in report["models"].items():
        out.append(f"## {r['model']}  —  {r['n_fp']} false positives, {r['n_fn']} false negatives\n")
        for kind in ("false_negatives", "false_positives"):
            out.append(f"**{kind.replace('_', ' ').title()}**\n")
            for e in r[kind]:
                text = e["text"].replace("\n", " ").replace("|", "\\|")
                out.append(f"- p(offensive)={e['prob_offensive']:.3f} · true={e['true']} · “{text}”")
            out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Collect failure examples for the report.")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    from src.data import load_splits
    _, _, test = load_splits()
    texts = test[C.TEXT_COL].astype(str).tolist()
    y = test[C.LABEL_COL].values

    models = load_models(device="cpu")
    report = {"models": {}}
    for key, (spec, model) in models.items():
        cached = C.RESULTS_DIR / "test_probs" / f"{key}.npy"
        if cached.exists() and len(np.load(cached)) == len(texts):
            print(f"Using cached test probabilities for {spec.display}")
            proba = np.load(cached)
        else:
            print(f"Scoring {spec.display} ...")
            proba = model.predict_proba(texts)
        r = failures(texts, y, proba, model.threshold, top=args.top)
        report["models"][key] = {"model": spec.display, **r}

    primary = next((k for k in ("bert_int8", "bert_fp32", "neural_net", "baseline")
                    if k in report["models"]), None)
    if primary:
        r = report["models"][primary]
        # Slide example: prefer a missed offensive message (the costlier error in moderation)
        ex = (r["false_negatives"] or r["false_positives"] or [None])[0]
        report["slide_example"] = {"model": r["model"], **ex} if ex else None

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.RESULTS_DIR / "failure_examples.json").write_text(json.dumps(report, indent=2))
    (C.RESULTS_DIR / "failure_examples.md").write_text(to_markdown(report), encoding="utf-8")
    print(f"Saved -> {C.RESULTS_DIR / 'failure_examples.md'}")
    if report.get("slide_example"):
        e = report["slide_example"]
        print(f"\nSlide example ({e['model']}): “{e['text']}” true={e['true']} "
              f"pred={e['pred']} p={e['prob_offensive']:.3f}")


if __name__ == "__main__":
    main()
