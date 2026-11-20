"""Run the whole ladder end-to-end, skipping steps whose artifacts already exist.

    python -m src.pipeline                      # everything
    python -m src.pipeline --force              # retrain everything
    python -m src.pipeline --bert-train-limit 40000
    python -m src.pipeline --bert-train-limit 20000 --bert-epochs 1   # quick check (~10-15 min on a laptop GPU)
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from src import config as C


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run([sys.executable, "-m", *cmd], check=True, cwd=C.ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--bert-train-limit", type=int, default=None,
                    help="stratified subsample of the TRAIN split for BERT (test split unchanged)")
    ap.add_argument("--bert-epochs", type=float, default=None)
    ap.add_argument("--cpu", action="store_true",
                    help="CPU-only machine: BERT tunes its threshold on a 4,000-message "
                         "validation subset, skips per-epoch eval, and every model is "
                         "scored on the (full) test split once, in src.evaluate")
    args = ap.parse_args()
    f = args.force

    if f or not C.TEST_CSV.exists():
        run(["src.data"])
    if f or not C.BASELINE_PATH.exists():
        run(["src.baseline"])
    if f or not C.NN_MODEL_PATH.exists():
        run(["src.neural_net"])
    if f or not C.BERT_DIR.exists():
        cmd = ["src.transformer_model"]
        if args.bert_train_limit:
            cmd += ["--train-limit", str(args.bert_train_limit)]
        if args.bert_epochs:
            cmd += ["--epochs", str(args.bert_epochs)]
        if args.cpu:
            cmd += ["--val-limit", "4000", "--no-epoch-eval", "--skip-test", "--workers", "0"]
        run(cmd)
    if f or not C.BERT_INT8_DIR.exists():
        run(["src.quantize"] + (["--val-limit", "4000", "--skip-test"] if args.cpu else []))
    run(["src.evaluate"])
    run(["src.error_analysis"])


if __name__ == "__main__":
    main()
