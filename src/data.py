"""Dataset loading and the single, frozen train/val/test split.

Usage (run once, before training any model):
    python -m src.data

Expects the Kaggle Jigsaw file at data/raw/train.csv:
    kaggle competitions download -c jigsaw-toxic-comment-classification-challenge
    unzip ... && mv train.csv data/raw/
"""
from __future__ import annotations

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config as C


def load_raw_jigsaw(path=C.RAW_TRAIN_CSV, max_rows: int | None = C.MAX_ROWS) -> pd.DataFrame:
    """Load Jigsaw train.csv and collapse its six labels into one binary target."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the Jigsaw Toxic Comment Classification "
            "Challenge data from Kaggle and place train.csv in data/raw/."
        )
    df = pd.read_csv(path)
    missing = [c for c in ["comment_text", *C.JIGSAW_LABELS] if c not in df.columns]
    if missing:
        raise ValueError(f"Unexpected Jigsaw schema, missing columns: {missing}")

    out = pd.DataFrame({
        C.TEXT_COL: df["comment_text"].fillna("").astype(str),
        C.LABEL_COL: (df[C.JIGSAW_LABELS].sum(axis=1) > 0).astype(int),
    })
    out = out[out[C.TEXT_COL].str.strip().str.len() > 0].drop_duplicates(C.TEXT_COL)

    if max_rows is not None and len(out) > max_rows:
        out, _ = train_test_split(
            out, train_size=max_rows, stratify=out[C.LABEL_COL], random_state=C.SEED
        )
    return out.reset_index(drop=True)


def make_splits(df: pd.DataFrame):
    """Stratified 80/10/10 split with a fixed seed."""
    train_val, test = train_test_split(
        df, test_size=C.TEST_FRACTION, stratify=df[C.LABEL_COL], random_state=C.SEED
    )
    val_rel = C.VAL_FRACTION / (1.0 - C.TEST_FRACTION)
    train, val = train_test_split(
        train_val, test_size=val_rel, stratify=train_val[C.LABEL_COL], random_state=C.SEED
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def load_splits():
    """Load the frozen splits used by *every* model. Raises if not prepared yet."""
    for p in (C.TRAIN_CSV, C.VAL_CSV, C.TEST_CSV):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing - run `python -m src.data` first.")
    read = lambda p: pd.read_csv(p, keep_default_na=False)
    return read(C.TRAIN_CSV), read(C.VAL_CSV), read(C.TEST_CSV)


def main():
    ap = argparse.ArgumentParser(description="Create frozen train/val/test splits.")
    ap.add_argument("--max-rows", type=int, default=C.MAX_ROWS,
                    help="Optional stratified subsample size before splitting.")
    args = ap.parse_args()

    df = load_raw_jigsaw(max_rows=args.max_rows)
    train, val, test = make_splits(df)
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(C.TRAIN_CSV, index=False)
    val.to_csv(C.VAL_CSV, index=False)
    test.to_csv(C.TEST_CSV, index=False)

    stats = {"source": "Jigsaw Toxic Comment Classification Challenge (train.csv)",
             "n_total": len(df), "seed": C.SEED, "max_rows": args.max_rows}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        pos = part[C.LABEL_COL].mean() * 100
        stats[f"n_{name}"] = len(part)
        stats[f"pct_offensive_{name}"] = pos
        print(f"{name:>5}: {len(part):>7,} rows | offensive = {pos:5.2f}%")
    stats["pct_offensive"] = df[C.LABEL_COL].mean() * 100
    from src.metrics import save_results
    print(f"stats -> {save_results('data_stats', stats)}")


if __name__ == "__main__":
    main()
