"""Tier 1 – Classical baseline: regex normalisation + TF-IDF + Logistic Regression.

Train + evaluate + benchmark (writes models/baseline_tfidf_lr.joblib and
models/results/baseline.json):
    python -m src.baseline
    python -m src.baseline --no-char      # word n-grams only (smaller, faster)

Features:
  * word 1-2 gram TF-IDF on normalised text
  * char_wb 2-5 gram TF-IDF (robust to obfuscations the regex misses)
Classifier: class-balanced Logistic Regression; the decision threshold is
tuned on the validation split for macro-F1 and stored with the model.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src import config as C
from src import metrics as M
from src.preprocessing import normalize_text


def build_pipeline(use_char: bool = True, C_reg: float = 4.0) -> Pipeline:
    word = TfidfVectorizer(
        preprocessor=normalize_text,       # module-level fn -> picklable
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        max_features=100_000,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    features = [("word", word)]
    if use_char:
        char = TfidfVectorizer(
            preprocessor=normalize_text,
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=3,
            max_features=150_000,
            sublinear_tf=True,
        )
        features.append(("char", char))
    clf = LogisticRegression(
        C=C_reg, class_weight="balanced", solver="liblinear", max_iter=2000,
        random_state=C.SEED,
    )
    return Pipeline([("features", FeatureUnion(features)), ("clf", clf)])


class BaselineModel:
    """Thin wrapper that bundles the sklearn pipeline with its tuned threshold."""

    name = "TF-IDF + LogReg"

    def __init__(self, pipeline: Pipeline, threshold: float = 0.5):
        self.pipeline = pipeline
        self.threshold = threshold

    def predict_proba(self, texts) -> np.ndarray:
        """Probability of the Offensive class."""
        return self.pipeline.predict_proba(list(texts))[:, 1]

    def predict(self, texts) -> np.ndarray:
        return (self.predict_proba(texts) >= self.threshold).astype(int)

    def predict_one(self, text: str) -> tuple[int, float]:
        p = float(self.predict_proba([text])[0])
        return int(p >= self.threshold), p

    def save(self, path: Path = C.BASELINE_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "threshold": self.threshold}, path, compress=3)
        return path

    @classmethod
    def load(cls, path: Path = C.BASELINE_PATH) -> "BaselineModel":
        obj = joblib.load(path)
        return cls(obj["pipeline"], obj["threshold"])


# Identity terms are never learned as "offensive": Jigsaw over-represents them in
# toxic comments, and masking them would penalise people for naming who they are.
IDENTITY_TERMS = {
    "gay", "gays", "lesbian", "lesbians", "homosexual", "homosexuals", "bisexual", "trans",
    "transgender", "queer", "black", "blacks", "white", "whites", "asian", "asians", "african",
    "jew", "jews", "jewish", "muslim", "muslims", "islam", "islamic", "christian", "christians",
    "arab", "arabs", "mexican", "mexicans", "indian", "indians", "chinese", "women", "woman",
    "female", "male", "girl", "girls", "men", "man", "catholic", "hindu", "atheist", "disabled",
}


def learned_lexicon_terms(model: BaselineModel, train_texts, train_labels, k: int = 150,
                          candidates: int = 600, min_count: int = 20,
                          min_precision: float = 0.6, min_len: int = 3) -> list[str]:
    """Data-driven lexicon for the rule baseline and masking (TRAIN split only).

    Candidates are the strongest positive TF-IDF+LR word weights, but a weight
    alone also picks words like "you" or "are" that *predict* insults without
    being offensive.  A term is kept only if, in the training messages that
    contain it, at least ``min_precision`` are offensive (seen >= ``min_count``
    times), and it is not an identity term.
    """
    import re
    from collections import Counter

    from src.preprocessing import normalize_text

    cands = [t for t in top_offensive_features(model, k=candidates, min_len=min_len)
             if t not in IDENTITY_TERMS]
    cand_set = set(cands)
    n_docs, n_off = Counter(), Counter()
    for text, y in zip(train_texts, train_labels):
        toks = set(re.findall(r"\w+", normalize_text(text))) & cand_set
        for t in toks:
            n_docs[t] += 1
            n_off[t] += int(y)
    keep = [t for t in cands
            if n_docs[t] >= min_count and n_off[t] / n_docs[t] >= min_precision]
    return keep[:k]


def write_learned_lexicon(model: BaselineModel, train: pd.DataFrame,
                          path: Path = C.LEARNED_LEXICON_PATH) -> list[str]:
    terms = learned_lexicon_terms(model, train[C.TEXT_COL].tolist(), train[C.LABEL_COL].values)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Auto-generated (src/baseline.py) from the TRAIN split: strongest TF-IDF+LR word\n"
        "# weights, kept only if >= 60% of training messages containing them are offensive.\n"
        "# Identity terms are excluded.\n" + "\n".join(terms) + "\n", encoding="utf-8")
    return terms


def top_offensive_features(model: BaselineModel, k: int = 150, min_len: int = 3) -> list[str]:
    """Strongest positive *word unigram* weights -> data-driven lexicon for masking."""
    union: FeatureUnion = model.pipeline.named_steps["features"]
    word_vec: TfidfVectorizer = dict(union.transformer_list)["word"]
    n_word = len(word_vec.vocabulary_)
    coefs = model.pipeline.named_steps["clf"].coef_[0][:n_word]
    vocab = word_vec.get_feature_names_out()
    order = np.argsort(coefs)[::-1]
    out = []
    for i in order:
        term = vocab[i]
        if coefs[i] <= 0:
            break
        if " " in term or len(term) < min_len or term.isdigit():
            continue
        out.append(term)
        if len(out) >= k:
            break
    return out


def train_and_evaluate(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
    use_char: bool = True, save_path: Path = C.BASELINE_PATH,
    lexicon_path: Path | None = C.LEARNED_LEXICON_PATH,
    latency_runs: int = C.LATENCY_RUNS, verbose: bool = True,
) -> dict:
    log = print if verbose else (lambda *a, **k: None)

    log(f"Training on {len(train):,} messages (char n-grams: {use_char}) ...")
    t0 = time.perf_counter()
    pipe = build_pipeline(use_char=use_char)
    pipe.fit(train[C.TEXT_COL].tolist(), train[C.LABEL_COL].values)
    train_s = time.perf_counter() - t0
    log(f"  done in {train_s:.1f}s")

    model = BaselineModel(pipe)
    val_proba = model.predict_proba(val[C.TEXT_COL])
    model.threshold, val_f1 = M.best_threshold(val[C.LABEL_COL].values, val_proba)
    log(f"  tuned threshold = {model.threshold:.2f} (val macro-F1 = {val_f1:.4f})")

    y_test = test[C.LABEL_COL].values
    y_pred = model.predict(test[C.TEXT_COL])
    test_f1 = M.macro_f1(y_test, y_pred)
    log("\nTest set:\n" + M.report(y_test, y_pred))

    model.save(save_path)
    size = M.size_mb(save_path)

    lat = M.benchmark_latency(lambda m: model.predict_one(m),
                              test[C.TEXT_COL].tolist(), runs=latency_runs)

    if lexicon_path is not None:
        terms = write_learned_lexicon(model, train, lexicon_path)
        log(f"  wrote {len(terms)} learned lexicon terms -> {lexicon_path}")

    results = {
        "model": BaselineModel.name,
        "macro_f1": test_f1,
        "val_macro_f1": val_f1,
        "threshold": model.threshold,
        "size_mb": size,
        "train_seconds": train_s,
        "n_train": len(train),
        "n_test": len(test),
        **lat,
    }
    log(f"\nMacro-F1 = {test_f1:.4f} | size = {size:.2f} MB | "
        f"latency = {lat['latency_mean_ms']:.3f} ms (mean of {lat['latency_runs']} runs)")
    return results


def main():
    ap = argparse.ArgumentParser(description="Train/evaluate the TF-IDF + LR baseline.")
    ap.add_argument("--no-char", action="store_true", help="disable char n-gram features")
    ap.add_argument("--runs", type=int, default=C.LATENCY_RUNS)
    ap.add_argument("--lexicon-only", action="store_true",
                    help="rebuild data/lexicon/learned_lexicon.txt from the saved model")
    args = ap.parse_args()

    from src.data import load_splits
    train, val, test = load_splits()
    if args.lexicon_only:
        terms = write_learned_lexicon(BaselineModel.load(), train)
        print(f"wrote {len(terms)} learned lexicon terms -> {C.LEARNED_LEXICON_PATH}")
        return
    results = train_and_evaluate(train, val, test, use_char=not args.no_char,
                                 latency_runs=args.runs)
    out = M.save_results("baseline", results)
    print(f"Results saved -> {out}")


if __name__ == "__main__":
    main()
