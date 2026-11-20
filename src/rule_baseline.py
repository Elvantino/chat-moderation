"""Rule baseline (Area B): the edit-distance lexicon matcher used as a classifier.

A message is "Offensive" iff the LexiconMatcher finds at least one term
(exact, prefix, phrase, or OSA edit-distance 1 match after regex
normalisation).  No training: the lexicon is the hand-written seed list plus
data/lexicon/learned_lexicon.txt, which src/baseline.py derives from the
TRAIN split only, so evaluating on the test split is leak-free.

The same matcher masks flagged words in the app.
"""
from __future__ import annotations

import numpy as np

from src.preprocessing import LexiconMatcher


class LexiconRuleModel:
    name = "Lexicon rule (edit distance)"

    def __init__(self, lexicon: LexiconMatcher | None = None):
        self.lexicon = lexicon or LexiconMatcher.default()
        self.threshold = 0.5

    def predict_proba(self, texts) -> np.ndarray:
        return np.array([1.0 if self.lexicon.find(t) else 0.0 for t in texts], dtype=np.float32)

    def predict(self, texts) -> np.ndarray:
        return (self.predict_proba(texts) >= self.threshold).astype(int)

    def predict_one(self, text: str) -> tuple[int, float]:
        hit = bool(self.lexicon.find(text))
        return int(hit), float(hit)
