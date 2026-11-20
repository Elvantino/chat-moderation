"""One place that knows how to find and load each tier of the model ladder.

Used by src/evaluate.py, src/error_analysis.py and app.py so they always load
identical artifacts.  Imports are lazy: the classical models work even if
torch/transformers are not installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src import config as C


def _load_rule(device="cpu"):
    from src.rule_baseline import LexiconRuleModel
    return LexiconRuleModel()


def _load_baseline(device="cpu"):
    from src.baseline import BaselineModel
    return BaselineModel.load(C.BASELINE_PATH)


def _load_nn(device="cpu"):
    from src.neural_net import NeuralModel
    return NeuralModel.load(C.NN_MODEL_PATH, device=device)


def _load_bert(device="cpu"):
    from src.transformer_model import BertClassifier
    return BertClassifier.load(C.BERT_DIR, device=device)


def _load_bert_int8(device="cpu"):
    from src.quantize import load_quantized
    return load_quantized(C.BERT_INT8_DIR)          # dynamic INT8 is CPU-only


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display: str
    tier: str
    area: str
    precision: str
    path: Path          # artifact counted for "Size (MB)"
    loader: Callable
    train_cmd: str

    def exists(self) -> bool:
        return self.path.exists()


SPECS: list[ModelSpec] = [
    ModelSpec("lexicon_rule", "Lexicon rule (edit distance)", "Rule baseline",
              "Area B – Classical NLP", "rules", C.LEXICON_DIR, _load_rule,
              "python -m src.baseline  (writes the learned lexicon)"),
    ModelSpec("baseline", "TF-IDF + LogReg", "Baseline", "Area B – Classical NLP",
              "fp64", C.BASELINE_PATH, _load_baseline, "python -m src.baseline"),
    ModelSpec("neural_net", "Word2Vec + FFNN (SGD)", "Neural Net", "Area A – Neural Networks",
              "fp32", C.NN_MODEL_PATH, _load_nn, "python -m src.neural_net"),
    ModelSpec("bert_fp32", "BERT-base (fp32)", "Transformer", "Area C – Transformers",
              "fp32", C.BERT_DIR, _load_bert, "python -m src.transformer_model"),
    ModelSpec("bert_int8", "BERT-base (dynamic INT8)", "Quantized", "Area D – Efficient Models",
              "int8 (Linear) + fp32 (embeddings)", C.BERT_INT8_DIR, _load_bert_int8,
              "python -m src.quantize"),
]
SPEC_BY_KEY = {s.key: s for s in SPECS}


def load_models(keys=None, device: str = "cpu", verbose: bool = True) -> dict:
    """Return {key: (spec, model)} for every requested model whose artifact exists."""
    log = print if verbose else (lambda *a, **k: None)
    out = {}
    for spec in SPECS:
        if keys is not None and spec.key not in keys:
            continue
        if not spec.exists():
            log(f"[skip] {spec.display}: {spec.path} not found (train with `{spec.train_cmd}`)")
            continue
        try:
            out[spec.key] = (spec, spec.loader(device))
            log(f"[ok]   {spec.display} loaded")
        except Exception as e:  # missing optional dependency, corrupt file, ...
            log(f"[fail] {spec.display}: {type(e).__name__}: {e}")
    return out
