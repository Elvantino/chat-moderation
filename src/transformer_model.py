"""Tier 3 – Area C Transformers: fine-tune bert-base-uncased with Hugging Face Trainer.

Run (after `python -m src.data`), on the RTX 3050 Ti (4 GB VRAM):
    python -m src.transformer_model                          # full train split, 2 epochs
    python -m src.transformer_model --train-limit 40000      # faster: subsample TRAIN only
    python -m src.transformer_model --batch-size 8 --grad-accum 4   # if you hit CUDA OOM

`--train-limit` only shrinks the training split; val/test stay the frozen
splits, so the results table remains comparable across all four models.

Outputs
    models/bert_finetuned/              save_pretrained() model + tokenizer +
                                        moderation_config.json (threshold, max_length)
    models/results/transformer_fp32.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src import config as C
from src import metrics as M
from src.preprocessing import normalize_text

MOD_CFG_NAME = "moderation_config.json"


# --------------------------------------------------------------------------- #
# Inference wrapper shared by the fp32 and INT8 models
# --------------------------------------------------------------------------- #
class BertClassifier:
    """Same interface as BaselineModel / NeuralModel: predict_proba, predict, predict_one."""

    def __init__(self, model, tokenizer, threshold: float = 0.5,
                 max_length: int = C.BERT_MAX_LENGTH, normalize: bool = True,
                 name: str = "BERT-base (fp32)", device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.threshold = float(threshold)
        self.max_length = int(max_length)
        self.normalize = bool(normalize)
        self.name = name

    def _prep(self, texts) -> list[str]:
        return [normalize_text(t) if self.normalize else str(t) for t in texts]

    @torch.inference_mode()
    def predict_proba(self, texts, batch_size: int = 64) -> np.ndarray:
        texts = self._prep(list(texts))
        probs = np.zeros(len(texts), dtype=np.float32)
        order = np.argsort([len(t) for t in texts])      # length-sorted -> less padding
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            enc = self.tokenizer([texts[j] for j in idx], truncation=True,
                                 max_length=self.max_length, padding=True,
                                 return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits
            probs[idx] = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
        return probs

    def predict(self, texts) -> np.ndarray:
        return (self.predict_proba(texts) >= self.threshold).astype(int)

    @torch.inference_mode()
    def predict_one(self, text: str) -> tuple[int, float]:
        # No padding for a single message: this is what keeps CPU latency low.
        enc = self.tokenizer(self._prep([text]), truncation=True, max_length=self.max_length,
                             return_tensors="pt").to(self.device)
        p = float(torch.softmax(self.model(**enc).logits.float(), dim=-1)[0, 1])
        return int(p >= self.threshold), p

    # ------------------------------------------------------------------ io
    def moderation_config(self) -> dict:
        return {"threshold": self.threshold, "max_length": self.max_length,
                "normalize": self.normalize}

    @classmethod
    def load(cls, path: Path = C.BERT_DIR, device: str = "cpu") -> "BertClassifier":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        path = Path(path)
        cfg = json.loads((path / MOD_CFG_NAME).read_text())
        model = AutoModelForSequenceClassification.from_pretrained(path)
        tok = AutoTokenizer.from_pretrained(path)
        return cls(model, tok, cfg["threshold"], cfg["max_length"], cfg["normalize"],
                   name="BERT-base (fp32)", device=device)


def write_moderation_config(path: Path, threshold: float, max_length: int, normalize: bool,
                            extra: dict | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    data = {"threshold": float(threshold), "max_length": int(max_length),
            "normalize": bool(normalize), **(extra or {})}
    (path / MOD_CFG_NAME).write_text(json.dumps(data, indent=2))


# --------------------------------------------------------------------------- #
# Fine-tuning
# --------------------------------------------------------------------------- #
def class_weights(labels) -> torch.Tensor:
    """'balanced' weights: n / (2 * count_c)."""
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    counts[counts == 0] = 1.0
    return torch.tensor(len(labels) / (2.0 * counts), dtype=torch.float32)


def _make_weighted_trainer_cls():
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        """Trainer with class-weighted cross-entropy (Jigsaw is ~90% clean)."""

        def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            w = None if self.class_weights is None else self.class_weights.to(logits.device)
            loss = nn.functional.cross_entropy(logits.float(), labels, weight=w)
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def _to_hf_dataset(df: pd.DataFrame, tokenizer, max_length: int, normalize: bool):
    from datasets import Dataset

    texts = df[C.TEXT_COL].astype(str)
    if normalize:
        texts = texts.map(normalize_text)
    ds = Dataset.from_dict({"text": texts.tolist(), "labels": df[C.LABEL_COL].astype(int).tolist()})
    return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=max_length),
                  batched=True, remove_columns=["text"])


def _stratified_limit(df: pd.DataFrame, n: int | None) -> pd.DataFrame:
    if n is None or n >= len(df):
        return df
    from sklearn.model_selection import train_test_split
    sub, _ = train_test_split(df, train_size=n, stratify=df[C.LABEL_COL], random_state=C.SEED)
    return sub.reset_index(drop=True)


def finetune(
    train: pd.DataFrame, val: pd.DataFrame,
    base_model: str = C.BERT_BASE_NAME, out_dir: Path = C.BERT_DIR,
    ckpt_dir: Path = C.BERT_CKPT_DIR, max_length: int = C.BERT_MAX_LENGTH,
    normalize: bool = True, epochs: float = 2.0, lr: float = 2e-5,
    batch_size: int = 16, grad_accum: int = 2, workers: int = 4,
    grad_checkpointing: bool = False, eval_during_training: bool = True,
    verbose: bool = True,
) -> dict:
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, EarlyStoppingCallback,
                              TrainingArguments, set_seed)

    log = print if verbose else (lambda *a, **k: None)
    set_seed(C.SEED)
    cuda = torch.cuda.is_available()
    log(f"CUDA: {cuda}" + (f" ({torch.cuda.get_device_name(0)})" if cuda else
                           " - WARNING: BERT fine-tuning on CPU will be very slow"))

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=2,
        id2label={0: C.LABEL_NAMES[0], 1: C.LABEL_NAMES[1]},
        label2id={C.LABEL_NAMES[0]: 0, C.LABEL_NAMES[1]: 1},
    )

    t0 = time.perf_counter()
    train_ds = _to_hf_dataset(train, tokenizer, max_length, normalize)
    val_ds = _to_hf_dataset(val, tokenizer, max_length, normalize)
    log(f"Tokenised {len(train_ds):,} train / {len(val_ds):,} val in {time.perf_counter()-t0:.1f}s")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return {"macro_f1": M.macro_f1(labels, np.argmax(logits, axis=-1))}

    # RTX 3050 Ti (4 GB): fp16 + batch 16 x accum 2 (effective 32) at max_len 128
    # peaks around ~3 GB.  Use --batch-size 8 --grad-accum 4 or --grad-checkpointing on OOM.
    args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 4,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=grad_checkpointing,
        warmup_ratio=0.06,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        fp16=cuda,
        # CPU runs (--no-epoch-eval): skip the per-epoch validation pass and
        # checkpointing; the threshold is still tuned on validation below.
        eval_strategy="epoch" if eval_during_training else "no",
        save_strategy="epoch" if eval_during_training else "no",
        save_total_limit=1,
        load_best_model_at_end=eval_during_training,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        group_by_length=True,               # bucket similar lengths -> faster epochs
        dataloader_num_workers=workers,
        dataloader_pin_memory=cuda,
        logging_steps=100,
        report_to="none",
        seed=C.SEED,
    )

    WeightedTrainer = _make_weighted_trainer_cls()
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if cuda else None),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)] if eval_during_training else [],
        class_weights=class_weights(train[C.LABEL_COL].values),
    )

    t1 = time.perf_counter()
    trainer.train()
    train_s = time.perf_counter() - t1

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.to("cpu").save_pretrained(out_dir)        # safetensors, fp32
    tokenizer.save_pretrained(out_dir)
    write_moderation_config(out_dir, 0.5, max_length, normalize,
                            extra={"base_model": base_model})

    # Threshold tuning on validation probabilities (same protocol as other tiers).
    # NOT trainer.predict(): with group_by_length=True it reorders the eval set by
    # length, so its predictions no longer line up with the labels.
    threshold, val_f1 = retune_threshold(out_dir, val, device="cuda" if cuda else "cpu")
    log(f"tuned threshold = {threshold:.2f} (val macro-F1 = {val_f1:.4f})")
    return {"threshold": threshold, "val_macro_f1": val_f1, "train_seconds": train_s,
            "n_train": len(train), "epochs": epochs,
            "peak_vram_gb": (torch.cuda.max_memory_allocated() / 1024**3) if cuda else 0.0}


def retune_threshold(path: Path, val: pd.DataFrame, device: str = "cpu") -> tuple[float, float]:
    """Score the validation split in order with the saved model, tune the threshold
    for macro-F1, and write it back into moderation_config.json."""
    clf = BertClassifier.load(path, device=device)
    threshold, val_f1 = M.best_threshold(val[C.LABEL_COL].values,
                                         clf.predict_proba(val[C.TEXT_COL].tolist()))
    cfg_path = Path(path) / MOD_CFG_NAME
    cfg = json.loads(cfg_path.read_text())
    cfg["threshold"] = float(threshold)
    cfg_path.write_text(json.dumps(cfg, indent=2))
    return threshold, val_f1


def evaluate_saved(path: Path, test: pd.DataFrame, f1_device: str = "cpu",
                   latency_runs: int = C.LATENCY_RUNS, verbose: bool = True) -> dict:
    """Test macro-F1 (optionally on GPU) + CPU latency of the saved fp32 model."""
    log = print if verbose else (lambda *a, **k: None)
    y = test[C.LABEL_COL].values
    clf = BertClassifier.load(path, device=f1_device)
    y_pred = clf.predict(test[C.TEXT_COL].tolist())
    f1 = M.macro_f1(y, y_pred)
    log("\nTest set:\n" + M.report(y, y_pred))
    if f1_device != "cpu":
        clf = BertClassifier.load(path, device="cpu")
    lat = M.benchmark_latency(clf.predict_one, test[C.TEXT_COL].tolist(), runs=latency_runs)
    return {"macro_f1": f1, "size_mb": M.size_mb(path), "latency_device": "cpu",
            "torch_threads": torch.get_num_threads(), **lat}


def main():
    ap = argparse.ArgumentParser(description="Fine-tune bert-base-uncased (Trainer).")
    ap.add_argument("--base-model", default=C.BERT_BASE_NAME)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=C.BERT_MAX_LENGTH)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--train-limit", type=int, default=None,
                    help="stratified subsample of the TRAIN split only")
    ap.add_argument("--val-limit", type=int, default=None,
                    help="stratified subsample of the VALIDATION split for threshold tuning "
                         "(CPU time saver; the test split is never subsampled)")
    ap.add_argument("--no-epoch-eval", action="store_true",
                    help="no per-epoch validation / checkpointing (faster on CPU)")
    ap.add_argument("--skip-test", action="store_true",
                    help="skip test-set scoring here; src.evaluate scores every model once")
    ap.add_argument("--retune-only", action="store_true",
                    help="no training: re-tune the saved model's threshold on validation")
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--no-normalize", action="store_true",
                    help="feed raw text to BERT instead of normalize_text()")
    ap.add_argument("--runs", type=int, default=C.LATENCY_RUNS)
    args = ap.parse_args()

    from src.data import load_splits
    train, val, test = load_splits()
    train = _stratified_limit(train, args.train_limit)
    val = _stratified_limit(val, args.val_limit)

    if args.retune_only:
        t, f = retune_threshold(C.BERT_DIR, val)
        res_path = C.RESULTS_DIR / "transformer_fp32.json"
        res = json.loads(res_path.read_text()) if res_path.exists() else {"model": "BERT-base (fp32)"}
        res.update({"threshold": t, "val_macro_f1": f, "n_val_for_threshold": len(val)})
        print(f"threshold = {t:.2f} (val macro-F1 = {f:.4f}) -> {M.save_results('transformer_fp32', res)}")
        return

    info = finetune(train, val, base_model=args.base_model, max_length=args.max_length,
                    normalize=not args.no_normalize, epochs=args.epochs, lr=args.lr,
                    batch_size=args.batch_size, grad_accum=args.grad_accum,
                    workers=args.workers, grad_checkpointing=args.grad_checkpointing,
                    eval_during_training=not args.no_epoch_eval)
    info["n_val_for_threshold"] = len(val)
    if args.skip_test:
        results = {"model": "BERT-base (fp32)", **info, "size_mb": M.size_mb(C.BERT_DIR)}
        print(f"Results saved -> {M.save_results('transformer_fp32', results)} "
              "(test metrics: run python -m src.evaluate)")
        return
    ev = evaluate_saved(C.BERT_DIR, test,
                        f1_device="cuda" if torch.cuda.is_available() else "cpu",
                        latency_runs=args.runs)
    results = {"model": "BERT-base (fp32)", "n_test": len(test), **info, **ev}
    print(f"\nMacro-F1 = {ev['macro_f1']:.4f} | size = {ev['size_mb']:.1f} MB | "
          f"latency = {ev['latency_mean_ms']:.2f} ms (CPU, {ev['latency_runs']} runs)")
    print(f"Results saved -> {M.save_results('transformer_fp32', results)}")


if __name__ == "__main__":
    main()
