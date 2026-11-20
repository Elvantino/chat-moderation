"""Tier 2 – Area A Neural Networks: custom Word2Vec + feed-forward net trained with SGD.

Pipeline
    raw text -> normalize_text/tokenize (src.preprocessing)
             -> token ids (Word2Vec vocabulary trained on the TRAIN split only)
             -> nn.EmbeddingBag(mode="mean")   # averaged word embeddings,
                                               # initialised from Word2Vec
             -> LayerNorm -> MLP (ReLU, dropout) -> logit (Offensive)

Training: torch.optim.SGD (momentum) + backprop, BCE loss with pos_weight for
class imbalance, ReduceLROnPlateau, early stopping on validation loss.
The decision threshold is tuned on the validation split for macro-F1
(same protocol as the baseline).

Run (after `python -m src.data`):
    python -m src.neural_net
    python -m src.neural_net --finetune-emb --epochs 40 --batch-size 1024

Outputs
    models/neural_net/model.pt          everything needed for inference
    models/neural_net/word2vec.kv       gensim KeyedVectors (for analysis)
    models/results/neural_net.json      metrics for the results table
"""
from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src import config as C
from src import metrics as M
from src.preprocessing import tokenize

NN_DIR = C.NN_DIR
NN_MODEL_PATH = C.NN_MODEL_PATH
W2V_PATH = NN_DIR / "word2vec.kv"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class W2VConfig:
    vector_size: int = 200
    window: int = 5
    min_count: int = 3
    max_final_vocab: int | None = 50_000   # caps embedding matrix (~40 MB at dim 200)
    sg: int = 1                            # skip-gram: better for rare/obfuscated words
    negative: int = 10
    epochs: int = 10
    workers: int = field(default_factory=lambda: max(1, os.cpu_count() or 1))


@dataclass
class NNConfig:
    hidden: tuple[int, ...] = (256, 64)
    dropout: float = 0.3
    finetune_emb: bool = False             # False = pure Word2Vec features
    max_len: int = 400                     # tokens kept per message
    batch_size: int = 512                  # <0.5 GB VRAM; safe on a 4 GB 3050 Ti
    eval_batch_size: int = 2048
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 1e-5
    max_epochs: int = 30
    patience: int = 4                      # early-stopping patience (epochs)
    min_delta: float = 1e-4
    grad_clip: float = 5.0
    num_workers: int = field(default_factory=lambda: min(4, os.cpu_count() or 1))


def set_seed(seed: int = C.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Word2Vec
# --------------------------------------------------------------------------- #
def tokenize_corpus(texts: Sequence[str]) -> list[list[str]]:
    return [tokenize(t) for t in texts]


def train_word2vec(token_lists: list[list[str]], cfg: W2VConfig = W2VConfig(),
                   seed: int = C.SEED):
    """Train gensim Word2Vec on the TRAIN split only and return KeyedVectors.

    Note: gensim is only bit-for-bit reproducible with workers=1 and a fixed
    PYTHONHASHSEED; multi-worker training is much faster and the variance in
    downstream F1 is negligible.
    """
    from gensim.models import Word2Vec

    w2v = Word2Vec(
        sentences=token_lists,
        vector_size=cfg.vector_size,
        window=cfg.window,
        min_count=cfg.min_count,
        max_final_vocab=cfg.max_final_vocab,
        sg=cfg.sg,
        negative=cfg.negative,
        epochs=cfg.epochs,
        workers=cfg.workers,
        seed=seed,
    )
    return w2v.wv


def embedding_matrix(kv) -> tuple[list[str], np.ndarray]:
    """(vocab list, float32 matrix [V, D]) in KeyedVectors index order."""
    vocab = list(kv.index_to_key)
    return vocab, np.asarray(kv.vectors, dtype=np.float32)


def standardize_embeddings(emb: np.ndarray) -> np.ndarray:
    """Centre and scale each dimension of the Word2Vec matrix.

    Raw word2vec vectors share a large common direction; after averaging a
    message's words, that direction dominates and every message looks alike
    (and ReLUs downstream can all die).  Removing the per-dimension mean and
    dividing by the std keeps what distinguishes words.
    """
    emb = np.asarray(emb, dtype=np.float32)
    return ((emb - emb.mean(0)) / (emb.std(0) + 1e-6)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Encoding / data loading
# --------------------------------------------------------------------------- #
def encode(tokens: list[str], stoi: dict[str, int], max_len: int) -> list[int]:
    """Map tokens to ids, dropping out-of-vocabulary words, truncating to max_len."""
    ids = [stoi[t] for t in tokens if t in stoi]
    return ids[:max_len]


class BagDataset(Dataset):
    def __init__(self, ids: list[list[int]], labels: Sequence[int] | None = None):
        self.ids = ids
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        y = 0.0 if self.labels is None else float(self.labels[i])
        return self.ids[i], y


def collate_bags(batch):
    """Flatten variable-length id lists into (flat_ids, offsets, labels) for EmbeddingBag."""
    id_lists, ys = zip(*batch)
    lengths = [len(x) for x in id_lists]
    flat = torch.tensor([i for x in id_lists for i in x], dtype=torch.long)
    offsets = torch.zeros(len(lengths), dtype=torch.long)
    if len(lengths) > 1:
        offsets[1:] = torch.cumsum(torch.tensor(lengths[:-1], dtype=torch.long), dim=0)
    return flat, offsets, torch.tensor(ys, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class AvgEmbeddingClassifier(nn.Module):
    """Averaged word embeddings -> LayerNorm -> MLP -> single logit.

    An empty bag (all tokens OOV / empty message) yields a zero vector, so the
    model still produces a valid prediction.
    """

    def __init__(self, emb: np.ndarray | torch.Tensor, hidden=(256, 64),
                 dropout: float = 0.3, finetune_emb: bool = False):
        super().__init__()
        emb_t = torch.as_tensor(np.asarray(emb, dtype=np.float32))
        self.embedding = nn.EmbeddingBag.from_pretrained(
            emb_t, freeze=not finetune_emb, mode="mean")
        layers: list[nn.Module] = [nn.LayerNorm(emb_t.shape[1])]
        d_in = emb_t.shape[1]
        for h in hidden:
            layers += [nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(dropout)]
            d_in = h
        layers.append(nn.Linear(d_in, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, flat_ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        pooled = self.embedding(flat_ids, offsets)          # [B, D] mean of word vectors
        return self.mlp(pooled).squeeze(-1)                 # [B] logits


# --------------------------------------------------------------------------- #
# Inference wrapper (same interface as BaselineModel)
# --------------------------------------------------------------------------- #
class NeuralModel:
    name = "Word2Vec + FFNN (SGD)"

    def __init__(self, net: AvgEmbeddingClassifier, vocab: list[str], cfg: NNConfig,
                 threshold: float = 0.5, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.net = net.to(self.device).eval()
        self.vocab = vocab
        self.stoi = {w: i for i, w in enumerate(vocab)}
        self.cfg = cfg
        self.threshold = threshold

    def _ids(self, texts) -> list[list[int]]:
        return [encode(tokenize(t), self.stoi, self.cfg.max_len) for t in texts]

    @torch.inference_mode()
    def predict_proba(self, texts) -> np.ndarray:
        texts = list(texts)
        ids = self._ids(texts)
        out = []
        bs = self.cfg.eval_batch_size
        for i in range(0, len(ids), bs):
            flat, offs, _ = collate_bags([(x, 0.0) for x in ids[i:i + bs]])
            logits = self.net(flat.to(self.device), offs.to(self.device))
            out.append(torch.sigmoid(logits).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def predict(self, texts) -> np.ndarray:
        return (self.predict_proba(texts) >= self.threshold).astype(int)

    @torch.inference_mode()
    def predict_one(self, text: str) -> tuple[int, float]:
        ids = encode(tokenize(text), self.stoi, self.cfg.max_len)
        flat = torch.tensor(ids, dtype=torch.long, device=self.device)
        offs = torch.zeros(1, dtype=torch.long, device=self.device)
        p = float(torch.sigmoid(self.net(flat, offs))[0])
        return int(p >= self.threshold), p

    # ------------------------------------------------------------------ io
    def save(self, path: Path = NN_MODEL_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
        torch.save({
            "state_dict": state,
            "vocab": self.vocab,
            "cfg": asdict(self.cfg),
            "threshold": float(self.threshold),
            "emb_shape": list(self.net.embedding.weight.shape),
        }, path)
        return path

    @classmethod
    def load(cls, path: Path = NN_MODEL_PATH, device: str = "cpu") -> "NeuralModel":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        cfg_d = dict(ckpt["cfg"])
        cfg_d["hidden"] = tuple(cfg_d["hidden"])
        cfg = NNConfig(**cfg_d)
        net = AvgEmbeddingClassifier(np.zeros(ckpt["emb_shape"], dtype=np.float32),
                                     hidden=cfg.hidden, dropout=cfg.dropout,
                                     finetune_emb=cfg.finetune_emb)
        net.load_state_dict(ckpt["state_dict"])
        return cls(net, ckpt["vocab"], cfg, ckpt["threshold"], device)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _make_loader(ids, labels, batch_size, shuffle, cfg: NNConfig, device, seed=C.SEED):
    workers = cfg.num_workers
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        BagDataset(ids, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_bags,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=workers > 0,
        generator=g if shuffle else None,
    )


@torch.no_grad()
def _evaluate_loss(net, loader, criterion, device) -> tuple[float, np.ndarray]:
    net.eval()
    total, n, probs = 0.0, 0, []
    for flat, offs, y in loader:
        flat, offs, y = (flat.to(device, non_blocking=True), offs.to(device, non_blocking=True),
                         y.to(device, non_blocking=True))
        logits = net(flat, offs)
        total += criterion(logits, y).item() * y.size(0)
        n += y.size(0)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return total / max(n, 1), np.concatenate(probs)


def fit(net: AvgEmbeddingClassifier, train_ids, y_train, val_ids, y_val,
        cfg: NNConfig, device: torch.device, verbose: bool = True) -> dict:
    """SGD + backprop with early stopping on validation loss. Restores best weights."""
    log = print if verbose else (lambda *a, **k: None)
    net.to(device)

    y_train = np.asarray(y_train, dtype=np.float32)
    pos = max(float(y_train.sum()), 1.0)
    neg = max(float(len(y_train) - y_train.sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, device=device))

    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum,
                                nesterov=True, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                           factor=0.5, patience=1)

    train_loader = _make_loader(train_ids, y_train, cfg.batch_size, True, cfg, device)
    val_loader = _make_loader(val_ids, y_val, cfg.eval_batch_size, False, cfg, device)

    best_loss, best_state, bad_epochs = float("inf"), None, 0
    history = []
    for epoch in range(1, cfg.max_epochs + 1):
        net.train()
        t0, run_loss, seen = time.perf_counter(), 0.0, 0
        for flat, offs, y in train_loader:
            flat = flat.to(device, non_blocking=True)
            offs = offs.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            loss = criterion(net(flat, offs), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optimizer.step()
            run_loss += loss.item() * y.size(0)
            seen += y.size(0)

        train_loss = run_loss / max(seen, 1)
        val_loss, val_proba = _evaluate_loss(net, val_loader, criterion, device)
        val_f1 = M.macro_f1(y_val, (val_proba >= 0.5).astype(int))
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_macro_f1@0.5": val_f1, "lr": lr})
        log(f"epoch {epoch:>2} | train {train_loss:.4f} | val {val_loss:.4f} | "
            f"val F1@0.5 {val_f1:.4f} | lr {lr:.4f} | {time.perf_counter() - t0:.1f}s")

        if val_loss < best_loss - cfg.min_delta:
            best_loss, bad_epochs = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                log(f"early stopping at epoch {epoch} (best val loss {best_loss:.4f})")
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    return {"history": history, "best_val_loss": best_loss, "epochs_trained": len(history)}


def train_and_evaluate(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
    w2v_cfg: W2VConfig = W2VConfig(), nn_cfg: NNConfig = NNConfig(),
    save_path: Path = NN_MODEL_PATH, w2v_path: Path | None = W2V_PATH,
    latency_runs: int = C.LATENCY_RUNS, verbose: bool = True,
) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    set_seed()
    device = get_device()
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "none"
    log(f"Device: {device} (GPU: {gpu})")

    # 1) Word2Vec on the training split only (no test leakage)
    t0 = time.perf_counter()
    train_tok = tokenize_corpus(train[C.TEXT_COL].tolist())
    val_tok = tokenize_corpus(val[C.TEXT_COL].tolist())
    log(f"Tokenised {len(train_tok):,} train / {len(val_tok):,} val messages "
        f"in {time.perf_counter() - t0:.1f}s")

    t1 = time.perf_counter()
    kv = train_word2vec(train_tok, w2v_cfg)
    w2v_s = time.perf_counter() - t1
    vocab, emb = embedding_matrix(kv)
    emb = standardize_embeddings(emb)
    log(f"Word2Vec: vocab {len(vocab):,} x dim {emb.shape[1]} in {w2v_s:.1f}s")
    if w2v_path is not None:
        w2v_path.parent.mkdir(parents=True, exist_ok=True)
        kv.save(str(w2v_path))

    stoi = {w: i for i, w in enumerate(vocab)}
    train_ids = [encode(t, stoi, nn_cfg.max_len) for t in train_tok]
    val_ids = [encode(t, stoi, nn_cfg.max_len) for t in val_tok]

    # 2) Feed-forward network with SGD
    net = AvgEmbeddingClassifier(emb, hidden=nn_cfg.hidden, dropout=nn_cfg.dropout,
                                 finetune_emb=nn_cfg.finetune_emb)
    t2 = time.perf_counter()
    info = fit(net, train_ids, train[C.LABEL_COL].values, val_ids, val[C.LABEL_COL].values,
               nn_cfg, device, verbose)
    nn_s = time.perf_counter() - t2

    # 3) Threshold tuning on validation, then save
    model = NeuralModel(net, vocab, nn_cfg, device=device)
    val_proba = model.predict_proba(val[C.TEXT_COL])
    model.threshold, val_f1 = M.best_threshold(val[C.LABEL_COL].values, val_proba)
    log(f"tuned threshold = {model.threshold:.2f} (val macro-F1 = {val_f1:.4f})")
    model.save(save_path)

    # 4) Evaluate the SAVED artifact on CPU (exactly what the app will load)
    cpu_model = NeuralModel.load(save_path, device="cpu")
    y_test = test[C.LABEL_COL].values
    y_pred = cpu_model.predict(test[C.TEXT_COL])
    test_f1 = M.macro_f1(y_test, y_pred)
    log("\nTest set:\n" + M.report(y_test, y_pred))

    lat = M.benchmark_latency(cpu_model.predict_one, test[C.TEXT_COL].tolist(),
                              runs=latency_runs)
    size = M.size_mb(save_path)

    results = {
        "model": NeuralModel.name,
        "macro_f1": test_f1,
        "val_macro_f1": val_f1,
        "threshold": model.threshold,
        "size_mb": size,
        "train_seconds": w2v_s + nn_s,
        "word2vec_seconds": w2v_s,
        "n_train": len(train),
        "n_test": len(test),
        "vocab_size": len(vocab),
        "embedding_dim": int(emb.shape[1]),
        "train_device": str(device),
        "gpu": gpu,
        "latency_device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "epochs_trained": info["epochs_trained"],
        "best_val_loss": info["best_val_loss"],
        "history": info["history"],
        "w2v_config": asdict(w2v_cfg),
        "nn_config": asdict(nn_cfg),
        **lat,
    }
    log(f"\nMacro-F1 = {test_f1:.4f} | size = {size:.2f} MB | "
        f"latency = {lat['latency_mean_ms']:.3f} ms (mean of {lat['latency_runs']} runs, CPU)")
    return results


def main():
    ap = argparse.ArgumentParser(description="Train Word2Vec + FFNN (SGD) classifier.")
    ap.add_argument("--dim", type=int, default=W2VConfig.vector_size)
    ap.add_argument("--w2v-epochs", type=int, default=W2VConfig.epochs)
    ap.add_argument("--epochs", type=int, default=NNConfig.max_epochs)
    ap.add_argument("--batch-size", type=int, default=NNConfig.batch_size)
    ap.add_argument("--lr", type=float, default=NNConfig.lr)
    ap.add_argument("--patience", type=int, default=NNConfig.patience)
    ap.add_argument("--max-len", type=int, default=NNConfig.max_len)
    ap.add_argument("--workers", type=int, default=None,
                    help="DataLoader workers (default min(4, cpu_count)); 0 = main process")
    ap.add_argument("--finetune-emb", action="store_true",
                    help="also update the Word2Vec embeddings with SGD")
    ap.add_argument("--runs", type=int, default=C.LATENCY_RUNS)
    args = ap.parse_args()

    w2v_cfg = W2VConfig(vector_size=args.dim, epochs=args.w2v_epochs)
    nn_cfg = NNConfig(max_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                      patience=args.patience, max_len=args.max_len,
                      finetune_emb=args.finetune_emb)
    if args.workers is not None:
        nn_cfg.num_workers = args.workers

    from src.data import load_splits
    train, val, test = load_splits()
    results = train_and_evaluate(train, val, test, w2v_cfg, nn_cfg, latency_runs=args.runs)
    out = M.save_results("neural_net", results)
    print(f"Results saved -> {out}")


if __name__ == "__main__":   # required for DataLoader workers on Windows (spawn)
    main()
