# Code Reference – Real-Time Chat Moderation

Every text file in the project, in dependency order. Binary files not shown: the slide decks, the PDF and the screenshot.

## Contents

1. `README.md`
2. `requirements.txt`
3. `pytest.ini`
4. `.gitignore`
5. `src/__init__.py`
6. `src/config.py`
7. `src/data.py`
8. `src/preprocessing.py`
9. `src/metrics.py`
10. `src/rule_baseline.py`
11. `src/baseline.py`
12. `src/neural_net.py`
13. `src/transformer_model.py`
14. `src/quantize.py`
15. `src/registry.py`
16. `src/moderation.py`
17. `src/evaluate.py`
18. `src/error_analysis.py`
19. `src/hub.py`
20. `src/pipeline.py`
21. `app.py`
22. `scripts/fill_slides.py`
23. `scripts/build_deck.js`
24. `data/lexicon/seed_lexicon.txt`
25. `data/lexicon/learned_lexicon.txt`
26. `docs/DEMO_SCRIPT.md`
27. `tests/test_baseline.py`
28. `tests/test_evaluate.py`
29. `tests/test_fill_slides.py`
30. `tests/test_moderation.py`
31. `tests/test_neural_net.py`
32. `tests/test_preprocessing.py`
33. `tests/test_rule_and_errors.py`
34. `tests/test_transformer.py`


---

## `README.md`

````markdown
# Real-Time Chat Moderation – How to Run

**SWE015 – Introduction to Large Language Models · Individual Homework**
**Student:** Matin Jebraeili · **ID:** 2309115105 · **Use Case 4:** Real-Time Chat Moderation (English)

This folder contains the full source code of a chat moderation application. Every chat message is
labelled **Clean** or **Offensive** with a confidence score, offensive words are masked, and a side
panel logs each model's response time. The application compares five models on the same held-out
test set and uses a **dynamic-INT8 quantized BERT** as its primary model. The latency budget is
**50 ms per message on CPU**.

| Tier | Model | Technique area |
|---|---|---|
| Rule baseline | Edit-distance lexicon matcher | B – Classical NLP |
| Baseline | TF-IDF + Logistic Regression | B – Classical NLP |
| Neural network | Own Word2Vec + feed-forward network trained with SGD | A – Neural networks |
| Transformer | `bert-base-uncased`, fine-tuned | C – Transformers |
| Quantized Transformer | the same BERT after dynamic INT8 quantization (**used by the app**) | D – Efficient models |

> **Slides:** `docs/presentation.pdf` (12 slides) holds the measured results below.
> `docs/presentation.pptx` is the template, filled by `scripts/fill_slides.py`.

---

## Quick reference

```bash
pip install -r requirements.txt                    # 1. install
# put Jigsaw train.csv into data/raw/              # 2. data
python -m src.pipeline                             # 3. train + quantize + evaluate everything
python app.py                                      # 4. run the app -> http://127.0.0.1:7860
```

The rest of this file explains each step.

---

## 1. Requirements

| | Minimum | Used for the reported results |
|---|---|---|
| Python | **3.10, 3.11 or 3.12** (3.13 is not supported by gensim / numpy 1.26) | 3.11 |
| RAM | 8 GB (16 GB recommended) | 7 GB |
| GPU | optional; speeds up BERT fine-tuning (NVIDIA, ≥ 4 GB VRAM, CUDA 12.x) | none – CPU only (Intel Xeon 2.3 GHz, 2 vCPUs) |
| Disk | ~3 GB (dataset, models, Python packages) | |
| OS | Windows 10/11, Linux or macOS | Windows 11 |

Inference (the app and all latency measurements) runs on the **CPU only**. The GPU is used for training only.

## 2. Installation

From the project folder (the one that contains `app.py`):

**Windows (PowerShell)**
```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
# NVIDIA GPU: install the CUDA build of PyTorch first
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Linux / macOS**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121   # NVIDIA GPU only
pip install -r requirements.txt
```

Without an NVIDIA GPU, skip the `torch ... cu121` line: `requirements.txt` installs the CPU build.

Check the installation:
```bash
python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
pytest            # expected: all tests pass (61 tests, < 1 minute)
```

## 3. Dataset

The project uses the **Jigsaw Toxic Comment Classification Challenge** dataset (Kaggle, English).
The dataset is released under CC0 (comment text: Wikipedia CC BY-SA 3.0). It is not included in this repository, so download it yourself:

1. Open https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
   (sign in and accept the competition rules), and download `train.csv.zip`.
2. Unzip it and place the file at **`data/raw/train.csv`**.

Or use the Kaggle CLI (needs an API token in `~/.kaggle/kaggle.json`):
```bash
kaggle competitions download -c jigsaw-toxic-comment-classification-challenge -f train.csv.zip -p data/raw
# then unzip data/raw/train.csv.zip into data/raw/
```

## 4. Train all models (one command)

```bash
python -m src.pipeline
```

This runs every step in order. It skips any step whose output already exists, so it can be
re-run safely after an interruption:

| # | Step | Command it runs | Output | Approx. time* |
|---|---|---|---|---|
| 1 | Clean labels, frozen stratified 80/10/10 split (seed 42) | `python -m src.data` | `data/processed/*.csv` | < 1 min |
| 2 | TF-IDF + LogReg baseline, plus the learned lexicon | `python -m src.baseline` | `models/baseline_tfidf_lr.joblib` | 2–5 min |
| 3 | Word2Vec + feed-forward net (SGD) | `python -m src.neural_net` | `models/neural_net/model.pt` | 5–10 min |
| 4 | Fine-tune BERT | `python -m src.transformer_model` | `models/bert_finetuned/` | ~1 h on a laptop GPU |
| 5 | Dynamic INT8 quantization | `python -m src.quantize` | `models/bert_int8/` | 5–10 min |
| 6 | Evaluate all models on the same test split | `python -m src.evaluate` | `models/results/results_table.md` | 10–20 min |
| 7 | Collect failure examples | `python -m src.error_analysis` | `models/results/failure_examples.md` | 5–10 min |

\* Rough guide for a 4-core laptop CPU and a 4 GB laptop GPU.

**Faster run (for checking that everything works).** This trains BERT on a 20 000-message subset of
the training split for one epoch. The validation and test splits stay exactly the same:
```bash
python -m src.pipeline --bert-train-limit 20000 --bert-epochs 1
```

**CPU-only machine (this is how the reported results were produced):**
```bash
python -m src.pipeline --cpu --bert-train-limit 6000 --bert-epochs 1
```
`--cpu` makes BERT skip the per-epoch validation pass and tune its decision threshold on a stratified
4,000-message subset of the validation split. The **test split is never subsampled**: every model is
scored once on all 15,958 test messages by `src.evaluate`. On 2 vCPUs the whole run took about
1.5 hours (Word2Vec ~9 min, BERT fine-tuning ~14 min, BERT test scoring ~30 min).

## 5. Results

Measured with the CPU-only command above, on the same 15,958-message test split
(`models/results/results_table.md`):

| Tier | Model | Technique Area | Precision | Macro-F1 | Size (MB) | Avg Latency (ms, CPU, n=200) | p95 (ms) | < 50 ms |
|---|---|---|---|---:|---:|---:|---:|:---:|
| Rule baseline | Lexicon rule (edit distance) | Area B – Classical NLP | rules | 0.7777 | 0.002 | 0.71 | 2.11 | ✅ |
| Baseline | TF-IDF + LogReg | Area B – Classical NLP | fp64 | 0.8925 | 4.33 | 2.84 | 5.33 | ✅ |
| Neural Net | Word2Vec + FFNN (SGD) | Area A – Neural Networks | fp32 | 0.8635 | 34.53 | 0.31 | 0.73 | ✅ |
| Transformer | BERT-base (fp32) | Area C – Transformers | fp32 | 0.8606 | 418.57 | 84.57 | 136.51 | ❌ |
| Quantized | BERT-base (dynamic INT8) | Area D – Efficient Models | int8 (Linear) + fp32 (embeddings) | 0.8521 | 173.99 | 38.67 | 74.42 | ✅ |

_Macro-F1 on the same 15,958-message test split. Latency: batch size 1, mean of 200 runs after warm-up, CPU only. Hardware: Intel(R) Xeon(R) Processor @ 2.30GHz (2 vCPUs, cloud VM), 7 GB RAM, CPU only._

**Reading the table.** TF-IDF + logistic regression has the best macro-F1. BERT was fine-tuned on
only 6,000 messages for one epoch because training ran on 2 CPU cores, while TF-IDF used all 127,655
training messages, so BERT does not beat the baseline here. Dynamic INT8 quantization makes BERT
2.4× smaller and 2.2× faster, which brings it under the 50 ms budget, at a cost of 0.0085 macro-F1.
With a GPU, train BERT on the full split (`python -m src.pipeline --force`) for a stronger Transformer.

After the pipeline finishes:

- **`models/results/results_table.md`** is the results table: one row per model with Macro-F1,
  size (MB), mean and p95 latency (ms, averaged over 200 CPU runs), and the hardware used.
- **`models/results/*.json`** holds the detailed metrics of every model.
- **`models/results/failure_examples.md`** lists misclassified test messages.

To state the hardware explicitly in the table:
```bash
python -m src.evaluate --hardware "Intel Xeon @ 2.30 GHz, 2 vCPUs, 7 GB RAM"
```

To put your measured numbers into the slides:
```bash
python scripts/fill_slides.py --hardware "Intel Xeon @ 2.30 GHz, 2 vCPUs, 7 GB RAM" \
    --repo "Submitted as source files" --video "<video link>" \
    --fail-why "<one-sentence explanation of the failure example>"
# -> docs/presentation_filled.pptx (every number comes from models/results/)
```

## 6. Run the application

```bash
python app.py
```

Open **http://127.0.0.1:7860** in a browser. Optional flags: `--port 8080`, `--share` (temporary
public link), `--primary baseline` (choose which model decides).

**How to use it:**
- Type a message and press **Enter**. Every trained model labels it live on the CPU.
- The chat shows the moderated message:
  - 🟢 **Clean** – allowed unchanged;
  - 🟠 **Masked** – offensive words replaced, e.g. `f******`;
  - 🔴 **Flagged** – offensive, but there is no single word to mask;
  - ⛔ **Blocked** – very confident offensive.
- The **side panel** shows each model's label, P(offensive), latency in ms and a ✅/❌ check against
  the 50 ms budget, plus session statistics and a message log.
- The **"Primary moderation model"** dropdown switches which model decides.
- **"▶ Stream test messages"** replays random messages from the held-out test split, like a live chat.
- The example buttons include a hard case of gaming slang that is not toxic.

The app loads whichever models have been trained, so it starts even if only some steps have run.
The INT8 BERT is the default primary model.

## 7. Project structure

```
app.py                     Gradio application (run command: python app.py)
requirements.txt           pinned dependencies
src/config.py              paths, random seed, latency budget and protocol
src/data.py                Jigsaw -> binary label -> frozen stratified split
src/preprocessing.py       regex normalization (leetspeak, repeated letters, URLs), OSA edit distance,
                           LexiconMatcher (finding and masking offensive words)
src/rule_baseline.py       lexicon matcher used as a rule-based classifier
src/baseline.py            TF-IDF (word + char n-grams) + logistic regression
src/neural_net.py          own Word2Vec + feed-forward network trained with SGD, early stopping
src/transformer_model.py   BERT fine-tuning with Hugging Face Trainer; inference wrapper
src/quantize.py            dynamic INT8 quantization (torch.ao.quantization.quantize_dynamic)
src/metrics.py             macro-F1, threshold tuning, model size, latency benchmark
src/evaluate.py            evaluates all models on the same test split -> results table
src/error_analysis.py      failure examples
src/registry.py            one loader for every model (used by app and evaluation)
src/moderation.py          moderation policy (allow / mask / flag / block)
src/pipeline.py            runs every step in order
src/hub.py                 optional: upload/download weights to/from the Hugging Face Hub
scripts/fill_slides.py     writes measured results into the slide deck
docs/presentation.pdf             final slides with measured results
docs/presentation_filled.pptx     the same deck, editable
docs/presentation.pptx            slide template (placeholders)
scripts/build_deck.js             generates the slide template (pptxgenjs)
docs/DEMO_SCRIPT.md        2-minute demo script
data/lexicon/seed_lexicon.txt     hand-written offensive-term lexicon
tests/                     unit tests (pytest)
```

## 8. Architecture

```
Browser (Gradio UI, app.py)
   │  typed message / test-stream replay / model dropdown
   ▼
Preprocessing: normalize_text() regex  (src/preprocessing.py; applied inside every model)
   ▼
Models, all on CPU ──► Lexicon rule (rules) · TF-IDF+LR (fp64) · Word2Vec+FFNN (fp32)
                       BERT-base (fp32) · BERT-base dynamic INT8 ★ primary
   ▼
Post-processing: moderation policy + LexiconMatcher masking  (src/moderation.py)
   ▼
Gradio output: masked chat message + verdict, side panel with per-model probability and latency
```

For the reported results, every model was trained and evaluated on the CPU (no GPU).
Every model is evaluated and served on the CPU.

## 9. Methodology

- **Same test set for all models:** `src/data.py` creates the split once; every script loads it.
- **No leakage:** vocabularies, Word2Vec, the learned lexicon and class weights come from the training
  split only. Decision thresholds are tuned for macro-F1 on the validation split, and the test split is
  used only for the final evaluation.
- **Class imbalance:** about 10 % of messages are offensive. The project uses class-weighted losses,
  threshold tuning, and **macro-F1** as the metric.
- **Latency protocol:** batch size 1 (one chat message), 10 warm-up calls, then the mean and p95 over
  200 runs on the CPU, with the same messages for every model.
- **Quantization:** all `nn.Linear` layers are converted to int8 weights, with activations quantized at
  run time; embeddings stay fp32. The threshold is re-tuned on validation, and the saved quantized
  model is reloaded before it is evaluated.
- **Component choice:** the use case suggests BERTurk or multilingual BERT. English `bert-base-uncased`
  is used because the dataset is English (the same technique area, C).

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `pip` fails building gensim or numpy | Use Python 3.10–3.12, not 3.13. |
| `torch.cuda.is_available()` is `False` on a GPU machine | Reinstall the CUDA build: `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall` |
| `CUDA out of memory` during BERT training | `python -m src.transformer_model --batch-size 8 --grad-accum 4` (or add `--grad-checkpointing`) |
| `data/raw/train.csv not found` | Download the dataset (step 3). |
| `data/processed/... missing - run python -m src.data first` | Run `python -m src.data`, or the full pipeline. |
| The app says "No trained models found" | Run `python -m src.pipeline` first. |
| DataLoader errors on Windows | Add `--workers 0` to `python -m src.neural_net` or `python -m src.transformer_model`. |
| Port 7860 is busy | `python app.py --port 7861` |

## 11. Sources

- Jigsaw / Conversation AI, *Toxic Comment Classification Challenge*, Kaggle, 2018.
- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL 2019.
- Mikolov et al., *Efficient Estimation of Word Representations in Vector Space*, 2013 (Word2Vec via gensim).
- Libraries: PyTorch (dynamic quantization), Hugging Face Transformers and Datasets, scikit-learn, gensim, Gradio.

## 12. AI usage

Claude (Anthropic) was used to draft the code, the tests, this README and the slides. Result values
are filled only from measured runs of the pipeline (step 4) and `scripts/fill_slides.py` (step 5).
````


---

## `requirements.txt`

````text
# --- Core numerical stack (numpy<2 / scipy<1.14 required by gensim 4.3.x wheels) ---
numpy==1.26.4
scipy==1.13.1
pandas==2.2.3
joblib==1.4.2

# --- Area B: classical NLP baseline ---
scikit-learn==1.5.2

# --- Area A: custom Word2Vec + PyTorch network ---
gensim==4.3.3
torch==2.5.1

# --- Area C/D: transformer fine-tuning + dynamic INT8 quantization ---
transformers==4.46.3
datasets==3.1.0
pyarrow==17.0.0          # newer pyarrow needs numpy>=2, which gensim 4.3 cannot use
accelerate==1.1.1
evaluate==0.4.3

# --- Application ---
gradio==5.9.1
pydantic==2.10.6         # gradio 5.9 API schema breaks with pydantic>=2.11

# --- Testing / utilities ---
pytest==8.3.4
tqdm==4.67.1
tabulate==0.9.0

# --- Weights hosting (Hugging Face Hub) and slide filling ---
huggingface_hub==0.26.2
python-pptx==1.0.2
````


---

## `pytest.ini`

````ini
[pytest]
testpaths = tests
pythonpath = .
````


---

## `.gitignore`

````text
__pycache__/
*.pyc
.venv/
.ipynb_checkpoints/
data/raw/*
data/processed/*
!data/**/.gitkeep
models/**/*.joblib
models/**/*.pt
models/**/*.bin
models/**/*.safetensors
models/bert*/
models/neural_net/
models/**/*.kv*
models/**/*.npy
*.egg-info/
docs/*_filled.pptx
~$*.pptx
````


---

## `src/__init__.py`

````python
"""Real-Time Chat Moderation (SWE015 – Use Case 4)."""
````


---

## `src/config.py`

````python
"""Central paths and hyper-parameters shared by every tier of the model ladder.

Keeping these in one place guarantees that all four models are trained and
evaluated on the *exact same* splits, which is a hard grading requirement.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LEXICON_DIR = DATA_DIR / "lexicon"

MODELS_DIR = ROOT / "models"
RESULTS_DIR = MODELS_DIR / "results"

# Kaggle "Jigsaw Toxic Comment Classification Challenge" train file.
RAW_TRAIN_CSV = RAW_DIR / "train.csv"

TRAIN_CSV = PROCESSED_DIR / "train.csv"
VAL_CSV = PROCESSED_DIR / "val.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

# Jigsaw multi-label columns -> collapsed into one binary "offensive" label.
JIGSAW_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
TEXT_COL = "text"
LABEL_COL = "label"            # 0 = Clean, 1 = Offensive
LABEL_NAMES = {0: "Clean", 1: "Offensive"}

SEED = 42
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10

# Optional cap on the number of rows used (None = full ~160k rows).
# Useful for BERT on limited hardware; the cap is applied *before* splitting
# so every model still sees identical train/val/test sets.
MAX_ROWS = None

# Latency protocol (assignment: average over >= 100 CPU runs).
LATENCY_RUNS = 200
LATENCY_WARMUP = 10
LATENCY_BUDGET_MS = 50.0

# Artifact paths
BASELINE_PATH = MODELS_DIR / "baseline_tfidf_lr.joblib"
NN_DIR = MODELS_DIR / "neural_net"
NN_MODEL_PATH = NN_DIR / "model.pt"
BERT_BASE_NAME = "bert-base-uncased"
BERT_DIR = MODELS_DIR / "bert_finetuned"          # fp32 fine-tuned model
BERT_INT8_DIR = MODELS_DIR / "bert_int8"          # dynamic INT8 quantized model
BERT_CKPT_DIR = MODELS_DIR / "bert_checkpoints"   # Trainer checkpoints (scratch)
BERT_MAX_LENGTH = 128
LEARNED_LEXICON_PATH = LEXICON_DIR / "learned_lexicon.txt"
SEED_LEXICON_PATH = LEXICON_DIR / "seed_lexicon.txt"

# Hugging Face Hub repo holding the trained weights (R7: no large files in git).
# After `python -m src.hub upload --repo <user>/<name>`, put the repo id here
# (or set CHATMOD_HF_REPO) so `python app.py` downloads missing weights itself.
HF_REPO_ID = os.environ.get("CHATMOD_HF_REPO", "")

# Optional human-readable hardware description for the results table,
# e.g. "Intel Core i7-11370H, 32 GB RAM". Falls back to auto-detected CPU name.
HARDWARE_DESC = os.environ.get("CHATMOD_HARDWARE", "")
````


---

## `src/data.py`

````python
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
````


---

## `src/preprocessing.py`

````python
"""Area B – Classical NLP: regex normalisation + edit-distance lexicon matcher.

Two public pieces:

* ``normalize_text(text)`` – deterministic cleaner used by *every* model tier.
    "Y0u are a  STUP1D   iiiiidiooot!!! http://x.co"  ->  "you are a stupid idiot"

* ``LexiconMatcher`` – finds offensive tokens in the *original* message
  (exact, prefix, bigram, or Damerau/OSA edit-distance match after
  normalisation) and returns character spans, so the app can mask them.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

# --------------------------------------------------------------------------- #
# Regex normalisation
# --------------------------------------------------------------------------- #
LEET_MAP = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b",
    "@": "a", "$": "s", "!": "i", "|": "l", "+": "t",
})
_LEET_CHARS = set("0134578@$!|+")

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_HTML_RE = re.compile(r"<[^>]+>|&[a-z]+;")
_TOKEN_RE = re.compile(r"\S+")
_TRAIL_PUNCT_RE = re.compile(r"^(.*?)([.,!?;:\"')\]]*)$")
_LEAD_PUNCT_RE = re.compile(r"^([\"'(\[]*)(.*)$")
_REPEAT_RE = re.compile(r"([a-z])\1{2,}")          # 3+ identical letters -> 1
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_SPACED_RE = re.compile(r"\b(?:[a-z] ){2,}[a-z]\b")  # "f u c k" -> "fuck"
_WS_RE = re.compile(r"\s+")


def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _deleet_token(token: str) -> str:
    """Convert leetspeak only inside tokens that also contain letters.

    Pure numbers ("2024", "50") are untouched; trailing punctuation such as
    the "!" in "idiot!" is *not* treated as leet.
    """
    lead, rest = _LEAD_PUNCT_RE.match(token).groups()
    core, trail = _TRAIL_PUNCT_RE.match(rest).groups()
    if core and any(c.isalpha() for c in core) and any(c in _LEET_CHARS for c in core):
        core = core.translate(LEET_MAP)
    return lead + core + trail


def deleet(text: str) -> str:
    return _TOKEN_RE.sub(lambda m: _deleet_token(m.group(0)), text)


def reduce_repeats(text: str) -> str:
    """"noooo" -> "no", "fuuuuck" -> "fuck"; double letters ("cool") are kept."""
    return _REPEAT_RE.sub(r"\1", text)


def normalize_text(text: str, join_spaced_letters: bool = True) -> str:
    """Full cleaning pipeline (order matters)."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    t = _strip_accents(text).lower()
    t = _URL_RE.sub(" ", t)
    t = _EMAIL_RE.sub(" ", t)
    t = _IP_RE.sub(" ", t)
    t = _HTML_RE.sub(" ", t)
    t = deleet(t)                       # before punctuation stripping ("sh!t", "@ss")
    t = reduce_repeats(t)
    t = t.replace("'", "").replace("’", "")   # "don't" -> "dont"
    t = _NON_ALNUM_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    if join_spaced_letters:
        t = _SPACED_RE.sub(lambda m: m.group(0).replace(" ", ""), t)
    return t


def tokenize(text: str) -> list[str]:
    """Normalise then whitespace-tokenise (used by Word2Vec / lexicon)."""
    return normalize_text(text).split()


# --------------------------------------------------------------------------- #
# Edit-distance lexicon matcher
# --------------------------------------------------------------------------- #
def osa_distance(a: str, b: str, max_dist: int | None = None) -> int:
    """Optimal-String-Alignment (restricted Damerau-Levenshtein) distance.

    Counts insertions, deletions, substitutions and adjacent transpositions,
    so "fcuk" -> "fuck" is distance 1.  Early-exits once ``max_dist`` is exceeded.
    """
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if max_dist is not None and abs(la - lb) > max_dist:
        return max_dist + 1
    prev2: list[int] | None = None
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if (prev2 is not None and i > 1 and j > 1
                    and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]):
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if max_dist is not None and min(cur) > max_dist:
            return max_dist + 1
        prev2, prev = prev, cur
    return prev[lb]


@dataclass(frozen=True)
class Match:
    start: int          # character span in the ORIGINAL message
    end: int
    surface: str        # original text, e.g. "Fcuk"
    term: str           # lexicon entry it matched, e.g. "fuck*"
    rule: str           # exact | prefix | fuzzy | phrase


class LexiconMatcher:
    """Lexicon lookup with conservative fuzzy matching.

    Lexicon file format (one entry per line, ``#`` comments allowed):
        idiot        exact word (after normalisation)
        fuck*        prefix entry: matches fuck, fucking, fucker ...
        kill yourself   two-word phrase

    Fuzzy rule (designed to keep false positives low – e.g. "duck" must not
    match "fuck", "shirt" must not match "shit"):
        * transposition, distance 1, word length >= 4   ("fcuk")
        * insertion/deletion, distance 1, word length >= 5   ("idiiot" -> handled,
          "biatch" style variants)
        * substitution, distance 1, only for long words (>= 7)   ("assholr")
    """

    def __init__(self, terms: Iterable[str]):
        self.exact: set[str] = set()
        self.prefixes: set[str] = set()
        self.phrases: set[tuple[str, ...]] = set()
        for raw in terms:
            t = raw.split("#", 1)[0].strip().lower()
            if not t:
                continue
            if " " in t:
                self.phrases.add(tuple(normalize_text(t).split()))
            elif t.endswith("*"):
                self.prefixes.add(normalize_text(t[:-1]))
            else:
                self.exact.add(normalize_text(t))
        self.exact.discard("")
        self.prefixes.discard("")
        self._fuzzy_pool = sorted(self.exact | self.prefixes)
        self._pool_by_len: dict[int, list[str]] = {}
        for w in self._fuzzy_pool:
            self._pool_by_len.setdefault(len(w), []).append(w)

    # ------------------------------------------------------------------ io
    @classmethod
    def from_files(cls, *paths: Path) -> "LexiconMatcher":
        terms: list[str] = []
        for p in paths:
            p = Path(p)
            if p.exists():
                terms.extend(p.read_text(encoding="utf-8").splitlines())
        return cls(terms)

    @classmethod
    def default(cls) -> "LexiconMatcher":
        from src import config as C
        return cls.from_files(C.SEED_LEXICON_PATH, C.LEARNED_LEXICON_PATH)

    # ------------------------------------------------------------ matching
    @staticmethod
    def _fuzzy_ok(tok: str, word: str) -> bool:
        d = osa_distance(tok, word, max_dist=1)
        if d != 1:
            return False
        if len(tok) == len(word):
            is_transposition = sorted(tok) == sorted(word)
            return (is_transposition and len(word) >= 4) or len(word) >= 7
        return len(word) >= 5  # insertion / deletion

    @lru_cache(maxsize=50_000)
    def match_token(self, norm_tok: str) -> tuple[str, str] | None:
        """Return (term, rule) for a *normalised* token, or None."""
        if not norm_tok or len(norm_tok) < 2:
            return None
        if norm_tok in self.exact:
            return norm_tok, "exact"
        for p in self.prefixes:
            if norm_tok.startswith(p):
                return p + "*", "prefix"
        if len(norm_tok) >= 4:
            # transposed prefix stems: "fcuking" -> fuck*
            for p in self.prefixes:
                if len(p) >= 4 and len(norm_tok) >= len(p):
                    head = norm_tok[:len(p)]
                    if head != p and sorted(head) == sorted(p) and osa_distance(head, p, 1) == 1:
                        return p + "*", "fuzzy"
            n = len(norm_tok)
            candidates = (self._pool_by_len.get(n - 1, []) + self._pool_by_len.get(n, [])
                          + self._pool_by_len.get(n + 1, []))
            for w in candidates:
                if self._fuzzy_ok(norm_tok, w):
                    return (w + "*" if w in self.prefixes else w), "fuzzy"
        return None

    def _spans(self, text: str):
        """Yield (start, end, normalised_token) for each original token's core."""
        for m in _TOKEN_RE.finditer(text):
            tok = m.group(0)
            lead, rest = _LEAD_PUNCT_RE.match(tok).groups()
            core, _ = _TRAIL_PUNCT_RE.match(rest).groups()
            norm = normalize_text(core, join_spaced_letters=False).replace(" ", "")
            if norm:
                s = m.start() + len(lead)
                yield s, s + len(core), norm

    def find(self, text: str) -> list[Match]:
        spans = list(self._spans(text))
        hits: dict[int, Match] = {}
        for i, (s, e, norm) in enumerate(spans):
            r = self.match_token(norm)
            if r:
                hits[i] = Match(s, e, text[s:e], r[0], r[1])
        for phrase in self.phrases:
            n = len(phrase)
            for i in range(len(spans) - n + 1):
                if tuple(sp[2] for sp in spans[i:i + n]) == phrase:
                    for k in range(i, i + n):
                        s, e, _ = spans[k]
                        hits.setdefault(k, Match(s, e, text[s:e], " ".join(phrase), "phrase"))
        return [hits[k] for k in sorted(hits)]

    def mask(self, text: str, keep_first: bool = True, char: str = "*") -> tuple[str, list[Match]]:
        """Return (masked_text, matches).  "you fcuking idiot" -> "you f****** i****"."""
        matches = self.find(text)
        out, last = [], 0
        for m in matches:
            out.append(text[last:m.start])
            w = m.surface
            out.append((w[0] + char * (len(w) - 1)) if keep_first and len(w) > 1 else char * len(w))
            last = m.end
        out.append(text[last:])
        return "".join(out), matches


if __name__ == "__main__":
    demos = [
        "Y0u are a  STUP1D   iiiiidiooot!!! http://x.co",
        "what the f.u.c.k is this sh!t",
        "u r such a l0$er, go kill yourself",
        "I love my duck and my new shirt, 50 points!",
        "Don't be an @$$hole",
    ]
    lex = LexiconMatcher.default()
    for d in demos:
        masked, hits = lex.mask(d)
        print(f"RAW : {d}\nNORM: {normalize_text(d)}\nMASK: {masked}  {[h.rule for h in hits]}\n")
````


---

## `src/metrics.py`

````python
"""Shared evaluation helpers so every model tier is measured identically.

* Macro-F1 (handles the ~10% offensive class imbalance)
* Model size in MB (file or directory on disk)
* Average single-message CPU latency over >= 100 runs
"""
from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import classification_report, f1_score

from src import config as C


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def report(y_true, y_pred) -> str:
    return classification_report(
        y_true, y_pred, target_names=[C.LABEL_NAMES[0], C.LABEL_NAMES[1]], digits=4
    )


def best_threshold(y_true, proba, grid=np.linspace(0.05, 0.95, 91)) -> tuple[float, float]:
    """Pick the decision threshold that maximises macro-F1 on the *validation* set."""
    y_true = np.asarray(y_true)
    best_t, best_f = 0.5, -1.0
    for t in grid:
        f = macro_f1(y_true, (proba >= t).astype(int))
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t, best_f


def size_mb(path: Path | str) -> float:
    p = Path(path)
    if p.is_file():
        return p.stat().st_size / 1024 ** 2
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return total / 1024 ** 2


def benchmark_latency(
    predict_one: Callable[[str], object],
    messages: Sequence[str],
    runs: int = C.LATENCY_RUNS,
    warmup: int = C.LATENCY_WARMUP,
) -> dict:
    """Time ``predict_one`` on single messages (batch size 1, like live chat).

    Messages are cycled from the test set so the timing reflects realistic
    lengths.  Returns mean / p50 / p95 in milliseconds.
    """
    if runs < 100:
        raise ValueError("Assignment requires latency averaged over >= 100 runs.")
    msgs = list(messages) or ["hello"]
    for i in range(warmup):
        predict_one(msgs[i % len(msgs)])
    times = []
    for i in range(runs):
        m = msgs[i % len(msgs)]
        t0 = time.perf_counter()
        predict_one(m)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "latency_mean_ms": statistics.fmean(times),
        "latency_p50_ms": times[len(times) // 2],
        "latency_p95_ms": times[int(len(times) * 0.95) - 1],
        "latency_runs": runs,
    }


def save_results(name: str, results: dict) -> Path:
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = C.RESULTS_DIR / f"{name}.json"
    out.write_text(json.dumps(results, indent=2))
    return out


def cpu_name() -> str:
    """Best-effort human-readable CPU model name (Windows registry, Linux, macOS)."""
    import platform
    import subprocess
    try:
        if platform.system() == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if platform.system() == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        pass
    return platform.processor() or "unknown CPU"


def hardware_description() -> str:
    """C.HARDWARE_DESC if set, else the detected CPU name + logical core count."""
    return C.HARDWARE_DESC or f"{cpu_name()} ({os.cpu_count()} logical CPUs)"
````


---

## `src/rule_baseline.py`

````python
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
````


---

## `src/baseline.py`

````python
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
````


---

## `src/neural_net.py`

````python
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
````


---

## `src/transformer_model.py`

````python
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
````


---

## `src/quantize.py`

````python
"""Tier 4 – Area D Efficient Models: post-training dynamic INT8 quantization.

    python -m src.quantize              # quantize models/bert_finetuned -> models/bert_int8
    python -m src.quantize --no-retune  # keep the fp32 model's threshold

torch.ao.quantization.quantize_dynamic converts every nn.Linear (attention
Q/K/V/output, FFN, classifier head) to int8 weights with dynamically
quantized activations.  The embedding tables stay fp32, so the size drops
from ~418 MB to roughly ~175-185 MB, and CPU matmuls run on int8 kernels (fbgemm/x86).

Saved layout (models/bert_int8/):
    config.json, tokenizer files, moderation_config.json
    quantized_state_dict.pt     int8 packed weights (loaded into a quantized skeleton)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn as nn

from src import config as C
from src import metrics as M
from src.transformer_model import (MOD_CFG_NAME, BertClassifier,
                                   write_moderation_config)

QSTATE_NAME = "quantized_state_dict.pt"
INT8_NAME = "BERT-base (dynamic INT8)"


def quantize_model(model: nn.Module) -> nn.Module:
    """Dynamic INT8 quantization of all Linear layers (CPU inference only)."""
    model = model.to("cpu").eval()
    return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def save_quantized(qmodel: nn.Module, tokenizer, config, out_dir: Path,
                   threshold: float, max_length: int, normalize: bool) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(qmodel.state_dict(), out_dir / QSTATE_NAME)
    config.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    write_moderation_config(out_dir, threshold, max_length, normalize,
                            extra={"quantization": "dynamic int8 (nn.Linear)"})
    return out_dir


def load_quantized(path: Path = C.BERT_INT8_DIR) -> BertClassifier:
    """Rebuild a quantized skeleton from config.json, then load the int8 weights."""
    import json

    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    path = Path(path)
    cfg = json.loads((path / MOD_CFG_NAME).read_text())
    skeleton = AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(path))
    qmodel = quantize_model(skeleton)
    # Our own trusted artifact; packed int8 params need the full unpickler.
    state = torch.load(path / QSTATE_NAME, map_location="cpu", weights_only=False)
    qmodel.load_state_dict(state)
    tok = AutoTokenizer.from_pretrained(path)
    return BertClassifier(qmodel, tok, cfg["threshold"], cfg["max_length"], cfg["normalize"],
                          name=INT8_NAME, device="cpu")


def main():
    ap = argparse.ArgumentParser(description="Dynamic INT8 quantization of fine-tuned BERT.")
    ap.add_argument("--src", type=Path, default=C.BERT_DIR)
    ap.add_argument("--out", type=Path, default=C.BERT_INT8_DIR)
    ap.add_argument("--no-retune", action="store_true",
                    help="reuse the fp32 threshold instead of re-tuning on validation")
    ap.add_argument("--runs", type=int, default=C.LATENCY_RUNS)
    ap.add_argument("--val-limit", type=int, default=None,
                    help="stratified validation subsample for threshold re-tuning (CPU time saver)")
    ap.add_argument("--skip-test", action="store_true",
                    help="skip test-set scoring here; src.evaluate scores every model once")
    args = ap.parse_args()

    from src.data import load_splits
    _, val, test = load_splits()
    from src.transformer_model import _stratified_limit
    val = _stratified_limit(val, args.val_limit)

    fp32 = BertClassifier.load(args.src, device="cpu")
    qmodel = quantize_model(fp32.model)
    q = BertClassifier(qmodel, fp32.tokenizer, fp32.threshold, fp32.max_length,
                       fp32.normalize, name=INT8_NAME, device="cpu")

    if not args.no_retune:
        print(f"Re-tuning threshold on {len(val):,} validation messages (CPU) ...")
        q.threshold, val_f1 = M.best_threshold(val[C.LABEL_COL].values,
                                               q.predict_proba(val[C.TEXT_COL].tolist()))
        print(f"  threshold = {q.threshold:.2f} (val macro-F1 = {val_f1:.4f})")
    else:
        val_f1 = None

    save_quantized(qmodel, fp32.tokenizer, fp32.model.config, args.out,
                   q.threshold, q.max_length, q.normalize)

    # Evaluate the artifact exactly as the app will load it
    q = load_quantized(args.out)
    if args.skip_test:
        res = {"model": INT8_NAME, "val_macro_f1": val_f1, "threshold": q.threshold,
               "n_val_for_threshold": len(val), "size_mb": M.size_mb(args.out),
               "size_fp32_mb": M.size_mb(args.src)}
        print(f"INT8 model saved and reloaded OK ({res['size_mb']:.1f} MB) -> "
              f"{M.save_results('transformer_int8', res)} (test metrics: python -m src.evaluate)")
        return
    y = test[C.LABEL_COL].values
    y_pred = q.predict(test[C.TEXT_COL].tolist())
    f1 = M.macro_f1(y, y_pred)
    print("\nTest set:\n" + M.report(y, y_pred))
    lat = M.benchmark_latency(q.predict_one, test[C.TEXT_COL].tolist(), runs=args.runs)

    size_fp32, size_int8 = M.size_mb(args.src), M.size_mb(args.out)
    results = {"model": INT8_NAME, "macro_f1": f1, "val_macro_f1": val_f1,
               "threshold": q.threshold, "size_mb": size_int8, "size_fp32_mb": size_fp32,
               "compression": size_fp32 / size_int8, "latency_device": "cpu",
               "torch_threads": torch.get_num_threads(),
               "within_budget": lat["latency_mean_ms"] < C.LATENCY_BUDGET_MS,
               "n_test": len(test), **lat}
    print(f"\nMacro-F1 = {f1:.4f} | size = {size_int8:.1f} MB (fp32 {size_fp32:.1f} MB, "
          f"{results['compression']:.2f}x smaller) | latency = {lat['latency_mean_ms']:.2f} ms "
          f"-> {'WITHIN' if results['within_budget'] else 'OVER'} {C.LATENCY_BUDGET_MS:.0f} ms budget")
    print(f"Results saved -> {M.save_results('transformer_int8', results)}")


if __name__ == "__main__":
    main()
````


---

## `src/registry.py`

````python
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
````


---

## `src/moderation.py`

````python
"""Framework-free moderation logic used by app.py (and unit-tested without Gradio/torch).

Policy for the message shown in chat (decided by the *primary* model):
    ALLOW  – primary model says Clean and the lexicon finds nothing
    MASK   – lexicon hits are masked (always, even for "clean" banter like "shit, gg")
    FLAG   – primary model says Offensive but no lexicon term to mask -> shown with a warning
    BLOCK  – primary model is very confident (p >= block_threshold) and nothing to mask
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from src import config as C
from src.preprocessing import LexiconMatcher, Match


@dataclass
class ModelOutput:
    key: str
    name: str
    label: int
    prob: float
    latency_ms: float

    @property
    def verdict(self) -> str:
        return C.LABEL_NAMES[self.label]

    @property
    def within_budget(self) -> bool:
        return self.latency_ms < C.LATENCY_BUDGET_MS


@dataclass
class ModerationResult:
    text: str
    display_text: str
    action: str                       # ALLOW | MASK | FLAG | BLOCK
    primary: ModelOutput
    outputs: list[ModelOutput]
    matches: list[Match] = field(default_factory=list)
    mask_ms: float = 0.0


def run_models(text: str, models: dict) -> list[ModelOutput]:
    """Run every loaded model on one message, timing each call (CPU, batch size 1)."""
    outs = []
    for key, (spec, model) in models.items():
        t0 = time.perf_counter()
        label, prob = model.predict_one(text)
        ms = (time.perf_counter() - t0) * 1000.0
        outs.append(ModelOutput(key, spec.display, int(label), float(prob), ms))
    return outs


def moderate(text: str, outputs: list[ModelOutput], primary_key: str,
             lexicon: LexiconMatcher, block_threshold: float = 0.97) -> ModerationResult:
    if not outputs:
        raise ValueError("no model outputs")
    primary = next((o for o in outputs if o.key == primary_key), outputs[0])
    t0 = time.perf_counter()
    masked, matches = lexicon.mask(text)
    mask_ms = (time.perf_counter() - t0) * 1000.0

    if matches:
        action, shown = "MASK", masked
    elif primary.label == 1 and primary.prob >= block_threshold:
        action, shown = "BLOCK", "[message hidden by moderator]"
    elif primary.label == 1:
        action, shown = "FLAG", text
    else:
        action, shown = "ALLOW", text
    return ModerationResult(text, shown, action, primary, outputs, matches, mask_ms)


_MD_SPECIAL = set("\\`*_{}[]<>#|~")


def escape_md(text: str) -> str:
    """Escape Markdown so masks like 'f***' render literally in the chat window."""
    return "".join("\\" + ch if ch in _MD_SPECIAL else ch for ch in text)


ACTION_BADGE = {"ALLOW": "🟢 Clean", "MASK": "🟠 Masked", "FLAG": "🔴 Flagged",
                "BLOCK": "⛔ Blocked"}


def verdict_line(res: ModerationResult) -> str:
    p = res.primary
    parts = [f"**{ACTION_BADGE[res.action]}**",
             f"{p.name}: {p.verdict} (p={p.prob:.2f})",
             f"{p.latency_ms:.1f} ms"]
    if res.matches:
        parts.append(f"{len(res.matches)} term(s) masked")
    return " · ".join(parts)
````


---

## `src/evaluate.py`

````python
"""Final benchmark: all four models on the SAME held-out test split.

    python -m src.evaluate                 # full test split, 200 latency runs
    python -m src.evaluate --runs 500
    python -m src.evaluate --models baseline neural_net

For each model:
  * Macro-F1 on the full frozen test split (fp32 BERT may use the GPU for this
    step only - same weights, same predictions; INT8 is CPU-only)
  * Size (MB) of the saved artifact on disk
  * CPU latency: mean / p95 over >= 200 single-message calls, every model timed
    on the SAME sampled messages, same thread count

Writes models/results/results_table.md and models/results/final_results.json.
"""
from __future__ import annotations

import argparse
import json
import os
import platform

import numpy as np

from src import config as C
from src import metrics as M
from src.registry import SPEC_BY_KEY, SPECS, load_models

MIN_RUNS = 200
PROBS_DIR = C.RESULTS_DIR / "test_probs"


def latency_messages(texts: list[str], n: int, seed: int = C.SEED) -> list[str]:
    """Fixed random sample of test messages so every model is timed on identical input."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(texts), size=min(n, len(texts)), replace=False)
    return [texts[i] for i in idx]


def fmt_mb(mb: float) -> str:
    return f"{mb:.2f}" if mb >= 0.1 else f"{mb:.3f}"


def results_to_markdown(rows: list[dict], runs: int, hardware: str | None = None,
                        n_test: int | None = None) -> str:
    header = ("| Tier | Model | Technique Area | Precision | Macro-F1 | Size (MB) | "
              f"Avg Latency (ms, CPU, n={runs}) | p95 (ms) | < {C.LATENCY_BUDGET_MS:.0f} ms |\n"
              "|---|---|---|---|---:|---:|---:|---:|:---:|\n")
    lines = []
    for r in rows:
        ok = "✅" if r["latency_mean_ms"] < C.LATENCY_BUDGET_MS else "❌"
        lines.append(f"| {r['tier']} | {r['model']} | {r['area']} | {r.get('precision', '')} | "
                     f"{r['macro_f1']:.4f} | {fmt_mb(r['size_mb'])} | {r['latency_mean_ms']:.2f} | "
                     f"{r['latency_p95_ms']:.2f} | {ok} |")
    md = header + "\n".join(lines) + "\n"
    if hardware:
        test = f" on the same {n_test:,}-message test split" if n_test else ""
        md += (f"\n_Macro-F1{test}. Latency: batch size 1, mean of {runs} runs after warm-up, "
               f"CPU only. Hardware: {hardware}._\n")
    return md


def _torch_info() -> dict:
    try:
        import torch
        return {"torch": torch.__version__, "torch_threads": torch.get_num_threads(),
                "cuda_available": torch.cuda.is_available()}
    except ImportError:
        return {"torch": None}


def main():
    ap = argparse.ArgumentParser(description="Evaluate all four models on the test split.")
    ap.add_argument("--runs", type=int, default=max(MIN_RUNS, C.LATENCY_RUNS))
    ap.add_argument("--models", nargs="*", default=[s.key for s in SPECS],
                    choices=[s.key for s in SPECS])
    ap.add_argument("--cpu-only", action="store_true",
                    help="compute fp32 BERT test predictions on CPU instead of GPU")
    ap.add_argument("--hardware", default=None,
                    help='e.g. "Intel Core i7-11370H, 32 GB RAM" (default: auto-detected CPU)')
    args = ap.parse_args()
    if args.runs < MIN_RUNS:
        ap.error(f"--runs must be >= {MIN_RUNS}")

    from src.data import load_splits
    _, _, test = load_splits()
    texts = test[C.TEXT_COL].tolist()
    y = test[C.LABEL_COL].values
    lat_msgs = latency_messages(texts, args.runs)
    print(f"Test split: {len(test):,} messages ({y.mean()*100:.2f}% offensive); "
          f"latency on {len(lat_msgs)} fixed messages x {args.runs} runs\n")

    # All latency is measured on CPU models.
    models = load_models(args.models, device="cpu")
    if not models:
        raise SystemExit("No trained models found - train them first (see README).")

    rows = []
    for key in [k for k in args.models if k in models]:
        spec, model = models[key]
        print(f"\n=== {spec.display} ===")

        f1_model = model
        if key == "bert_fp32" and not args.cpu_only:
            try:
                import torch
                if torch.cuda.is_available():
                    f1_model = spec.loader("cuda")
                    print("  (test predictions on GPU; latency on CPU)")
            except ImportError:
                pass
        proba = np.asarray(f1_model.predict_proba(texts), dtype=np.float32)
        y_pred = (proba >= model.threshold).astype(int)
        PROBS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(PROBS_DIR / f"{key}.npy", proba)          # reused by src.error_analysis
        if f1_model is not model:
            del f1_model
        f1 = M.macro_f1(y, y_pred)
        print(M.report(y, y_pred))

        lat = M.benchmark_latency(model.predict_one, lat_msgs, runs=args.runs)
        size = M.size_mb(spec.path)
        print(f"  Macro-F1 {f1:.4f} | {size:.2f} MB | {lat['latency_mean_ms']:.2f} ms mean "
              f"/ {lat['latency_p95_ms']:.2f} ms p95")
        rows.append({"key": key, "tier": spec.tier, "model": spec.display, "area": spec.area,
                     "precision": spec.precision, "macro_f1": f1, "size_mb": size, "threshold": model.threshold, **lat})

    # Re-evaluating a subset (e.g. --models lexicon_rule): keep the other rows.
    prev_path = C.RESULTS_DIR / "final_results.json"
    if set(args.models) != {s.key for s in SPECS} and prev_path.exists():
        prev = json.loads(prev_path.read_text())
        if prev.get("runs") == args.runs and prev.get("n_test") == len(test):
            done = {r["key"] for r in rows}
            rows += [r for r in prev["rows"] if r["key"] not in done]
            order = [s.key for s in SPECS]
            rows.sort(key=lambda r: order.index(r["key"]))
    missing = [s.display for s in SPECS if s.key not in {r["key"] for r in rows}]
    hardware = args.hardware or M.hardware_description()
    table = results_to_markdown(rows, args.runs, hardware, len(test))
    env = {"hardware": hardware, "platform": platform.platform(), "cpu": M.cpu_name(),
           "cpu_count": os.cpu_count(), **_torch_info()}
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.RESULTS_DIR / "results_table.md").write_text(table, encoding="utf-8")
    (C.RESULTS_DIR / "final_results.json").write_text(
        json.dumps({"n_test": len(test), "runs": args.runs, "env": env, "rows": rows,
                    "missing": missing}, indent=2))

    print("\n" + table)
    if missing:
        print("Not evaluated (artifact missing): " + ", ".join(missing))
    print(f"Saved -> {C.RESULTS_DIR / 'results_table.md'}")


if __name__ == "__main__":
    main()
````


---

## `src/error_analysis.py`

````python
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
````


---

## `src/hub.py`

````python
"""Host trained weights on the Hugging Face Hub (R7: no large weight files in git).

    huggingface-cli login                                   # once, with a write token
    python -m src.hub upload   --repo <hf-user>/chat-moderation-models
    python -m src.hub download --repo <hf-user>/chat-moderation-models

After uploading, set HF_REPO_ID in src/config.py (or the CHATMOD_HF_REPO env
var): `python app.py` then downloads any missing weights automatically, so a
grader can run the app straight after `pip install -r requirements.txt`.
"""
from __future__ import annotations

import argparse

from src import config as C

# Everything the app and src.evaluate need (no Trainer checkpoints, no raw data).
ARTIFACT_PATTERNS = [
    "models/baseline_tfidf_lr.joblib",
    "models/neural_net/model.pt",
    "models/bert_finetuned/*",
    "models/bert_int8/*",
    "models/results/*",
    "data/lexicon/learned_lexicon.txt",
]


def missing_artifacts() -> list[str]:
    from src.registry import SPECS
    return [s.display for s in SPECS if not s.exists()] + (
        [] if C.LEARNED_LEXICON_PATH.exists() else ["learned lexicon"])


def upload(repo: str, private: bool = False) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(C.ROOT),
                      allow_patterns=ARTIFACT_PATTERNS,
                      commit_message="Upload trained chat-moderation models")
    print(f"Uploaded -> https://huggingface.co/{repo}")


def download(repo: str) -> None:
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, repo_type="model", local_dir=str(C.ROOT),
                      allow_patterns=ARTIFACT_PATTERNS)
    print(f"Downloaded weights from https://huggingface.co/{repo}")


def ensure_models(repo: str | None = None) -> None:
    """Download weights if any artifact is missing and a Hub repo is configured."""
    repo = repo or C.HF_REPO_ID
    missing = missing_artifacts()
    if not missing:
        return
    if not repo:
        print("Missing artifacts: " + ", ".join(missing) +
              "\n  -> train them (python -m src.pipeline) or set HF_REPO_ID in src/config.py")
        return
    print(f"Missing artifacts ({', '.join(missing)}); downloading from {repo} ...")
    try:
        download(repo)
    except Exception as e:
        print(f"  download failed: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser(description="Upload/download weights to/from the HF Hub.")
    ap.add_argument("action", choices=["upload", "download"])
    ap.add_argument("--repo", default=C.HF_REPO_ID or None, required=not C.HF_REPO_ID)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    upload(args.repo, args.private) if args.action == "upload" else download(args.repo)


if __name__ == "__main__":
    main()
````


---

## `src/pipeline.py`

````python
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
````


---

## `app.py`

````python
"""Real-Time Chat Moderation – Gradio web app.

    python app.py                    # http://127.0.0.1:7860
    python app.py --share            # temporary public link
    python app.py --primary baseline # choose the model that drives moderation

Each message is scored live by every trained model (CPU, batch size 1).  The
primary model (default: dynamic-INT8 BERT) decides the moderation action;
lexicon terms are masked with the edit-distance matcher from src/preprocessing.py.
The "Primary moderation model" dropdown switches which model decides.
The side panel shows per-model verdicts, probabilities and latency for the
latest message, a running session log, and the offline benchmark table.
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gradio as gr
import pandas as pd

from src import config as C
from src.moderation import escape_md, moderate, run_models, verdict_line
from src.preprocessing import LexiconMatcher
from src.registry import SPECS, load_models

LOG_LIMIT = 50
EXAMPLES = [
    "gg wp everyone, that was a close match",
    "you are such a STUP1D n00b, uninstall",
    "fcuking idiot can't even aim",
    "go kill yourself loser",
    "shit, I missed that shot lol",
    "can someone explain how ranked works?",
    "ur trash, absolute garbage player",
    "that boss fight was sick, killed it!",          # hard case: gaming slang, not toxic
]

# --------------------------------------------------------------------------- #
# Load everything once at start-up
# --------------------------------------------------------------------------- #
from src.hub import ensure_models  # noqa: E402

ensure_models()                     # pulls weights from the HF Hub if configured & missing
print("Loading models ...")
MODELS = load_models(device="cpu")
LEXICON = LexiconMatcher.default()

# Warm-up: the first PyTorch call is much slower (lazy initialisation), which would
# make the first live message look over budget.  Run each model a few times first.
for _spec, _model in MODELS.values():
    for _ in range(3):
        _model.predict_one("warm up message for the moderation models")
DEFAULT_PRIMARY = next((k for k in ("bert_int8", "bert_fp32", "neural_net", "baseline")
                        if k in MODELS), None)


def _test_messages() -> list[str]:
    try:
        df = pd.read_csv(C.TEST_CSV, keep_default_na=False)
        return [t for t in df[C.TEXT_COL].astype(str) if 3 <= len(t) <= 300]
    except Exception:
        return EXAMPLES


TEST_MESSAGES = _test_messages()


def _benchmark_md() -> str:
    p = C.RESULTS_DIR / "results_table.md"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return "_No benchmark yet – run `python -m src.evaluate`._"


# --------------------------------------------------------------------------- #
# Side-panel renderers
# --------------------------------------------------------------------------- #
LAST_COLS = ["Model", "Verdict", "P(offensive)", "Latency (ms)", f"< {C.LATENCY_BUDGET_MS:.0f} ms"]
LOG_COLS = ["Time", "Action", "Message (masked)"] + [s.display for s in SPECS if s.key in MODELS]


def _last_df(res) -> pd.DataFrame:
    rows = [[o.name + (" ★" if o.key == res.primary.key else ""), o.verdict, round(o.prob, 4),
             round(o.latency_ms, 2), "✅" if o.within_budget else "❌"] for o in res.outputs]
    rows.append(["Lexicon matcher (masking)", f"{len(res.matches)} hit(s)", "—",
                 round(res.mask_ms, 2), "✅" if res.mask_ms < C.LATENCY_BUDGET_MS else "❌"])
    return pd.DataFrame(rows, columns=LAST_COLS)


def _log_df(log: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([[r.get(c, "") for c in LOG_COLS] for r in reversed(log)],
                        columns=LOG_COLS)


def _stats_md(log: list[dict]) -> str:
    if not log:
        return "_No messages yet._"
    n = len(log)
    actions = {a: sum(r["Action"] == a for r in log) for a in ("ALLOW", "MASK", "FLAG", "BLOCK")}
    lines = [f"**Session:** {n} message(s) · " +
             " · ".join(f"{a}: {c}" for a, c in actions.items()), "",
             "| Model | Mean latency (ms) | Max (ms) | Flagged offensive |",
             "|---|---:|---:|---:|"]
    for s in SPECS:
        if s.key not in MODELS:
            continue
        lats = [r["_lat"][s.key] for r in log]
        off = sum(r["_lab"][s.key] for r in log)
        lines.append(f"| {s.display} | {statistics.fmean(lats):.2f} | {max(lats):.2f} | {off}/{n} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Core handler
# --------------------------------------------------------------------------- #
def process(text: str, history: list, log: list, primary_key: str):
    outputs = run_models(text, MODELS)
    res = moderate(text, outputs, primary_key or DEFAULT_PRIMARY, LEXICON)

    history = list(history or []) + [
        {"role": "user", "content": escape_md(res.display_text)},
        {"role": "assistant", "content": verdict_line(res)},
    ]
    entry = {"Time": time.strftime("%H:%M:%S"), "Action": res.action,
             "Message (masked)": res.display_text,
             "_lat": {o.key: o.latency_ms for o in outputs},
             "_lab": {o.key: o.label for o in outputs}}
    for o in outputs:
        entry[o.name] = f"{o.verdict} {o.prob:.2f} · {o.latency_ms:.1f} ms"
    log = (log + [entry])[-LOG_LIMIT:]
    return history, log, _last_df(res), _log_df(log), _stats_md(log)


def on_send(text, history, log, primary_key):
    if not text or not text.strip():
        return gr.update(), history, log, gr.update(), gr.update(), gr.update()
    return ("", *process(text.strip(), history, log, primary_key))


def on_stream(history, log, primary_key, n):
    """Simulate an incoming chat stream with random held-out test messages."""
    for text in random.sample(TEST_MESSAGES, k=min(int(n), len(TEST_MESSAGES))):
        history, log, last, log_df, stats = process(text, history, log, primary_key)
        yield history, log, last, log_df, stats
        time.sleep(0.35)


def on_clear():
    return [], [], pd.DataFrame(columns=LAST_COLS), pd.DataFrame(columns=LOG_COLS), _stats_md([])


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def build_ui() -> gr.Blocks:
    choices = [(s.display, s.key) for s in SPECS if s.key in MODELS]
    with gr.Blocks(title="Real-Time Chat Moderation", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🛡️ Real-Time Chat Moderation\n"
            "SWE015 · Use Case 4 · English · every message is scored by all models on CPU "
            f"(budget < {C.LATENCY_BUDGET_MS:.0f} ms/message)")
        if not MODELS:
            gr.Markdown("### ⚠️ No trained models found\nRun `python -m src.pipeline` first.")
            return demo

        log = gr.State([])
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(type="messages", height=520, label="Live chat (moderated view)")
                with gr.Row():
                    msg = gr.Textbox(placeholder="Type a chat message and press Enter …",
                                     show_label=False, scale=5, autofocus=True)
                    send = gr.Button("Send", variant="primary", scale=1)
                gr.Examples(EXAMPLES, inputs=msg, label="Try these")
                with gr.Row():
                    primary = gr.Dropdown(choices, value=DEFAULT_PRIMARY,
                                          label="Primary moderation model", scale=3)
                    n_stream = gr.Slider(5, 50, value=10, step=5, label="Messages", scale=2)
                    stream = gr.Button("▶ Stream test messages", scale=2)
                    clear = gr.Button("Clear", scale=1)

            with gr.Column(scale=2):
                gr.Markdown("### Latest message – all models  (★ = primary)")
                last_df = gr.Dataframe(headers=LAST_COLS, interactive=False, wrap=True)
                stats = gr.Markdown(_stats_md([]))
                gr.Markdown("### Message log")
                log_df = gr.Dataframe(headers=LOG_COLS, interactive=False, wrap=True)
                with gr.Accordion("Offline benchmark (test split)", open=False):
                    gr.Markdown(_benchmark_md())

        outs = [chat, log, last_df, log_df, stats]
        # The Chatbot's own value is the conversation history (input and output).
        send.click(on_send, [msg, chat, log, primary], [msg] + outs)
        msg.submit(on_send, [msg, chat, log, primary], [msg] + outs)
        stream.click(on_stream, [chat, log, primary, n_stream], outs)
        clear.click(on_clear, None, outs)
    return demo


def main():
    global DEFAULT_PRIMARY
    ap = argparse.ArgumentParser(description="Launch the chat moderation demo.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--primary", choices=[s.key for s in SPECS], default=None)
    args = ap.parse_args()
    if args.primary and args.primary in MODELS:
        DEFAULT_PRIMARY = args.primary
    build_ui().queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
````


---

## `scripts/fill_slides.py`

````python
"""Fill docs/presentation.pptx with the numbers YOUR runs produced.

The template contains {{TOKENS}} wherever a measured value belongs, so the
deck never carries a number that did not come from models/results/
(assignment rule: un-measured numbers count as fabricated results).

    python -m src.pipeline                      # produces models/results/*.json
    python scripts/fill_slides.py \
        --repo  https://github.com/<you>/chat-moderation \
        --video https://youtu.be/<id> \
        --hardware "Intel Core i7-11370H, 32 GB RAM" \
        --fail-why "No swear word, so the lexicon cannot help, and ..."

Writes docs/presentation_filled.pptx; open it, check it, export to PDF.
Any token that could not be filled is listed at the end and left visible.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402

TOKEN_RE = re.compile(r"\{\{[A-Za-z0-9_.]+\}\}")
MODEL_KEYS = ["lexicon_rule", "baseline", "neural_net", "bert_fp32", "bert_int8"]
CHART_LABELS = ["Lexicon rule", "TF-IDF + LR", "W2V + FFNN", "BERT fp32", "BERT INT8"]


def _load(name: str) -> dict | None:
    p = C.RESULTS_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def fmt_mb(mb: float) -> str:
    if mb >= 10:
        return f"{mb:.1f}"
    return f"{mb:.2f}" if mb >= 0.1 else f"{mb:.3f}"


def build_tokens(final: dict | None, data: dict | None, failures: dict | None,
                 repo: str | None, video: str | None, hardware: str | None,
                 fail_why: str | None, fail_index: int = 0) -> tuple[dict, dict]:
    """Return (tokens, chart_values). Only values present in the result files are set."""
    t: dict[str, str] = {}
    if repo:
        t["REPO_URL"] = repo
    if video:
        t["VIDEO_URL"] = video

    if data:
        t["N_TOTAL"] = f"{data['n_total']:,}"
        for s in ("train", "val", "test"):
            t[f"N_{s.upper()}"] = f"{data[f'n_{s}']:,}"
        t["PCT_OFFENSIVE"] = f"{data['pct_offensive']:.1f}%"

    rows = {r["key"]: r for r in (final or {}).get("rows", [])}
    charts = {"chart macro-f1": [], "chart latency": []}
    for k in MODEL_KEYS:
        r = rows.get(k)
        charts["chart macro-f1"].append(round(r["macro_f1"], 4) if r else 0.0)
        charts["chart latency"].append(round(r["latency_mean_ms"], 2) if r else 0.0)
        if not r:
            continue
        t[f"{k}.F1"] = f"{r['macro_f1']:.4f}"
        t[f"{k}.SIZE"] = fmt_mb(r["size_mb"])
        t[f"{k}.LAT"] = f"{r['latency_mean_ms']:.2f}"
        t[f"{k}.P95"] = f"{r['latency_p95_ms']:.2f}"
        t[f"{k}.OK"] = "✅" if r["latency_mean_ms"] < C.LATENCY_BUDGET_MS else "❌"

    if final:
        t["RUNS"] = str(final["runs"])
        if "N_TEST" not in t:
            t["N_TEST"] = f"{final['n_test']:,}"
        hw = hardware or final.get("env", {}).get("hardware")
        if hw:
            t["HARDWARE"] = hw
    elif hardware:
        t["HARDWARE"] = hardware

    if rows:
        best = max(rows.values(), key=lambda r: r["macro_f1"])
        fast = min(rows.values(), key=lambda r: r["latency_mean_ms"])
        short = dict(zip(MODEL_KEYS, CHART_LABELS))
        t["BEST_F1_MODEL"], t["BEST_F1"] = short[best["key"]], f"{best['macro_f1']:.4f}"
        t["FASTEST_MODEL"], t["FASTEST_LAT"] = short[fast["key"]], f"{fast['latency_mean_ms']:.2f}"
        t["N_WITHIN"] = str(sum(r["latency_mean_ms"] < C.LATENCY_BUDGET_MS for r in rows.values()))
        t["N_MODELS"] = str(len(rows))
    if "bert_fp32" in rows and "bert_int8" in rows:
        a, b = rows["bert_fp32"], rows["bert_int8"]
        t["INT8_SPEEDUP"] = f"{a['latency_mean_ms'] / b['latency_mean_ms']:.1f}"
        t["INT8_SIZE_RATIO"] = f"{a['size_mb'] / b['size_mb']:.1f}"
        t["INT8_F1_DELTA"] = f"{b['macro_f1'] - a['macro_f1']:+.4f}"

    ex = None
    if failures:
        if fail_index == 0:
            ex = failures.get("slide_example")
        if ex is None or fail_index:
            prim = next((failures["models"][k] for k in ("bert_int8", "bert_fp32")
                         if k in failures.get("models", {})), None)
            pool = (prim["false_negatives"] + prim["false_positives"]) if prim else []
            if pool:
                ex = {"model": prim["model"], **pool[min(fail_index, len(pool) - 1)]}
    if ex:
        text = " ".join(ex["text"].split())
        t["FAIL_MODEL"] = ex["model"]
        t["FAIL_TEXT"] = text if len(text) <= 160 else text[:157] + "…"
        t["FAIL_TRUE"], t["FAIL_PRED"] = ex["true"], ex["pred"]
        t["FAIL_PROB"] = f"{ex['prob_offensive']:.2f}"
    if fail_why:
        t["FAIL_WHY"] = fail_why
    return t, charts


def _replace_in_paragraph(par, tokens: dict) -> None:
    def sub(s: str) -> str:
        return TOKEN_RE.sub(lambda m: tokens.get(m.group(0)[2:-2], m.group(0)), s)

    for run in par.runs:
        if "{{" in run.text:
            run.text = sub(run.text)
    # token split across runs (e.g. after manual editing): merge into the first run
    full = "".join(r.text for r in par.runs)
    if TOKEN_RE.search(full) and not any(TOKEN_RE.search(r.text) for r in par.runs):
        new = sub(full)
        if new != full and par.runs:
            par.runs[0].text = new
            for r in par.runs[1:]:
                r.text = ""


def _text_frames(shape):
    if shape.has_text_frame:
        yield shape.text_frame
    if getattr(shape, "has_table", False) and shape.has_table:
        for row in shape.table.rows:
            for cell in row.cells:
                yield cell.text_frame
    if shape.shape_type == 6:  # group
        for s in shape.shapes:
            yield from _text_frames(s)


def fill(template: Path, out: Path, tokens: dict, charts: dict) -> list[str]:
    from pptx import Presentation
    from pptx.chart.data import CategoryChartData

    prs = Presentation(str(template))
    remaining: set[str] = set()
    for slide in prs.slides:
        for shape in slide.shapes:
            for tf in _text_frames(shape):
                for par in tf.paragraphs:
                    _replace_in_paragraph(par, tokens)
                    remaining.update(TOKEN_RE.findall("".join(r.text for r in par.runs)))
            if getattr(shape, "has_chart", False) and shape.has_chart and shape.name in charts:
                cd = CategoryChartData(number_format="0.000" if "f1" in shape.name else "0.0")
                cd.categories = CHART_LABELS
                title = "Macro-F1" if "f1" in shape.name else "Latency (ms)"
                cd.add_series(title, charts[shape.name])
                shape.chart.replace_data(cd)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    return sorted(remaining)


def main():
    ap = argparse.ArgumentParser(description="Insert measured results into the slide deck.")
    ap.add_argument("--template", type=Path, default=ROOT / "docs" / "presentation.pptx")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "presentation_filled.pptx")
    ap.add_argument("--repo", help="public GitHub/GitLab URL")
    ap.add_argument("--video", help="unlisted YouTube / Google Drive URL")
    ap.add_argument("--hardware", help='e.g. "Intel Core i7-11370H, 32 GB RAM"')
    ap.add_argument("--fail-why", help="your one-sentence explanation of the failure example")
    ap.add_argument("--fail-index", type=int, default=0,
                    help="pick another failure from failure_examples.json (0 = default)")
    args = ap.parse_args()

    tokens, charts = build_tokens(_load("final_results.json"), _load("data_stats.json"),
                                  _load("failure_examples.json"), args.repo, args.video,
                                  args.hardware, args.fail_why, args.fail_index)
    remaining = fill(args.template, args.out, tokens, charts)
    print(f"Filled {len(tokens)} values -> {args.out}")
    if remaining:
        print("\nSTILL UNFILLED (do not submit until these are gone):")
        for r in remaining:
            print("  " + r)
        print("Run the missing pipeline steps or pass the matching --options, then re-run.")
        sys.exit(1)
    print("All placeholders filled. Review the deck, then export it to PDF.")


if __name__ == "__main__":
    main()
````


---

## `scripts/build_deck.js`

````javascript
// Builds docs/presentation.pptx — results are {{TOKENS}} filled by scripts/fill_slides.py
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { applyTheme } = require(process.env.SKILL_DIR + "/scripts/apply_theme.js");

const OUT = process.argv[2] || "presentation.pptx";

const HEX = {
  ink: "1B1F2A", slate: "2B3245", white: "FFFFFF", mist: "F1F3F7",
  teal: "2EC4B6", red: "E63946", amber: "F4A261", blue: "4361EE", muted: "8D99AE", grey: "5C6475",
};
const THEME = {
  name: "Moderation Signal", headFontFace: "Arial", bodyFontFace: "Calibri",
  colors: {
    dk1: HEX.ink, lt1: HEX.white, dk2: HEX.slate, lt2: HEX.mist,
    accent1: HEX.teal, accent2: HEX.red, accent3: HEX.amber, accent4: HEX.blue,
    accent5: HEX.muted, accent6: HEX.grey, hlink: HEX.blue, folHlink: HEX.grey,
  },
};

async function icon(Comp, hex, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + hex, size }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9"; // 10 x 5.625
  pres.author = "Matin Jebraeili";
  pres.title = "Real-Time Chat Moderation – SWE015 Use Case 4";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;

  const FOOTER = "SWE015 · Use Case 4 · Real-Time Chat Moderation · Matin Jebraeili";

  // ---------------------------------------------------------------- layouts
  pres.defineSlideMaster({
    title: "TITLE_DARK",
    background: { color: C.text1 },
    objects: [
      { placeholder: { options: { name: "kicker", type: "body", x: 0.6, y: 0.55, w: 8.8, h: 0.35,
        fontSize: 11, bold: true, color: C.accent1, charSpacing: 2, margin: 0 }, text: "" } },
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.05, w: 5.6, h: 1.3,
        fontSize: 38, bold: true, color: C.background1, align: "left", valign: "top", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "subtitle", type: "body", x: 0.6, y: 2.45, w: 5.4, h: 0.65,
        fontSize: 16, color: C.accent5, valign: "top", margin: 0 }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENT",
    background: { color: C.background1 },
    margin: [0.5, 0.5, 0.5, 0.5],
    objects: [
      { placeholder: { options: { name: "kicker", type: "body", x: 0.5, y: 0.28, w: 6, h: 0.28,
        fontSize: 10, bold: true, color: C.accent4, charSpacing: 2, margin: 0 }, text: "" } },
      { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.56, w: 9, h: 0.55,
        fontSize: 24, bold: true, color: C.text1, align: "left", valign: "top", margin: 0 }, text: "" } },
      { text: { text: FOOTER, options: { x: 0.5, y: 5.27, w: 7.5, h: 0.25, fontSize: 9,
        color: C.accent6, margin: 0 } } },
    ],
    slideNumber: { x: 9.0, y: 5.27, w: 0.5, h: 0.25, fontSize: 9, color: C.accent6, align: "right" },
  });
  pres.defineSlideMaster({
    title: "CLOSING_DARK",
    background: { color: C.text1 },
    objects: [
      { placeholder: { options: { name: "kicker", type: "body", x: 0.5, y: 0.28, w: 6, h: 0.28,
        fontSize: 10, bold: true, color: C.accent1, charSpacing: 2, margin: 0 }, text: "" } },
      { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.56, w: 9, h: 0.55,
        fontSize: 24, bold: true, color: C.background1, align: "left", valign: "top", margin: 0 }, text: "" } },
      { text: { text: FOOTER, options: { x: 0.5, y: 5.27, w: 7.5, h: 0.25, fontSize: 9,
        color: C.accent5, margin: 0 } } },
    ],
    slideNumber: { x: 9.0, y: 5.27, w: 0.5, h: 0.25, fontSize: 9, color: C.accent5, align: "right" },
  });

  const card = (slide, name, x, y, w, h, fill = C.background2) =>
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { objectName: name, x, y, w, h,
      fill: { color: fill }, line: { color: fill, width: 0 }, rectRadius: 0.08 });
  const txt = (slide, name, text, o) => slide.addText(text, { objectName: name, isTextBox: true,
    margin: 0, valign: "top", fontSize: 14, color: C.text1, ...o });
  const iconCircle = async (slide, name, Comp, x, y, d, circle, glyph) => {
    slide.addShape(pres.shapes.OVAL, { objectName: name + " circle", x, y, w: d, h: d,
      fill: { color: circle }, line: { color: circle, width: 0 } });
    const pad = d * 0.24;
    slide.addImage({ objectName: name + " icon", data: await icon(Comp, glyph),
      x: x + pad, y: y + pad, w: d - 2 * pad, h: d - 2 * pad });
  };
  const arrow = (slide, name, x, y, w, h = 0, color = HEX.muted) =>
    slide.addShape(pres.shapes.LINE, { objectName: name, x, y, w, h,
      line: { color, width: 1.5, endArrowType: "triangle" } });

  // ================================================================ 1 TITLE
  pres.addSection({ title: "Title" });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Title" });
  s.addText("SWE015 · INTRODUCTION TO LARGE LANGUAGE MODELS · INDIVIDUAL HOMEWORK", { placeholder: "kicker" });
  s.addText("Real-Time Chat Moderation", { placeholder: "title" });
  s.addText("A four-model ladder for English chat, served by a quantized BERT under a 50 ms CPU budget",
    { placeholder: "subtitle" });
  const info = [
    ["Student", "Matin Jebraeili"],
    ["Student ID", "2309115105"],
    ["Use case", "Use Case 4 – Real-Time Chat Moderation (English)"],
    ["Repository", "{{REPO_URL}}"],
    ["Demo video", "{{VIDEO_URL}}"],
  ];
  info.forEach(([k, v], i) => {
    txt(s, "info label " + i, k, { x: 0.6, y: 3.25 + i * 0.36, w: 1.3, h: 0.3, fontSize: 12,
      bold: true, color: C.accent1 });
    txt(s, "info value " + i, v, { x: 1.9, y: 3.25 + i * 0.36, w: 4.3, h: 0.3, fontSize: 12,
      color: C.background1 });
  });
  // chat-bubble motif: what the app does
  const bubbles = [
    ["gg wp everyone, close match", HEX.teal, "ALLOW"],
    ["you f****** i****", HEX.amber, "MASK"],
    ["[message hidden by moderator]", HEX.red, "BLOCK"],
  ];
  for (let i = 0; i < bubbles.length; i++) {
    const [t, col, tag] = bubbles[i];
    const y = 1.35 + i * 1.05;
    card(s, "bubble " + i, 6.5, y, 2.9, 0.7, HEX.slate);
    txt(s, "bubble text " + i, t, { x: 6.68, y: y + 0.12, w: 2.55, h: 0.3, fontSize: 13, color: C.background1 });
    txt(s, "bubble tag " + i, tag, { x: 6.68, y: y + 0.42, w: 2.55, h: 0.2, fontSize: 9, bold: true,
      color: col, charSpacing: 2 });
  }

  // ================================================================ 2 PROBLEM
  pres.addSection({ title: "Introduction" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("INTRODUCTION", { placeholder: "kicker" });
  s.addText("Gaming chat needs a verdict before the message lands", { placeholder: "title" });
  txt(s, "problem", [
    { text: "The problem", options: { bold: true, fontSize: 16, breakLine: true } },
    { text: "A gaming platform must flag offensive messages in a high-velocity chat stream, mask them, and do it on CPU-only servers within a strict time limit.",
      options: { breakLine: true } },
    { text: " ", options: { fontSize: 8, breakLine: true } },
    { text: "Who it is for", options: { bold: true, fontSize: 16, breakLine: true } },
    { text: "Players", options: { bullet: { indent: 12 }, bold: true } },
    { text: " – safer chat without waiting for a human", options: { breakLine: true } },
    { text: "Moderators", options: { bullet: { indent: 12 }, bold: true } },
    { text: " – fewer messages to review by hand", options: { breakLine: true } },
    { text: "Platform", options: { bullet: { indent: 12 }, bold: true } },
    { text: " – runs on cheap CPU servers, no GPU at inference" },
  ], { x: 0.5, y: 1.4, w: 4.4, h: 3.6, paraSpaceAfter: 4 });
  txt(s, "success head", "What success looks like", { x: 5.3, y: 1.4, w: 4.2, h: 0.35, fontSize: 16, bold: true });
  const succ = [
    [fa.FaBolt, HEX.teal, "< 50 ms per message", "Quantized BERT meets the latency budget on CPU (mean over ≥ 200 runs)"],
    [fa.FaBalanceScale, HEX.blue, "Macro-F1 on one test set", "Every model scored on the same held-out split; macro-F1 because only a minority is offensive"],
    [fa.FaComments, HEX.amber, "A live app", "Chat UI labels and masks each message, logs latency, runs the INT8 model"],
  ];
  for (let i = 0; i < succ.length; i++) {
    const [Ic, col, h, d] = succ[i];
    const y = 1.85 + i * 1.08;
    card(s, "success card " + i, 5.3, y, 4.2, 0.95);
    await iconCircle(s, "success " + i, Ic, 5.45, y + 0.2, 0.55, col, HEX.white);
    txt(s, "success title " + i, h, { x: 6.15, y: y + 0.1, w: 3.25, h: 0.3, fontSize: 14, bold: true });
    txt(s, "success desc " + i, d, { x: 6.15, y: y + 0.4, w: 3.25, h: 0.5, fontSize: 11, color: C.accent6 });
  }

  // ================================================================ 3 DATA
  pres.addSection({ title: "Development" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Development" });
  s.addText("DEVELOPMENT · DATA", { placeholder: "kicker" });
  s.addText("Jigsaw comments, one frozen split, one normalizer", { placeholder: "title" });
  card(s, "data card", 0.5, 1.4, 4.3, 3.65);
  await iconCircle(s, "data", fa.FaDatabase, 0.7, 1.55, 0.5, HEX.blue, HEX.white);
  txt(s, "data head", "Jigsaw Toxic Comment Classification", { x: 1.35, y: 1.6, w: 3.35, h: 0.4, fontSize: 14, bold: true });
  txt(s, "data body", [
    { text: "Source: ", options: { bold: true } },
    { text: "Kaggle (Jigsaw / Conversation AI), English Wikipedia talk-page comments", options: { breakLine: true } },
    { text: "Label: ", options: { bold: true } },
    { text: "Offensive = any of toxic, severe_toxic, obscene, threat, insult, identity_hate", options: { breakLine: true } },
    { text: "Size: ", options: { bold: true } },
    { text: "{{N_TOTAL}} comments after de-duplication", options: { breakLine: true } },
    { text: "Splits: ", options: { bold: true } },
    { text: "stratified 80 / 10 / 10, seed 42", options: { breakLine: true } },
    { text: "train {{N_TRAIN}} · val {{N_VAL}} · test {{N_TEST}}", options: { breakLine: true } },
    { text: "Offensive share: ", options: { bold: true } },
    { text: "{{PCT_OFFENSIVE}}" },
  ], { x: 0.75, y: 2.15, w: 3.85, h: 2.8, fontSize: 13, paraSpaceAfter: 6 });

  txt(s, "pre head", "Regex normalization (src/preprocessing.py)", { x: 5.2, y: 1.4, w: 4.3, h: 0.35, fontSize: 14, bold: true });
  const steps = ["Lowercase, strip accents", "Remove URLs, e-mails, IPs, HTML",
    "Leetspeak → letters, only inside words", "Collapse 3+ repeated letters", "Strip punctuation, join “f u c k”"];
  steps.forEach((t, i) => {
    const y = 1.85 + i * 0.42;
    s.addShape(pres.shapes.OVAL, { objectName: "step dot " + i, x: 5.2, y: y + 0.03, w: 0.28, h: 0.28,
      fill: { color: C.accent4 }, line: { color: C.accent4, width: 0 } });
    txt(s, "step num " + i, String(i + 1), { x: 5.2, y: y + 0.03, w: 0.28, h: 0.28, fontSize: 10, bold: true,
      color: C.background1, align: "center", valign: "middle" });
    txt(s, "step " + i, t, { x: 5.6, y: y + 0.02, w: 3.9, h: 0.3, fontSize: 13 });
  });
  card(s, "example card", 5.2, 4.0, 4.3, 1.05, HEX.slate);
  txt(s, "example", [
    { text: "Y0u are a  STUP1D   iiiiidiooot!!!", options: { fontFace: "Courier New", color: C.accent3, breakLine: true } },
    { text: "→  you are a stupid idiot", options: { fontFace: "Courier New", color: C.accent1, bold: true } },
  ], { x: 5.4, y: 4.15, w: 3.95, h: 0.8, fontSize: 12, valign: "middle", paraSpaceAfter: 6 });

  // ================================================================ 4 ARCHITECTURE
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Development" });
  s.addText("DEVELOPMENT · ARCHITECTURE", { placeholder: "kicker" });
  s.addText("One message in, five models, one moderation decision", { placeholder: "title" });
  const colW = 1.62, gap = 0.27, top = 1.45, H = 3.05;
  const xs = [0.5, 0.5 + (colW + gap), 0.5 + 2 * (colW + gap), 0.5 + 3 * (colW + gap), 0.5 + 4 * (colW + gap)];
  const block = (i, head, sub, body, fill = C.background2, headCol = C.text1) => {
    card(s, "arch col " + i, xs[i], top, colW, H, fill);
    txt(s, "arch head " + i, head, { x: xs[i] + 0.1, y: top + 0.12, w: colW - 0.2, h: 0.5, fontSize: 12, bold: true, color: headCol });
    txt(s, "arch sub " + i, sub, { x: xs[i] + 0.1, y: top + 0.62, w: colW - 0.2, h: 0.3, fontSize: 9, color: C.accent6 });
    if (body) txt(s, "arch body " + i, body, { x: xs[i] + 0.1, y: top + 0.98, w: colW - 0.2, h: H - 1.1, fontSize: 10.5, paraSpaceAfter: 4 });
  };
  block(0, "User input", "app.py · Gradio · browser", [
    { text: "Typed message", options: { bullet: { indent: 12 }, breakLine: true } },
    { text: "“Stream test messages” replay", options: { bullet: { indent: 12 }, breakLine: true } },
    { text: "Primary-model dropdown", options: { bullet: { indent: 12 } } }]);
  block(1, "Preprocessing", "src/preprocessing.py", [
    { text: "normalize_text(): regex leetspeak, repeats, URLs, punctuation", options: { breakLine: true } },
    { text: "Applied inside every model; BERT then uses its WordPiece tokenizer" }]);
  card(s, "arch col 2", xs[2], top, colW, H, C.background2);
  txt(s, "arch head 2", "Models (all on CPU)", { x: xs[2] + 0.1, y: top + 0.12, w: colW - 0.2, h: 0.3, fontSize: 12, bold: true });
  const mods = [["Lexicon rule", "rules · OSA ≤ 1"], ["TF-IDF + LogReg", "fp64 · sklearn"], ["Word2Vec + FFNN", "fp32 · PyTorch"],
    ["BERT-base", "fp32 · Transformers"], ["BERT-base ★", "dynamic INT8 · primary"]];
  mods.forEach(([m, p], i) => {
    const y = top + 0.5 + i * 0.5;
    const isQ = i === 4;
    card(s, "model box " + i, xs[2] + 0.08, y, colW - 0.16, 0.44, isQ ? C.accent1 : C.background1);
    txt(s, "model name " + i, m, { x: xs[2] + 0.16, y: y + 0.03, w: colW - 0.3, h: 0.2, fontSize: 10, bold: true });
    txt(s, "model prec " + i, p, { x: xs[2] + 0.16, y: y + 0.22, w: colW - 0.3, h: 0.18, fontSize: 8.5, color: isQ ? C.text1 : C.accent6 });
  });
  block(3, "Post-processing", "src/moderation.py", [
    { text: "Primary model → ALLOW / MASK / FLAG / BLOCK", options: { breakLine: true } },
    { text: "LexiconMatcher masks offensive words: “f******”" }]);
  block(4, "Output", "Gradio chat + side panel", [
    { text: "Masked message + verdict, p(offensive)", options: { bullet: { indent: 12 }, breakLine: true } },
    { text: "Per-model latency vs 50 ms", options: { bullet: { indent: 12 }, breakLine: true } },
    { text: "Session log + benchmark table", options: { bullet: { indent: 12 } } }]);
  for (let i = 0; i < 4; i++) arrow(s, "arrow " + i, xs[i] + colW + 0.03, top + H / 2, gap - 0.06);
  card(s, "runs on", 0.5, 4.62, 9.0, 0.48, HEX.slate);
  txt(s, "runs on text", [
    { text: "Where it runs: ", options: { bold: true, color: C.accent1 } },
    { text: "everything on CPU – measured runs: cloud VM, Intel Xeon 2.3 GHz, 2 vCPUs · no GPU used · app: local Python process · weights: Hugging Face Hub → models/",
      options: { color: C.background1 } }],
    { x: 0.65, y: 4.66, w: 8.75, h: 0.4, fontSize: 10.5, valign: "middle" });

  // ================================================================ 5 TECHNIQUE MAP
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Development" });
  s.addText("DEVELOPMENT · TECHNIQUE MAP", { placeholder: "kicker" });
  s.addText("Technique map: every area has a working component", { placeholder: "title" });
  const areaCell = (label, color) => ({ text: label, options: { bold: true, color: C.background1, fill: { color } } });
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 } } });
  const rows = [
    [hdr("Area"), hdr("Technique"), hdr("Component in the application (file)")],
    [areaCell("A · Neural networks", HEX.blue), "Feedforward net, SGD + backprop, dropout", "Word2Vec + FFNN classifier (neural_net.py)"],
    [areaCell("B · Classical NLP", HEX.teal), "Regex normalization (leetspeak, repeated letters)", "Input of every model (preprocessing.py)"],
    [areaCell("B · Classical NLP", HEX.teal), "Minimum edit distance (OSA) lexicon matcher", "Rule baseline + masking (rule_baseline.py)"],
    [areaCell("B · Classical NLP", HEX.teal), "tf-idf (word + char n-grams) + logistic regression", "Classical baseline (baseline.py)"],
    [areaCell("B · Classical NLP", HEX.teal), "word2vec trained on the training messages", "FFNN embedding layer (neural_net.py)"],
    [areaCell("C · Transformers", HEX.amber), "BERT fine-tuning with a task head; WordPiece", "bert-base-uncased (transformer_model.py)"],
    [areaCell("D · Efficient models", HEX.red), "Dynamic INT8 quantization; size/latency/F1 benchmark", "Primary app model (quantize.py, evaluate.py)"],
  ];
  s.addTable(rows, { objectName: "technique table", x: 0.5, y: 1.35, w: 9.0, colW: [1.75, 3.75, 3.5],
    fontSize: 12, color: C.text1, fontFace: "Calibri", rowH: 0.38, valign: "middle",
    border: { type: "solid", pt: 0.75, color: "D5D9E2" }, margin: [2, 5, 2, 5] });
  txt(s, "swap note", [
    { text: "Component swap (allowed within an area): ", options: { bold: true } },
    { text: "the brief suggests BERTurk or multilingual BERT; I use English bert-base-uncased because the data (Jigsaw) is English." }],
    { x: 0.5, y: 4.6, w: 9.0, h: 0.45, fontSize: 11, color: C.accent6 });

  // ================================================================ 6 MODELS
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Development" });
  s.addText("DEVELOPMENT · MODELS", { placeholder: "kicker" });
  s.addText("The ladder, and how each model was trained", { placeholder: "title" });
  const mcards = [
    [fa.FaSpellCheck, HEX.teal, "Lexicon rule", "No training. Seed list + 88 terms mined from the train split (TF-IDF/LR weight, ≥ 60% offensive); offensive if any token matches, incl. OSA distance 1."],
    [fa.FaChartBar, HEX.teal, "TF-IDF + LogReg", "Word 1–2-grams + char_wb 2–5-grams, sublinear tf. Logistic regression (liblinear, C = 4, class-balanced)."],
    [fa.FaProjectDiagram, HEX.blue, "Word2Vec + FFNN", "Skip-gram 200-d, window 5, 10 epochs. Mean-pooled embeddings → LayerNorm → 256 → 64 → 1. SGD (momentum 0.9), lr 0.05, early stopping."],
    [fa.FaBrain, HEX.amber, "BERT-base (fp32)", "bert-base-uncased, 128 tokens. HF Trainer on CPU: 6,000-message stratified train subset, 1 epoch, AdamW 2e-5, batch 16 × 2, class-weighted loss."],
    [fa.FaCompressArrowsAlt, HEX.red, "BERT-base (INT8)", "Fine-tuned BERT after post-training dynamic INT8 quantization (next slide). No retraining."],
    [fa.FaClipboardCheck, HEX.grey, "Shared protocol", "Same full test split (15,958) for all. Threshold tuned on validation for macro-F1 (BERT: 4,000-message val subset)."],
  ];
  for (let i = 0; i < mcards.length; i++) {
    const [Ic, col, h, d] = mcards[i];
    const cx = 0.5 + (i % 3) * 3.07, cy = 1.4 + Math.floor(i / 3) * 1.85;
    card(s, "model card " + i, cx, cy, 2.86, 1.7);
    await iconCircle(s, "model " + i, Ic, cx + 0.15, cy + 0.15, 0.45, col, HEX.white);
    txt(s, "model card title " + i, h, { x: cx + 0.7, y: cy + 0.22, w: 2.05, h: 0.32, fontSize: 14, bold: true });
    txt(s, "model card body " + i, d, { x: cx + 0.15, y: cy + 0.7, w: 2.6, h: 0.95, fontSize: 10.5, color: C.text2 });
  }

  // ================================================================ 7 QUANTIZATION
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Development" });
  s.addText("DEVELOPMENT · QUANTIZATION", { placeholder: "kicker" });
  s.addText("Dynamic INT8: int8 Linear layers, fp32 embeddings", { placeholder: "title" });
  txt(s, "quant method", [
    { text: "torch.ao.quantization.quantize_dynamic(",
      options: { fontFace: "Courier New", fontSize: 11, color: C.accent4, breakLine: true } },
    { text: "    model, {nn.Linear}, dtype=torch.qint8)",
      options: { fontFace: "Courier New", fontSize: 11, color: C.accent4, breakLine: true } },
    { text: " ", options: { fontSize: 6, breakLine: true } },
    { text: "Weights of every Linear layer (attention Q/K/V/output, feed-forward, classifier head) stored as int8", options: { bullet: { indent: 12 }, breakLine: true } },
    { text: "Activations quantized on the fly per input; int8 matmuls on x86 (fbgemm) kernels", options: { bullet: { indent: 12 }, breakLine: true } },
    { text: "Embeddings and LayerNorm stay fp32, so the file shrinks by less than 4×", options: { bullet: { indent: 12 }, breakLine: true } },
    { text: "Decision threshold re-tuned on validation after quantization", options: { bullet: { indent: 12 }, breakLine: true } },
    { text: "Saved as an int8 state_dict and reloaded into a quantized skeleton: the app loads exactly what was evaluated", options: { bullet: { indent: 12 } } },
  ], { x: 0.5, y: 1.4, w: 4.6, h: 3.65, fontSize: 13, paraSpaceAfter: 6 });
  const stats = [
    ["Size (MB)", "{{bert_fp32.SIZE}}", "{{bert_int8.SIZE}}"],
    ["Latency (ms, CPU)", "{{bert_fp32.LAT}}", "{{bert_int8.LAT}}"],
    ["Macro-F1", "{{bert_fp32.F1}}", "{{bert_int8.F1}}"],
  ];
  txt(s, "fp32 label", "fp32", { x: 6.85, y: 1.4, w: 1.2, h: 0.3, fontSize: 11, bold: true, color: C.accent6, align: "center" });
  txt(s, "int8 label", "INT8", { x: 8.25, y: 1.4, w: 1.2, h: 0.3, fontSize: 11, bold: true, color: C.accent1, align: "center" });
  stats.forEach(([k, a, b], i) => {
    const y = 1.75 + i * 1.1;
    card(s, "stat card " + i, 5.45, y, 4.05, 0.95);
    txt(s, "stat label " + i, k, { x: 5.6, y: y + 0.1, w: 1.25, h: 0.75, fontSize: 12, bold: true, valign: "middle" });
    txt(s, "stat fp32 " + i, a, { x: 6.85, y: y + 0.1, w: 1.2, h: 0.75, fontSize: 22, color: C.accent6, align: "center", valign: "middle" });
    txt(s, "stat arrow " + i, "→", { x: 8.0, y: y + 0.1, w: 0.3, h: 0.75, fontSize: 18, color: C.accent5, align: "center", valign: "middle" });
    txt(s, "stat int8 " + i, b, { x: 8.25, y: y + 0.1, w: 1.2, h: 0.75, fontSize: 22, bold: true, color: C.accent1, align: "center", valign: "middle" });
  });

  // ================================================================ 8 RESULTS TABLE
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Development" });
  s.addText("DEVELOPMENT · RESULTS", { placeholder: "kicker" });
  s.addText("Results on the same held-out test split", { placeholder: "title" });
  const H2 = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, align: "center" } });
  const R = (k, name, area, prec, hl = false) => {
    const o = hl ? { fill: { color: "D5F5F2" }, bold: true } : {};
    const c = (t, align = "center") => ({ text: t, options: { align, ...o } });
    return [c(name, "left"), c(area, "left"), c(prec), c(`{{${k}.F1}}`), c(`{{${k}.SIZE}}`), c(`{{${k}.LAT}}`), c(`{{${k}.P95}}`), c(`{{${k}.OK}}`)];
  };
  s.addTable([
    [H2("Model"), H2("Area"), H2("Precision"), H2("Macro-F1"), H2("Size (MB)"), H2("Avg latency (ms)"), H2("p95 (ms)"), H2("< 50 ms")],
    R("lexicon_rule", "Lexicon rule (edit distance)", "B", "rules"),
    R("baseline", "TF-IDF + LogReg", "B", "fp64"),
    R("neural_net", "Word2Vec + FFNN (SGD)", "A", "fp32"),
    R("bert_fp32", "BERT-base-uncased", "C", "fp32"),
    R("bert_int8", "BERT-base-uncased ★ (app)", "D", "INT8", true),
  ], { objectName: "results table", x: 0.5, y: 1.4, w: 9.0, colW: [2.35, 0.6, 0.85, 1.0, 1.0, 1.35, 0.95, 0.9],
    fontSize: 12, color: C.text1, fontFace: "Calibri", rowH: 0.46, valign: "middle",
    border: { type: "solid", pt: 0.75, color: "D5D9E2" }, margin: [2, 5, 2, 5] });
  card(s, "protocol card", 0.5, 4.3, 9.0, 0.78);
  txt(s, "protocol", [
    { text: "Protocol: ", options: { bold: true } },
    { text: "macro-F1 on all {{N_TEST}} test messages · latency = mean of {{RUNS}} single-message calls (batch size 1) after warm-up, CPU only, same messages for every model", options: { breakLine: true } },
    { text: "Hardware: ", options: { bold: true } },
    { text: "{{HARDWARE}}" }],
    { x: 0.65, y: 4.37, w: 8.7, h: 0.65, fontSize: 11, valign: "middle", paraSpaceAfter: 2 });

  // ================================================================ 9 COMPARISON
  pres.addSection({ title: "Conclusion" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Conclusion" });
  s.addText("CONCLUSION", { placeholder: "kicker" });
  s.addText("What the comparison showed", { placeholder: "title" });
  const labels = ["Lexicon rule", "TF-IDF + LR", "W2V + FFNN", "BERT fp32", "BERT INT8"];
  const chartBase = (title, color) => ({
    barDir: "bar", showTitle: true, title, titleFontSize: 12, titleColor: HEX.ink, titleFontFace: "+mj-lt",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 9, dataLabelColor: HEX.slate, dataLabelFontFace: "+mn-lt",
    chartColors: [color], catAxisLabelColor: HEX.slate, catAxisLabelFontSize: 9, catAxisLabelFontFace: "+mn-lt",
    valAxisLabelColor: HEX.grey, valAxisLabelFontSize: 8, valAxisLabelFontFace: "+mn-lt",
    valGridLine: { color: "E3E6ED", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
    catAxisOrientation: "maxMin", valAxisMinVal: 0,
  });
  s.addChart(pres.charts.BAR, [{ name: "Macro-F1", labels, values: [0, 0, 0, 0, 0] }],
    { objectName: "chart macro-f1", x: 0.5, y: 1.3, w: 4.4, h: 2.45, ...chartBase("Macro-F1 (test split)", HEX.blue), valAxisMaxVal: 1, dataLabelFormatCode: "0.000", valAxisLabelFormatCode: "0.0" });
  s.addChart(pres.charts.BAR, [{ name: "Latency (ms)", labels, values: [0, 0, 0, 0, 0] }],
    { objectName: "chart latency", x: 5.1, y: 1.3, w: 4.4, h: 2.45, ...chartBase("Mean CPU latency per message (ms)", HEX.teal), dataLabelFormatCode: "0.0", valAxisLabelFormatCode: "0" });
  const takeaways = [
    [fa.FaTrophy, HEX.blue, "Most accurate", "{{BEST_F1_MODEL}}", "macro-F1 {{BEST_F1}}"],
    [fa.FaTachometerAlt, HEX.teal, "Fastest", "{{FASTEST_MODEL}}", "{{FASTEST_LAT}} ms per message"],
    [fa.FaCompressArrowsAlt, HEX.red, "INT8 vs fp32 BERT", "{{INT8_SPEEDUP}}× faster", "{{INT8_SIZE_RATIO}}× smaller · ΔF1 {{INT8_F1_DELTA}}"],
    [fa.FaStopwatch, HEX.amber, "Latency budget", "{{N_WITHIN}} of {{N_MODELS}} models", "under 50 ms on CPU"],
  ];
  for (let i = 0; i < takeaways.length; i++) {
    const [Ic, col, k, v, d] = takeaways[i];
    const cx = 0.5 + i * 2.29;
    card(s, "takeaway " + i, cx, 3.9, 2.14, 1.18);
    await iconCircle(s, "takeaway " + i, Ic, cx + 0.12, cy0(3.9), 0.36, col, HEX.white);
    txt(s, "takeaway key " + i, k, { x: cx + 0.56, y: 3.99, w: 1.5, h: 0.25, fontSize: 10, bold: true, color: C.accent6 });
    txt(s, "takeaway val " + i, v, { x: cx + 0.12, y: 4.35, w: 1.95, h: 0.32, fontSize: 13, bold: true });
    txt(s, "takeaway desc " + i, d, { x: cx + 0.12, y: 4.67, w: 1.95, h: 0.35, fontSize: 10, color: C.text2 });
  }
  function cy0(y) { return y + 0.08; }

  // ================================================================ 10 LIMITATIONS
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Conclusion" });
  s.addText("CONCLUSION · LIMITATIONS", { placeholder: "kicker" });
  s.addText("Limitations, with a real failure from the test split", { placeholder: "title" });
  card(s, "failure card", 0.5, 1.4, 4.3, 3.65, HEX.slate);
  await iconCircle(s, "failure", fa.FaExclamationTriangle, 0.7, 1.55, 0.45, HEX.red, HEX.white);
  txt(s, "failure head", "Failure example · {{FAIL_MODEL}}", { x: 1.3, y: 1.6, w: 3.4, h: 0.4, fontSize: 13, bold: true, color: C.background1 });
  txt(s, "failure text", "“{{FAIL_TEXT}}”", { x: 0.7, y: 2.15, w: 3.9, h: 1.15, fontSize: 13, italic: true, color: C.background1 });
  txt(s, "failure verdict", "True: {{FAIL_TRUE}} · Predicted: {{FAIL_PRED}} (p = {{FAIL_PROB}})",
    { x: 0.7, y: 3.35, w: 3.9, h: 0.3, fontSize: 11, bold: true, color: C.accent3 });
  txt(s, "failure why", [
    { text: "Why: ", options: { bold: true, color: C.accent1 } },
    { text: "{{FAIL_WHY}}", options: { color: C.background1 } }],
    { x: 0.7, y: 3.75, w: 3.9, h: 1.2, fontSize: 11 });
  txt(s, "limits", [
    { text: "Domain shift", options: { bullet: { indent: 12 }, bold: true } },
    { text: " – Wikipedia talk comments, not game chat (“killed it”, “trash”)", options: { breakLine: true } },
    { text: "English only", options: { bullet: { indent: 12 }, bold: true } },
    { text: " – Turkish needs BERTurk and OffensEval-TR", options: { breakLine: true } },
    { text: "CPU-only training", options: { bullet: { indent: 12 }, bold: true } },
    { text: " – BERT saw 6k messages (1 epoch) vs 128k for TF-IDF", options: { breakLine: true } },
    { text: "One message at a time", options: { bullet: { indent: 12 }, bold: true } },
    { text: " – no conversation context or user history", options: { breakLine: true } },
    { text: "Masking needs the lexicon", options: { bullet: { indent: 12 }, bold: true } },
    { text: " – implicit toxicity is flagged but has no word to mask", options: { breakLine: true } },
    { text: "Dataset bias", options: { bullet: { indent: 12 }, bold: true } },
    { text: " – identity terms are over-represented in toxic Jigsaw comments" },
  ], { x: 5.15, y: 1.4, w: 4.35, h: 3.65, fontSize: 12.5, paraSpaceAfter: 7 });

  // ================================================================ 11 NEXT
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Conclusion" });
  s.addText("CONCLUSION · NEXT STEPS", { placeholder: "kicker" });
  s.addText("What I would do next", { placeholder: "title" });
  const nexts = [
    [fa.FaFeatherAlt, HEX.red, "Distil + ONNX Runtime", "Distil BERT into a smaller student (e.g. DistilBERT), export to ONNX Runtime with INT8 for more latency headroom."],
    [fa.FaLanguage, HEX.blue, "Turkish support", "BERTurk on OffensEval-TR 2020; add Turkish-character rules (ı/i, ş/s) to the normalizer."],
    [fa.FaHistory, HEX.amber, "Conversation context", "Feed the previous messages and the user's recent history so sarcasm and targeted harassment become visible."],
    [fa.FaUserCheck, HEX.teal, "Moderator feedback loop", "Log moderator overrides from the app, retrain on them, and split the label into insult / threat / hate."],
  ];
  for (let i = 0; i < nexts.length; i++) {
    const [Ic, col, h, d] = nexts[i];
    const cx = 0.5 + (i % 2) * 4.6, cy = 1.4 + Math.floor(i / 2) * 1.85;
    card(s, "next card " + i, cx, cy, 4.4, 1.65);
    await iconCircle(s, "next " + i, Ic, cx + 0.2, cy + 0.25, 0.6, col, HEX.white);
    txt(s, "next title " + i, h, { x: cx + 1.0, y: cy + 0.25, w: 3.25, h: 0.35, fontSize: 15, bold: true });
    txt(s, "next body " + i, d, { x: cx + 1.0, y: cy + 0.65, w: 3.25, h: 0.9, fontSize: 12, color: C.text2 });
  }

  // ================================================================ 12 AI USAGE
  s = pres.addSlide({ masterName: "CLOSING_DARK", sectionTitle: "Conclusion" });
  s.addText("CONCLUSION · AI USAGE STATEMENT", { placeholder: "kicker" });
  s.addText("How AI tools were used in this project", { placeholder: "title" });
  const ai = [
    [fa.FaRobot, HEX.blue, "Tool", "Claude (Anthropic), used through claude.ai."],
    [fa.FaCode, HEX.amber, "Used for", "Drafting the code of every module (preprocessing, the four models, quantization, evaluation, Gradio app), the unit tests, the README and the first draft of these slides."],
    [fa.FaUserCheck, HEX.teal, "My own work", "Claude also executed the training, evaluation and latency runs in a cloud sandbox (Intel Xeon, 2 vCPUs) at my request. I reviewed the code and results and can explain every component."],
    [fa.FaCheckCircle, HEX.red, "Numbers", "Every number was measured by running this project's code and inserted from models/results/ by scripts/fill_slides.py. None comes from an AI answer, paper or model card."],
  ];
  for (let i = 0; i < ai.length; i++) {
    const [Ic, col, h, d] = ai[i];
    const y = 1.4 + i * 0.93;
    card(s, "ai card " + i, 0.5, y, 9.0, 0.8, HEX.slate);
    await iconCircle(s, "ai " + i, Ic, 0.65, y + 0.14, 0.52, col, HEX.white);
    txt(s, "ai head " + i, h, { x: 1.35, y: y + 0.12, w: 1.6, h: 0.55, fontSize: 14, bold: true, color: C.background1, valign: "middle" });
    txt(s, "ai body " + i, d, { x: 3.0, y: y + 0.1, w: 6.35, h: 0.6, fontSize: 12, color: C.background1, valign: "middle" });
  }
  s.addNotes("Edit this slide if you used any other AI tool (e.g. Copilot, ChatGPT) or used Claude for other parts.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote " + OUT);
})();
````


---

## `data/lexicon/seed_lexicon.txt`

````text
# Seed lexicon for the edit-distance matcher (used for masking in the app).
# One entry per line.  "word*" = prefix match, "two words" = phrase match.
# Entries are normalised with src.preprocessing.normalize_text before use.
# src/baseline.py also writes data/lexicon/learned_lexicon.txt from the
# strongest positive TF-IDF + LR features; both files are merged at load time.
#
# Extend this list for your deployment (e.g. slurs / identity-based hate terms).

# --- profanity (prefix forms catch -ing, -er, -ed ...) ---
fuck*
motherfuck*
shit*
bullshit
bitch*
bastard*
asshole*
dumbass
jackass
cunt*
dick
dickhead*
cock
cocksuck*
pussy
whore*
slut*
wanker*
twat*
prick
crap
damn
piss*
douche*

# --- insults ---
idiot*
moron*
stupid*
retard*
imbecile*
loser*
dumb
pathetic
scum*
trash
worthless
ugly
fag*
nazi*
pig

# --- threats / harassment phrases ---
kill yourself
kill you
kys
go die
shut up
i hope you die
````


---

## `data/lexicon/learned_lexicon.txt`

````text
# Auto-generated (src/baseline.py) from the TRAIN split: strongest TF-IDF+LR word
# weights, kept only if >= 60% of training messages containing them are offensive.
# Identity terms are excluded.
ass
fuck
shit
idiot
stupid
bitch
suck
fucking
cunt
dick
moron
loser
pig
sucks
jerk
penis
wtf
asshole
pussy
cock
bullshit
fat
dumb
balls
whore
twat
piss
pathetic
arse
scum
bastard
nigger
idiots
anus
faggot
cum
vagina
semen
whiny
morons
retard
nerd
stfu
dumbass
pervert
lick
fucked
imbecile
moms
fuk
nerds
wanker
losers
fag
tits
faggots
smelly
dipshit
dildo
bastards
jackass
dicks
hairy
homo
assholes
bitches
poop
niggers
donkey
tosser
masturbating
cretin
retarded
dickhead
prick
commie
jerks
bongwarrior
asses
mommy
cunts
chink
arsehole
slut
fucks
goddamn
shove
shitty
````


---

## `docs/DEMO_SCRIPT.md`

````markdown
# 2-minute demo video script

Record the screen with your voice (OBS, Xbox Game Bar `Win+Alt+R`, or Loom). Only the first 2:00 count.
Upload as **unlisted** on YouTube (or Google Drive with "anyone with the link") and test the link in a
private browser window.

| Time | Show | Say (roughly) |
|---|---|---|
| 0:00–0:15 | Title slide | "Gaming platforms must flag offensive chat on CPU-only servers in under 50 ms. I built a moderation app that compares four models and runs a quantized BERT." |
| 0:15–0:30 | `python app.py` running, browser at 127.0.0.1:7860 | "This is the live app. The primary model is the INT8-quantized BERT; the side panel shows every model's verdict and latency." |
| 0:30–0:45 | Type a **new** clean message, e.g. `nice clutch, gg everyone` | "A clean message is allowed; every model runs in a few milliseconds on CPU." |
| 0:45–1:00 | Type a new obfuscated insult, e.g. `ur such a fcuking 1d10t` | "Leetspeak and a misspelling: the regex normalizer and edit-distance matcher still catch it, and the words are masked." |
| 1:00–1:15 | **Difficult case**, e.g. `that boss fight was sick, I killed it` or an insult without swear words. Switch the dropdown to the TF-IDF baseline and send it again | "A hard case: gaming slang. Here the models disagree. You can switch the deciding model live." |
| 1:15–1:45 | Architecture slide, then the results table slide | "Input goes through regex normalization into five models; the policy masks and labels. On the same test split, INT8 BERT reached ___ macro-F1 at ___ ms, ___× faster than fp32." (read **your** numbers) |
| 1:45–2:00 | Conclusion slide | "Quantization made BERT fit the budget at a small accuracy cost; the next step is distillation, ONNX Runtime and Turkish support." |
````


---

## `tests/test_baseline.py`

````python
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
````


---

## `tests/test_evaluate.py`

````python
from src import config as C
from src.evaluate import latency_messages, results_to_markdown
from src.registry import SPECS, load_models


def _row(name, f1, lat):
    return {"tier": "T", "model": name, "area": "A", "macro_f1": f1, "size_mb": 1.5,
            "latency_mean_ms": lat, "latency_p95_ms": lat * 1.5}


def test_markdown_table_structure():
    md = results_to_markdown([_row("fast", 0.9, 3.0), _row("slow", 0.95, 80.0)], runs=200)
    lines = md.strip().splitlines()
    assert len(lines) == 4
    assert "Macro-F1" in lines[0] and "Size (MB)" in lines[0] and "n=200" in lines[0]
    assert "| 0.9000 |" in lines[2] and "✅" in lines[2]
    assert "❌" in lines[3]


def test_markdown_states_hardware():
    md = results_to_markdown([_row("m", 0.9, 3.0)], runs=200, hardware="Test CPU", n_test=1234)
    assert "Hardware: Test CPU" in md and "1,234-message" in md and "200 runs" in md


def test_latency_messages_deterministic():
    texts = [f"m{i}" for i in range(1000)]
    a, b = latency_messages(texts, 200), latency_messages(texts, 200)
    assert a == b and len(a) == 200 and len(set(a)) == 200
    assert len(latency_messages(texts[:50], 200)) == 50


def test_registry_covers_ladder():
    assert [s.key for s in SPECS] == ["lexicon_rule", "baseline", "neural_net",
                                      "bert_fp32", "bert_int8"]
    assert C.LATENCY_BUDGET_MS == 50.0


def test_load_models_skips_missing(monkeypatch, tmp_path):
    import src.registry as R
    fake = [R.ModelSpec("x", "X", "t", "a", "fp32", tmp_path / "nope", lambda d: None, "cmd")]
    monkeypatch.setattr(R, "SPECS", fake)
    assert R.load_models(verbose=False) == {}
    assert load_models is not None
````


---

## `tests/test_fill_slides.py`

````python
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
````


---

## `tests/test_moderation.py`

````python
"""App logic tests with fake models (no torch / gradio needed)."""
import pytest

from src.moderation import escape_md, moderate, run_models, verdict_line
from src.preprocessing import LexiconMatcher
from src.registry import SPEC_BY_KEY


class FakeModel:
    def __init__(self, prob, threshold=0.5):
        self.prob, self.threshold = prob, threshold

    def predict_one(self, text):
        return int(self.prob >= self.threshold), self.prob


@pytest.fixture
def lex():
    return LexiconMatcher(["idiot*", "shit*"])


def _models(p_int8, p_base=0.1):
    return {"bert_int8": (SPEC_BY_KEY["bert_int8"], FakeModel(p_int8)),
            "baseline": (SPEC_BY_KEY["baseline"], FakeModel(p_base))}


def test_run_models_times_every_model():
    outs = run_models("hello", _models(0.2))
    assert [o.key for o in outs] == ["bert_int8", "baseline"]
    assert all(o.latency_ms >= 0 for o in outs)
    assert outs[0].verdict == "Clean"


@pytest.mark.parametrize("text, p, action", [
    ("good game", 0.10, "ALLOW"),
    ("you idiot", 0.90, "MASK"),
    ("shit, gg", 0.10, "MASK"),          # profanity masked even when classified clean
    ("you are worthless garbage", 0.80, "FLAG"),
    ("you are worthless garbage", 0.99, "BLOCK"),
])
def test_moderation_actions(lex, text, p, action):
    outs = run_models(text, _models(p))
    res = moderate(text, outs, "bert_int8", lex)
    assert res.action == action
    assert res.primary.key == "bert_int8"
    if action == "MASK":
        assert "*" in res.display_text
    if action == "BLOCK":
        assert text not in res.display_text


def test_primary_falls_back_to_first(lex):
    outs = run_models("hi", _models(0.1))
    assert moderate("hi", outs, "missing", lex).primary.key == "bert_int8"


def test_escape_md_and_verdict(lex):
    assert escape_md("f*** _x_") == "f\\*\\*\\* \\_x\\_"
    res = moderate("you idiot", run_models("you idiot", _models(0.9)), "bert_int8", lex)
    line = verdict_line(res)
    assert "Masked" in line and "p=0.90" in line and "1 term(s)" in line
````


---

## `tests/test_neural_net.py`

````python
"""Tests for Tier 2: Word2Vec trainer, encoding, forward pass, SGD training, save/load.

Skipped automatically if torch / gensim are not installed.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src import config as C  # noqa: E402
from src.neural_net import (AvgEmbeddingClassifier, NeuralModel, NNConfig,  # noqa: E402
                            W2VConfig, collate_bags, embedding_matrix, encode,
                            standardize_embeddings,
                            tokenize_corpus, train_and_evaluate, train_word2vec)
from tests.test_baseline import _toy  # noqa: E402

TINY_W2V = W2VConfig(vector_size=16, window=3, min_count=1, max_final_vocab=None,
                     epochs=5, workers=1)
TINY_NN = NNConfig(hidden=(16,), dropout=0.0, batch_size=32, eval_batch_size=64,
                   lr=0.1, max_epochs=25, patience=5, num_workers=0)


# ------------------------------------------------------------------ word2vec
def test_word2vec_trains_and_exports_matrix():
    pytest.importorskip("gensim")
    toks = tokenize_corpus(["you stupid idiot", "good game well played", "Y0u 1d10t!!"] * 20)
    kv = train_word2vec(toks, TINY_W2V)
    vocab, emb = embedding_matrix(kv)
    assert emb.shape == (len(vocab), 16)
    assert emb.dtype == np.float32
    assert "idiot" in vocab            # leetspeak "1d10t" normalised before training
    assert "1d10t" not in vocab
    assert np.isfinite(emb).all()


def test_standardize_embeddings():
    emb = np.random.RandomState(0).randn(50, 8).astype(np.float32) + 5.0
    z = standardize_embeddings(emb)
    assert np.allclose(z.mean(0), 0, atol=1e-5) and np.allclose(z.std(0), 1, atol=1e-3)
    assert z.dtype == np.float32


# ------------------------------------------------------------------ encoding
def test_encode_drops_oov_and_truncates():
    stoi = {"a": 0, "b": 1, "c": 2}
    assert encode(["a", "zzz", "b", "c"], stoi, max_len=10) == [0, 1, 2]
    assert encode(["a", "b", "c"], stoi, max_len=2) == [0, 1]
    assert encode(["zzz"], stoi, max_len=5) == []


def test_collate_offsets():
    flat, offs, y = collate_bags([([1, 2, 3], 1.0), ([], 0.0), ([4], 1.0)])
    assert flat.tolist() == [1, 2, 3, 4]
    assert offs.tolist() == [0, 3, 3]
    assert y.tolist() == [1.0, 0.0, 1.0]


# ------------------------------------------------------------------ forward pass
def test_forward_shape_and_mean_pooling():
    emb = np.arange(12, dtype=np.float32).reshape(4, 3)
    net = AvgEmbeddingClassifier(emb, hidden=(8,), dropout=0.0).eval()
    flat, offs, _ = collate_bags([([0, 1], 0.0), ([], 0.0), ([2, 3, 3], 0.0)])
    pooled = net.embedding(flat, offs)
    assert torch.allclose(pooled[0], torch.tensor(emb[[0, 1]].mean(0)))
    assert torch.allclose(pooled[1], torch.zeros(3))          # empty message -> zeros
    out = net(flat, offs)
    assert out.shape == (3,)
    assert torch.isfinite(out).all()


def test_embeddings_frozen_by_default():
    net = AvgEmbeddingClassifier(np.ones((5, 4), np.float32))
    assert not net.embedding.weight.requires_grad
    net_ft = AvgEmbeddingClassifier(np.ones((5, 4), np.float32), finetune_emb=True)
    assert net_ft.embedding.weight.requires_grad


def test_backprop_updates_mlp_not_frozen_embeddings():
    torch.manual_seed(0)
    emb = np.random.RandomState(0).randn(6, 4).astype(np.float32)
    net = AvgEmbeddingClassifier(emb, hidden=(8,), dropout=0.0)
    before_emb = net.embedding.weight.clone()
    before_w = net.mlp[1].weight.clone()
    opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=0.5)
    flat, offs, y = collate_bags([([0, 1], 1.0), ([2, 3], 0.0)])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(net(flat, offs), y)
    loss.backward()
    opt.step()
    assert torch.equal(before_emb, net.embedding.weight)
    assert not torch.equal(before_w, net.mlp[1].weight)


# ------------------------------------------------------------------ end-to-end
def test_neural_net_end_to_end(tmp_path):
    pytest.importorskip("gensim")
    train, val, test = _toy(400, 0), _toy(100, 1), _toy(100, 2)
    path = tmp_path / "model.pt"
    res = train_and_evaluate(train, val, test, TINY_W2V, TINY_NN, save_path=path,
                             w2v_path=None, latency_runs=100, verbose=False)
    assert res["macro_f1"] > 0.8
    assert res["size_mb"] > 0
    assert res["latency_runs"] >= 100 and res["latency_mean_ms"] > 0
    assert res["latency_device"] == "cpu"
    assert 1 <= res["epochs_trained"] <= TINY_NN.max_epochs

    model = NeuralModel.load(path)
    label, p = model.predict_one("you stupid idiot")
    assert label == 1 and 0.0 <= p <= 1.0
    # single-message and batched paths agree
    assert abs(model.predict_proba(["you stupid idiot"])[0] - p) < 1e-5
    # all-OOV / empty input does not crash
    assert model.predict_one("")[1] >= 0.0
    assert set(np.unique(model.predict(test[C.TEXT_COL]))) <= {0, 1}
````


---

## `tests/test_preprocessing.py`

````python
import pytest

from src.preprocessing import (LexiconMatcher, deleet, normalize_text, osa_distance,
                               reduce_repeats)


@pytest.mark.parametrize("raw, expected", [
    ("noooo", "no"),
    ("fuuuuuck", "fuck"),
    ("cool", "cool"),                       # double letters preserved
])
def test_reduce_repeats(raw, expected):
    assert reduce_repeats(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("h4t3", "hate"),
    ("sh!t", "shit"),
    ("@$$", "@$$"),                         # no letters -> untouched
    ("2024", "2024"),
    ("idiot!", "idiot!"),                   # trailing "!" is punctuation, not leet
])
def test_deleet(raw, expected):
    assert deleet(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("Y0u are a  STUP1D   iiiiidiooot!!!", "you are a stupid idiot"),
    ("check http://spam.com now", "check now"),
    ("from 192.168.0.1 hello", "from hello"),
    ("f.u.c.k off", "fuck off"),
    ("Don't", "dont"),
    ("Café naïve", "cafe naive"),
    ("", ""),
    (None, ""),
    ("I scored 50 points", "i scored 50 points"),
])
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


def test_osa_distance():
    assert osa_distance("fcuk", "fuck") == 1      # transposition
    assert osa_distance("kitten", "sitting") == 3
    assert osa_distance("abc", "abc") == 0
    assert osa_distance("a", "abcdef", max_dist=1) == 2


@pytest.fixture
def lex():
    return LexiconMatcher(["fuck*", "idiot*", "shit*", "loser*", "kill yourself", "bastard"])


@pytest.mark.parametrize("msg, n_hits", [
    ("you idiot", 1),
    ("you 1d10t!!!", 1),
    ("fcuk this", 1),               # fuzzy transposition
    ("fcuking hell", 1),            # transposed prefix stem
    ("bastrad", 1),                 # fuzzy transposition on long word
    ("go kill yourself", 2),        # phrase
    ("I love my duck", 0),          # substitution on short word NOT matched
    ("nice shirt", 0),              # insertion on short word NOT matched
    ("hello world", 0),
])
def test_lexicon_find(lex, msg, n_hits):
    assert len(lex.find(msg)) == n_hits


def test_mask_preserves_punctuation_and_spacing(lex):
    masked, hits = lex.mask("You IDIOT!! what the sh1t.")
    assert masked == "You I****!! what the s***."
    assert [h.surface for h in hits] == ["IDIOT", "sh1t"]


def test_mask_full(lex):
    masked, _ = lex.mask("idiot", keep_first=False)
    assert masked == "*****"
````


---

## `tests/test_rule_and_errors.py`

````python
import numpy as np

from src.error_analysis import failures, to_markdown
from src.preprocessing import LexiconMatcher
from src.rule_baseline import LexiconRuleModel


def test_rule_baseline_predicts_from_lexicon():
    m = LexiconRuleModel(LexiconMatcher(["idiot*", "kill yourself"]))
    assert m.predict_one("you 1d10t") == (1, 1.0)
    assert m.predict_one("good game") == (0, 0.0)
    assert m.predict(["go kill yourself", "hello"]).tolist() == [1, 0]


def test_failures_picks_confident_errors():
    texts = ["a", "b", "c", "d", "x" * 500]
    y = [0, 0, 1, 1, 1]
    p = [0.9, 0.2, 0.1, 0.8, 0.05]
    r = failures(texts, y, p, threshold=0.5, top=2, max_chars=100)
    assert r["n_fp"] == 1 and r["n_fn"] == 2
    assert r["false_positives"][0]["text"] == "a"
    assert [e["text"] for e in r["false_negatives"]] == ["c"]   # long message deprioritised
    md = to_markdown({"models": {"k": {"model": "M", **r}}})
    assert "False Negatives" in md and "“a”" in md
````


---

## `tests/test_transformer.py`

````python
"""Transformer + quantization tests on a TINY randomly initialised BERT (no download).

Skipped automatically if torch / transformers are not installed.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import BertConfig, BertForSequenceClassification, BertTokenizer  # noqa: E402

from src.quantize import load_quantized, quantize_model, save_quantized  # noqa: E402
from src.transformer_model import (BertClassifier, class_weights,  # noqa: E402
                                   write_moderation_config)

WORDS = ["you", "are", "an", "idiot", "good", "game", "well", "played", "stupid", "hello"]


@pytest.fixture
def tiny(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]) + "\n")
    tok = BertTokenizer(vocab_file=str(vocab), do_lower_case=True)
    cfg = BertConfig(vocab_size=len(WORDS) + 5, hidden_size=32, num_hidden_layers=2,
                     num_attention_heads=2, intermediate_size=64, max_position_embeddings=64,
                     num_labels=2)
    torch.manual_seed(0)
    model = BertForSequenceClassification(cfg).eval()
    return model, tok


def test_class_weights_balanced():
    w = class_weights([0] * 90 + [1] * 10)
    assert w.shape == (2,)
    assert torch.allclose(w, torch.tensor([100 / 180, 100 / 20]))


def test_wrapper_single_and_batch_agree(tiny):
    model, tok = tiny
    clf = BertClassifier(model, tok, threshold=0.5, max_length=32)
    texts = ["you are an idiot", "good game well played", "", "hello"]
    probs = clf.predict_proba(texts, batch_size=3)     # padding + length sorting
    assert probs.shape == (4,) and np.all((probs >= 0) & (probs <= 1))
    for t, p in zip(texts, probs):
        label, p1 = clf.predict_one(t)
        assert abs(p - p1) < 1e-4
        assert label == int(p1 >= 0.5)


def test_fp32_save_load_roundtrip(tiny, tmp_path):
    model, tok = tiny
    d = tmp_path / "fp32"
    model.save_pretrained(d)
    tok.save_pretrained(d)
    write_moderation_config(d, 0.42, 32, True)
    clf = BertClassifier.load(d)
    assert clf.threshold == pytest.approx(0.42)
    ref = BertClassifier(model, tok, 0.42, 32).predict_proba(["you idiot"])
    assert np.allclose(clf.predict_proba(["you idiot"]), ref, atol=1e-5)


def test_quantize_replaces_linear_layers(tiny):
    model, _ = tiny
    q = quantize_model(model)
    qlinear = torch.ao.nn.quantized.dynamic.Linear
    assert isinstance(q.classifier, qlinear)
    assert isinstance(q.bert.encoder.layer[0].attention.self.query, qlinear)
    assert not any(type(m) is torch.nn.Linear for m in q.modules())


def test_quantized_save_load_roundtrip(tiny, tmp_path):
    model, tok = tiny
    q = quantize_model(model)
    d = save_quantized(q, tok, model.config, tmp_path / "int8", 0.3, 32, True)
    assert (d / "quantized_state_dict.pt").exists()
    clf = load_quantized(d)
    assert clf.threshold == pytest.approx(0.3)
    texts = ["you are an idiot", "good game"]
    ref = BertClassifier(q, tok, 0.3, 32).predict_proba(texts)
    assert np.allclose(clf.predict_proba(texts), ref, atol=1e-5)
    label, p = clf.predict_one("stupid")
    assert label in (0, 1) and 0 <= p <= 1
````
