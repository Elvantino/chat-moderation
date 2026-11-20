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
