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
