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
