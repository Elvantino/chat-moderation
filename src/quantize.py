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
