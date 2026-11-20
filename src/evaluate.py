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
