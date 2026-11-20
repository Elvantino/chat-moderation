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
