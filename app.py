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
