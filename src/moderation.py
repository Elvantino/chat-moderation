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
