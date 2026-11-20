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
