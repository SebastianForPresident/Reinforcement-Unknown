"""Plain-language readouts derived from recorded metrics: "is it learning?" and the next suggested action.

Deliberately conservative: it reports numbers and says "no clear change" unless the difference between the earliest and the latest
window of updates is larger than its own noise. It never turns a rising reward into a claim of success.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .runs import MetricsStore


@dataclass
class LearningSignal:
    state: str                  # none | too_early | no_change | improving | worse
    headline: str
    detail: str
    first: float | None = None
    latest: float | None = None


def _window_stats(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    return mean, math.sqrt(var / n)


def learning_signal(store: MetricsStore, window: int = 10, min_updates: int = 20) -> LearningSignal:
    """Compare goals reached per 1,000 decisions in the first and in the latest ``window`` updates of a run."""
    key = "nav/arrivals_per_1k_decisions"
    if not store.has(key):
        return LearningSignal("none", "No navigation statistics in this run",
                              "This run did not record goals reached; start a new run with the current trainer.")
    values = [float(v) for v in store.column(key) if isinstance(v, (int, float))]
    n = len(values)
    if n < min_updates:
        return LearningSignal("too_early", f"Too early to tell ({n} updates)",
                              f"At least {min_updates} updates are needed before a trend means anything; at the validated speed (64 worlds, about 1,700 decisions/s) that is roughly {min_updates * 4096 / 1700:.0f} seconds of training.")
    w = min(window, n // 3)
    first, last = values[:w], values[-w:]
    m1, s1 = _window_stats(first)
    m2, s2 = _window_stats(last)
    se = math.sqrt(s1 ** 2 + s2 ** 2)
    diff = m2 - m1
    first_text = f"{m1:.2f}"
    last_text = f"{m2:.2f}"
    if se > 0 and abs(diff) > 2 * se:
        if diff > 0:
            return LearningSignal("improving", "Goals reached are increasing",
                                  f"Goals per 1,000 decisions: first {w} updates {first_text}, latest {w} updates {last_text} (difference {diff:+.2f}, "
                                  f"more than twice its uncertainty). This is a training-time signal on the training terrains; confirm it with a held-out evaluation.", m1, m2)
        return LearningSignal("worse", "Goals reached are decreasing",
                              f"Goals per 1,000 decisions: first {w} updates {first_text}, latest {w} updates {last_text} (difference {diff:+.2f}).", m1, m2)
    return LearningSignal("no_change", "No clear change in goals reached yet",
                          f"Goals per 1,000 decisions: first {w} updates {first_text}, latest {w} updates {last_text}; the difference ({diff:+.2f}) is within the "
                          "noise of the measurement. That is normal early in training and not a failure by itself.", m1, m2)


def mean_of_last(store: MetricsStore, key: str, n: int = 10) -> float | None:
    vals = [float(v) for v in store.column(key)[-n:] if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None
