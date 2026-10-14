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


def outside_world_share(store: MetricsStore, n: int = 3) -> float | None:
    vals = [float(v) for v in store.column("nav/active_outside_world_fraction")[-n:] if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None


def learning_signal(store: MetricsStore, window: int = 20, min_updates: int = 20) -> LearningSignal:
    signal = _learning_signal(store, window, min_updates)
    lost = outside_world_share(store)
    if lost is not None and lost >= 0.5 and signal.state in ("improving", "no_change", "worse"):
        signal = LearningSignal("drained", f"Training data is drained: {100 * lost:.0f} % of the agents have left the world",
                                "Most episodes belong to agents that fell through the floor and cannot reach any goal, so training-curve trends are not interpretable any more. "
                                + signal.detail.split(" This is a training-time")[0], signal.first, signal.latest)
    if lost is not None and lost >= 0.10:
        signal.detail += (f" Caution: {100 * lost:.0f} % of the agents have left the world (fell through the floor) and cannot contribute any more, "
                          "so the signal gets weaker the longer the run goes on.")
    return signal


def _trend(store: MetricsStore, key: str, window: int):
    """(first-window mean, latest-window mean, uncertainty of the difference, window, number of samples) for one recorded metric."""
    values = [float(v) for v in store.column(key) if isinstance(v, (int, float))]
    n = len(values)
    w = min(window, n // 3)
    if w < 3:
        return None
    m1, s1 = _window_stats(values[:w])
    m2, s2 = _window_stats(values[-w:])
    return m1, m2, math.sqrt(s1 ** 2 + s2 ** 2), w, n


def _learning_signal(store: MetricsStore, window: int = 20, min_updates: int = 20) -> LearningSignal:
    """Compare the first and the latest window of updates for two indicators and say only what the numbers support.

    Goals reached per 1,000 decisions is the direct measure but sparse (about one arrival per update early on, so single updates are mostly noise); the progress
    toward the goal of episodes in progress is averaged over every running episode and moves earlier. Both are training-time statistics on the training terrains.
    """
    key = "nav/arrivals_per_1k_decisions"
    if store.rows == 0:
        return LearningSignal("too_early", "Waiting for the first update",
                              "Nothing has been measured yet. The first update comes after Isaac Sim has started and the worlds are built.")
    if not store.has(key):
        return LearningSignal("none", "No navigation statistics in this run",
                              "This run did not record goals reached; start a new run with the current trainer.")
    n = store.rows
    if n < min_updates:
        return LearningSignal("too_early", f"Too early to tell ({n} updates)",
                              f"At least {min_updates} updates are needed before a trend means anything; at the validated speed (64 worlds, about 1,700 decisions/s) "
                              f"that is roughly {min_updates * 4096 / 1700:.0f} seconds of training.")
    parts, states = [], []
    for label, k, fmt in (("Goals per 1,000 decisions", key, "{:.2f}"), ("Progress toward the goal", "nav/active_progress_mean", "{:.0%}")):
        t = _trend(store, k, window)
        if t is None:
            continue
        m1, m2, se, w, _n = t
        diff = m2 - m1
        state = "improving" if (se > 0 and diff > 2 * se) else ("worse" if (se > 0 and diff < -2 * se) else "no_change")
        states.append(state)
        parts.append(f"{label}: first {w} updates {fmt.format(m1)}, latest {w} updates {fmt.format(m2)} "
                     f"({'clearly higher' if state == 'improving' else 'clearly lower' if state == 'worse' else 'within the noise'}).")
    if "improving" in states:
        state, headline = "improving", ("Goals reached are increasing" if states[0] == "improving" else "Progress toward the goal is increasing")
        tail = " This is a training-time signal on the training terrains; confirm it with a held-out evaluation."
    elif "worse" in states:
        state, headline, tail = "worse", "Goals reached or progress are decreasing", ""
    else:
        state, headline = "no_change", "No clear change yet"
        tail = " Within the noise is normal early in training and not a failure by itself; a held-out evaluation is more sensitive than these training curves."
    return LearningSignal(state, headline, " ".join(parts) + tail)


def mean_of_last(store: MetricsStore, key: str, n: int = 10) -> float | None:
    vals = [float(v) for v in store.column(key)[-n:] if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None
