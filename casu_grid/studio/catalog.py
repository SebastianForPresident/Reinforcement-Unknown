"""Which charts exist, which recorded metric feeds each one, and how series are built for one or several runs.

Only metrics that the trainer (or the Studio's monitor) actually records appear here. A chart whose metric is absent from a run shows an
explicit explanation (``ChartSpec.empty``) instead of a blank plot, and a metric that has no sample in an update is a gap, never a zero.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from . import theme
from .charts import ChartSpec, Series
from .runs import MetricsStore, human_count

CATEGORIES = [("navigation", "Navigation & task"), ("ppo", "PPO health"), ("system", "Simulator & system")]

X_AXES = {
    "update": ("Update", True),
    "decisions": ("Decisions (all worlds)", False),
    "time": ("Training time", False),
}

NO_NAV = ("This run did not record navigation statistics (it was made before the trainer reported them). Start a new run to get this chart.")


@dataclass
class SeriesDef:
    label: str
    key: str                       # a column of metrics.jsonl, or "derived:<name>"
    dashed: bool = False
    smoothable: bool = True
    source: str = "metrics"        # metrics | system


@dataclass
class ChartDef:
    spec: ChartSpec
    category: str
    series: list[SeriesDef]
    detail: str = ""               # longer interpretation shown in the focus view


def _spec(key, title, ylabel, help_key, caption, empty=NO_NAV, **kw) -> ChartSpec:
    return ChartSpec(key=key, title=title, ylabel=ylabel, help_key=help_key, caption=caption, empty=empty, **kw)


CHARTS: list[ChartDef] = [
    # --------------------------------------------------------------------------------------------------- navigation
    ChartDef(_spec("nav_success", "Goals reached per 1,000 decisions", "goals / 1,000 decisions", "arrivals_per_1k",
                   "The most direct sign of learning during training. A random policy reaches a few goals per thousand decisions.",
                   y_min=0, include_zero=True),
             "navigation", [SeriesDef("goals reached", "nav/arrivals_per_1k_decisions")]),
    ChartDef(_spec("nav_outside", "Agents that left the world", "share of running episodes", "out_of_world",
                   "Agents that dug through the floor and fell out of the 128 × 128 world. Nothing ends such an episode, so they are lost to training; rising values degrade every other chart.",
                   y_min=0, y_max=1, include_zero=True, percent=True),
             "navigation", [SeriesDef("outside the world", "nav/active_outside_world_fraction")]),
    ChartDef(_spec("nav_progress", "Progress toward the goal", "share of start distance closed", "progress",
                   "For episodes still running: how much of the starting distance the agent has closed at its best. 0 % = none, 100 % = arrived.",
                   y_min=0, y_max=1, include_zero=True, percent=True),
             "navigation", [SeriesDef("progress (episodes in progress)", "nav/active_progress_mean"),
                            SeriesDef("progress (finished episodes)", "nav/completed_progress_mean", dashed=True)]),
    ChartDef(_spec("nav_distance", "Distance to goal", "tiles", "distance",
                   "Gap between 'current' and 'closest so far' shows agents that get near the goal and then drift away.", include_zero=True),
             "navigation", [SeriesDef("current distance (agents inside the world)", "nav/active_distance_mean"), SeriesDef("closest so far", "nav/active_best_distance_mean"),
                            SeriesDef("distance at episode start", "nav/active_initial_distance_mean", dashed=True)]),
    ChartDef(_spec("nav_rate", "Arrival rate among finished episodes", "share of finished episodes", "arrival_rate",
                   "Gaps mean no episode finished in that update. With 100,000-decision episodes this stays near 100 % and says little; "
                   "use goals per 1,000 decisions.", empty="No episode has finished yet in this run, so this rate does not exist (it is not zero).",
                   y_min=0, y_max=1.05, include_zero=True, percent=True),
             "navigation", [SeriesDef("arrivals ÷ finished episodes", "nav/arrival_rate")]),
    ChartDef(_spec("nav_time", "Decisions needed to reach the goal", "decisions", "decisions_to_arrival",
                   "Only episodes that reached their goal are counted; lower is better (compare with the starting distance).",
                   empty="No goal has been reached yet in this run.", include_zero=True),
             "navigation", [SeriesDef("mean", "nav/decisions_to_arrival_mean"), SeriesDef("median", "nav/decisions_to_arrival_median", dashed=True)]),
    ChartDef(_spec("nav_episodes", "Episodes finished per update", "episodes", "truncation",
                   "Episodes end by reaching the goal (arrival) or by running out of their decision budget (timeout).", y_min=0, include_zero=True),
             "navigation", [SeriesDef("finished", "nav/completed_episodes"), SeriesDef("arrivals", "nav/arrivals"),
                            SeriesDef("timeouts", "nav/timeouts", dashed=True), SeriesDef("left the world (variant only)", "nav/out_of_world_resets", dashed=True)]),
    ChartDef(_spec("nav_dig", "Excavation activity", "blocks destroyed / 1,000 decisions", "excavation",
                   "Blocks actually destroyed by the agent, per 1,000 decisions in all worlds.", y_min=0, include_zero=True),
             "navigation", [SeriesDef("blocks destroyed", "nav/blocks_destroyed_per_1k_decisions")]),
    ChartDef(_spec("nav_attack", "Attack frequency", "share of decisions", "attack_rate",
                   "Share of decisions in which the policy chose to attack. A random policy is near 50 %.",
                   empty="No attack statistics in this run.", y_min=0, y_max=1, include_zero=True, percent=True,
                   references=[("random policy", 0.5)]),
             "navigation", [SeriesDef("attack chosen", "action/attack/choice_1_fraction")]),
    ChartDef(_spec("nav_reward", "Reward components", "mean reward per decision", "reward",
                   "Distance improvement is paid whenever the agent gets closer to the goal than ever before in the episode; arrival bonus once per arrival.",
                   empty="No reward-term statistics in this run."),
             "navigation", [SeriesDef("distance improvement (×0.1)", "reward_term/distance_improvement"),
                            SeriesDef("arrival bonus (10)", "reward_term/arrival_bonus", dashed=True)]),
    ChartDef(_spec("nav_return", "Return of finished episodes", "mean return", "return",
                   "A rising return alone is not proof of reliable navigation; read it together with goals reached and progress.",
                   empty="No episode has finished yet in this run."),
             "navigation", [SeriesDef("mean return", "mean_completed_return")]),
    # --------------------------------------------------------------------------------------------------- PPO health
    ChartDef(_spec("ppo_entropy", "Policy entropy", "nats (sum over 7 heads)", "entropy",
                   "Falls as the policy becomes decisive. A collapse within a few updates would mean exploration stopped.", empty="No entropy recorded."),
             "ppo", [SeriesDef("entropy", "entropy")]),
    ChartDef(_spec("ppo_kl", "Approximate KL divergence", "KL", "kl",
                   "How far one update moved the policy. Healthy: 0.001–0.02; updates stop early above the 0.03 target.",
                   y_min=0, include_zero=True, references=[("target 0.03", 0.03)], empty="No KL recorded."),
             "ppo", [SeriesDef("approx KL", "approx_kl")]),
    ChartDef(_spec("ppo_clip", "Clip fraction", "share of samples", "clip_fraction",
                   "Share of samples hitting the PPO clip limit. Near 0: tiny steps; above ~0.3: aggressive steps.",
                   empty="Clip fraction is recorded by runs made with the current trainer; this run predates it.", y_min=0, include_zero=True, percent=True),
             "ppo", [SeriesDef("clip fraction", "clip_fraction")]),
    ChartDef(_spec("ppo_value", "Value loss", "loss", "value_loss", "How wrong the critic's return predictions are. Spikes appear when rewards become larger, e.g. at the first arrivals.",
                   y_min=0, include_zero=True),
             "ppo", [SeriesDef("value loss", "value_loss")]),
    ChartDef(_spec("ppo_ev", "Explained variance of returns", "explained variance", "explained_variance",
                   "How much of the variation in returns the critic explains: 1 = perfectly; 0 or below = no better than the average.",
                   empty="Explained variance is recorded by runs made with the current trainer; this run predates it.", references=[("no skill", 0.0)]),
             "ppo", [SeriesDef("explained variance", "explained_variance")]),
    ChartDef(_spec("ppo_policy", "Policy loss", "loss", "policy_loss", "Hovers near zero because advantages are normalised; watch for non-finite values, not for a trend."),
             "ppo", [SeriesDef("policy loss", "policy_loss")]),
    ChartDef(_spec("ppo_grad", "Gradient norm (before clipping)", "norm", "grad_norm", "Clipped to 0.5. Persistent values far above that, or sudden growth, indicate instability.",
                   y_min=0, include_zero=True, references=[("clip limit 0.5", 0.5)]),
             "ppo", [SeriesDef("gradient norm", "grad_norm")]),
    # --------------------------------------------------------------------------------------------------- system
    ChartDef(_spec("sys_throughput", "Decisions per second", "decisions / s", "throughput",
                   "End-to-end includes the PPO optimisation; rollout counts only data collection. Reference for the validated 64 × 8 layout: about 1,700 end-to-end.",
                   y_min=0, include_zero=True, references=[("reference 1,700", 1700.0)]),
             "system", [SeriesDef("end-to-end", "derived:e2e_decisions_per_s"), SeriesDef("rollout only", "policy_decisions_per_second", dashed=True)]),
    ChartDef(_spec("sys_time", "Time per update", "seconds", "update",
                   "Collection is the CPU physics work; optimisation is the GPU PPO update.", y_min=0, include_zero=True),
             "system", [SeriesDef("collection", "collection_seconds"), SeriesDef("optimisation", "optimization_seconds")]),
    ChartDef(_spec("sys_physics", "Physics ticks per second", "ticks / s", "decision", "Simulated physics ticks per second of collection (10 ticks per decision)."),
             "system", [SeriesDef("physics ticks / s", "physics_ticks_per_second")]),
    ChartDef(_spec("sys_memory", "Memory of the training processes", "GiB", "worlds",
                   "Unique memory (USS) and proportional memory (PSS) of the whole process tree; flat or slowly rising is healthy. Sampled by the Studio while it supervised the run.",
                   empty="Memory is sampled by the Studio while it watches a run; there are no samples for this run.", y_min=0, include_zero=True),
             "system", [SeriesDef("unique (USS)", "uss_gib", source="system"), SeriesDef("proportional (PSS)", "pss_gib", dashed=True, source="system")]),
    ChartDef(_spec("sys_gpu", "GPU memory", "MiB", "worlds", "GPU memory in use on the machine while the run was supervised.",
                   empty="GPU memory is sampled by the Studio while it watches a run; there are no samples for this run.", y_min=0, include_zero=True),
             "system", [SeriesDef("GPU memory", "gpu_mem_mib", source="system")]),
    ChartDef(_spec("sys_health", "Simulation processes alive", "processes", "worlds",
                   "Number of simulator worker processes alive at each sample; a drop means a worker died.",
                   empty="Process health is sampled by the Studio while it watches a run; there are no samples for this run.", y_min=0, include_zero=True),
             "system", [SeriesDef("workers alive", "workers_alive", source="system", smoothable=False)]),
]

CHART_BY_KEY = {c.spec.key: c for c in CHARTS}


class RunData:
    """What the charts need from one run: its metrics store, system samples and derived columns (kept incrementally)."""

    def __init__(self, label: str, metrics: MetricsStore, system: MetricsStore | None = None, color: str = theme.SERIES_COLORS[0]):
        self.label, self.metrics, self.system, self.color = label, metrics, system, color
        self._time: list[float] = []

    def refresh(self) -> int:
        added = self.metrics.refresh()
        if self.system is not None:
            self.system.refresh()
        col = self.metrics.column("rollout_seconds")
        total = self._time[-1] if self._time else 0.0
        for v in col[len(self._time):]:
            total += float(v) if isinstance(v, (int, float)) else 0.0
            self._time.append(total)
        return added

    def derived(self, name: str) -> list[float | None]:
        m = self.metrics
        if name == "e2e_decisions_per_s":
            steps, coll, opt = m.column("environment_steps"), m.column("collection_seconds"), m.column("optimization_seconds")
            out, prev = [], 0
            for s, c, o in zip(steps, coll, opt):
                if s is None or c is None or o is None or (c + o) <= 0:
                    out.append(None)
                else:
                    out.append((s - prev) / (c + o))
                    prev = s
            return out
        raise KeyError(name)

    def x_values(self, axis: str, source: str = "metrics") -> list[float]:
        m = self.metrics if source == "metrics" else self.system
        if source == "system":
            key = {"update": "update", "decisions": "decisions", "time": "train_s"}[axis]
            return [float(v) if v is not None else float("nan") for v in m.column(key)]
        if axis == "update":
            return [float(v) if v is not None else float(i + 1) for i, v in enumerate(m.column("update"))]
        if axis == "decisions":
            return [float(v) if v is not None else float("nan") for v in m.column("environment_steps")]
        return list(self._time)

    def info_fn(self, source: str = "metrics") -> Callable[[int], str]:
        m = self.metrics if source == "metrics" else self.system

        def info(i: int) -> str:
            upd = m.column("update")[i] if i < m.rows else None
            dec = m.column("environment_steps" if source == "metrics" else "decisions")[i] if i < m.rows else None
            parts = []
            if upd is not None:
                parts.append(f"update {int(upd):,}")
            if dec is not None:
                parts.append(f"{int(dec):,} decisions")
            return " · ".join(parts)
        return info


DASHES = [False, True]


def build_series(chart: ChartDef, runs: list[RunData], axis: str) -> list[Series]:
    series: list[Series] = []
    multi = len(runs) > 1
    for r_index, run in enumerate(runs):
        for s_index, sd in enumerate(chart.series):
            store = run.metrics if sd.source == "metrics" else run.system
            if store is None:
                continue
            if sd.key.startswith("derived:"):
                ys = run.derived(sd.key.split(":", 1)[1])
            else:
                ys = store.column(sd.key)
                ys = [v if isinstance(v, (int, float)) and not isinstance(v, bool) else None for v in ys]
            xs = run.x_values(axis, sd.source)
            n = min(len(xs), len(ys))
            xs, ys = xs[:n], ys[:n]
            # drop samples without a valid x position, keeping order
            pairs = [(x, y) for x, y in zip(xs, ys) if x == x]
            if not pairs:
                continue
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            color = run.color if multi else theme.SERIES_COLORS[s_index % len(theme.SERIES_COLORS)]
            series.append(Series(key=f"{run.label}:{sd.key}", label=sd.label, xs=xs, ys=ys, color=color, run=run.label if multi else "",
                                 dashed=sd.dashed, smoothable=sd.smoothable, info=run.info_fn(sd.source)))
    return series


def series_stats(series: list[Series], x_range: tuple[float, float] | None) -> list[dict]:
    """Per-series numbers for the detail panel: samples, gaps, min / mean / last in the visible x range."""
    out = []
    for s in series:
        pts = [(x, y) for x, y in zip(s.xs, s.ys) if (x_range is None or x_range[0] <= x <= x_range[1])]
        vals = [y for _, y in pts if y is not None and y == y]
        out.append({"label": (s.run + " · " if s.run else "") + s.label, "samples": len(vals), "missing": len(pts) - len(vals),
                    "min": min(vals) if vals else None, "max": max(vals) if vals else None,
                    "mean": sum(vals) / len(vals) if vals else None, "last": vals[-1] if vals else None, "color": s.color})
    return out
