"""Helpers for building synthetic run folders (no simulator) for the Studio's tests."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path


_FIXTURE = Path(__file__).with_name("fixtures") / "trainer_metrics_row.json"


def real_row_keys() -> set[str]:
    """The keys a real trainer run writes (captured from an actual run), so tests cannot drift from what the trainer records."""
    return set(json.loads(_FIXTURE.read_text()))


def metrics_row(update: int, *, arrivals: float | None = None, episodes: int = 0, seed: int = 0, progress: float | None = None) -> dict:
    rnd = random.Random(seed * 1000 + update)
    arrivals = 0.5 + 0.02 * update + rnd.uniform(-0.2, 0.2) if arrivals is None else arrivals
    row = {"update": update, "environment_steps": update * 4096, "environment_count": 64, "collection_seconds": 1.7, "optimization_seconds": 0.65,
           "rollout_seconds": 2.4, "policy_loss": -0.008, "value_loss": 0.1 / update, "entropy": 8.78 - 0.002 * update, "approx_kl": 0.004, "grad_norm": 1.0,
           "clip_fraction": 0.05, "explained_variance": 0.3, "learning_rate": 3e-4, "policy_decisions_per_second": 2400.0, "physics_ticks_per_second": 25000.0,
           "completed_episodes": episodes, "mean_completed_return": 10.5 if episodes else None,
           "nav/arrivals_per_1k_decisions": arrivals, "nav/arrival_rate": 1.0 if episodes else None, "nav/arrivals": episodes, "nav/timeouts": 0,
           "nav/completed_episodes": episodes, "nav/active_progress_mean": progress if progress is not None else min(1.0, 0.2 + 0.005 * update), "nav/active_distance_mean": 30.0, "nav/active_best_distance_mean": 18.0,
           "nav/active_initial_distance_mean": 22.0, "nav/blocks_destroyed_per_1k_decisions": 40.0, "action/attack/choice_1_fraction": 0.5,
           "reward_term/distance_improvement": 0.01, "reward_term/arrival_bonus": 0.001}
    base = json.loads(_FIXTURE.read_text())
    base.update(row)
    return base


def make_run(root: Path, name: str, *, updates: int = 30, status: str = "completed", fingerprint: str | None = "fp-test", seed: int = 7, worlds: int = 64,
             workers: int = 8, config_path: str | None = None, config_sha: str = "abc", rows: list[dict] | None = None, pid: int | None = None,
             termination: str | None = "xy1-termination-v2-layer-exit") -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({
        "schema": "casu.training-run.v1", "run_id": name, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seed": seed, "git_revision": "0123456789abcdef",
        "config_path": config_path or "/x/xy1_point_navigation_mini.py", "config_sha256": config_sha,
        "experiment": {"name": "XY1 Point Navigation · Mini World 128 × 128"}, "environment": {"num_envs": worlds, **({"termination_contract": termination} if termination else {})}, "run_kind": "research_training"}))
    (run / "environment.json").write_text(json.dumps({"performance": {"num_workers": workers, "physx_threads_per_worker": 2}}))
    rows = rows if rows is not None else [metrics_row(i, seed=seed, episodes=1 if i % 5 == 0 else 0) for i in range(1, updates + 1)]
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if rows:
        (run / "latest.pt").write_bytes(b"x" * 1024)
        (run / "latest.pt.json").write_text(json.dumps({"update_count": len(rows), "total_steps": len(rows) * 4096, "contract_fingerprint": fingerprint,
                                                         "world_partition": [worlds // workers] * workers, "saved_utc": "2026-10-14T00:00:00Z",
                                                         **({"environment_contract": {"termination": termination}} if termination else {})}))
    if status == "completed":
        (run / "COMPLETE").write_text("done\n")
    elif status == "failed":
        (run / "failure.json").write_text(json.dumps({"status": "failed", "error_type": "WorkerFailure", "message": "worker 0 closed its pipe", "log_tail": "WorkerFailure: worker 0 closed its pipe"}))
    elif status == "interrupted":
        (run / "INTERRUPTED").write_text("stop\n")
    elif status == "running":
        (run / "RUNNING").write_text(f"{pid or os.getpid()}\n")
    return run
