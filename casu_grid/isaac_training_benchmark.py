"""End-to-end PPO training throughput (launch ``workbench.train`` and read its metrics).

    python casu_grid/isaac_training_benchmark.py --envs 16 --workers 4 --updates 6 --output out.json
    python casu_grid/isaac_training_benchmark.py --config casu_grid/perf/synthetic_large_model.py ...

Separates simulation-only, rollout (simulation + policy inference) and end-to-end (rollout + PPO
update) decisions per second from the trainer's own per-update timers. The first update (JIT
compilation, cache warm-up) is dropped. Uses the engineering-only smoke configuration unless
``--config`` says otherwise; results say nothing about learning quality.
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(os.environ.get("CASU_SRC_ROOT", Path(__file__).resolve().parents[1]))   # CASU_SRC_ROOT: benchmark another checkout
p = argparse.ArgumentParser()
p.add_argument("--envs", type=int, default=16)
p.add_argument("--workers", type=int, default=1)
p.add_argument("--updates", type=int, default=6)
p.add_argument("--episode-steps", type=int, default=None)
p.add_argument("--terrain-mode", choices=("shared", "independent"), default="shared")
p.add_argument("--config", type=Path, default=ROOT / "casu_grid/workbench/engineering_smoke.py")
p.add_argument("--backend", default=None, help="CASU_BACKEND for the run (auto|numba|reference-python)")
p.add_argument("--label", default="")
p.add_argument("--keep-run", action="store_true")
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()

run_dir = Path(tempfile.mkdtemp(prefix="casu_trainbench_")) / "run"
env = dict(os.environ)
env["CASU_NUM_WORKERS"] = str(args.workers)
if args.backend:
    env["CASU_BACKEND"] = args.backend
cmd = [sys.executable, "-m", "casu_grid.workbench.train", "--config", str(args.config), "--updates", str(args.updates),
       "--num-envs", str(args.envs), "--run-dir", str(run_dir), "--terrain-mode", args.terrain_mode]
if args.episode_steps:
    cmd += ["--episode-steps", str(args.episode_steps)]
started = time.perf_counter()
proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
wall = time.perf_counter() - started
if proc.returncode != 0:
    sys.exit(f"training run failed ({proc.returncode}):\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines() if l.strip()]
steady = rows[1:] if len(rows) > 1 else rows


def total(key):
    return sum(r[key] for r in steady)


steps = rows[-1]["environment_steps"] - rows[0]["environment_steps"] if len(rows) > 1 else rows[0]["environment_steps"]
sim, rollout = total("physics_step_seconds"), total("collection_seconds")
optimize = total("optimization_seconds")
result = {
    "schema": "casu.isaac-training-benchmark.v1", "label": args.label, "envs": args.envs, "workers": args.workers,
    "terrain_mode": args.terrain_mode, "config": str(args.config.relative_to(ROOT)) if args.config.is_relative_to(ROOT) else str(args.config),
    "backend_env": args.backend or os.environ.get("CASU_BACKEND", "auto"), "updates": args.updates,
    "measured_updates": len(steady), "measured_decisions": steps,
    "simulation_only_decisions_per_second": steps / sim,
    "rollout_decisions_per_second": steps / rollout,
    "end_to_end_decisions_per_second": steps / (rollout + optimize),
    "simulation_only_ticks_per_second": steps * 10 / sim,
    "end_to_end_ticks_per_second": steps * 10 / (rollout + optimize),
    "seconds": {"simulation": sim, "policy_inference_and_rollout_overhead": rollout - sim,
                "ppo_optimization": optimize, "rollout_total": rollout},
    "optimizer_overhead_fraction": optimize / (rollout + optimize),
    "policy_inference_fraction_of_rollout": (rollout - sim) / rollout,
    "process_wall_seconds_including_startup": wall,
    "finite_losses": all(all(v == v and abs(v) != float("inf") for v in (r["policy_loss"], r["value_loss"], r["grad_norm"])) for r in rows),
    "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
if not args.keep_run:
    shutil.rmtree(run_dir.parent, ignore_errors=True)
print(f"envs={args.envs} workers={args.workers}: sim-only {result['simulation_only_decisions_per_second']:.1f}, "
      f"rollout {result['rollout_decisions_per_second']:.1f}, end-to-end {result['end_to_end_decisions_per_second']:.1f} "
      f"decisions/s; optimizer {100 * result['optimizer_overhead_fraction']:.1f}% of rollout+update", flush=True)
