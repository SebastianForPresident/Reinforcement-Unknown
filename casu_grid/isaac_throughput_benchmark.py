"""Batched throughput benchmark with a per-phase time breakdown (one process per batch size).

Reports aggregate physics ticks/s, ten-tick policy decisions/s, per-environment throughput,
reset / snapshot / restore latency, build time and peak memory. It measures simulation only;
policy inference and PPO update overhead are reported by the training workbench metrics.
"""
from __future__ import annotations
import argparse, json, resource, subprocess, sys, time, traceback
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, nargs="+", default=[1, 2, 4, 8, 16])
parser.add_argument("--macrosteps", type=int, default=30)
parser.add_argument("--terrain-mode", choices=("shared", "independent"), default="shared")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--single", type=int, default=0, help=argparse.SUPPRESS)
args, rest = parser.parse_known_args()

if not args.single:
    # one Kit SimulationContext per process
    rows = []
    for n in args.num_envs:
        part = args.output.with_name(f"{args.output.stem}_n{n}.json")
        subprocess.run([sys.executable, __file__, "--single", str(n), "--macrosteps", str(args.macrosteps),
                        "--terrain-mode", args.terrain_mode, "--output", str(part), "--headless"], check=True)
        rows.append(json.loads(part.read_text()))
        part.unlink()
    args.output.write_text(json.dumps({"schema": "casu.isaac-throughput.v1", "terrain_mode": args.terrain_mode,
                                       "rows": rows}, indent=2) + "\n")
    print(json.dumps(rows, indent=1))
    raise SystemExit(0)

from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from casu_grid.isaac_navigation_env import IsaacLayer1EnvConfig
from casu_grid.isaac_vector_env import BatchedIsaacLayer1Env

try:
    n = args.single
    cfg = IsaacLayer1EnvConfig(num_envs=n, spawn_mode="fixture", fixture_grid_xy=(483, 1010))
    started = time.perf_counter()
    env = BatchedIsaacLayer1Env(cfg, terrain_mode=args.terrain_mode)
    construct = time.perf_counter() - started
    t = time.perf_counter(); env.reset(); reset_all = time.perf_counter() - t
    rng = np.random.default_rng(0)
    def random_action():
        return {"move_x": rng.integers(0, 3, n), "jump": rng.integers(0, 2, n), "crouch": rng.integers(0, 2, n)}
    for _ in range(3):
        env.step(random_action())
    env.timers.clear()
    t = time.perf_counter()
    for _ in range(args.macrosteps):
        env.step(random_action())
    elapsed = time.perf_counter() - t
    breakdown = {k: v for k, v in env.timers.items()}
    t = time.perf_counter(); snap = env.snapshot(); snapshot_s = time.perf_counter() - t
    t = time.perf_counter(); env.restore(snap); restore_s = time.perf_counter() - t
    t = time.perf_counter(); env.reset_done(np.asarray([True] + [False] * (n - 1))); reset_one = time.perf_counter() - t
    ticks = args.macrosteps * cfg.action_decimation * n
    row = {"num_envs": n, "terrain_mode": args.terrain_mode, "macrosteps": args.macrosteps,
           "construct_seconds": construct, "geometry_build_seconds": env.geometry_build_seconds,
           "elapsed_seconds": elapsed, "aggregate_physics_ticks_per_second": ticks / elapsed,
           "aggregate_policy_decisions_per_second": args.macrosteps * n / elapsed,
           "per_env_physics_ticks_per_second": ticks / elapsed / n,
           "phase_seconds": breakdown, "phase_fraction": {k: v / elapsed for k, v in breakdown.items()},
           "reset_all_seconds": reset_all, "reset_one_world_seconds": reset_one,
           "snapshot_seconds": snapshot_s, "restore_seconds": restore_s,
           "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)}
    Path(args.output).write_text(json.dumps(row, indent=2) + "\n")
    env.close()
except BaseException:
    traceback.print_exc()
    raise
finally:
    app.close()
