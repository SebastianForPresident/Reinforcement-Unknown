"""Terrain-mutation and lifecycle latency in independent-terrain mode (one process, Kit started here).

    python casu_grid/perf/mutation_latency.py --envs 4 --output casu_grid/perf/results/mutation_latency.json
"""
import argparse, json, statistics, sys, time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--envs", type=int, default=4)
p.add_argument("--repeats", type=int, default=6)
p.add_argument("--output", type=Path, required=True)
args, _ = p.parse_known_args()
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from casu_grid.isaac_navigation_env import IsaacLayer1EnvConfig
from casu_grid.isaac_vector_env import BatchedIsaacLayer1Env

try:
    n = args.envs
    cfg = IsaacLayer1EnvConfig(num_envs=n, spawn_mode="fixture", fixture_grid_xy=(7, 372), num_workers=1)
    t = time.perf_counter()
    env = BatchedIsaacLayer1Env(cfg, terrain_mode="independent")
    build = time.perf_counter() - t
    env.reset()
    snap = env.snapshot()
    edit, restore, reset_one = [], [], []
    for _ in range(args.repeats):
        t = time.perf_counter(); env.set_block(0, 7, 372, 0); edit.append(time.perf_counter() - t)
        env.step({"move_x": np.ones(n, int), "jump": np.zeros(n, int), "crouch": np.zeros(n, int)})
        t = time.perf_counter(); env.restore(snap); restore.append(time.perf_counter() - t)
        t = time.perf_counter(); env.reset_done(np.asarray([True] + [False] * (n - 1))); reset_one.append(time.perf_counter() - t)
    out = {"schema": "casu.mutation-latency.v1", "envs": n, "terrain_mode": "independent", "build_seconds": build,
           "set_block_seconds": {"median": statistics.median(edit), "samples": edit},
           "restore_after_edit_seconds": {"median": statistics.median(restore), "samples": restore},
           "reset_one_world_seconds": {"median": statistics.median(reset_one), "samples": reset_one}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print("MUTATION_LATENCY", json.dumps({k: (v["median"] if isinstance(v, dict) else v) for k, v in out.items() if k not in ("schema",)}), flush=True)
    env.close()
finally:
    app.close()
