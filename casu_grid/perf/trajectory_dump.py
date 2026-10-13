"""Dump deterministic batched-environment trajectories for pre/post-optimisation comparison (Gate C).

    python casu_grid/perf/trajectory_dump.py --envs 16 --macrosteps 60 --workload random --output traj.json

Runs the full-world batched environment on a seeded action tape and records every macrostep's
per-world state, reward, termination and a SHA-256 of the whole sequence. Compare two dumps with
``casu_grid/tools/compare_trajectories.py`` (Isaac is not needed for the comparison).
The package under test is whichever ``casu_grid`` is first on ``PYTHONPATH`` (so a baseline
revision extracted into a scratch directory can be dumped with the same script).
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--envs", type=int, default=4)
p.add_argument("--macrosteps", type=int, default=60)
p.add_argument("--workload", choices=("random", "run"), default="random")
p.add_argument("--terrain-mode", choices=("shared", "independent"), default="shared")
p.add_argument("--spawn", choices=("fixture", "canonical"), default="fixture")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--physx-threads", type=int, default=None, help="in-process PhysX CPU threads (experiment)")
p.add_argument("--workers", type=int, default=None, help="override num_workers")
p.add_argument("--worker-threads", type=int, default=None, help="PhysX threads per worker")
p.add_argument("--output", type=Path, required=True)
args, _ = p.parse_known_args()

from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
if args.physx_threads is not None:
    import carb.settings
    carb.settings.get_settings().set("/persistent/physics/numThreads", int(args.physx_threads))
sys.path.insert(0, os.environ.get("CASU_SRC_ROOT", str(Path(__file__).resolve().parents[2])))
import numpy as np
import casu_grid
from casu_grid.isaac_navigation_env import IsaacLayer1EnvConfig
from casu_grid.isaac_vector_env import BatchedIsaacLayer1Env

try:
    n = args.envs
    kw = dict(spawn_mode="fixture", fixture_grid_xy=(483, 1010)) if args.spawn == "fixture" else dict(spawn_mode="canonical")
    if args.workers is not None:
        kw["num_workers"] = args.workers
    if args.worker_threads is not None:
        kw["worker_physx_threads"] = args.worker_threads
    env = BatchedIsaacLayer1Env(IsaacLayer1EnvConfig(num_envs=n, **kw), terrain_mode=args.terrain_mode)
    env.reset()
    rng = np.random.default_rng(args.seed)
    frames, h = [], hashlib.sha256()
    t = time.perf_counter()
    for step in range(args.macrosteps):
        if args.workload == "random":
            action = {"move_x": rng.integers(0, 3, n), "jump": rng.integers(0, 2, n), "crouch": rng.integers(0, 2, n)}
        else:
            action = {"move_x": np.full(n, 2), "jump": np.full(n, 1 if step % 4 == 3 else 0), "crouch": np.zeros(n, int)}
        _, reward, terminated, truncated, info = env.step(action)
        # per-tick body records are not exposed by the batched env, so frame = end of macrostep
        summaries = env.slot_summaries() if hasattr(env, "slot_summaries") else [
            {"crouch": float(s.model.s.crouch_amount), "grounded": bool(s.model.s.grounded),
             "fluid_tick": int(s.grid.tick), "fluid_digest": hashlib.sha256(s.grid.fluids.tobytes()).hexdigest()[:16]}
            for s in env.slots]
        frame = {"step": step, "players": info["player"], "reward": [float(r) for r in reward],
                 "terminated": [bool(x) for x in terminated], "truncated": [bool(x) for x in truncated],
                 "reasons": [r for r in info["termination_reason"]],
                 "crouch": [x["crouch"] for x in summaries],
                 "grounded": [x["grounded"] for x in summaries],
                 "fluid_ticks": [x["fluid_tick"] for x in summaries],
                 "fluid_digest": [x["fluid_digest"] for x in summaries]}
        frames.append(frame)
        h.update(json.dumps(frame, sort_keys=True).encode())
        done = np.logical_or(terminated, truncated)
        if done.any():
            env.reset_done(done)
    out = {"schema": "casu.trajectory-dump.v1", "envs": n, "macrosteps": args.macrosteps, "workload": args.workload,
           "terrain_mode": args.terrain_mode, "spawn": args.spawn, "seed": args.seed,
           "backend": casu_grid.backend.describe() if hasattr(casu_grid, "backend") else "pre-backend-baseline",
           "casu_grid_path": casu_grid.__file__, "wall_seconds": time.perf_counter() - t,
           "sha256": h.hexdigest(), "frames": frames}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out) + "\n")
    print("TRAJECTORY_DUMP", args.output, out["sha256"][:16], flush=True)
    env.close()
finally:
    app.close()
