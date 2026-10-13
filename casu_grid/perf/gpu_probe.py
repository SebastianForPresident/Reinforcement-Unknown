"""GPU-PhysX feasibility probe (opt-in research tool, never used by production training).

    python casu_grid/perf/gpu_probe.py --envs 4 --steps 20 --cuda

Builds the batched environment with the PhysX GPU pipeline (``--cuda``) or the CPU reference, runs a seeded
tape and reports whether it builds/steps, the physics settings in effect, timings and any errors.
"""
import argparse, json, sys, time, traceback
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--envs", type=int, default=4)
p.add_argument("--steps", type=int, default=20)
p.add_argument("--workload", choices=("random", "run", "crouch"), default="run")
p.add_argument("--cuda", action="store_true")
p.add_argument("--no-ccd", action="store_true")
p.add_argument("--output", type=Path, default=None)
args, _ = p.parse_known_args()
sys.stdout.reconfigure(line_buffering=True)
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from casu_grid.isaac_navigation_env import IsaacLayer1EnvConfig
from casu_grid.isaac_vector_env import BatchedIsaacLayer1Env

result = {"cuda": args.cuda, "envs": args.envs}
try:
    n = args.envs
    cfg = IsaacLayer1EnvConfig(num_envs=n, spawn_mode="fixture", fixture_grid_xy=(483, 1010), use_cuda=args.cuda, physx_ccd=not args.no_ccd, num_workers=1)
    t = time.perf_counter()
    env = BatchedIsaacLayer1Env(cfg, terrain_mode="shared")
    result["construct_seconds"] = time.perf_counter() - t
    result["device"] = str(env.sim.device)
    px = env.sim.cfg.physx
    result["physx_cfg"] = {k: getattr(px, k) for k in ("enable_ccd", "solver_type", "use_gpu", "enable_stabilization",
                                                         "bounce_threshold_velocity", "gpu_max_rigid_contact_count",
                                                         "gpu_max_rigid_patch_count", "gpu_found_lost_pairs_capacity") if hasattr(px, k)}
    rng = np.random.default_rng(0)
    env.reset()
    frames = []
    t = time.perf_counter()
    for s in range(args.steps):
        if args.workload == "random":
            a = {"move_x": rng.integers(0, 3, n), "jump": rng.integers(0, 2, n), "crouch": rng.integers(0, 2, n)}
        elif args.workload == "crouch":   # settle, crouch for 8 decisions, stand again
            a = {"move_x": np.ones(n, int), "jump": np.zeros(n, int), "crouch": np.full(n, 1 if 3 <= s < 11 else 0)}
        else:
            a = {"move_x": np.full(n, 2), "jump": np.full(n, 1 if s % 4 == 3 else 0), "crouch": np.zeros(n, int)}
        _, r, term, trunc, info = env.step(a)
        frames.append(info["player"])
        if args.workload == "crouch":
            print("  step", s, "y", round(info["player"][0]["y"], 3), flush=True)
    result["step_seconds"] = time.perf_counter() - t
    result["ticks_per_second"] = args.steps * 10 * n / result["step_seconds"]
    result["final_players"] = frames[-1]
    result["status"] = "ran"
    env.close()
except BaseException as error:
    result["status"] = "failed"
    result["error"] = f"{type(error).__name__}: {error}"
    result["traceback"] = traceback.format_exc()[-2000:]
print("GPU_PROBE", json.dumps(result)[:3000], flush=True)
if args.output:
    args.output.write_text(json.dumps(result, indent=2) + "\n")
app.close()
