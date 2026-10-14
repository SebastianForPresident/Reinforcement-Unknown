"""Deterministic XY1 mini-world trajectory dump (all seven action heads, including excavation), for bit-exact comparisons.

    CASU_COMPILED_CONTROLLER=0 python casu_grid/perf/xy1_trajectory_dump.py --output ref.json
    python casu_grid/perf/xy1_trajectory_dump.py --output new.json        # compare with tools/compare_trajectories.py or cmp

Per decision and world: position, velocity, crouch, grounded, reward terms, termination flags, a hash of the logical terrain
and of the physical (collider) terrain. A SHA-256 of the whole record is printed.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--envs", type=int, default=4)
p.add_argument("--steps", type=int, default=60)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); args.headless = True
app = AppLauncher(args).app
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from casu_grid.isaac_navigation_env import IsaacLayer1EnvConfig
from casu_grid.isaac_vector_env import BatchedIsaacLayer1Env
from casu_grid.workbench.experiments import xy1_point_navigation_mini as cfg


def main():
    cb = cfg.callbacks()
    env_cfg = IsaacLayer1EnvConfig(seed=391728405, num_envs=args.envs, spawn_mode="canonical", navigation_mode=True,
                                   world_variant="xy1_mini_128x128", observation_window_tiles=(64, 64),
                                   player_collider="compound", episode_length_steps=100000, num_workers=1)
    env = BatchedIsaacLayer1Env(env_cfg, action_decoder=cb[0], observation_encoder=cb[1], reward_function=cb[2], terrain_mode="independent")
    rng = np.random.default_rng(args.seed)
    env.reset(seed=args.seed)
    sizes = [h.size for h in cfg.ACTION.heads]
    names = [h.name for h in cfg.ACTION.heads]
    frames = []
    for step in range(args.steps):
        action = {n: rng.integers(0, s, args.envs) for n, s in zip(names, sizes)}
        if step % 5 < 3:
            action["attack"] = np.ones(args.envs, dtype=int)            # keep excavating so terrain edits are exercised
        obs, reward, term, trunc, info = env.step(action)
        frames.append({"step": step,
                       "players": [[float(p["x"]), float(p["y"]), float(p["vx"]), float(p["vy"])] for p in info["player"]],
                       "crouch": [float(s.model.s.crouch_amount) for s in env.slots],
                       "grounded": [bool(s.model.s.grounded) for s in env.slots],
                       "reward": [float(r) for r in reward], "terminated": [bool(t) for t in term],
                       "terrain": [hashlib.sha256(s.grid.blocks.tobytes()).hexdigest()[:16] for s in env.slots],
                       "physical": [hashlib.sha256(s.physical_blocks.tobytes()).hexdigest()[:16] for s in env.slots],
                       "obs": hashlib.sha256(np.ascontiguousarray(obs["spatial"]).tobytes()).hexdigest()[:16]})
    record = {"envs": args.envs, "steps": args.steps, "seed": args.seed, "frames": frames}
    digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    args.output.write_text(json.dumps(record))
    edits = sum(len(set(f["terrain"])) for f in frames)
    print("XY1_TRAJECTORY", digest, "frames", len(frames), "compiled", env._fast is not None, "terrain_variants", edits, flush=True)
    env.close()

import traceback
try:
    main()
except BaseException:
    traceback.print_exc(); sys.stderr.flush(); raise
finally:
    sys.stdout.flush(); app.close()
