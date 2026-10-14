"""Locality of terrain edits: destroying a block far from the player must not change the player's motion.

    CASU_INCREMENTAL_TERRAIN=0|1 python casu_grid/perf/terrain_edit_locality.py [--distance 15]

Runs the same idle/jumping action tape twice from identical worlds -- once without an edit and once with a block far from the player
removed (through the same path an excavation uses) -- and reports the largest difference of the player's state over the tape.
A physically local update leaves the control trajectory unchanged (difference 0).
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--distance", type=int, default=15)
p.add_argument("--steps", type=int, default=12)
p.add_argument("--edit-step", type=int, default=4)
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
    env_cfg = IsaacLayer1EnvConfig(seed=391728405, num_envs=1, spawn_mode="canonical", navigation_mode=True,
                                   world_variant="xy1_mini_128x128", observation_window_tiles=(64, 64), num_workers=1)
    env = BatchedIsaacLayer1Env(env_cfg, action_decoder=cb[0], observation_encoder=cb[1], reward_function=cb[2], terrain_mode="independent")
    names = [h.name for h in cfg.ACTION.heads]
    results = {}
    for label in ("control", "edit"):
        env.reset(seed=3)
        slot = env.slots[0]
        px, py = slot.grid.world_to_cell(slot.model.s.x, slot.model.s.y, slot.grid.width, slot.grid.height)
        # a solid cell `distance` tiles to the right in the player's chunk and not adjacent to the spawn column
        target = next((x, y) for x in range(px + args.distance, min(px + args.distance + 12, 126)) for y in range(py - 3, py + 8)
                      if slot.grid.get_block(x, y) and (x // 64, y // 64) == (px // 64, py // 64))
        frames = []
        for step in range(args.steps):
            if label == "edit" and step == args.edit_step:
                env.set_block(0, target[0], target[1], 0)
            action = {n: np.asarray([v]) for n, v in zip(names, (1, 1 if step % 6 == 0 else 0, 1, 0, 4, 5, 0))}
            obs, reward, term, trunc, info = env.step(action)
            s = slot.model.s
            frames.append([s.x, s.y, s.vx, s.vy])
        results[label] = np.asarray(frames)
        results["target"] = [int(target[0]), int(target[1])]
    diff = np.abs(results["control"] - results["edit"])
    report = {"incremental": os.environ.get("CASU_INCREMENTAL_TERRAIN", "1"), "distance": args.distance, "target_cell": results["target"],
              "max_abs_state_difference": float(diff.max()), "first_differing_decision": next((int(i) for i in range(len(diff)) if diff[i].max() > 0), None)}
    print("LOCALITY", json.dumps(report), flush=True)
    env.close()

import traceback
try:
    main()
except BaseException:
    traceback.print_exc(); sys.stderr.flush(); raise
finally:
    sys.stdout.flush(); app.close()
