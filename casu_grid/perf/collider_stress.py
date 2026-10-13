"""Collider robustness probe on the real terrain: embedding depth, resets and mobility under long seeded tapes.

    python casu_grid/perf/collider_stress.py --player-collider convex_hull --envs 64 --decisions 600 --tape random --output out.json

No CasU reference is involved. It looks for hull- or compound-specific *pathologies* on canonical terrain: the player collider
overlapping solid tiles (tunnelling / ghost-contact artefacts), non-finite states, implausible per-decision jumps and
abnormal terminations. Trajectories diverge between colliders after the first differing contact, so only distributions
(depth percentiles, event counts) are comparable, not individual worlds.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--player-collider", choices=("compound", "convex_hull"), default="compound")
parser.add_argument("--hull-segments", type=int, default=8)
parser.add_argument("--envs", type=int, default=64)
parser.add_argument("--decisions", type=int, default=600)
parser.add_argument("--tape", choices=("random", "run", "climb"), default="random")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--fixture", type=int, nargs=2, default=(483, 1010))
parser.add_argument("--depth-threshold", type=float, default=0.15, help="overlap (m) counted as embedded")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from casu_grid.isaac_navigation_env import IsaacLayer1EnvConfig
from casu_grid.isaac_vector_env import BatchedIsaacLayer1Env

HALF = 512.0


def embedding_depth(slot) -> float:
    """Largest min(overlap_x, overlap_y) between the player's rounded box and any solid tile cell."""
    s, c = slot.model.s, slot.model.cfg
    half_w = s.col_size[0] * 0.5 + c.edge_radius
    half_h = s.col_size[1] * 0.5 + c.edge_radius
    cx, cy = s.x, s.y + s.col_offset[1]
    x0, x1, y0, y1 = cx - half_w, cx + half_w, cy - half_h, cy + half_h
    blocks = slot.ground_world.blocks
    worst = 0.0
    for tx in range(int(math.floor(x0 + HALF)), int(math.floor(x1 + HALF)) + 1):
        for ty in range(int(math.floor(y0 + HALF)), int(math.floor(y1 + HALF)) + 1):
            if 0 <= tx < blocks.shape[0] and 0 <= ty < blocks.shape[1] and blocks[tx, ty]:
                ox = min(x1, tx - HALF + 1.0) - max(x0, tx - HALF)
                oy = min(y1, ty - HALF + 1.0) - max(y0, ty - HALF)
                if ox > 0 and oy > 0:
                    worst = max(worst, min(ox, oy))
    return worst


def main():
    n = args.envs
    cfg = IsaacLayer1EnvConfig(num_envs=n, spawn_mode="fixture", fixture_grid_xy=tuple(args.fixture),
                               player_collider=args.player_collider, player_hull_segments=args.hull_segments, num_workers=1)
    env = BatchedIsaacLayer1Env(cfg)
    rng = np.random.default_rng(args.seed)
    depths, jumps, nonfinite, resets = [], 0, 0, 0
    previous_depth = np.zeros(n)
    fast_samples, fast_worlds = 0, set()
    persistent = 0
    reasons = {}
    ends = []                                  # (reason, x, y, vy) of every episode end
    travelled = np.zeros(n)
    try:
        env.reset(seed=args.seed)
        previous = np.array([[s.model.s.x, s.model.s.y] for s in env.slots])
        for step in range(args.decisions):
            if args.tape == "random":
                action = {"move_x": rng.integers(0, 3, n), "jump": rng.integers(0, 2, n), "crouch": rng.integers(0, 2, n)}
            elif args.tape == "run":
                direction = np.where(rng.random(n) < 0.5, 1, 2)
                action = {"move_x": np.where(step % 40 < 20, direction * 0 + 2, direction * 0 + 1),
                          "jump": np.full(n, 1 if step % 4 == 3 else 0), "crouch": np.zeros(n, int)}
            else:                                   # climb: always pushing one way, jumping on most decisions
                action = {"move_x": np.where(np.arange(n) % 2 == 0, 2, 1), "jump": (rng.random(n) < 0.8).astype(int),
                          "crouch": np.zeros(n, int)}
            _, _, term, trunc, info = env.step(action)
            done = np.logical_or(term, trunc)
            for i, slot in enumerate(env.slots):
                s = slot.model.s
                values = (s.x, s.y, s.vx, s.vy)
                if not all(math.isfinite(v) for v in values):
                    nonfinite += 1
                    continue
                if abs(s.vy) > 40.0 or abs(s.vx) > 40.0:       # CasU never exceeds 31 m/s in 20,000 recorded ticks
                    fast_samples += 1
                    fast_worlds.add(i)
                if not done[i]:
                    depth = embedding_depth(slot)
                    depths.append(depth)
                    persistent += depth > args.depth_threshold and previous_depth[i] > args.depth_threshold
                    previous_depth[i] = depth
                    step_move = math.hypot(s.x - previous[i][0], s.y - previous[i][1])
                    jumps += step_move > 14.0           # > 10 ticks at the 25 m/s speed limit + margin
                    travelled[i] += abs(s.x - previous[i][0])
                previous[i] = (s.x, s.y)
            for i, reason in enumerate(info["termination_reason"]):
                if reason:
                    reasons[reason] = reasons.get(reason, 0) + 1
                    s = env.slots[i].model.s
                    ends.append((reason, round(float(s.x), 2), round(float(s.y), 2), round(float(s.last_time_step_vy), 1)))
            resets += int(done.sum())
            if done.any():
                env.reset_done(done)
                for i in np.flatnonzero(done):
                    previous_depth[i] = 0.0
                    previous[i] = (env.slots[i].model.s.x, env.slots[i].model.s.y)
    finally:
        env.close()
    d = np.asarray(depths)
    result = {"schema": "casu.collider-stress.v1", "collider": args.player_collider, "hull_segments": args.hull_segments,
              "envs": n, "decisions": args.decisions, "tape": args.tape, "seed": args.seed, "samples": int(d.size),
              "embedded_threshold_m": args.depth_threshold, "embedded_fraction": float((d > args.depth_threshold).mean()),
              "embedded_samples": int((d > args.depth_threshold).sum()),
              "persistent_embedded_samples": int(persistent),          # embedded at two consecutive decisions of the same world
              "depth_percentiles": {str(q): float(np.percentile(d, q)) for q in (50, 90, 99, 99.9, 100)},
              "samples_with_speed_over_40_at_decision_boundary": int(fast_samples),
              "speed_over_40_rate_per_1000_decisions": 1000.0 * fast_samples / (args.envs * args.decisions),
              "worlds_with_speed_over_40": len(fast_worlds),
              "implausible_jumps": int(jumps), "nonfinite_states": int(nonfinite), "episode_ends": int(resets),
              "termination_reasons": reasons, "episode_end_positions": ends, "mean_total_abs_x_travel": float(travelled.mean())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print("STRESS", json.dumps({k: v for k, v in result.items() if k != "depth_percentiles"}), flush=True)


import traceback
try:
    main()
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    app.close()
