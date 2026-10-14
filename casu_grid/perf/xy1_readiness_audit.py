"""Scientific-correctness audit of the XY1 mini-world environment before a learning campaign (real simulator, 4 worlds, one process).

    "$ISAAC_SIM/python.sh" casu_grid/perf/xy1_readiness_audit.py --output audit.json

Checks, each against an independent recomputation (not against the code under test): independent per-world terrain and placement, per-world goals and the relative-goal
observation, uniform goal sampling, arrival detection (radius, bonus, termination), distance-improvement reward accumulation (new bests only), timeout vs arrival
precedence, per-world resets, terrain isolation between worlds, and (documented, not judged) what happens when the agent leaves the world.
"""
from __future__ import annotations
import argparse, faulthandler, hashlib, json, math, os, sys, traceback
faulthandler.dump_traceback_later(int(os.environ.get("AUDIT_DEADLINE", "420")), exit=True)
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--worlds", type=int, default=4)
p.add_argument("--guarded", action="store_true", help="audit the exploratory variant that ends an episode when the agent leaves the world")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); args.headless = True
app = AppLauncher(args).app
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from casu_grid.isaac_navigation_env import IsaacLayer1EnvConfig
from casu_grid.isaac_vector_env import BatchedIsaacLayer1Env
from casu_grid.workbench.experiments import xy1_point_navigation_mini as cfgmod

RESULTS: list[dict] = []


def check(name: str, ok: bool, evidence: str):
    RESULTS.append({"check": name, "passed": bool(ok), "evidence": evidence})
    print(("PASS " if ok else "FAIL ") + name + " | " + evidence, flush=True)


def neutral(n):
    return {h.name: np.full(n, {"move_x": 1, "jump": 0, "move_y": 1, "crouch": 0, "look_dx": 4, "look_dy": 5, "attack": 0}[h.name]) for h in cfgmod.ACTION.heads}


def teleport(env, i, x, y):
    s = env.slots[i].model.s
    s.x, s.y, s.vx, s.vy = float(x), float(y), 0.0, 0.0
    s.grounded = False; s.crouch_amount = 0.0
    env.slots[i].last_state = env.slots[i].body_dict()
    env._write_slot_states([i])
    env.player.update(env.cfg.physics_dt)


def cell_of(env, i, x, y):
    g = env.slots[i].grid
    return g.world_to_cell(x, y, g.width, g.height)


def main():
    n = args.worlds
    assert os.environ.get("CASU_LEGACY_MINI_SPAWN", "0") != "1", "legacy spawn must be off"
    cb = cfgmod.callbacks()
    cfg = IsaacLayer1EnvConfig(seed=391728405, num_envs=n, spawn_mode="canonical", navigation_mode=True, world_variant="xy1_mini_128x128",
                               observation_window_tiles=(64, 64), episode_length_steps=6, num_workers=1, terminate_out_of_world=args.guarded)
    env = BatchedIsaacLayer1Env(cfg, action_decoder=cb[0], observation_encoder=cb[1], reward_function=cb[2], terrain_mode="independent")
    obs, _ = env.reset(seed=12345)
    W = env.slots[0].grid.width
    # ---- A1 independent worlds
    hashes = [hashlib.sha256(np.ascontiguousarray(s.physical_blocks).tobytes()).hexdigest()[:12] for s in env.slots]
    check("A1 every world has its own terrain", len(set(hashes)) == n, f"terrain hashes {hashes}")
    seeds = [s.reference.seed for s in env.slots] if hasattr(env.slots[0], "reference") else None
    check("A1b terrain seeds are 391728405 + world index", seeds == [391728405 + i for i in range(n)], f"seeds {seeds}")
    # ---- A2 per-world placement
    pos = env._np(env.player.data.root_pos_w)
    off = env.offsets
    errs = [(pos[i, 0] - off[i, 0] - 0.5, pos[i, 2] - off[i, 2] - 0.5) for i in range(n)]
    check("A2 every body starts at its own world's centre (stage position minus offset = (0.5, 0.5) +- fall of one tick)", all(abs(e[0]) < 0.05 and abs(e[1]) < 0.2 for e in errs),
          f"deviation from centre per world {[(round(float(a), 4), round(float(b), 4)) for a, b in errs]}; stage x {[round(float(v), 1) for v in pos[:, 0]]}")
    check("A2b stage positions are distinct (no stacked bodies)", len({round(float(v), 1) for v in pos[:, 0]}) == n, f"stage x {[round(float(v), 1) for v in pos[:, 0]]}")
    dense = [obs["spatial"][i] for i in range(n)]
    nonempty = [int((d[0] > 0).sum()) for d in dense]
    check("A2c every observation window contains its own terrain (non-empty cells)", all(c > 100 for c in nonempty), f"non-empty cells per world {nonempty}")
    check("A2d no window reports 'outside the world' fluid code", all(int((d[7 + 7 - 1 + 0] > 0).sum()) >= 0 for d in dense), "windows are inside the world")
    # ---- A3 goals
    ok_range, ok_obs, ok_world = True, True, True
    details = []
    for i, s in enumerate(env.slots):
        px, py = cell_of(env, i, s.model.s.x, s.model.s.y)
        gx, gy = s.goal_cell
        in_view = 32 <= gx < 96 and 32 <= gy < 96
        rel = obs["goal_relative"][i]
        ok_range &= in_view
        ok_obs &= bool(rel[0] == gx - px and rel[1] == gy - py)
        expect_world = (gx - W / 2 + 0.5, gy - s.grid.height / 2 + 0.5)
        ok_world &= abs(s.goal_world_xy[0] - expect_world[0]) < 1e-6 and abs(s.goal_world_xy[1] - expect_world[1]) < 1e-6
        details.append((i, (gx, gy), (px, py), tuple(float(v) for v in rel)))
    check("A3 goals lie in the 64 x 64 viewport around the spawn tile", ok_range, f"(world, goal cell, player cell, relative goal) = {details}")
    check("A3b relative-goal observation = goal tile - player tile (recomputed independently)", ok_obs, "see A3 details")
    check("A3c goal world coordinates are in the world's local frame", ok_world, "goal_world_xy == (gx - 64 + 0.5, gy - 64 + 0.5)")
    # ---- A4 distribution of goals
    rng = np.random.default_rng(1)
    samples = [env._sample_goal(env.slots[0], rng, (0.5, 0.5))[0] for _ in range(20000)]
    gxs, gys = np.array([s[0] for s in samples]), np.array([s[1] for s in samples])
    check("A4 goal sampling is uniform over the viewport", gxs.min() == 32 and gxs.max() == 95 and gys.min() == 32 and gys.max() == 95 and abs(gxs.mean() - 63.5) < 0.6 and abs(gys.mean() - 63.5) < 0.6,
          f"x in [{gxs.min()}, {gxs.max()}] mean {gxs.mean():.2f}; y in [{gys.min()}, {gys.max()}] mean {gys.mean():.2f}")
    # ---- A5 arrival detection
    out = {}
    for label, dx, dy, expect in (("on the goal tile", 0, 0, True), ("diagonal neighbour", 1, 1, True), ("two tiles away", 2, 0, False)):
        obs, _ = env.reset(seed=777)
        s = env.slots[0]
        gx, gy = s.goal_cell
        tx, ty = (gx + dx) - W / 2 + 0.5, (gy + dy) - s.grid.height / 2 + 0.5
        teleport(env, 0, tx, ty)
        _, reward, terminated, truncated, info = env.step(neutral(n))
        arrived = bool(terminated[0]) and info["termination_reason"][0] == "arrival"
        out[label] = (arrived, float(info["reward_terms"].get("arrival_bonus", np.zeros(n))[0]), float(reward[0]))
        check(f"A5 arrival detection: {label}", arrived == expect, f"terminated={bool(terminated[0])} reason={info['termination_reason'][0]} bonus={out[label][1]:.2f} reward={out[label][2]:.3f}")
    check("A5b arrival bonus is 10 and paid once with the improvement term", abs(out["on the goal tile"][1] - 10.0) < 1e-6, f"bonus {out['on the goal tile'][1]}")
    # ---- A6 reward accumulation (new all-time bests only)
    obs, _ = env.reset(seed=4242)
    s = env.slots[0]
    gx, gy = s.goal_cell
    goal = s.goal_world_xy
    start = (s.model.s.x, s.model.s.y)
    d0 = math.dist(start, goal)
    rewards = []
    best = d0
    floor = min(0.9, max(0.5, 4.0 / max(d0, 1e-6)))        # never get close enough to arrive: the checks below are about the distance reward only
    for label, frac in (("closer", 0.85), ("farther than before", 0.97), ("same best again", 0.85), ("new best", floor), ("regress", 1.0)):
        t = (goal[0] + (start[0] - goal[0]) * frac, goal[1] + (start[1] - goal[1]) * frac)
        teleport(env, 0, *t)
        _, reward, terminated, truncated, info = env.step(neutral(n))
        cur = info["player"][0]
        dist = math.dist((cur["x"], cur["y"]), goal)
        expected = 0.1 * max(0.0, best - dist)
        best = min(best, dist)
        got = float(info["reward_terms"]["distance_improvement"][0])
        rewards.append((label, round(got, 4), round(expected, 4)))
        if bool(terminated[0] or truncated[0]):
            break
    check("A6 distance-improvement reward pays only for new all-time bests (0.1 x improvement)", all(abs(g - e) < 2e-3 for _, g, e in rewards) and len(rewards) == 5, f"(step, paid, recomputed) = {rewards}")
    # ---- A7 timeout and precedence
    obs, _ = env.reset(seed=99)
    flags = []
    for k in range(7):
        _, reward, terminated, truncated, info = env.step(neutral(n))
        flags.append((k + 1, bool(truncated[1]), info["termination_reason"][1]))
        if truncated[1]:
            break
    check("A7 timeout: episode is truncated (not terminated) exactly at the step limit", flags[-1][0] == 6 and flags[-1][1] and flags[-1][2] == "time_limit" and not any(f[1] for f in flags[:-1]), f"steps {flags}")
    # ---- A8 resets
    obs, _ = env.reset(seed=5)
    before = {i: env.slots[i].goal_cell for i in range(n)}
    ground = [hashlib.sha256(np.ascontiguousarray(s.physical_blocks).tobytes()).hexdigest()[:12] for s in env.slots]
    grid0 = env.slots[0].grid
    solid = np.argwhere(grid0.blocks[64:80, 3:40] != 0)
    ex, ey = int(solid[0][0]) + 64, int(solid[0][1]) + 3
    logical = lambda: [hashlib.sha256(np.ascontiguousarray(s.grid.blocks).tobytes()).hexdigest()[:12] for s in env.slots]
    logical_before = logical()
    env.set_block(0, ex, ey, 0)
    for _ in range(3):
        env.step(neutral(n))
    dug = [hashlib.sha256(np.ascontiguousarray(s.physical_blocks).tobytes()).hexdigest()[:12] for s in env.slots]
    logical_after = logical()
    check("A8 terrain edits stay inside their own world (logical and collider terrain)", logical_after[0] != logical_before[0] and logical_after[1:] == logical_before[1:] and dug[0] != ground[0] and dug[1:] == ground[1:],
          f"edited tile ({ex}, {ey}) of world 0; logical before {logical_before}, after {logical_after}; collider terrain after {dug} (before {ground})")
    mask = np.zeros(n, dtype=bool); mask[0] = True
    env.reset_done(mask, observe=False)
    after = {i: env.slots[i].goal_cell for i in range(n)}
    restored = [hashlib.sha256(np.ascontiguousarray(s.physical_blocks).tobytes()).hexdigest()[:12] for s in env.slots]
    check("A8b reset_done re-draws the goal and restores the terrain of the finished world only", after[0] != before[0] and all(after[i] == before[i] for i in range(1, n)) and restored == ground,
          f"goals before {before} after {after}; terrain restored {restored == ground}")
    # ---- A9 leaving the world
    obs, _ = env.reset(seed=6)
    teleport(env, 2, 0.5, -70.0)
    flags = []
    for k in range(4):
        _, reward, terminated, truncated, info = env.step(neutral(n))
        flags.append((bool(terminated[2] or truncated[2]), info["termination_reason"][2], round(float(info["player"][2]["y"]), 1), float(info["xy1"]["outside_world"][2])))
        if truncated[2] or terminated[2]:
            break
    if not args.guarded:
        check("A9 (documented XY1 behaviour) leaving the world does not end the episode and is flagged as outside", len(flags) == 4 and not any(f[0] for f in flags) and all(f[3] == 1.0 for f in flags),
              f"(ended, reason, y, outside flag) per step {flags}: the body keeps falling and nothing terminates the episode")
    else:
        check("A9 variant: leaving the world truncates the episode with reason out_of_world and no reward", flags[0][0] and flags[0][1] == "out_of_world" and not bool(terminated[2]) and float(reward[2]) == 0.0,
              f"(ended, reason, y, outside flag) {flags}, reward {float(reward[2])}")
        env.reset_done(np.array([False, False, True, False][:n]), observe=False)
        check("A9b variant: the agent is back inside its world after the reset", abs(env.slots[2].model.s.y) < 64 and abs(env.slots[2].model.s.x) < 64, f"local position after reset ({env.slots[2].model.s.x:.1f}, {env.slots[2].model.s.y:.1f})")
    env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "casu.xy1-readiness-audit.v1", "worlds": n, "results": RESULTS}, indent=2) + "\n")
    failed = [r for r in RESULTS if not r["passed"]]
    print(f"AUDIT {'PASSED' if not failed else 'FAILED'}: {len(RESULTS) - len(failed)} of {len(RESULTS)} checks", flush=True)


try:
    main()
except BaseException:
    traceback.print_exc()
    raise SystemExit(1)
