"""Scenario suite for comparing PhysX backends (CPU reference vs experimental GPU pipeline).

    python casu_grid/perf/gpu_scenarios.py --output cpu.json            # reference
    python casu_grid/perf/gpu_scenarios.py --cuda --output gpu.json     # EXPERIMENTAL GPU pipeline
    python casu_grid/tools/compare_scenarios.py cpu.json gpu.json

Each scenario runs on one independent-terrain world from the flat fixture (483,1010), records the body state
after every decision and any failure, so differences can be read as a function of simulated time.
"""
import argparse, json, sys, time, traceback
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--cuda", action="store_true")
p.add_argument("--player-contact-offset", type=float, default=0.04, help="EXPERIMENT: protected default 0.04")
p.add_argument("--player-collider", choices=("compound", "convex_hull"), default="compound")
p.add_argument("--no-ccd", action="store_true", help="EXPERIMENT: disable CCD")
p.add_argument("--only", required=True, help="one scenario per process (one SimulationContext per process)")
p.add_argument("--output", type=Path, required=True)
args, _ = p.parse_known_args()
sys.stdout.reconfigure(line_buffering=True)
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from casu_grid.isaac_navigation_env import IsaacLayer1EnvConfig
from casu_grid.isaac_vector_env import BatchedIsaacLayer1Env


def act(move=1, jump=0, crouch=0):
    return {"move_x": np.array([move]), "jump": np.array([jump]), "crouch": np.array([crouch])}


def tape_crouch_tunnel():
    return [act()] * 2 + [act(1, 0, 1)] * 6 + [act(2, 0, 1)] * 14


def tape_support_removal():
    return [act()] * 3 + [act()] * 10


def tape_landing():
    return [act()] * 2 + [act(1, 1, 0)] + [act(1, 0, 0)] * 10


def tape_wall():
    return [act()] * 2 + [act(2)] * 8 + [act(2, 1)] * 2 + [act(2)] * 6


def tape_junction_run():
    return [act()] * 2 + [act(2)] * 10 + [act(0)] * 3 + [act(2)] * 6


def tape_stand_tunnel():
    return [act()] * 2 + [act(2)] * 12


def tape_wall_slow():
    return [act()] * 2 + [act(2)] * 3 + [act(1)] * 2 + [act(2)] * 6


SCENARIOS = {
    "stand_tunnel": ([(x, 1015, 1) for x in range(485, 493)], tape_stand_tunnel(), {}),
    "wall_thick": ([(x, y, 1) for x in (487, 488, 489, 490) for y in (1011, 1012, 1013)], tape_wall(), {}),
    "wall_noscript": ([(487, y, 1) for y in (1011, 1012, 1013)], [act()] * 2 + [act(2)] * 8, {}),
    "crouch_tunnel": ([(x, 1015, 1) for x in range(485, 493)], tape_crouch_tunnel(), {}),
    "support_removal": ([], tape_support_removal(), {3: [(482, 1010, 0), (483, 1010, 0), (484, 1010, 0)]}),
    "landing": ([], tape_landing(), {}),
    "wall": ([(487, y, 1) for y in (1011, 1012, 1013)], tape_wall(), {}),
    "junction_run": ([], tape_junction_run(), {}),
}

results = {"cuda": args.cuda, "scenarios": {}}
for name, (edits, tape, timed_edits) in SCENARIOS.items():
    if args.only and name not in args.only.split(","):
        continue
    entry = {"frames": [], "status": "ok"}
    env = None
    try:
        cfg = IsaacLayer1EnvConfig(num_envs=1, spawn_mode="fixture", fixture_grid_xy=(483, 1010), use_cuda=args.cuda, physx_ccd=not args.no_ccd, player_collider=args.player_collider, player_contact_offset=args.player_contact_offset,
                                   num_workers=1, terrain_edit_scene_reset=args.cuda)
        env = BatchedIsaacLayer1Env(cfg, terrain_mode="independent")
        env.reset()
        for x, y, b in edits:
            env.slots[0].edit_block(x, y, b, immediate=True)
        if edits:
            env._rebuild_dirty([0])
        for k, a in enumerate(tape):
            for x, y, b in timed_edits.get(k, []):
                env.set_block(0, x, y, b)
            env.step(a)
            s = env._body_state(0)
            entry["frames"].append({**s, "crouch": float(env.slots[0].model.s.crouch_amount),
                                    "level": env.colliders[0].active_level, "grounded": bool(env.slots[0].model.s.grounded)})
    except BaseException as error:
        entry["status"] = f"failed: {type(error).__name__}: {str(error)[:200]}"
        entry["traceback"] = traceback.format_exc()[-800:]
    results["scenarios"][name] = entry
    print(name, entry["status"], len(entry["frames"]), flush=True)
    try:
        if env is not None:
            env.close()
    except BaseException:
        pass
args.output.write_text(json.dumps(results, indent=1) + "\n")
print("SCENARIOS_DONE", flush=True)
app.close()
