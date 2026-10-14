"""Profile the XY1 mini-world vector environment (one process) under near-uniform random policy actions.

    python casu_grid/perf/xy1_env_profile.py --envs 4 --steps 60 --cprofile --output out.json
"""
from __future__ import annotations
import argparse, cProfile, io, json, pstats, sys, time
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--envs", type=int, default=4)
p.add_argument("--steps", type=int, default=60)
p.add_argument("--warmup", type=int, default=5)
p.add_argument("--seed", type=int, default=3)
p.add_argument("--cprofile", action="store_true")
p.add_argument("--top", type=int, default=35)
p.add_argument("--player-collider", default="compound")
p.add_argument("--empty-terrain", action="store_true", help="DIAGNOSTIC ONLY: floor-only worlds, to size terrain-collider cost")
p.add_argument("--crouch-levels", type=int, default=None, help="DIAGNOSTIC ONLY: changes physics")
p.add_argument("--carb", action="append", default=[], help="NAME=VALUE carb setting applied after construction (diagnostic)")
p.add_argument("--no-ccd", action="store_true", help="DIAGNOSTIC ONLY (changes physics)")
p.add_argument("--physx-threads", type=int, default=None)
p.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); args.headless = True
app = AppLauncher(args).app
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from casu_grid.isaac_navigation_env import IsaacLayer1EnvConfig
from casu_grid.isaac_vector_env import BatchedIsaacLayer1Env
from casu_grid.workbench.experiments import xy1_point_navigation_mini as cfgmod

SIZES = [h.size for h in cfgmod.ACTION.heads]
NAMES = [h.name for h in cfgmod.ACTION.heads]


def main():
    if args.empty_terrain:
        import casu_grid.xy1_mini_world as mw
        orig = mw.generate_mini_terrain
        def floor_only(seed):
            t = orig(seed); keep = np.zeros_like(t); keep[:, :2] = t[:, :2]; keep[:7, :] = t[:7, :]; keep[-7:, :] = t[-7:, :]
            c = mw.MINI_WIDTH // 2; keep[c-1:c+2, 2:] = 0; return keep * (np.arange(t.shape[1])[None, :] < 3)
        mw.generate_mini_terrain = floor_only
    cb = cfgmod.callbacks()
    env_cfg = IsaacLayer1EnvConfig(seed=391728405, num_envs=args.envs, spawn_mode="canonical", navigation_mode=True,
                                   world_variant="xy1_mini_128x128", observation_window_tiles=(64, 64),
                                   player_collider=args.player_collider, episode_length_steps=100000, num_workers=1, physx_ccd=not args.no_ccd, **({'crouch_levels': args.crouch_levels} if args.crouch_levels else {}))
    if args.physx_threads:
        import carb.settings; carb.settings.get_settings().set('/persistent/physics/numThreads', int(args.physx_threads))
    started = time.perf_counter()
    env = BatchedIsaacLayer1Env(env_cfg, action_decoder=cb[0], observation_encoder=cb[1], reward_function=cb[2], terrain_mode="independent")
    construct = time.perf_counter() - started
    import carb.settings
    st = carb.settings.get_settings()
    for item in args.carb:
        k, v = item.split("=", 1)
        v = {"true": True, "false": False}.get(v.lower(), v)
        if isinstance(v, str):
            try: v = int(v)
            except ValueError: v = float(v)
        print("CARB", k, st.get(k), "->", v, flush=True); st.set(k, v)
    for k in ("/physics/updateToUsd", "/physics/updateVelocitiesToUsd", "/physics/fabricEnabled", "/physics/suppressReadback", "/physics/cudaDevice", "/physics/numThreads", "/physics/outputVelocitiesLocalSpace"):
        print("CARBSETTING", k, st.get(k), flush=True)
    rng = np.random.default_rng(args.seed)
    env.reset(seed=args.seed)

    def act():
        return {n: rng.integers(0, s, args.envs) for n, s in zip(NAMES, SIZES)}
    for _ in range(args.warmup):
        env.step(act())
    prof = cProfile.Profile() if args.cprofile else None
    t0 = time.perf_counter()
    if prof: prof.enable()
    for _ in range(args.steps):
        env.step(act())
    if prof: prof.disable()
    wall = time.perf_counter() - t0
    result = {"envs": args.envs, "steps": args.steps, "construct_seconds": construct, "step_wall_seconds": wall,
              "decisions_per_second": args.envs * args.steps / wall, "ms_per_env_decision": 1e3 * wall / (args.envs * args.steps),
              "counters": dict(getattr(env, "counters", {})), "timers_per_tick_us": {k: 1e6 * v / ((args.steps + args.warmup) * 10) for k, v in dict(env.timers).items()}}
    if prof:
        out = io.StringIO()
        pstats.Stats(prof, stream=out).sort_stats("tottime").print_stats(args.top)
        result["profile_tottime"] = out.getvalue()
        out = io.StringIO()
        pstats.Stats(prof, stream=out).sort_stats("cumulative").print_stats(args.top)
        result["profile_cumulative"] = out.getvalue()
    print("XY1PROFILE", json.dumps({k: v for k, v in result.items() if not k.startswith("profile")}, default=str), flush=True)
    if prof:
        print(result["profile_tottime"]); print(result["profile_cumulative"])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=1, default=str))
    env.close()

import traceback
try:
    main()
except BaseException:
    traceback.print_exc(); sys.stderr.flush(); raise
finally:
    sys.stdout.flush(); app.close()
