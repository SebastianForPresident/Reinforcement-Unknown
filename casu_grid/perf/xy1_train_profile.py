"""Profile the trainer process of the XY1 mini-world pipeline (sharded workers + PPO), without launching a research run.

    CASU_NUM_WORKERS=4 python casu_grid/perf/xy1_train_profile.py --envs 16 --updates 4 --cprofile --output out.json
"""
from __future__ import annotations
import argparse, cProfile, io, json, os, pstats, sys, time
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--envs", type=int, default=16)
p.add_argument("--updates", type=int, default=4)
p.add_argument("--seed", type=int, default=391728405)
p.add_argument("--train-device", default="cpu")
p.add_argument("--cprofile", action="store_true")
p.add_argument("--torch-profile", action="store_true", help="torch.profiler on the optimizer phase of the last update")
p.add_argument("--top", type=int, default=40)
p.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); args.headless = True
app = AppLauncher(args).app
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from gymnasium import spaces
from casu_grid.isaac_navigation_env import IsaacLayer1EnvConfig
from casu_grid.isaac_vector_env import BatchedIsaacLayer1Env
from casu_grid.workbench.ppo import PPOTrainer
from casu_grid.workbench.experiments import xy1_point_navigation_mini as cfg


def main():
    import torch; print("TORCH_THREADS", torch.get_num_threads(), torch.get_num_interop_threads(), flush=True)
    cb = cfg.callbacks()
    env_cfg = IsaacLayer1EnvConfig(seed=391728405, num_envs=args.envs, spawn_mode="canonical", navigation_mode=True,
                                   world_variant="xy1_mini_128x128", observation_window_tiles=(64, 64),
                                   player_collider="compound", episode_length_steps=100000)
    t0 = time.perf_counter()
    env = BatchedIsaacLayer1Env(env_cfg, action_decoder=cb[0], observation_encoder=cb[1], reward_function=cb[2], terrain_mode="independent")
    construct = time.perf_counter() - t0
    env.action_space = spaces.Dict({h.name: spaces.Discrete(h.size) for h in cfg.ACTION.heads})
    env.observation_space = spaces.Dict({f.name: spaces.Box(-np.inf, np.inf, shape=f.shape, dtype=np.float32)
                                         for f in cfg.OBSERVATION.spatial + cfg.OBSERVATION.scalar})
    trainer = PPOTrainer(env, cfg.ACTION, cfg.OBSERVATION, cfg.MODEL, cfg.PPO, device=args.train_device, seed=args.seed)
    rows = []
    prof = cProfile.Profile() if args.cprofile else None
    try:
        for u in range(args.updates):
            if prof and u >= 1: prof.enable()
            if args.torch_profile and u == args.updates - 1:
                import torch
                orig = trainer._update
                def traced(records, last_value, _orig=orig):
                    from torch.profiler import profile, ProfilerActivity
                    torch.cuda.synchronize()
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
                        out = _orig(records, last_value); torch.cuda.synchronize()
                    print(pr.key_averages().table(sort_by="cuda_time_total", row_limit=18, max_name_column_width=60), flush=True)
                    print(pr.key_averages().table(sort_by="cpu_time_total", row_limit=12, max_name_column_width=60), flush=True)
                    return out
                trainer._update = traced
            m = trainer.train_update()
            if prof and u >= 1: prof.disable()
            rows.append({k: m[k] for k in ("collection_seconds", "physics_step_seconds", "optimization_seconds")})
            print("UPDATE", u + 1, json.dumps(rows[-1]), flush=True)
        timers = dict(getattr(env, "timers", {}))
    finally:
        pass
    result = {"envs": args.envs, "construct_seconds": construct, "updates": rows, "env_timers": timers, "device": args.train_device}
    if prof:
        out = io.StringIO(); pstats.Stats(prof, stream=out).sort_stats("tottime").print_stats(args.top); result["tottime"] = out.getvalue()
        out = io.StringIO(); pstats.Stats(prof, stream=out).sort_stats("cumulative").print_stats(args.top); result["cumulative"] = out.getvalue()
        print(result["tottime"]); print(result["cumulative"])
    print("ENV_TIMERS", json.dumps(timers, default=float), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=1, default=float))
    trainer.close(); env.close()

import traceback
try:
    main()
except BaseException:
    traceback.print_exc(); sys.stderr.flush(); raise
finally:
    sys.stdout.flush(); app.close()
