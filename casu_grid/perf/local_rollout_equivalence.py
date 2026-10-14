"""Worker-local rollout vs central rollout on the real simulator (deterministic action selection).

    python casu_grid/perf/local_rollout_equivalence.py --envs 4 --workers 2 --steps 40

Both paths start from the same restored world state. The central path is the trainer's original loop (full float observations
through the pipe, policy evaluated by the parent); the local path is ``LocalRollout`` running inside the workers. With argmax actions every
recorded quantity must agree: observations (bit for bit), actions, rewards, terminations, values (floating-point tolerance).
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--envs", type=int, default=4)
p.add_argument("--workers", type=int, default=2)
p.add_argument("--steps", type=int, default=40)
p.add_argument("--output", type=Path)
p.add_argument("--no-tf32", action="store_true")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); args.headless = True
os.environ["CASU_NUM_WORKERS"] = str(args.workers)
app = AppLauncher(args).app
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from casu_grid.isaac_navigation_env import IsaacLayer1EnvConfig
from casu_grid.isaac_vector_env import BatchedIsaacLayer1Env
from casu_grid.workbench.ppo import PPOTrainer, _as_tensor_obs
from casu_grid.workbench.runtime_validation import checked_callbacks
from casu_grid.workbench.experiments import xy1_point_navigation_mini as cfg


def main():
    if args.no_tf32:
        torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    cb = checked_callbacks(cfg.callbacks(), cfg.OBSERVATION, cfg.REWARD)
    env_cfg = IsaacLayer1EnvConfig(seed=391728405, num_envs=args.envs, spawn_mode="canonical", navigation_mode=True,
                                   world_variant="xy1_mini_128x128", observation_window_tiles=(64, 64),
                                   player_collider="compound", episode_length_steps=100000)

    def make_env():
        return BatchedIsaacLayer1Env(env_cfg, action_decoder=cb[0], observation_encoder=cb[1], reward_function=cb[2], terrain_mode="independent")
    env = make_env()
    assert env.has_stock_xy1_observation and env.num_workers > 1, "use >= 2 workers (two sequential environments)"
    trainer = PPOTrainer(env, cfg.ACTION, cfg.OBSERVATION, cfg.MODEL, cfg.PPO, device="cuda", seed=5)
    model = trainer.model.eval()
    names = [h.name for h in cfg.ACTION.heads]

    def run_central(env):
        obs0, _ = env.reset(seed=11)
        obs = _as_tensor_obs(obs0, "cuda")
        central = {"latent": [], "value": [], "reward": [], "done": [], "obs_spatial": [], "next_value": []}
        for _ in range(args.steps):
            with torch.no_grad():
                action, log_prob, value, latent = model.act(obs, deterministic=True, return_latent=True)
            env_action = {k: v.cpu().numpy() for k, v in action.items()}
            next_obs, reward, terminated, truncated, info = env.step(env_action)
            next_t = _as_tensor_obs(next_obs, "cuda")
            with torch.no_grad():
                _, next_value = model.distributions(next_t)
            central["latent"].append(torch.stack([latent[n] for n in names], 1).cpu().numpy())
            central["value"].append(value.cpu().numpy()); central["next_value"].append(next_value.cpu().numpy())
            central["reward"].append(np.asarray(reward)); central["done"].append(np.logical_or(terminated, truncated))
            central["obs_spatial"].append(obs["spatial"].cpu().numpy())
            if np.logical_or(terminated, truncated).any():
                next_t = _as_tensor_obs(env.reset_done(np.logical_or(terminated, truncated)), "cuda")
            obs = next_t
        return {k: np.stack(v) for k, v in central.items()}

    central = run_central(env)
    env.close()
    env = make_env()                      # fresh workers: identical start state
    trainer.env = env
    env.reset(seed=11, observe=False)

    # --- worker-local rollout from the same state
    env.local_rollout_setup(cfg.ACTION, cfg.OBSERVATION, cfg.MODEL, "cuda")
    weights = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    res = env.local_rollout_run(weights, args.steps, [1] * env.num_workers, True, np.zeros(args.envs, np.float32), np.zeros(args.envs, np.int64))
    from casu_grid.xy1 import expand_compact_observation
    local_obs = np.stack([expand_compact_observation(res["block_ids"][t], res["fluid_ids"][t], res["cursor"][t]).numpy() for t in range(args.steps)])
    differing = np.flatnonzero((res["latent"] != central["latent"]).reshape(args.steps, -1).any(axis=1))
    first = int(differing[0]) if differing.size else args.steps
    value_scale = float(np.abs(central["value"]).mean())
    report = {
        "first_step_with_a_different_action": first,
        "mean_abs_value": value_scale,
        "all_quantities_identical_before_first_difference": bool(
            np.array_equal(local_obs[:first + 1], central["obs_spatial"][:first + 1]) and np.array_equal(res["reward"][:first], central["reward"][:first])
            and np.array_equal(res["done"][:first] > 0, central["done"][:first])),
        "steps": args.steps, "envs": args.envs, "workers": args.workers,
        "observations_bit_identical": bool(np.array_equal(local_obs, central["obs_spatial"])),
        "actions_identical_fraction": float((res["latent"] == central["latent"]).mean()),
        "rewards_identical": bool(np.array_equal(res["reward"], central["reward"])),
        "dones_identical": bool(np.array_equal(res["done"] > 0, central["done"])),
        "max_abs_value_difference": float(np.abs(res["value"] - central["value"]).max()),
        "max_abs_next_value_difference": float(np.abs(res["next_value"] - central["next_value"]).max()),
        "seconds_per_worker": res["seconds"],
    }
    print("EQUIVALENCE", json.dumps(report), flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=1))
    trainer.close(); env.close()

import traceback
try:
    main()
except BaseException:
    traceback.print_exc(); sys.stderr.flush(); raise
finally:
    sys.stdout.flush(); app.close()
