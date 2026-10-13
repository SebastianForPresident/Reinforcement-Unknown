"""Bounded policy transfer / behavioural-sensitivity check across player colliders (no training, no weight updates).

    python casu_grid/validation/policy_transfer_check.py --checkpoint RUN/latest.pt --config RUN/research_config.py \
        --player-collider convex_hull --envs 64 --decisions 400 --seed 11 --output out.json

Loads an existing checkpoint, keeps its observation/action/reward contract, environment settings and seeds fixed, and only
collects rollouts (``PPOTrainer._collect``): completed-episode returns/lengths, termination causes and action frequencies.
The checkpoint available in this repository is an *engineering-smoke* policy (binary stay/advance head); this is a
transfer check of that policy, not an evaluation of a research policy.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--player-collider", choices=("compound", "convex_hull"), default="compound")
parser.add_argument("--envs", type=int, default=64)
parser.add_argument("--decisions", type=int, default=400)
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from gymnasium import spaces
from casu_grid.isaac_navigation_env import IsaacLayer1EnvConfig
from casu_grid.isaac_vector_env import BatchedIsaacLayer1Env
from casu_grid.workbench.ppo import PPOTrainer


def main():
    spec = importlib.util.spec_from_file_location("policy_config", args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    callbacks = config.callbacks()
    env = BatchedIsaacLayer1Env(
        IsaacLayer1EnvConfig(seed=391728405, num_envs=args.envs, spawn_mode="canonical", player_collider=args.player_collider),
        action_decoder=callbacks[0], observation_encoder=callbacks[1], reward_function=callbacks[2])
    env.action_space = spaces.Dict({h.name: spaces.Discrete(h.size) for h in config.ACTION.heads})
    env.observation_space = spaces.Dict({f.name: spaces.Box(-np.inf, np.inf, shape=f.shape, dtype=np.float32)
                                         for f in config.OBSERVATION.spatial + config.OBSERVATION.scalar})
    trainer = PPOTrainer(env, config.ACTION, config.OBSERVATION, config.MODEL, config.PPO, device="cpu", seed=args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "casu.ppo-checkpoint.v1" or checkpoint.get("contract_fingerprint") != trainer.contract_fingerprint:
        raise ValueError("checkpoint does not match the learning contract of --config")
    trainer.model.load_state_dict(checkpoint["model"])       # weights only: no optimizer, RNG or environment state
    trainer.update_count = int(checkpoint["update_count"])
    torch.manual_seed(args.seed)
    trainer.rng = np.random.default_rng(args.seed)
    trainer._observation = None
    try:
        records, _, returns, lengths, terminations, action_freq, _, _, ticks = trainer._collect(args.decisions)
        goal = None
        terms = records["reward_terms"]
        if terms and "goal" in terms[0]:
            goal = float(sum(float(t["goal"].sum()) for t in terms))
        result = {"schema": "casu.policy-transfer-check.v1", "collider": args.player_collider, "seed": args.seed,
                  "envs": args.envs, "decisions": args.decisions, "checkpoint": str(args.checkpoint),
                  "checkpoint_update_count": int(trainer.update_count),
                  "completed_episodes": len(returns),
                  "mean_return": float(np.mean(returns)) if returns else None,
                  "mean_length": float(np.mean(lengths)) if lengths else None,
                  "median_length": float(np.median(lengths)) if lengths else None,
                  "terminations": terminations, "action_frequencies": action_freq, "goal_reward_sum": goal,
                  "physics_ticks": int(ticks),
                  "mean_reward_per_decision": float(torch.stack(records["reward"]).mean())}
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print("POLICY_CHECK", json.dumps(result), flush=True)


import traceback
try:
    main()
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    app.close()
