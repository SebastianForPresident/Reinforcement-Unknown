"""Create budget-conditioned PPO_6 without modifying its PPO_5 source."""

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.save_util import load_from_zip_file, save_to_zip_file

import CasualtiesEnv
import ResourceBudget
import Train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError(args.destination)
    if Train.CheckpointStep(args.source) != Train.CB1_INTERVENTION_SOURCE_STEP:
        raise ValueError("Expected the accepted 51,843,364-step PPO_5 source")

    torch.set_num_threads(2)
    data, params, variables = load_from_zip_file(args.source, device="cpu")
    old_space = data["observation_space"]
    if old_space["general"].shape != (139,):
        raise ValueError("Source checkpoint does not use 139 general scalars")
    new_space = copy.deepcopy(old_space)
    new_space.spaces["general"] = gym.spaces.Box(
        -1.0, 1.0, (140,), np.float32
    )
    data["observation_space"] = new_space

    expanded_weights = []
    for key, value in list(params["policy"].items()):
        if key.endswith("general_encoder.net.0.weight"):
            if value.shape != (256, 139):
                raise ValueError(f"Unexpected encoder weight {key}: {value.shape}")
            params["policy"][key] = torch.nn.functional.pad(value, (0, 1))
            expanded_weights.append(key)
    if not expanded_weights:
        raise RuntimeError("General encoder input weight was not found")

    optimizer_tensors_expanded = 0
    for state in params["policy.optimizer"]["state"].values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor) and value.shape == (256, 139):
                state[key] = torch.nn.functional.pad(value, (0, 1))
                optimizer_tensors_expanded += 1
    data["_last_obs"] = None
    data["_last_original_obs"] = None

    args.destination.mkdir(parents=True)
    target = args.destination / args.source.name
    save_to_zip_file(target, data=data, params=params, pytorch_variables=variables)
    del data, params, variables

    old = Train.PausingPPO.load(args.source, device="cpu")
    new = Train.AttackPriorPPO.load(target, device="cpu")
    old.policy.set_training_mode(False)
    new.policy.set_training_mode(False)
    torch.manual_seed(8617)
    max_probability_error = 0.0
    max_value_error = 0.0
    observations_tested = 0
    with torch.no_grad():
        for _ in range(4):
            obs = {
                "general": torch.rand(2, 139) * 2 - 1,
                "spatial": torch.rand(2, *old_space["spatial"].shape),
            }
            old_dist = old.policy.get_distribution(obs)
            old_value = old.policy.predict_values(obs)
            for remaining in (0, 1_500, 3_000):
                conditioned = {
                    **obs,
                    "general": torch.cat([
                        obs["general"],
                        torch.full(
                            (2, 1),
                            ResourceBudget.normalize_remaining(remaining),
                        ),
                    ], dim=1),
                }
                new_dist = new.policy.get_distribution(conditioned)
                for before, after in zip(
                    old_dist.distribution, new_dist.distribution
                ):
                    error = (before.probs - after.probs).abs().max().item()
                    max_probability_error = max(max_probability_error, error)
                    torch.testing.assert_close(before.probs, after.probs)
                new_value = new.policy.predict_values(conditioned)
                error = (old_value - new_value).abs().max().item()
                max_value_error = max(max_value_error, error)
                torch.testing.assert_close(old_value, new_value)
                observations_tested += len(obs["general"])

    environment = {
        "PPO_ATTACK_BUDGET": "1",
        "PPO_ATTACK_LOGIT_PRIOR": "1",
        "PPO_MISSED_ATTACK_PENALTY": "0",
        "PPO_ATTACK_PERMISSION": "0",
        "PPO_ATTACK_PENALTY": "0",
        "PPO_ATTACK_TERMINAL": "0",
        "PPO_ATTACK_PHASES": "0",
        "PPO_ATTACK_CURRICULUM": "0",
        "PPO_ATTACK_EXPLORATION": "0",
    }
    os.environ.update(environment)
    env = CasualtiesEnv.Env()
    Train.ValidateOrWriteProtocolManifest(args.destination, env, resuming=False)
    Train.ValidateOrWriteInterventionManifest(args.destination, target)
    Train.ValidateOrWriteProtocolManifest(args.destination, env, resuming=True)
    env.close()

    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    report = {
        "source": str(args.source.resolve()),
        "source_sha256": source_hash,
        "destination": str(target.resolve()),
        "expanded_weights": expanded_weights,
        "optimizer_tensors_expanded": optimizer_tensors_expanded,
        "optimizer_moments_preserved": True,
        "general_shape": [140],
        "initial_outputs_preserved_for_all_budget_values": True,
        "synthetic_observations_tested": observations_tested,
        "max_initial_action_probability_error": max_probability_error,
        "max_initial_value_error": max_value_error,
    }
    (args.destination / "resource_budget_surgery.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Resource-budget preparation failed: {exc}", file=sys.stderr)
        raise
