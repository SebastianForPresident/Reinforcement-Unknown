"""Create the separate PPO_6 attack-cost model without altering its source."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.save_util import (
    load_from_zip_file,
    save_to_zip_file,
)

import CasualtiesEnv
import Train
import AttackPermission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--terminal", action="store_true")
    parser.add_argument("--terminal-exploration", action="store_true")
    parser.add_argument("--curriculum", action="store_true")
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError(args.destination)
    if Train.CheckpointStep(args.source) != Train.CB1_INTERVENTION_SOURCE_STEP:
        raise ValueError("Expected the agreed 51,843,364-step PPO_5 source")

    torch.set_num_threads(2)
    data, params, variables = load_from_zip_file(args.source, device="cpu")
    old_space = data["observation_space"]
    if old_space["general"].shape != (139,):
        raise ValueError("Source does not have the PPO_5 139-scalar schema")
    new_space = copy.deepcopy(old_space)
    new_space.spaces["general"] = gym.spaces.Box(
        -1.0, 1.0, (140,), np.float32
    )
    data["observation_space"] = new_space
    if args.curriculum:
        args.terminal_exploration = True
    if args.terminal_exploration:
        args.terminal = True
        data["policy_class"] = AttackPermission.TerminalExplorationPolicy

    expanded = []
    for key, value in list(params["policy"].items()):
        if key.endswith("general_encoder.net.0.weight"):
            if value.shape != (256, 139):
                raise ValueError(f"Unexpected first-layer shape for {key}")
            params["policy"][key] = torch.nn.functional.pad(value, (0, 1))
            expanded.append(key)
    if not expanded:
        raise RuntimeError("No general encoder weights were expanded")
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
    save_to_zip_file(
        target, data=data, params=params, pytorch_variables=variables
    )
    del data, params, variables

    old = Train.PausingPPO.load(args.source, device="cpu")
    new = Train.PausingPPO.load(target, device="cpu")
    if args.terminal_exploration:
        new.policy.attack_temperature = (
            AttackPermission.EXPLORATION_INITIAL_TEMPERATURE
        )
    old.policy.set_training_mode(False)
    new.policy.set_training_mode(False)
    torch.manual_seed(6027)
    max_probability_error = 0.0
    max_value_error = 0.0
    for _ in range(4):
        obs = {
            "general": torch.rand(2, 139) * 2 - 1,
            "spatial": torch.rand(2, *old_space["spatial"].shape),
        }
        old_dist = old.policy.get_distribution(obs)
        old_value = old.policy.predict_values(obs)
        for attack_cost in (0.0, 1.0):
            conditioned = {
                **obs,
                "general": torch.cat(
                    [obs["general"], torch.full((2, 1), attack_cost)], dim=1
                ),
            }
            new_dist = new.policy.get_distribution(conditioned)
            for before, after in zip(
                old_dist.distribution, new_dist.distribution
            ):
                error = (before.probs - after.probs).abs().max().item()
                if attack_cost == 0.0:
                    max_probability_error = max(max_probability_error, error)
                    torch.testing.assert_close(before.probs, after.probs)
            new_value = new.policy.predict_values(conditioned)
            error = (old_value - new_value).abs().max().item()
            max_value_error = max(max_value_error, error)
            torch.testing.assert_close(old_value, new_value)

    os.environ.pop("PPO_ATTACK_PERMISSION", None)
    os.environ.pop("PPO_ATTACK_PENALTY", None)
    os.environ["PPO_ATTACK_TERMINAL" if args.terminal else "PPO_ATTACK_PENALTY"] = "1"
    if args.terminal_exploration:
        os.environ["PPO_ATTACK_EXPLORATION"] = "1"
    if args.curriculum:
        os.environ["PPO_ATTACK_CURRICULUM"] = "1"
    env = CasualtiesEnv.Env()
    Train.ValidateOrWriteProtocolManifest(args.destination, env, resuming=False)
    Train.ValidateOrWriteInterventionManifest(args.destination, target)
    Train.ValidateOrWriteProtocolManifest(args.destination, env, resuming=True)
    env.close()

    with args.source.open("rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    report = {
        "source": str(args.source.resolve()),
        "source_sha256": source_hash,
        "expanded_weights": expanded,
        "optimizer_tensors_expanded": optimizer_tensors_expanded,
        "optimizer_moments_preserved": True,
        "general_shape": [140],
        "spatial_shape": list(old_space["spatial"].shape),
        "synthetic_observations_tested_per_cost": 8,
        "costs_tested": [0.0, 1.0],
        "max_initial_action_probability_error": max_probability_error,
        "max_initial_value_error": max_value_error,
        "attack_mask_present": False,
        "attack_contract": (
            "adaptive_terminal_attack_curriculum"
            if args.curriculum
            else "terminal_violation_with_annealed_logit_temperature"
            if args.terminal_exploration
            else "terminal_violation" if args.terminal
            else "per_swing_0.001"
        ),
        "success_window": "starts fresh",
    }
    (args.destination / "surgery_validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
