"""Create a separate, verified PPO_6 checkpoint; never overwrite a run."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.save_util import load_from_zip_file, save_to_zip_file

import AttackPermission
import CasualtiesEnv
import Train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError(args.destination)
    if Train.CheckpointStep(args.source) != 51843364:
        raise ValueError("Expected the agreed 51,843,364-step source")
    torch.set_num_threads(2)
    data, params, variables = load_from_zip_file(args.source, device="cpu")
    old_space = data["observation_space"]
    assert old_space["general"].shape == (139,)
    new_space = copy.deepcopy(old_space)
    new_space.spaces["general"] = gym.spaces.Box(-1, 1, (140,), np.float32)
    data["observation_space"] = new_space
    data["policy_class"] = AttackPermission.PermissionPolicy
    # Preserve every existing weight and Adam moment. Only the new input
    # column and its first/second moments begin at zero.
    expanded = []
    for key, value in list(params["policy"].items()):
        if key.endswith("general_encoder.net.0.weight"):
            assert value.shape == (256, 139)
            params["policy"][key] = torch.nn.functional.pad(value, (0, 1))
            expanded.append(key)
    assert expanded
    for state in params["policy.optimizer"]["state"].values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor) and value.shape == (256, 139):
                state[key] = torch.nn.functional.pad(value, (0, 1))
    data["_last_obs"] = None
    data["_last_original_obs"] = None
    args.destination.mkdir(parents=True)
    target = args.destination / args.source.name
    save_to_zip_file(target, data=data, params=params, pytorch_variables=variables)
    del data, params, variables

    old = Train.PausingPPO.load(args.source, device="cpu")
    new = Train.PausingPPO.load(target, device="cpu")
    old.policy.set_training_mode(False)
    new.policy.set_training_mode(False)
    torch.manual_seed(6026)
    max_error = 0.0
    for _ in range(4):
        obs = {"general": torch.rand(2, 139) * 2 - 1,
               "spatial": torch.rand(2, *old_space["spatial"].shape)}
        allowed = {**obs, "general": torch.cat([obs["general"], torch.zeros(2, 1)], 1)}
        forbidden = {**obs, "general": torch.cat([obs["general"], torch.ones(2, 1)], 1)}
        with torch.no_grad():
            before = old.policy.get_distribution(obs)
            before_probs = [d.probs.clone() for d in before.distribution]
            after = new.policy.get_distribution(allowed)
            for a, b in zip(before_probs, after.distribution):
                error = (a - b.probs).abs().max().item()
                max_error = max(max_error, error)
                torch.testing.assert_close(a, b.probs, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(old.policy.predict_values(obs), new.policy.predict_values(allowed), atol=1e-6, rtol=1e-5)
            dist = new.policy.get_distribution(forbidden)
            assert (dist.distribution[-1].probs[:, 1] == 0).all()
            actions, values, logs = new.policy(forbidden)
            assert (actions[:, 6] == 0).all()
            _, evaluated, entropy = new.policy.evaluate_actions(forbidden, actions)
            torch.testing.assert_close(logs, evaluated)
            assert torch.isfinite(entropy).all()
        # Verify masked PPO backward pass remains finite.
        _, log_prob, entropy = new.policy.evaluate_actions(forbidden, actions)
        new.policy.optimizer.zero_grad()
        (-log_prob.mean() - .01 * entropy.mean()).backward()
        assert all(torch.isfinite(p.grad).all() for p in new.policy.parameters() if p.grad is not None)

    os.environ["PPO_ATTACK_PERMISSION"] = "1"
    env = CasualtiesEnv.Env()
    Train.ValidateOrWriteProtocolManifest(args.destination, env, resuming=False)
    Train.ValidateOrWriteInterventionManifest(args.destination, target)
    Train.ValidateOrWriteProtocolManifest(args.destination, env, resuming=True)
    env.close()
    report = {"source": str(args.source.resolve()),
              "source_sha256": hashlib.file_digest(args.source.open("rb"), "sha256").hexdigest(),
              "expanded_weights": expanded, "optimizer_moments_preserved": True,
              "general_shape": [140], "spatial_shape": list(old_space["spatial"].shape),
              "synthetic_observations_tested": 8, "max_allowed_probability_error": max_error,
              "masked_sampling_evaluation_and_backward": "passed",
              "success_window": "starts fresh; mixed permissions are a new task distribution"}
    (args.destination / "surgery_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
