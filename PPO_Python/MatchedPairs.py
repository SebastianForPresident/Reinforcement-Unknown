"""Matched-checkpoint evaluation on reproducible CB1 cave seeds."""

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

import CasualtiesEnv


RESULT_FIELDS = [
    "speed", "checkpoint", "checkpoint_path", "seed", "action_seed",
    "deterministic", "max_steps", "steps", "sim_seconds", "wall_seconds",
    "completed", "terminated", "truncated", "final_progress",
    "best_depth", "border_steps",
]


def _load_config(path):
    config_path = Path(path).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    models = config.get("checkpoints", {})
    seeds = config.get("seeds", [])
    if not models:
        raise ValueError("matched evaluation requires at least one checkpoint")
    if len(seeds) != 10 or len(set(seeds)) != 10:
        raise ValueError("matched evaluation requires exactly ten unique seeds")
    config["seeds"] = [CasualtiesEnv.NormalizeWorldSeed(seed) for seed in seeds]
    config["max_steps"] = int(config.get("max_steps", 54_000))
    config["deterministic"] = bool(config.get("deterministic", False))
    config["action_seed"] = int(config.get("action_seed", 20260815))
    if config["max_steps"] <= 0:
        raise ValueError("max_steps must be positive")
    config["checkpoints"] = {
        name: str((config_path.parent / model_path).resolve())
        if not Path(model_path).is_absolute() else str(Path(model_path).resolve())
        for name, model_path in models.items()
    }
    return config_path, config


def _completed_keys(output_path):
    if not output_path.is_file():
        return set()
    with output_path.open(newline="", encoding="utf-8") as handle:
        return {
            (row["speed"], row["checkpoint"], int(row["seed"]))
            for row in csv.DictReader(handle)
        }


def _append_result(output_path, result):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(result)
        handle.flush()


def _trial_order(model_names, seeds):
    names = list(model_names)
    for index, seed in enumerate(seeds):
        order = names if index % 2 == 0 else list(reversed(names))
        for name in order:
            yield name, seed


def Run(env, config_path, speed_label):
    """Evaluate frozen policies on the same ten seeds at one game speed."""
    config_path, config = _load_config(config_path)
    output_value = config.get("output", "matched_pairs_results.csv")
    output_path = Path(output_value)
    if not output_path.is_absolute():
        output_path = (config_path.parent / output_path).resolve()
    result_speed = str(config.get("result_speed", speed_label))

    models = {
        name: PPO.load(path, env=env, device="auto")
        for name, path in config["checkpoints"].items()
    }
    completed = _completed_keys(output_path)

    try:
        for model_name, world_seed in _trial_order(models, config["seeds"]):
            key = (result_speed, model_name, world_seed)
            if key in completed:
                print(f"Skipping completed matched trial: {key}")
                continue

            action_seed = config["action_seed"] + world_seed
            random.seed(action_seed)
            np.random.seed(action_seed % (2 ** 32))
            torch.manual_seed(action_seed)

            obs, reset_info = env.reset(options={"world_seed": world_seed})
            if int(reset_info["world_seed"]) != world_seed:
                raise RuntimeError("Unity applied the wrong matched-pair seed")

            started = time.monotonic()
            terminated = truncated = completed_episode = False
            border_steps = 0
            final_progress = best_depth = 0.0
            steps = 0
            for steps in range(1, config["max_steps"] + 1):
                action, _ = models[model_name].predict(
                    obs, deterministic=config["deterministic"]
                )
                obs, _, terminated, truncated, info = env.step(action)
                raw = env.latest_obs
                final_progress = float(raw["LayerProgress"])
                best_depth = max(best_depth, float(raw["BestLayerDepth"]))
                x = int(raw["PlayerTilePosition"]["X"])
                width = int(raw["WorldDimensions"]["X"])
                border_steps += int(width > 0 and (x <= 63 or x >= width - 64))
                if terminated or truncated:
                    completed_episode = bool(info.get("completion", 0.0) > 0.0)
                    break

            result = {
                "speed": result_speed,
                "checkpoint": model_name,
                "checkpoint_path": config["checkpoints"][model_name],
                "seed": world_seed,
                "action_seed": action_seed,
                "deterministic": config["deterministic"],
                "max_steps": config["max_steps"],
                "steps": steps,
                "sim_seconds": steps * 0.2,
                "wall_seconds": time.monotonic() - started,
                "completed": completed_episode,
                "terminated": terminated,
                "truncated": truncated,
                "final_progress": final_progress,
                "best_depth": best_depth,
                "border_steps": border_steps,
            }
            _append_result(output_path, result)
            completed.add(key)
            print(f"Matched trial complete: {result}")
    finally:
        env.close()
