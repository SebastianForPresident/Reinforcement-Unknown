"""Held-out policy evaluation for the XY1 navigation experiments.

    python -m casu_grid.workbench.evaluate --config casu_grid/workbench/experiments/xy1_point_navigation_mini.py \
        --policy untrained --policy checkpoint=runs/x/checkpoints/update_000100.pt --output eval.json

Every policy is evaluated under *identical* conditions, which is what makes the numbers comparable:

* held-out worlds: the terrain of world ``i`` is generated from ``391728405 + world_offset + i`` with an offset far outside the range used
  for training (training uses offsets 0..63), so the policy has never seen these terrains;
* identical goals: round ``r`` resets every world with ``reset(seed = eval_seed + r)``, which draws the goals from that seed only; only the *first*
  episode of each world per round is scored, so what a policy does cannot change which goals the next policies see;
* a fixed horizon (``--horizon`` decisions) after which an unfinished episode counts as a failure ("timeout"): success rate is therefore
  "reached the goal within H decisions", which is comparable between checkpoints and is not biased by censoring as training statistics are;
* the policy is either the arg-max of every head (deterministic) or sampled from the learned distribution (stochastic); both are reported.

Nothing here changes training: the environment, reward, termination rule (arrival within one tile) and controller are the training ones;
only the episode cap differs from the 100,000-decision training ceiling, and it is recorded in the result.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, subprocess, sys, time, traceback
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--config", required=True, type=Path)
p.add_argument("--policy", action="append", required=True,
               help="'untrained' (the freshly initialised network of --init-seed) or 'checkpoint=PATH' / a path to a .pt checkpoint; repeatable")
p.add_argument("--output", required=True, type=Path)
p.add_argument("--worlds", type=int, default=16, help="held-out worlds evaluated in parallel")
p.add_argument("--rounds", type=int, default=3, help="goal draws per world (episodes per world and policy mode = rounds)")
p.add_argument("--horizon", type=int, default=1500, help="decisions allowed per episode")
p.add_argument("--eval-seed", type=int, default=20261014, help="seed of the goal draws")
p.add_argument("--world-offset", type=int, default=1_000_000,
               help="world index of the first evaluated world; the terrain and fluid seeds are 391728405 + offset + i (training uses offsets 0..63)")
p.add_argument("--init-seed", type=int, default=7, help="seed that initialised the 'untrained' network (match the training run's seed)")
p.add_argument("--modes", default="deterministic,stochastic")
p.add_argument("--train-device", default=os.environ.get("CASU_TRAIN_DEVICE", "auto"))
AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); args.headless = True
app = AppLauncher(args).app
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import importlib.util
import torch
from gymnasium import spaces

from casu_grid.isaac_navigation_env import IsaacLayer1EnvConfig
from casu_grid.workbench.ppo import PPOTrainer
from casu_grid.workbench.contracts import validate_contracts
from casu_grid.workbench.runtime_validation import checked_callbacks


def wilson(successes: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    phat = successes / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def summarize(episodes: list[dict]) -> dict:
    n = len(episodes)
    wins = [e for e in episodes if e["outcome"] == "arrival"]
    mean = lambda xs: (float(np.mean(xs)) if len(xs) else None)
    return {
        "episodes": n, "arrivals": len(wins), "timeouts": n - len(wins),
        "success_rate": (len(wins) / n) if n else None, "success_rate_ci95": wilson(len(wins), n),
        "decisions_to_arrival_median": (float(np.median([e["decisions"] for e in wins])) if wins else None),
        "decisions_to_arrival_mean": mean([e["decisions"] for e in wins]),
        "mean_initial_distance": mean([e["initial_distance"] for e in episodes]),
        "mean_best_distance": mean([e["best_distance"] for e in episodes]),
        "mean_final_distance": mean([e["final_distance"] for e in episodes]),
        "mean_progress": mean([1.0 - e["best_distance"] / e["initial_distance"] for e in episodes if e["initial_distance"] > 0]),
        "mean_blocks_destroyed": mean([e["blocks_destroyed"] for e in episodes]),
        "mean_return": mean([e["return"] for e in episodes]),
    }


def main():
    spec = importlib.util.spec_from_file_location("research_config", args.config)
    config = importlib.util.module_from_spec(spec); spec.loader.exec_module(config)
    validate_contracts(config.ACTION, config.OBSERVATION, config.REWARD, config.MODEL, config.PPO)
    if not bool(getattr(config, "ENVIRONMENT", {}).get("navigation_mode", False)):
        raise ValueError("evaluate.py scores XY1 navigation experiments only")
    if os.environ.get("CASU_LEGACY_MINI_SPAWN", "0") == "1":
        raise ValueError("CASU_LEGACY_MINI_SPAWN=1 is not allowed for evaluation")
    callbacks = config.callbacks() if hasattr(config, "callbacks") else (None, None, None)
    from casu_grid.workbench.components import resolve_reward_reference
    callbacks = resolve_reward_reference(config, args.config, callbacks)
    callbacks = checked_callbacks(callbacks, config.OBSERVATION, config.REWARD)
    environment = getattr(config, "ENVIRONMENT", {})
    env_config = IsaacLayer1EnvConfig(
        num_envs=args.worlds, spawn_mode="canonical", navigation_mode=True,
        world_variant=environment.get("world_variant", "canonical"),
        observation_window_tiles=tuple(environment.get("observation_window_tiles", (85, 49))),
        player_collider=environment.get("player_collider", "compound"),
        terminate_out_of_world=bool(environment.get("terminate_out_of_world", False)), episode_length_steps=args.horizon)
    from casu_grid.isaac_vector_env import BatchedIsaacLayer1Env
    env = trainer = None
    started = time.time()
    try:
        env = BatchedIsaacLayer1Env(env_config, action_decoder=callbacks[0], observation_encoder=callbacks[1],
                                    reward_function=callbacks[2], terrain_mode="independent", world_offset=args.world_offset)
        env.action_space = spaces.Dict({h.name: spaces.Discrete(h.size) for h in config.ACTION.heads})
        device = "cuda" if (args.train_device == "auto" and torch.cuda.is_available()) else ("cpu" if args.train_device == "auto" else args.train_device)
        trainer = PPOTrainer(env, config.ACTION, config.OBSERVATION, config.MODEL, config.PPO, device=device, seed=args.init_seed, log_dir=None)
        if not trainer._use_local_rollout():
            raise RuntimeError("worker-local rollouts are required for evaluation (CUDA trainer, XY1 stock observation)")
        initial_weights = {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
        results = []
        policies = []
        for spec_text in args.policy:
            if spec_text == "untrained":
                policies.append(("untrained", None))
            else:
                path = Path(spec_text.split("=", 1)[1] if spec_text.startswith("checkpoint=") else spec_text).expanduser().resolve()
                policies.append((path.stem if path.parent.name != "checkpoints" or path.stem != "initial" else "initial", path))
        modes = [m for m in args.modes.split(",") if m]
        total = len(policies) * len(modes) * args.rounds
        done = 0
        for label, path in policies:
            if path is None:
                trainer.model.load_state_dict(initial_weights)
                info = {"kind": "untrained", "init_seed": args.init_seed, "update_count": 0, "total_steps": 0}
            else:
                loaded = trainer.load_policy(path)
                info = {"kind": "checkpoint", "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "update_count": loaded["update_count"], "total_steps": loaded["total_steps"]}
            weights = {k: v.detach().cpu().numpy() for k, v in trainer.model.state_dict().items()}
            for mode in modes:
                episodes = []
                for r in range(args.rounds):
                    env.reset(seed=args.eval_seed + r, observe=False)
                    res = env.local_rollout_run(weights, args.horizon, [args.eval_seed * 7919 + r * 131 + w for w in range(getattr(env, "num_workers", 1))],
                                                mode == "deterministic", np.zeros(args.worlds, np.float32), np.zeros(args.worlds, np.int64))
                    nav = res["nav_episodes"]
                    seen = set()
                    for i, world in enumerate(nav["world"]):
                        if world in seen:
                            continue
                        seen.add(world)          # first episode of each world in this round
                        episodes.append({"round": r, "world": int(world), "outcome": nav["reason"][i], "decisions": int(nav["length"][i]),
                                         "initial_distance": nav["initial_distance"][i], "best_distance": nav["best_distance"][i],
                                         "final_distance": nav["final_distance"][i], "blocks_destroyed": nav["blocks_destroyed"][i],
                                         "return": nav["return"][i]})
                    done += 1
                    print(f"CASU_EVAL_PROGRESS {done}/{total} policy={label} mode={mode} round={r}", flush=True)
                results.append({"policy": label, "mode": mode, "checkpoint": info, "summary": summarize(episodes), "episodes": episodes})
        output = {
            "schema": "casu.xy1-evaluation.v1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": str(args.config.resolve()), "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "source_revision": revision,
            "protocol": {"worlds": args.worlds, "rounds": args.rounds, "horizon_decisions": args.horizon, "eval_seed": args.eval_seed,
                         "world_offset": args.world_offset, "terrain_seed_first": 391728405 + args.world_offset, "training_terrain_seed_first": 391728405,
                         "held_out": args.world_offset >= 4096, "scored": "first episode of every world in every round",
                         "success": "arrival within one tile of the goal within the horizon"},
            "wall_seconds": time.time() - started, "results": results}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(output, indent=2) + "\n")
        os.replace(tmp, args.output)
        print("CASU_EVAL_DONE " + str(args.output), flush=True)
    finally:
        for close in ((trainer.close if trainer is not None else None), (env.close if env is not None else None)):
            if close is not None:
                try:
                    close()
                except BaseException:
                    traceback.print_exc()


try:
    main()
except BaseException:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    raise SystemExit(1)
