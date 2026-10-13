"""Headless construction, reset, action, snapshot, and replay smoke for NavEnv."""
from __future__ import annotations

import argparse
import json
import resource
from pathlib import Path
import sys
import time
import traceback
from dataclasses import replace

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path("casu_grid/results/isaac_navigation_lifecycle_smoke.json"))
parser.add_argument("--spawn-mode", choices=("canonical", "fixture"), default="canonical")
parser.add_argument("--episode-steps", type=int, default=40)
parser.add_argument("--use-cuda", action="store_true")
parser.add_argument("--geometry-scope", choices=("full", "spawn_chunk"), default="full")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
launcher = AppLauncher(args)
app = launcher.app

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from casu_grid.workbench.engineering_smoke import ACTION, MODEL, OBSERVATION, PPO, REWARD
from casu_grid.workbench.ppo import PPOTrainer, write_metrics
from casu_grid.isaac_navigation_env import (
    IsaacLayer1EnvConfig, IsaacLayer1NavigationEnv,
    engineering_navigation_callbacks,
)


def state(env):
    return env._body_state()


def action(move=1, jump=0, crouch=0):
    return {"move_x": np.asarray([move]), "jump": np.asarray([jump]),
            "crouch": np.asarray([crouch])}


def main():
    env = None
    trainer = None
    start = time.perf_counter()
    try:
        decode, observe, reward = engineering_navigation_callbacks()
        env = IsaacLayer1NavigationEnv(IsaacLayer1EnvConfig(
            spawn_mode=args.spawn_mode, episode_length_steps=args.episode_steps,
            geometry_scope=args.geometry_scope,
            use_cuda=args.use_cuda, rounded_player_collider=True), action_decoder=decode,
            observation_encoder=observe, reward_function=reward)
        observation0, reset_info = env.reset(seed=391728405)
        mutation_result = {"status": "skipped", "reason": "requires full geometry scope"}
        if args.geometry_scope == "full":
            mutation_chunk_xy = (0, 0)
            region = env.grid.blocks[:64, :64]
            mutation_cells = np.argwhere(region != 0)
            if mutation_cells.size == 0:
                raise RuntimeError("canonical mutation test chunk has no solid cell")
            mutation_x, mutation_y = map(int, mutation_cells[0])
            mutation_snapshot = env.snapshot()
            mutation_chunk_path = f"/World/CasUTerrain/Chunk_{mutation_chunk_xy[0]}_{mutation_chunk_xy[1]}"
            initial_chunk_children = sorted(str(child.GetPath()) for child in
                                            env.stage.GetPrimAtPath(mutation_chunk_path).GetChildren())
            mutation_block_id = env.grid.get_block(mutation_x, mutation_y)
            env.set_block(mutation_x, mutation_y, 0)
            mutated_chunk_children = sorted(str(child.GetPath()) for child in
                                            env.stage.GetPrimAtPath(mutation_chunk_path).GetChildren())
            env.restore(mutation_snapshot)
            restored_chunk_children = sorted(str(child.GetPath()) for child in
                                             env.stage.GetPrimAtPath(mutation_chunk_path).GetChildren())
            mutation_restore_ok = (
                env.grid.get_block(mutation_x, mutation_y) == mutation_block_id and
                initial_chunk_children == restored_chunk_children and
                initial_chunk_children != mutated_chunk_children)
            if not mutation_restore_ok:
                raise RuntimeError("terrain mutation collider restore did not reproduce canonical chunk")
            mutation_result = {
                "status": "passed", "cell_xy": [mutation_x, mutation_y],
                "initial_block_id": mutation_block_id,
                "initial_chunk_collider_count": len(initial_chunk_children),
                "mutated_chunk_collider_count": len(mutated_chunk_children),
                "restored_chunk_collider_count": len(restored_chunk_children),
            }
            env.reset(seed=391728405)
        first0 = state(env)
        from pxr import UsdPhysics
        standing_collision = UsdPhysics.CollisionAPI(env._standing_collider_prim).GetCollisionEnabledAttr()
        crouch_collision = UsdPhysics.CollisionAPI(env._crouch_collider_prim).GetCollisionEnabledAttr()
        crouch_size_before = [bool(standing_collision.Get()), bool(crouch_collision.Get())]
        env.step(action(1, crouch=1))
        crouch_size_during = [bool(standing_collision.Get()), bool(crouch_collision.Get())]
        env.controller.state.crouch_amount = 0.0
        env.step(action(1, crouch=0))
        crouch_size_recovered = [bool(standing_collision.Get()), bool(crouch_collision.Get())]
        crouch_shape_ok = crouch_size_before == [True, False] and crouch_size_during[0] is False and crouch_size_during[1] is True and crouch_size_recovered[0] is True and crouch_size_recovered[1] is False
        if not crouch_shape_ok:
            raise RuntimeError(f"player collider did not shrink and recover during crouch input: {crouch_size_before}, {crouch_size_during}, {crouch_size_recovered}, amount={env.controller.state.crouch_amount}")
        tape_a = [action(1), action(2), action(2), action(1)]
        states_a = []
        for sample in tape_a:
            observation, reward, terminated, truncated, info = env.step(sample)
            states_a.append(state(env))
        snapshot = env.snapshot()
        grid_tick_at_snapshot = env.grid.tick
        branch_actions = [action(2), action(1), action(0)]
        branch_a = []
        for sample in branch_actions:
            env.step(sample)
            branch_a.append(state(env))
        env.restore(snapshot)
        post_restore_state = state(env)
        branch_b = []
        for sample in branch_actions:
            env.step(sample)
            branch_b.append(state(env))
        replay_error = max(abs(branch_a[-1][key] - branch_b[-1][key])
                           for key in branch_a[-1])
        # Restored-vs-restored continuations start from identical (cold) PhysX
        # contact caches, so they must agree exactly; the original (warm-cache)
        # branch may differ slightly after contact-resolving motion.
        env.restore(snapshot)
        branch_c = []
        for sample in branch_actions:
            env.step(sample)
            branch_c.append(state(env))
        restored_replay_error = max(abs(b_state[key] - c_state[key])
                                    for b_state, c_state in zip(branch_b, branch_c)
                                    for key in b_state)
        env.reset(seed=391728405)
        first1 = state(env)
        reset_error = max(abs(first0[key] - first1[key]) for key in first0)

        ppo_metrics = []
        checkpoint_path = args.output.with_name(args.output.stem + "_engineering_ppo.pt")
        metrics_path = args.output.with_name(args.output.stem + "_engineering_ppo.jsonl")
        smoke_ppo = replace(PPO, rollout_steps=12, minibatch_size=12,
                            epochs=2, target_kl=None)
        tensorboard_dir = args.output.parent / (args.output.stem + "_tensorboard")
        trainer = PPOTrainer(env, ACTION, OBSERVATION, MODEL, smoke_ppo,
                             device="cpu", seed=391728405, log_dir=tensorboard_dir)
        first_update = trainer.train_update(rollout_steps=12)
        trainer.save(checkpoint_path, metadata={
            "purpose": "engineering-only canonical Layer 1 PPO pipeline smoke",
            "not_user_approved_timmy_experiment": True,
            "canonical_content_id": env.reference.manifest["content_id"],
        })
        restored = PPOTrainer(env, ACTION, OBSERVATION, MODEL, smoke_ppo,
                              device="cpu", seed=19, log_dir=tensorboard_dir)
        restored.load(checkpoint_path)
        second_update = restored.train_update(rollout_steps=12)
        for metrics in (first_update, second_update):
            write_metrics(metrics_path, metrics)
        ppo_metrics = [first_update, second_update]
        metrics = {
            "schema": "casu.isaac-navigation-lifecycle-smoke.v1",
            "status": "passed" if (replay_error <= 1e-3 and restored_replay_error == 0.0
                                   and reset_error <= 1e-4) else "failed",
            "canonical_fixture_id": env.reference.manifest["fixture_id"],
            "canonical_content_id": env.reference.manifest["content_id"],
            "spawn_mode": args.spawn_mode,
            "initial_observation_keys": sorted(observation0),
            "reset_info": reset_info,
            "terrain_mutation_restore": mutation_result,
            "geometry": env.geometry_info,
            "geometry_scope": args.geometry_scope,
            "geometry_total_seconds": env.geometry_build_seconds,
            "physics_dt_s": env.cfg.physics_dt,
            "physics_ticks_per_action": env.cfg.action_decimation,
            "rounded_player_collider": env.cfg.rounded_player_collider,
            "crouch_shape_probe": {"status": "passed", "scale_standing": crouch_size_before,
                                   "scale_crouched": crouch_size_during,
                                   "scale_recovering": crouch_size_recovered},
            "physics_ticks_elapsed": env.physics_ticks,
            "fluid_scheduler_ticks": env.grid.tick,
            "fluid_tick_delta_during_branch_a": 3 * env.cfg.action_decimation,
            "fluid_rng_decisions_consumed": env.grid.rng.position,
            "fluid_decision_contract": "seeded PCG64 replay tape; Unity RNG parity not claimed",
            "liquid_affect_phase": env.cfg.liquid_affect_phase,
            "fluid_grid_matches_physics_tick_count": env.grid.tick == env.physics_ticks,
            "snapshot_grid_tick": grid_tick_at_snapshot,
            "action_tape_states": states_a,
            "snapshot_replay_max_abs_state_error": replay_error,
            "snapshot_replay_note": "original (warm PhysX contact cache) branch versus restored branch",
            "restored_vs_restored_max_abs_state_error": restored_replay_error,
            "snapshot_player_state": snapshot["player_root_state_wxyz"].tolist(),
            "snapshot_body_state": {
                "x": float(snapshot["player_root_state_wxyz"][0]),
                "y": float(snapshot["player_root_state_wxyz"][2]),
                "vx": float(snapshot["player_root_state_wxyz"][7]),
                "vy": float(snapshot["player_root_state_wxyz"][9]),
            },
            "post_restore_body_state": post_restore_state,
            "snapshot_replay_branch_a": branch_a,
            "snapshot_replay_branch_b": branch_b,
            "reset_max_abs_state_error": reset_error,
            "reset_restored_fluid_tick": env.grid.tick,
            "final_player_state": state(env),
            "engineering_ppo": {
                "updates": ppo_metrics,
                "checkpoint": str(checkpoint_path),
                "metrics_jsonl": str(metrics_path),
                "tensorboard_dir": str(tensorboard_dir),
                "checkpoint_restored_and_training_continued": True,
                "research_status": "temporary engineering-only contract; not approved Timmy experiment",
            },
            "elapsed_seconds": time.perf_counter() - start,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2) + "\n")
        print("ISAAC_NAVIGATION_SMOKE", metrics["status"], args.output, flush=True)
        print("ISAAC_NAVIGATION_LIFECYCLE_ERRORS", reset_error, replay_error, restored_replay_error, flush=True)
        if metrics["status"] != "passed":
            raise RuntimeError("Isaac lifecycle replay or reset exceeded tolerance")
    finally:
        if trainer is not None:
            trainer.close()
            trainer.env = None
        if env is not None:
            env.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    app.close()
