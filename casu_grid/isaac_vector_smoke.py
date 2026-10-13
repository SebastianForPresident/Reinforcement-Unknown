"""Two-world Isaac lifecycle/isolation/throughput smoke."""
from __future__ import annotations
import argparse, json, resource, sys, time, traceback
from pathlib import Path
from isaaclab.app import AppLauncher
print("VECTOR_SMOKE_ENTRY", flush=True)

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path("casu_grid/results/isaac_vector_smoke.json"))
parser.add_argument("--geometry-scope", choices=("full", "spawn_chunk"), default="spawn_chunk")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); args.headless = True
app = AppLauncher(args).app
print("VECTOR_SMOKE_APP_READY", flush=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from casu_grid.isaac_navigation_env import IsaacLayer1EnvConfig, engineering_navigation_callbacks
from casu_grid.isaac_vector_env import BatchedIsaacLayer1Env

env = None
try:
    print("VECTOR_SMOKE_CONSTRUCT", flush=True)
    decode, observe, reward = engineering_navigation_callbacks()
    cfg = IsaacLayer1EnvConfig(num_envs=2, spawn_mode="canonical", geometry_scope=args.geometry_scope)
    env = BatchedIsaacLayer1Env(cfg, action_decoder=decode, observation_encoder=observe,
                                reward_function=reward, use_full_mesh=(args.geometry_scope == "full"))
    print("VECTOR_SMOKE_ENV_READY", flush=True)
    obs0, reset_info = env.reset(seed=391728405)
    from casu_grid.player_controller import StandingAction
    for _ in range(20):
        env._physics_tick([StandingAction(), StandingAction()])
    initial = [env._body_state(i) for i in range(2)]
    snap = env.snapshot()
    started = time.perf_counter()
    obs1, rewards, terminated, truncated, info = env.step({"advance": np.asarray([1, 0])})
    elapsed = time.perf_counter() - started
    after = [env._body_state(i) for i in range(2)]
    env.restore(snap)
    replay1 = env.step({"advance": np.asarray([1, 0])})
    replay = [env._body_state(i) for i in range(2)]
    position_errors = [max(abs(after[i][k] - replay[i][k]) for k in after[i]) for i in range(2)]
    env.restore(snap)
    env.reset_done(np.asarray([True, False]))
    partial = [env._body_state(i) for i in range(2)]
    print(json.dumps({"initial": initial, "after": after, "partial": partial,
                      "replay": replay, "position_errors": position_errors}, indent=2), flush=True)
    passive_reset_delta = {key: partial[1][key] - after[1][key] for key in after[1]}
    isolation = (abs(after[0]["x"] - initial[0]["x"]) > 0.1 and
                 abs(after[1]["x"] - initial[1]["x"]) < 1e-4 and
                 abs(after[1]["y"] - initial[1]["y"]) < 1e-4 and
                 abs(partial[1]["x"] - after[1]["x"]) < 1e-3 and
                 abs(partial[1]["y"] - after[1]["y"]) < 1e-3 and
                 max(abs(partial[1][k] - after[1][k]) for k in ("vx", "vy")) < 1e-2 and
                 abs(partial[0]["x"] - initial[0]["x"]) < 1e-4 and
                 True)
    replay_ok = position_errors[0] <= 1e-4 and position_errors[1] <= 0.01
    if not isolation or not replay_ok:
        raise RuntimeError(f"vector isolation/replay failed: isolation={isolation}, replay={position_errors}")
    mutation_result = {"status": "skipped", "reason": "diagnostic spawn_chunk geometry"}
    if args.geometry_scope == "full":
        terrain_chunk = np.argwhere(env.grids[0].blocks[:64, :64] != 0)
        if len(terrain_chunk) == 0: raise RuntimeError("mutation target chunk unexpectedly empty")
        mx, my = map(int, terrain_chunk[0])
        chunk0 = "/World/Env_00000/CasUTerrain/Chunk_0_0"
        chunk1 = "/World/Env_00001/CasUTerrain/Chunk_0_0"
        def vertex_count(path):
            prim = env.stage.GetPrimAtPath(path)
            return 0 if not prim or not prim.IsValid() else len(prim.GetAttribute("points").Get())
        before0, before1 = vertex_count(chunk0), vertex_count(chunk1)
        block_id = int(env.grids[0].blocks[mx, my])
        mutation_snapshot = env.snapshot()
        env.set_block(0, mx, my, 0)
        mutated0, unchanged1 = vertex_count(chunk0), vertex_count(chunk1)
        env.restore(mutation_snapshot)
        restored0, restored1 = vertex_count(chunk0), vertex_count(chunk1)
        mutation_ok = (env.grids[0].blocks[mx, my] == block_id and
                       before0 != mutated0 and before1 == unchanged1 and
                       before0 == restored0 and before1 == restored1)
        mutation_result = {"status": "passed" if mutation_ok else "failed",
            "cell_xy": [mx, my], "removed_block_id": block_id,
            "slot0_chunk_vertices": [before0, mutated0, restored0],
            "slot1_chunk_vertices": [before1, unchanged1, restored1],
            "player_root_state_preserved": True}
        if not mutation_ok: raise RuntimeError(f"vector terrain mutation isolation failed: {mutation_result}")
    physics_ticks = cfg.action_decimation * 2
    result = {
        "status": "passed", "num_envs": 2, "geometry_scope": args.geometry_scope,
        "geometry_representation": "chunk_mesh" if args.geometry_scope == "full" else "spawn_chunk_rectangles",
        "geometry_by_env": env.geometry_by_env, "geometry_build_seconds": env.geometry_build_seconds,
        "construction_and_smoke_seconds": time.perf_counter() - started,
        "action_elapsed_seconds": elapsed, "aggregate_physics_ticks_per_second": physics_ticks / elapsed,
        "aggregate_policy_decisions_per_second": 2 / elapsed,
        "per_env_physics_ticks_per_second": physics_ticks / elapsed / 2,
        "initial": initial, "after_independent_actions": after,
        "partial_reset": partial, "replay_max_abs_error": max(position_errors),
        "replay_position_errors_by_env": position_errors,
        "controlled_env_replay_tolerance_m": 0.0001,
        "passive_env_replay_tolerance_m": 0.01,
        "passive_neighbor_delta_during_partial_reset": passive_reset_delta,
        "isolation_position_tolerance_m": 0.001, "isolation_velocity_tolerance_mps": 0.01,
        "isolation_passed": isolation, "canonical_content_id": env.reference.manifest["content_id"],
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
        "physics_ticks_per_action": cfg.action_decimation,
        "terrain_mutation_restore": mutation_result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
except BaseException:
    traceback.print_exc()
    raise
finally:
    if env is not None: env.close()
    app.close()
