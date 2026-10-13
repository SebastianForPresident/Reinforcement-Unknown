"""Validation of the multi-process environment (``ShardedBatchedEnv``): API parity, isolation,
snapshot/restore, partial reset, doors, per-world terrain edits, callbacks and worker-failure handling.

    python casu_grid/isaac_sharded_smoke.py --workers 2 --num-envs 4 --output casu_grid/results/isaac_sharded_validation.json

The launching process needs no Kit application of its own; every worker starts one.
"""
from __future__ import annotations
import argparse, json, os, signal, sys, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from casu_grid.isaac_navigation_env import IsaacLayer1EnvConfig, engineering_navigation_callbacks
from casu_grid.isaac_sharded_env import ShardedBatchedEnv, WorkerFailure

p = argparse.ArgumentParser()
p.add_argument("--workers", type=int, default=2)
p.add_argument("--num-envs", type=int, default=4)
p.add_argument("--output", type=Path, default=Path("casu_grid/results/isaac_sharded_validation.json"))
args = p.parse_args()
checks = {}


def move(n, moves, jumps=None):
    return {"move_x": np.asarray(moves), "jump": np.asarray(jumps if jumps is not None else [0] * n),
            "crouch": np.zeros(n, dtype=int)}


def states(env):
    return [env._body_state(i) for i in range(env.count)]


def err(a, b):
    return max(abs(a[k] - b[k]) for k in a)


def alive(pids):
    out = []
    for pid in pids:
        try:
            os.kill(pid, 0)
            # a zombie is reaped by Popen.wait in the parent; treat existing directory as alive
            out.append(os.path.exists(f"/proc/{pid}") and open(f"/proc/{pid}/stat").read().split()[2] != "Z")
        except OSError:
            out.append(False)
    return out


def flat_cfg(n, **kw):
    return IsaacLayer1EnvConfig(num_envs=n, spawn_mode="fixture", fixture_grid_xy=(483, 1010),
                                num_workers=args.workers, **kw)


try:
    n = args.num_envs
    started = time.perf_counter()
    env = ShardedBatchedEnv(flat_cfg(n), terrain_mode="shared")
    checks["startup"] = {"status": "passed", "seconds": time.perf_counter() - started, "sizes": env.sizes,
                         "pids": env.worker_pids, "log_dir": str(env.log_dir)}
    obs, info = env.reset(seed=1)
    ok = obs["player_state"].shape == (n, 8) and info["canonical_content_id"] == env.reference.manifest["content_id"]
    for _ in range(3):
        obs, reward, term, trunc, step_info = env.step(move(n, [1] * n))
    ok = ok and reward.shape == (n,) and len(step_info["player"]) == n and step_info["physics_ticks"] == 10 * n
    checks["api_shapes"] = {"status": "passed" if ok else "failed"}

    base = env.snapshot()
    moves_a, jumps_a = ([2, 1, 0, 2] + [1] * n)[:n], ([0, 1, 0, 1] + [0] * n)[:n]
    for _ in range(4):
        env.step(move(n, moves_a, jumps_a))
    a_states = states(env)
    env.restore(base)
    for _ in range(4):
        env.step(move(n, [2] + [0] * (n - 1), [0] + [1] * (n - 1)))
    b_states = states(env)
    checks["world_isolation_across_workers"] = {
        "status": "passed" if err(a_states[0], b_states[0]) <= 1e-4 else "failed",
        "world0_error": err(a_states[0], b_states[0]),
        "worlds_differ": err(a_states[-1], a_states[0]) > 0.1}
    env.restore(base)
    for _ in range(4):
        env.step(move(n, moves_a, jumps_a))
    first = states(env)
    env.restore(base)
    for _ in range(4):
        env.step(move(n, moves_a, jumps_a))
    second = states(env)
    exact = max(err(x, y) for x, y in zip(first, second))
    checks["restore_continuation_exact"] = {"status": "passed" if exact == 0.0 else "failed", "max_abs_error": exact}

    before = states(env)
    mask = np.zeros(n, dtype=bool)
    mask[n - 1] = True           # last world lives in the last worker
    env.reset_done(mask)
    after = states(env)
    untouched = max(err(before[i], after[i]) for i in range(n - 1))
    checks["partial_reset_across_workers"] = {"status": "passed" if untouched <= 1e-6 and err(before[-1], after[-1]) > 0 else "failed",
                                               "untouched_worlds_error": untouched}
    try:
        env.set_block(0, 7, 372, 0)
        blocked = False
    except NotImplementedError:
        blocked = True
    checks["shared_terrain_edit_rejected"] = {"status": "passed" if blocked else "failed"}

    # callbacks (engineering navigation callbacks are closures of a factory) run inside the workers
    pids = list(env.worker_pids)
    env.close()
    checks["clean_shutdown"] = {"status": "passed" if not any(alive(pids)) else "failed", "alive": alive(pids)}

    decoder, encoder, reward_fn = engineering_navigation_callbacks()
    env = ShardedBatchedEnv(flat_cfg(n), terrain_mode="shared", action_decoder=decoder,
                            observation_encoder=encoder, reward_function=reward_fn)
    obs, _ = env.reset()
    ok = True
    for _ in range(3):
        obs, reward, term, trunc, step_info = env.step({"advance": np.ones(n, dtype=int)})
    ok = obs["player_state"].shape[0] == n and "reward_terms" in step_info and len(step_info["reward_terms"]) > 0
    checks["callbacks_run_in_workers"] = {"status": "passed" if ok else "failed",
                                          "reward_terms": sorted(step_info["reward_terms"])}
    env.close()

    # door state per world, canonical spawn: walk world 0 to the right trigger, others idle
    cfg = IsaacLayer1EnvConfig(num_envs=2, spawn_mode="canonical", num_workers=2)
    env = ShardedBatchedEnv(cfg, terrain_mode="shared")
    env.reset()
    for _ in range(4):
        env.step(move(2, [2, 1]))
    for _ in range(20):
        env.step(move(2, [1, 1]))
    doors = [x["shuttle_doors"] for x in env.slot_summaries()]
    checks["door_state_per_world"] = {"status": "passed" if doors[0]["activated"] and not doors[1]["activated"] else "failed",
                                      "world0": doors[0], "world1": doors[1]}
    env.close()

    # independent terrain: delete the support under world 0's player only
    cfg = IsaacLayer1EnvConfig(num_envs=2, spawn_mode="fixture", fixture_grid_xy=(7, 372), num_workers=2)
    env = ShardedBatchedEnv(cfg, terrain_mode="independent")
    env.reset()
    snap = env.snapshot()
    env.set_block(0, 7, 372, 0)
    for _ in range(6):
        env.step(move(2, [1, 1]))
    fell = env._body_state(0)["y"] < env._body_state(1)["y"] - 1.0
    env.restore(snap)
    for _ in range(6):
        env.step(move(2, [1, 1]))
    restored = abs(env._body_state(0)["y"] - env._body_state(1)["y"]) < 1e-3
    checks["independent_terrain_edit_and_restore"] = {"status": "passed" if fell and restored else "failed",
                                                      "edited_world_fell": bool(fell), "restore_back_on_support": bool(restored)}
    env.close()

    # worker failure: kill one worker, the next call must raise and tear everything down
    env = ShardedBatchedEnv(flat_cfg(n), terrain_mode="shared")
    env.reset()
    pids = list(env.worker_pids)
    os.kill(pids[-1], signal.SIGKILL)
    raised = False
    try:
        env.step(move(n, [1] * n))
    except WorkerFailure as failure:
        raised = True
    time.sleep(1.0)
    checks["worker_failure_fails_fast_and_cleans_up"] = {"status": "passed" if raised and not any(alive(pids)) else "failed",
                                                         "raised": raised, "alive_after": alive(pids)}
except BaseException:
    traceback.print_exc()
    checks["unexpected_error"] = {"status": "failed", "traceback": traceback.format_exc()}
status = "passed" if checks and all(c["status"] == "passed" for c in checks.values()) else "failed"
result = {"schema": "casu.isaac-sharded-validation.v1", "workers": args.workers, "num_envs": args.num_envs,
          "status": status, "checks": checks}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print("SHARDED_VALIDATION", status, {k: v["status"] for k, v in checks.items()}, flush=True)
sys.exit(0 if status == "passed" else 1)
