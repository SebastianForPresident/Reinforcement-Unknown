"""Multi-process execution of the batched Layer 1 environment.

``ShardedBatchedEnv`` has the same API as :class:`~casu_grid.isaac_vector_env.BatchedIsaacLayer1Env`
(``reset``, ``step``, ``reset_done``, ``snapshot``/``restore``, ``set_block`` ...) but spreads the
worlds over ``config.num_workers`` independent Isaac processes. Each worker owns a contiguous shard
(its own Kit application, PhysX scene and Python interpreter), so the Python-side controller, the USD
shape switching and PhysX all run in parallel instead of sharing one interpreter. Simulated physics,
action cadence and per-world semantics are unchanged; only the execution layout differs.

Select it with ``IsaacLayer1EnvConfig(num_workers=N)`` or ``CASU_NUM_WORKERS=N`` (default 1 =
in-process). ``BatchedIsaacLayer1Env(config)`` returns this class automatically when
``config.num_workers > 1``, so existing launchers need no change.

Design notes
  * Workers are fresh interpreters started with ``subprocess.Popen`` (never ``fork``/``spawn`` of the
    Kit-initialised training process) and talk over an anonymous pipe.
  * World ``i`` keeps its global index in every layout, so fluid seeds (and, in shared-terrain mode,
    the whole simulation) do not depend on how worlds are partitioned.
  * Researcher callbacks (action decoder, observation encoder, reward function) execute inside the
    workers. They are shipped as ``(module, file, qualname)`` references and rebuilt there: top-level
    functions of importable modules or of a config file work directly; closures work when they come
    from a no-argument factory such as ``callbacks()``. Anything else (lambdas, bound methods) is
    rejected with an explanatory error before any process starts.
  * Fail fast: if a worker dies or reports an error, every worker is terminated (exact-owned
    processes only) and the exception is raised; no silent restarts that would alter the data.
"""
from __future__ import annotations

import atexit
import dataclasses
import inspect
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
import weakref
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .isaac_shard_worker import resolve_callable
from .reference import load_layer1_reference

REPO_ROOT = str(Path(__file__).resolve().parents[1])


def callable_spec(fn):
    """``(module, file, qualname)`` reference for a callback, or ``None`` for the default."""
    if fn is None:
        return None
    qualname = getattr(fn, "__qualname__", "")
    module_name = getattr(fn, "__module__", None)
    if (not inspect.isfunction(fn)) or qualname.endswith("<lambda>") or not module_name:
        raise ValueError(
            f"callback {fn!r} cannot be sent to worker processes; define it as a top-level function or "
            "inside a no-argument factory such as callbacks(), or run with num_workers=1")
    file_path = inspect.getsourcefile(fn)
    spec = (module_name, file_path, qualname)
    try:   # fail early, in the parent, if the worker could not rebuild it
        rebuilt = resolve_callable(spec)
    except Exception as error:
        raise ValueError(f"callback {qualname} cannot be rebuilt in a worker process ({error}); "
                         "use a top-level function or a no-argument factory, or num_workers=1") from error
    if rebuilt is None:
        raise ValueError(f"callback {qualname} cannot be located by a worker process; "
                         "use a top-level function or a no-argument factory, or num_workers=1")
    return spec


def shard_sizes(total: int, workers: int) -> list[int]:
    workers = max(1, min(workers, total))
    base, extra = divmod(total, workers)
    return [base + (1 if k < extra else 0) for k in range(workers)]


class _ShardedTimers(dict):
    """Per-phase seconds averaged over workers; ``clear()`` resets the workers' counters too."""

    def __init__(self, owner):
        super().__init__()
        self._owner = owner

    def clear(self):
        super().clear()
        if self._owner is not None:
            self._owner._straggler = self._owner._ipc = 0.0
            self._owner._broadcast("clear_timers", None)


class WorkerFailure(RuntimeError):
    """A worker process failed or exited; all workers have been terminated."""


def _terminate(processes):
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.time() + 10
    for proc in processes:
        try:
            proc.wait(max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


class ShardedBatchedEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config, *, action_decoder=None, observation_encoder=None, reward_function=None,
                 terrain_mode: str = "shared", world_offset: int = 0):
        if config.num_envs < 1:
            raise ValueError("ShardedBatchedEnv requires num_envs >= 1")
        if config.geometry_scope != "full":
            raise ValueError("the batched environment builds full canonical geometry")
        self.cfg = config
        self.terrain_mode = terrain_mode
        self.count = int(config.num_envs)
        self.world_offset = int(world_offset)
        self.sizes = shard_sizes(self.count, config.num_workers)
        self.num_workers = len(self.sizes)
        self.reference = load_layer1_reference()
        self.closed = False
        self.timers = _ShardedTimers(self)
        specs = {"action_decoder": callable_spec(action_decoder),
                 "observation_encoder": callable_spec(observation_encoder),
                 "reward_function": callable_spec(reward_function)}
        self.action_space = spaces.Dict({"move_x": spaces.MultiDiscrete([3]), "jump": spaces.MultiDiscrete([2]),
                                         "crouch": spaces.MultiDiscrete([2])})
        self.observation_space = spaces.Dict({"player_state": spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)})
        self._procs: list[subprocess.Popen] = []
        self._conns = []
        self.log_dir = Path(tempfile.mkdtemp(prefix=f"casu_workers_{os.getpid()}_"))
        self._finalizer = weakref.finalize(self, _terminate, self._procs)
        atexit.register(self._finalizer)
        started = time.perf_counter()
        base = self.world_offset
        env = dict(os.environ)
        env["PYTHONPATH"] = REPO_ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        try:
            for k, size in enumerate(self.sizes):
                parent_end, child_end = mp.Pipe()
                log = open(self.log_dir / f"worker_{k}.log", "wb")
                proc = subprocess.Popen(
                    [sys.executable, "-m", "casu_grid.isaac_shard_worker", str(child_end.fileno())],
                    pass_fds=[child_end.fileno()], stdout=log, stderr=subprocess.STDOUT, env=env,
                    cwd=os.getcwd())
                child_end.close()
                log.close()
                self._procs.append(proc)
                self._conns.append(parent_end)
                parent_end.send({"config": config, "count": size, "world_offset": base,
                                 "terrain_mode": terrain_mode, "callbacks": specs,
                                 "repo_root": REPO_ROOT, "physx_threads": config.worker_physx_threads,
                                 "environ": {}})
                base += size
            ready = [self._receive(k) for k in range(self.num_workers)]
        except BaseException:
            _terminate(self._procs)
            raise
        self.geometry_build_seconds = max(r["geometry_build_seconds"] for r in ready)
        self.construct_seconds = time.perf_counter() - started
        self.worker_pids = [r["pid"] for r in ready]
        # per-terrain-copy geometry statistics, as BatchedIsaacLayer1Env exposes (launcher metadata)
        self.geometry_by_env = [g for r in ready for g in r["geometry_by_env"]]
        content_ids = {r["content_id"] for r in ready}
        if content_ids != {self.reference.manifest["content_id"]}:
            _terminate(self._procs)
            raise RuntimeError("workers loaded a different canonical world")

    # ------------------------------------------------------------------ plumbing
    def _receive(self, k):
        conn = self._conns[k]
        try:
            while not conn.poll(1.0):
                if self._procs[k].poll() is not None:
                    self._fail(f"worker {k} exited with code {self._procs[k].returncode}", k)
            kind, payload = conn.recv()
        except (EOFError, OSError):   # ConnectionResetError / BrokenPipeError: the worker is gone
            self._fail(f"worker {k} closed its pipe (exit code {self._procs[k].poll()})", k)
        if kind == "error":
            self._fail(f"worker {k} reported an error:\n{payload}", k)
        return payload

    def _send(self, k, message):
        try:
            self._conns[k].send(message)
        except (EOFError, OSError):
            self._fail(f"worker {k} is not accepting requests (exit code {self._procs[k].poll()})", k)

    def _fail(self, message, k=None):
        tail = ""
        if k is not None:
            try:
                tail = (self.log_dir / f"worker_{k}.log").read_text(errors="replace")[-2000:]
            except Exception:
                pass
        _terminate(self._procs)
        self.closed = True
        raise WorkerFailure(f"{message}\n--- worker log tail ({self.log_dir}) ---\n{tail}")

    def _broadcast(self, kind, payload):
        for k in range(self.num_workers):
            self._send(k, (kind, payload))
        return [self._receive(k) for k in range(self.num_workers)]

    def _scatter(self, kind, payloads):
        for k, payload in enumerate(payloads):
            self._send(k, (kind, payload))
        return [self._receive(k) for k in range(self.num_workers)]

    def _slices(self):
        out, start = [], 0
        for size in self.sizes:
            out.append(slice(start, start + size))
            start += size
        return out

    @staticmethod
    def _concat_obs(parts):
        return {key: np.concatenate([p[key] for p in parts], axis=0) for key in parts[0]}

    def _locate(self, index):
        if not 0 <= index < self.count:
            raise IndexError("environment index outside vector batch")
        for k, sl in enumerate(self._slices()):
            if sl.start <= index < sl.stop:
                return k, index - sl.start

    # ----------------------------------------------------------------- env API
    def reset(self, *, seed=None, options=None):
        parts = self._broadcast("reset", {"seed": seed, "options": options})
        return self._concat_obs([p[0] for p in parts]), {
            "seed": seed, "canonical_content_id": self.reference.manifest["content_id"]}

    def reset_done(self, mask):
        mask = np.asarray(mask, dtype=bool).reshape(self.count)
        return self._concat_obs(self._scatter("reset_done", [mask[sl] for sl in self._slices()]))

    def step(self, action_dict):
        actions = {key: np.asarray(value) for key, value in action_dict.items()}
        for key, value in actions.items():
            if value.shape[0] != self.count:
                raise ValueError(f"action {key} has {value.shape[0]} rows for {self.count} environments")
        started = time.perf_counter()
        replies = self._scatter("step", [{k: v[sl] for k, v in actions.items()} for sl in self._slices()])
        wall = time.perf_counter() - started
        results = [r[0] for r in replies]
        phase = defaultdict(float)
        for _, worker_timers, _ in replies:
            for key, value in worker_timers.items():
                phase[key] += value / self.num_workers
        busy = [r[2] for r in replies]
        # cumulative diagnostics: time lost waiting for the slowest worker, and request/response overhead
        self._straggler = getattr(self, "_straggler", 0.0) + (max(busy) - sum(busy) / len(busy))
        self._ipc = getattr(self, "_ipc", 0.0) + (wall - max(busy))
        phase["sharded_straggler_wait"], phase["sharded_ipc_overhead"] = self._straggler, self._ipc
        self.timers.update(phase)
        obs = self._concat_obs([r[0] for r in results])
        rewards = np.concatenate([r[1] for r in results])
        terminated = np.concatenate([r[2] for r in results])
        truncated = np.concatenate([r[3] for r in results])
        infos = [r[4] for r in results]
        terms: dict[str, np.ndarray] = {}
        for key in infos[0]["reward_terms"]:
            terms[key] = np.concatenate([i["reward_terms"][key] for i in infos])
        info = {"reward_terms": terms,
                "termination_reason": [x for i in infos for x in i["termination_reason"]],
                "physics_ticks": sum(i["physics_ticks"] for i in infos),
                "player": [x for i in infos for x in i["player"]],
                "canonical_content_id": infos[0]["canonical_content_id"]}
        return obs, rewards, terminated, truncated, info

    def set_block(self, env_index: int, x: int, y: int, block_id: int):
        if self.terrain_mode != "independent":
            raise NotImplementedError("terrain edits need terrain_mode='independent' (shared terrain is static)")
        k, local = self._locate(env_index)
        self._send(k, ("set_block", (local, x, y, block_id)))
        self._receive(k)

    def snapshot(self):
        shards = self._broadcast("snapshot", None)
        return {"schema": "casu.isaac-sharded-state.v1", "sizes": list(self.sizes), "shards": shards,
                "terrain_mode": self.terrain_mode,
                "canonical_content_id": self.reference.manifest["content_id"]}

    def restore(self, snapshot):
        if snapshot.get("schema") != "casu.isaac-sharded-state.v1" or list(snapshot["sizes"]) != list(self.sizes):
            raise ValueError("snapshot has wrong sharded schema or world partition")
        self._scatter("restore", snapshot["shards"])

    def _body_state(self, index):
        k, local = self._locate(index)
        self._send(k, ("body_state", local))
        return self._receive(k)

    def slot_summaries(self):
        return [item for part in self._broadcast("slot_summaries", None) for item in part]

    def performance_info(self) -> dict:
        from . import backend
        return {"backend": backend.describe(), "num_workers": self.num_workers, "worlds": self.count,
                "terrain_mode": self.terrain_mode, "worlds_per_worker": list(self.sizes),
                "physx_threads_per_worker": self.cfg.worker_physx_threads,
                "worker_log_dir": str(self.log_dir)}

    def close(self):
        if self.closed:
            return
        self.closed = True
        for conn in self._conns:
            try:
                conn.send(("close", None))
            except Exception:
                pass
        deadline = time.time() + 20
        for proc in self._procs:
            try:
                proc.wait(max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                pass
        _terminate(self._procs)
        for conn in self._conns:
            conn.close()
