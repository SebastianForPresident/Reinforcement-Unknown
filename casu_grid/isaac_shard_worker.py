"""Worker process of :class:`casu_grid.isaac_sharded_env.ShardedBatchedEnv`.

Started as ``python -m casu_grid.isaac_shard_worker <pipe-fd>`` by the parent, never by hand. It owns
exactly one Isaac Kit application and one :class:`BatchedIsaacLayer1Env` holding a contiguous shard
of the worlds, and serves requests over a ``multiprocessing`` pipe until told to close or until the
parent disappears (parent-death signal + pipe EOF), so crashed trainers cannot leave orphan Kit
processes behind.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import signal
import sys
import time
import traceback
from pathlib import Path


def _die_with_parent():
    """Ask the kernel to SIGTERM this process when the parent exits (Linux)."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)   # PR_SET_PDEATHSIG
    except Exception:
        pass


def _load_module(module_name, file_path):
    if file_path and Path(file_path).is_file():
        # a module that is not importable by name (e.g. a researcher config loaded from a path)
        try:
            module = importlib.import_module(module_name)
            if getattr(module, "__file__", None) and Path(module.__file__).resolve() == Path(file_path).resolve():
                return module
        except Exception:
            pass
        spec = importlib.util.spec_from_file_location(module_name or "casu_worker_callbacks", file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(module_name)


def resolve_callable(spec):
    """Rebuild a callback from ``(module_name, file_path, qualname)`` (see ``callable_spec``)."""
    if spec is None:
        return None
    module_name, file_path, qualname = spec
    module = _load_module(module_name, file_path)
    if "<locals>" not in qualname:
        obj = module
        for part in qualname.split("."):
            obj = getattr(obj, part)
        return obj
    # closure produced by a no-argument factory (e.g. ``callbacks()``): call the factory, find it
    outer = qualname.split(".<locals>")[0]
    factory = module
    for part in outer.split("."):
        factory = getattr(factory, part)
    return _find_by_qualname(factory(), qualname)


def _find_by_qualname(produced, qualname, depth=0):
    if callable(produced) and getattr(produced, "__qualname__", None) == qualname:
        return produced
    if depth < 3:
        items = produced.values() if isinstance(produced, dict) else produced if isinstance(produced, (list, tuple)) else ()
        for item in items:
            found = _find_by_qualname(item, qualname, depth + 1)
            if found is not None:
                return found
    return None


def _plain(snapshot):
    """Snapshots hold torch tensors; send NumPy so pipes never use torch's fd-sharing reducers."""
    import torch
    for slot in snapshot["slots"]:
        for key, value in list(slot.items()):
            if torch.is_tensor(value):
                slot[key] = value.detach().cpu().numpy().copy()
    return snapshot


def _tensors(snapshot):
    import torch
    for slot in snapshot["slots"]:
        for key, value in list(slot.items()):
            if key == "player_root_state_wxyz" and not torch.is_tensor(value):
                slot[key] = torch.from_numpy(value.copy())
    return snapshot


def main() -> None:
    import multiprocessing.connection as mpc

    _die_with_parent()
    conn = mpc.Connection(int(sys.argv[1]))
    init = conn.recv()
    for key, value in init.get("environ", {}).items():
        os.environ[key] = value
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True).app
        import carb.settings
        carb.settings.get_settings().set("/persistent/physics/numThreads", int(init["physx_threads"]))
        sys.path.insert(0, init["repo_root"])
        import dataclasses
        import numpy as np
        from casu_grid.isaac_vector_env import BatchedIsaacLayer1Env

        callbacks = {name: resolve_callable(spec) for name, spec in init["callbacks"].items()}
        cfg = dataclasses.replace(init["config"], num_envs=init["count"], num_workers=1)
        env = BatchedIsaacLayer1Env(cfg, terrain_mode=init["terrain_mode"], world_offset=init["world_offset"],
                                    action_decoder=callbacks.get("action_decoder"),
                                    observation_encoder=callbacks.get("observation_encoder"),
                                    reward_function=callbacks.get("reward_function"))
        conn.send(("ready", {"geometry_build_seconds": env.geometry_build_seconds,
                             "content_id": env.reference.manifest["content_id"], "pid": os.getpid(),
                             "geometry_by_env": env.geometry_by_env}))
    except BaseException:
        conn.send(("error", traceback.format_exc()))
        return

    def timers():
        return {k: float(v) for k, v in env.timers.items()}

    try:
        while True:
            try:
                kind, payload = conn.recv()
            except EOFError:        # parent is gone
                break
            if kind == "close":
                conn.send(("ok", None))
                break
            try:
                if kind == "step":
                    started = time.perf_counter()
                    result = env.step(payload)
                    reply = (result, timers(), time.perf_counter() - started)
                elif kind == "reset":
                    reply = env.reset(**payload)
                elif kind == "reset_done":
                    reply = env.reset_done(payload)
                elif kind == "snapshot":
                    reply = _plain(env.snapshot())
                elif kind == "restore":
                    env.restore(_tensors(payload))
                    reply = None
                elif kind == "set_block":
                    env.set_block(*payload)
                    reply = None
                elif kind == "body_state":
                    reply = env._body_state(payload)
                elif kind == "clear_timers":
                    env.timers.clear()
                    reply = None
                elif kind == "timers":
                    reply = timers()
                elif kind == "slot_summaries":
                    reply = env.slot_summaries()
                else:
                    raise ValueError(f"unknown request {kind!r}")
                conn.send(("ok", reply))
            except BaseException:
                conn.send(("error", traceback.format_exc()))
    finally:
        try:
            env.close()
        finally:
            app.close()


if __name__ == "__main__":
    main()
