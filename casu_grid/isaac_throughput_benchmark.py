"""Batched throughput benchmark with a per-phase time breakdown (one process per batch size).

Schema ``casu.isaac-throughput.v2``. For every batch size it reports aggregate physics ticks/s,
ten-tick policy decisions/s, per-environment throughput, repeated-window variability, CPU
utilisation, VRAM/RSS, build time and reset / snapshot / restore latency. It measures
simulation only; policy inference and PPO update overhead come from the workbench metrics
(see ``casu_grid/isaac_training_benchmark.py`` for end-to-end training throughput).

Each measured window starts from ``reset()`` with the same seeded action tape, so windows are
identical workloads and the spread between them is timing noise. ``--workload random`` is the
baseline tape (random move/jump/crouch); ``--workload run`` holds right with periodic jumps.
Compare two result files with ``casu_grid/tools/compare_throughput.py``.
"""
from __future__ import annotations
import argparse, json, os, platform, resource, subprocess, sys, time, traceback
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, nargs="+", default=[1, 2, 4, 8, 16])
parser.add_argument("--macrosteps", type=int, default=30, help="measured macrosteps per window")
parser.add_argument("--windows", type=int, default=3, help="repeated measurement windows per batch size")
parser.add_argument("--warmup", type=int, default=3, help="untimed macrosteps before each window")
parser.add_argument("--terrain-mode", choices=("shared", "independent"), default="shared")
parser.add_argument("--workload", choices=("random", "run"), default="random")
parser.add_argument("--label", default="", help="free-text tag stored in the result (e.g. the optimisation under test)")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--single", type=int, default=0, help=argparse.SUPPRESS)
args, rest = parser.parse_known_args()


def backend_info() -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        from casu_grid import backend
        return backend.describe()
    except Exception as error:   # pre-backend revisions
        return {"active": "reference-python", "note": f"backend module unavailable: {error}"}


def system_info() -> dict:
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    cpu = ""
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        pass
    status = run(["git", "status", "--porcelain", "--", "casu_grid"])
    return {"git_revision": run(["git", "rev-parse", "HEAD"]) or None,
            "casu_grid_dirty_files": [l[3:] for l in status.splitlines()],
            "cpu": cpu, "cpu_count": os.cpu_count(), "platform": platform.platform(),
            "python": sys.version.split()[0],
            "gpu": run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
            "backend": backend_info()}


def gpu_sample() -> dict:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().split(",")
        return {"utilization_percent": float(out[0]), "memory_used_mib": float(out[1])}
    except Exception:
        return {}


if not args.single:
    # one Kit SimulationContext per process
    rows = []
    for n in args.num_envs:
        part = args.output.with_name(f"{args.output.stem}_n{n}.json")
        subprocess.run([sys.executable, __file__, "--single", str(n), "--macrosteps", str(args.macrosteps),
                        "--windows", str(args.windows), "--warmup", str(args.warmup),
                        "--terrain-mode", args.terrain_mode, "--workload", args.workload,
                        "--output", str(part), "--headless"], check=True)
        rows.append(json.loads(part.read_text()))
        part.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "schema": "casu.isaac-throughput.v2", "label": args.label, "terrain_mode": args.terrain_mode,
        "workload": args.workload, "system": system_info(), "rows": rows}, indent=2) + "\n")
    for r in rows:
        print(f"{args.terrain_mode} n={r['num_envs']:>3} ticks/s={r['aggregate_physics_ticks_per_second']['mean']:8.1f} "
              f"(±{r['aggregate_physics_ticks_per_second']['std']:.1f}) decisions/s="
              f"{r['aggregate_policy_decisions_per_second']['mean']:7.1f} rss={r['peak_rss_gib']:.2f} GiB", flush=True)
    raise SystemExit(0)

from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from casu_grid.isaac_navigation_env import IsaacLayer1EnvConfig
from casu_grid.isaac_vector_env import BatchedIsaacLayer1Env


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()), "max": float(a.max()),
            "samples": [float(v) for v in a]}


try:
    n = args.single
    cfg = IsaacLayer1EnvConfig(num_envs=n, spawn_mode="fixture", fixture_grid_xy=(483, 1010))
    started = time.perf_counter()
    env = BatchedIsaacLayer1Env(cfg, terrain_mode=args.terrain_mode)
    construct = time.perf_counter() - started
    vram_after_build = gpu_sample()
    t = time.perf_counter(); env.reset(); reset_all = time.perf_counter() - t

    def tape(step, rng):
        if args.workload == "random":
            return {"move_x": rng.integers(0, 3, n), "jump": rng.integers(0, 2, n), "crouch": rng.integers(0, 2, n)}
        return {"move_x": np.full(n, 2), "jump": np.full(n, 1 if step % 4 == 3 else 0), "crouch": np.zeros(n, int)}

    ticks_per_s, decisions_per_s, cpu_util, breakdowns = [], [], [], []
    gpu_during = {}
    for window in range(args.windows):
        env.reset()
        rng = np.random.default_rng(0)
        for step in range(args.warmup):
            env.step(tape(step, rng))
        env.timers.clear()
        cpu0, t = time.process_time(), time.perf_counter()
        for step in range(args.macrosteps):
            env.step(tape(args.warmup + step, rng))
        elapsed, cpu = time.perf_counter() - t, time.process_time() - cpu0
        if window == args.windows - 1:
            gpu_during = gpu_sample()
        ticks = args.macrosteps * cfg.action_decimation * n
        ticks_per_s.append(ticks / elapsed)
        decisions_per_s.append(args.macrosteps * n / elapsed)
        cpu_util.append(cpu / elapsed)
        breakdowns.append({k: v / elapsed for k, v in env.timers.items()})
    phase_fraction = {k: float(np.mean([b.get(k, 0.0) for b in breakdowns])) for k in breakdowns[0]}
    t = time.perf_counter(); snap = env.snapshot(); snapshot_s = time.perf_counter() - t
    t = time.perf_counter(); env.restore(snap); restore_s = time.perf_counter() - t
    t = time.perf_counter(); env.reset_done(np.asarray([True] + [False] * (n - 1))); reset_one = time.perf_counter() - t
    row = {"num_envs": n, "terrain_mode": args.terrain_mode, "workload": args.workload,
           "macrosteps_per_window": args.macrosteps, "windows": args.windows, "warmup_macrosteps": args.warmup,
           "construct_seconds": construct, "geometry_build_seconds": env.geometry_build_seconds,
           "aggregate_physics_ticks_per_second": stats(ticks_per_s),
           "aggregate_policy_decisions_per_second": stats(decisions_per_s),
           "per_env_physics_ticks_per_second": stats(np.asarray(ticks_per_s) / n),
           "cpu_utilization_of_one_core": stats(cpu_util),
           "phase_fraction": phase_fraction,
           "reset_all_seconds": reset_all, "reset_one_world_seconds": reset_one,
           "snapshot_seconds": snapshot_s, "restore_seconds": restore_s,
           "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
           "gpu_after_build": vram_after_build, "gpu_during_last_window": gpu_during}
    Path(args.output).write_text(json.dumps(row, indent=2) + "\n")
    env.close()
except BaseException:
    traceback.print_exc()
    raise
finally:
    app.close()
