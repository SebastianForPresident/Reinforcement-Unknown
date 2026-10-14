"""End-to-end XY1 mini-world training throughput with resource monitoring (peak RSS of the whole process tree, GPU memory).

    python casu_grid/perf/xy1_bench.py --envs 64 --workers 6 --updates 8 --label opt3 --output out.json

Runs ``workbench.train`` on the XY1 mini-world configuration exactly like a research launch (no engineering smoke shortcuts
except the update count) and reads the trainer's own per-update timers. Accounting (all figures decisions/s):
  end-to-end = decisions / (collection_seconds + optimization_seconds)   over the steady updates (first update dropped)
  rollout    = decisions / collection_seconds
  optimizer  = decisions / optimization_seconds
  wall       = decisions / (wall time between the first and last update's completion)  -- includes logging/checkpoint overhead
"""
from __future__ import annotations
import argparse, json, os, shutil, statistics, subprocess, sys, tempfile, threading, time
from pathlib import Path

import psutil

ROOT = Path(os.environ.get("CASU_SRC_ROOT", Path(__file__).resolve().parents[2]))
p = argparse.ArgumentParser()
p.add_argument("--envs", type=int, default=16)
p.add_argument("--workers", type=int, default=4)
p.add_argument("--updates", type=int, default=8)
p.add_argument("--physx-threads", type=int, default=None)
p.add_argument("--train-device", default=None)
p.add_argument("--config", type=Path, default=ROOT / "casu_grid/workbench/experiments/xy1_point_navigation_mini.py")
p.add_argument("--label", default="")
p.add_argument("--keep-run", action="store_true")
p.add_argument("--series", action="store_true", help="store the RSS / available memory / GPU / update-progress time series in the result (soak runs)")
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()

run_dir = Path(tempfile.mkdtemp(prefix="casu_xy1bench_")) / "run"
env = dict(os.environ)
env["CASU_NUM_WORKERS"] = str(args.workers)
if args.physx_threads:
    env["CASU_WORKER_PHYSX_THREADS"] = str(args.physx_threads)
if args.train_device:
    env["CASU_TRAIN_DEVICE"] = args.train_device
cmd = [sys.executable, "-m", "casu_grid.workbench.train", "--config", str(args.config), "--updates", str(args.updates),
       "--num-envs", str(args.envs), "--run-dir", str(run_dir), "--terrain-mode", "independent"]
log = open(run_dir.parent / "train.log", "wb")
started = time.perf_counter()
proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
samples = []
progress = []          # (seconds since launch, completed updates)
per_process = []       # (seconds since launch, {pid: rss bytes}) every ~30 s with --series
stop = threading.Event()


def gpu_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        used, util = out.stdout.strip().splitlines()[0].split(",")
        gpu_util.append(int(util))
        return int(used)
    except Exception:
        return None


gpu_util = []
unique_mem = []         # (seconds, PSS bytes, USS bytes, system used bytes) of the whole process tree, every ~10 s


def tree_unique(tree):
    pss = uss = 0
    for q in tree:
        try:
            with open(f"/proc/{q.pid}/smaps_rollup") as f:
                for line in f:
                    if line.startswith("Pss:"):
                        pss += int(line.split()[1]) * 1024
                    elif line.startswith(("Private_Clean:", "Private_Dirty:")):
                        uss += int(line.split()[1]) * 1024
        except OSError:
            pass
    return pss, uss


def monitor():
    root = psutil.Process(proc.pid)
    while not stop.is_set():
        try:
            tree = [root] + root.children(recursive=True)
            rss = sum(q.memory_info().rss for q in tree if q.is_running())
            if args.series and int(time.perf_counter() - started) % 30 == 0:
                per_process.append((round(time.perf_counter() - started), {q.pid: q.memory_info().rss for q in tree if q.is_running()}))
            if int(time.perf_counter() - started) % 10 == 0 and (not unique_mem or time.perf_counter() - started - unique_mem[-1][0] > 5):
                pss, uss = tree_unique(tree)
                vm = psutil.virtual_memory()
                unique_mem.append((time.perf_counter() - started, pss, uss, vm.total - vm.available))
            avail = psutil.virtual_memory().available
            samples.append((time.perf_counter() - started, rss, avail, gpu_used_mib(), psutil.cpu_percent(interval=None)))
            try:
                done = sum(1 for _ in open(run_dir / "metrics.jsonl"))
            except OSError:
                done = 0
            if not progress or progress[-1][1] != done:
                progress.append((time.perf_counter() - started, done))
        except psutil.Error:
            pass
        stop.wait(1.0)


thread = threading.Thread(target=monitor, daemon=True)
thread.start()
returncode = proc.wait()
stop.set(); thread.join()
wall_total = time.perf_counter() - started
if returncode != 0:
    sys.exit(f"training run failed ({returncode}); see {run_dir.parent / 'train.log'}\n" + (run_dir.parent / "train.log").read_text(errors="replace")[-1500:])
rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines() if l.strip()]
steady = rows[1:]
decisions = rows[-1]["environment_steps"] - rows[0]["environment_steps"]       # decisions of the steady updates
collect = sum(r["collection_seconds"] for r in steady)
optimize = sum(r["optimization_seconds"] for r in steady)
env_step = sum(r["physics_step_seconds"] for r in steady)
mtimes = (run_dir / "metrics.jsonl").stat().st_mtime
result = {
    "schema": "casu.xy1-bench.v1", "label": args.label, "envs": args.envs, "workers": args.workers,
    "physx_threads": args.physx_threads or int(env.get("CASU_WORKER_PHYSX_THREADS", 2)), "updates": args.updates,
    "measured_updates": len(steady), "decisions": decisions,
    "end_to_end_decisions_per_second": decisions / (collect + optimize),
    "rollout_decisions_per_second": decisions / collect, "optimizer_decisions_per_second": decisions / optimize,
    "simulation_only_decisions_per_second": decisions / env_step,
    "seconds_per_update": {"collection": collect / len(steady), "optimization": optimize / len(steady), "environment_step": env_step / len(steady)},
    "per_update_end_to_end": [(b["environment_steps"] - a["environment_steps"]) / (b["collection_seconds"] + b["optimization_seconds"])
                              for a, b in zip(rows, rows[1:])],
    "worker_seconds": {k: statistics.fmean(r[k] for r in steady if k in r) for k in (rows[-1].keys()) if k.startswith("worker_")},
    "process_wall_seconds_including_startup": wall_total,
    "peak_rss_gib": max(s[1] for s in samples) / 2**30, "min_available_memory_gib": min(s[2] for s in samples) / 2**30,
    "peak_gpu_memory_mib": max([s[3] for s in samples if s[3] is not None] or [None]),
    "baseline_gpu_memory_mib": min([s[3] for s in samples if s[3] is not None] or [None]),
    "peak_pss_gib": max([u[1] for u in unique_mem] or [0]) / 2**30, "peak_uss_gib": max([u[2] for u in unique_mem] or [0]) / 2**30,
    "peak_system_used_gib": max([u[3] for u in unique_mem] or [0]) / 2**30,
    "gpu_utilization_percent_mean": (statistics.mean(gpu_util[len(gpu_util) // 3:]) if gpu_util else None),
    "mean_cpu_percent_steady": statistics.mean([s[4] for s in samples[len(samples) // 3:]] or [0]),
    "finite_losses": all(all(v == v and abs(v) != float("inf") for v in (r["policy_loss"], r["value_loss"], r["grad_norm"])) for r in rows),
    "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
    "environment": {k: env[k] for k in env if k.startswith("CASU_")},
}
second = [item for item in progress if item[1] >= 2]
if len(second) >= 2:     # wall-clock rate from the completion of update 2 to the last update (includes logging and checkpoint overhead)
    (t_first, n_first), (t_last, n_last) = second[0], progress[-1]
    per_update = decisions / max(len(steady), 1)
    if n_last > n_first:
        result["wall_decisions_per_second"] = (n_last - n_first) * per_update / (t_last - t_first)
if args.series:
    stride = max(1, len(samples) // 1500)
    result["series"] = {"t": [round(s[0], 1) for s in samples[::stride]], "rss_gib": [round(s[1] / 2**30, 3) for s in samples[::stride]],
                        "available_gib": [round(s[2] / 2**30, 3) for s in samples[::stride]], "gpu_mib": [s[3] for s in samples[::stride]],
                        "updates_completed": progress, "unique_memory": [(round(a, 1), b, c, d) for a, b, c, d in unique_mem], "per_process_rss": per_process, "per_update_end_to_end": result["per_update_end_to_end"]}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
if not args.keep_run:
    shutil.rmtree(run_dir.parent, ignore_errors=True)
print(f"envs={args.envs} workers={args.workers}: sim {result['simulation_only_decisions_per_second']:.0f}, rollout {result['rollout_decisions_per_second']:.0f}, "
      f"end-to-end {result['end_to_end_decisions_per_second']:.0f} decisions/s; opt {optimize / len(steady):.2f} s/update; "
      f"peak RSS {result['peak_rss_gib']:.1f} GiB (PSS {result['peak_pss_gib']:.1f}, USS {result['peak_uss_gib']:.1f}), GPU {result['peak_gpu_memory_mib']} MiB", flush=True)
