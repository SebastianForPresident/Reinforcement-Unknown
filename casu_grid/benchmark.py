"""Reproducible CPU benchmark for the source-faithful CasU grid core.

Run with the bundled Isaac Python, but without starting Isaac/Kit. All grid
work is sequential NumPy/Python on CPU; GPU is only sampled for context.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import platform
import resource
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from casu_grid import CasUGridState


WORLD_DIMS = (1024, 1024)
WORLD_CELLS = 1024 * 1024
SCAN_CELLS = 128 * 16
FIXED_DT = 0.02


def proc_memory_mib():
    values = {}
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    key, rest = line.split(":", 1)
                    values[key] = int(rest.strip().split()[0]) / 1024.0
    except OSError:
        pass
    return values


def peak_rss_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    return value / (1024.0 if sys.platform.startswith("linux") else 1024.0 ** 2)


def gpu_snapshot():
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits"], text=True, timeout=2)
        rows = []
        for row in output.strip().splitlines():
            util, memory = (part.strip() for part in row.split(",", 1))
            rows.append({"gpu_utilization_percent": int(util),
                         "gpu_memory_used_mib": int(memory)})
        return rows
    except Exception as exc:
        return {"unavailable": str(exc)}


def pct(values, percentile):
    ordered = sorted(values)
    if not ordered:
        return 0
    index = min(len(ordered) - 1, max(0, int(np.ceil(percentile * len(ordered))) - 1))
    return ordered[index]


def make_state(kind):
    s = CasUGridState(*WORLD_DIMS, camera_x=512, camera_y=512,
                      decisions=([False, True] * 50_000
                                 if kind == "moderate_fluid_11pct_dynamic" else ()))
    if kind == "static_terrain_25pct":
        s.blocks[::2, ::2] = 1
    elif kind == "sparse_fluid_0p1pct":
        for x in range(32, 1024, 32):
            for y in range(32, 1024, 32):
                s.fluids[x, y] = 3
                s.blocks[x - 1, y] = 1
                s.blocks[x + 1, y] = 1
                s.blocks[x, y - 1] = 1
    elif kind == "moderate_fluid_11pct_dynamic":
        s.fluids[::3, ::3] = 3
    elif kind == "dense_fluid_100pct":
        s.fluids.fill(3)
    elif kind == "mixed_stable_25pct_fluid_75pct_solid":
        s.blocks.fill(1)
        s.fluids[::2, ::2] = 3
        s.blocks[::2, ::2] = 0
    elif kind == "repeated_damage_partial":
        s.set_block(500, 500, 14)
        s.damage_block(500, 500, 3000.0)
    elif kind == "repeated_destruction":
        pass
    elif kind != "empty":
        raise ValueError(kind)
    return s


def density_info(s):
    fluid_count = int(np.count_nonzero(s.fluids))
    block_count = int(np.count_nonzero(s.blocks))
    x0, x1 = max(1, s.camera_x - 64), min(s.width - 2, s.camera_x + 64)
    y0, y1 = max(1, s.camera_y - 64), min(s.height - 2, s.camera_y - 48)
    active = s.fluids[x0:x1, y0:y1]
    return {
        "world_fluid_cells": fluid_count,
        "world_fluid_density": fluid_count / float(WORLD_CELLS),
        "initial_band_fluid_cells": int(np.count_nonzero(active)),
        "initial_band_cells": int(active.size),
        "initial_band_fluid_density": (float(np.count_nonzero(active)) / active.size
                                        if active.size else 0.0),
        "world_solid_cells": block_count,
        "world_solid_density": block_count / float(WORLD_CELLS),
    }


def perform_extra(s, kind):
    if kind == "repeated_damage_partial":
        for _ in range(10):
            s.damage_block(500, 500, 0.001)
    elif kind == "repeated_destruction":
        for i in range(20):
            x = 500 + i
            s.set_block(x, 500, 1)
            s.damage_block(x, 500, 100.0)


def benchmark_single(kind, warmup, samples):
    alloc_start = time.perf_counter_ns()
    s = make_state(kind)
    allocation_ns = time.perf_counter_ns() - alloc_start
    density = density_info(s)
    for _ in range(warmup):
        perform_extra(s, kind)
        s.simulation_step()
    latencies = []
    cpu_before = time.process_time()
    for _ in range(samples):
        start = time.perf_counter_ns()
        perform_extra(s, kind)
        s.simulation_step()
        latencies.append(time.perf_counter_ns() - start)
    wall = sum(latencies) / 1e9
    cpu = time.process_time() - cpu_before
    tick_rate = samples / wall if wall else 0.0
    return {
        "workload": kind,
        "warmup_ticks": warmup,
        "measured_ticks": samples,
        "allocation_and_setup_ms": allocation_ns / 1e6,
        "density_at_measurement_start": density,
        "latency_ns_raw": latencies,
        "step_latency_ns": {"median": int(statistics.median(latencies)),
                            "p95": pct(latencies, 0.95), "p99": pct(latencies, 0.99)},
        "physics_ticks_per_second": tick_rate,
        "policy_macrosteps_10_ticks_per_second": tick_rate / 10.0,
        "simulation_seconds_per_wall_second": tick_rate * FIXED_DT,
        "process_cpu_core_percent_during_measurement": cpu / wall * 100.0 if wall else 0,
        "steady_process_memory_mib": proc_memory_mib(),
        "process_peak_rss_mib": peak_rss_mib(),
        "grid_arrays_bytes": int(s.blocks.nbytes + s.fluids.nbytes),
        "execution": "NumPy/Python CPU; no Isaac Kit/physics started; sequential single world",
    }


def benchmark_snapshot(samples):
    s = make_state("mixed_stable_25pct_fluid_75pct_solid")
    snap = s.snapshot()
    encoded = pickle.dumps(snap, protocol=5)
    capture, restore = [], []
    for _ in range(samples):
        start = time.perf_counter_ns()
        snap = s.snapshot()
        capture.append(time.perf_counter_ns() - start)
        start = time.perf_counter_ns()
        s.restore(snap)
        restore.append(time.perf_counter_ns() - start)
    return {
        "samples": samples,
        "snapshot_arrays_bytes": int(s.blocks.nbytes + s.fluids.nbytes),
        "snapshot_pickle_bytes_protocol_5": len(encoded),
        "capture_latency_ns_raw": capture,
        "restore_latency_ns_raw": restore,
        "capture_latency_ns": {"median": int(statistics.median(capture)),
                               "p95": pct(capture, 0.95), "p99": pct(capture, 0.99)},
        "restore_latency_ns": {"median": int(statistics.median(restore)),
                               "p95": pct(restore, 0.95), "p99": pct(restore, 0.99)},
        "steady_process_memory_mib": proc_memory_mib(),
        "process_peak_rss_mib": peak_rss_mib(),
    }


def multi_worker(count, warmup, samples):
    allocation_start = time.perf_counter_ns()
    worlds = [CasUGridState(1024, 1024, camera_x=512, camera_y=512)
              for _ in range(count)]
    allocation_ns = time.perf_counter_ns() - allocation_start
    setup_start = time.perf_counter_ns()
    for s in worlds:
        s.blocks.fill(1)
        s.fluids[::2, ::2] = 3
        s.blocks[::2, ::2] = 0
    setup_ns = time.perf_counter_ns() - setup_start
    density = density_info(worlds[0])
    for _ in range(warmup):
        for s in worlds:
            s.simulation_step()
    latencies = []
    cpu_before = time.process_time()
    for _ in range(samples):
        start = time.perf_counter_ns()
        for s in worlds:  # Explicitly sequential; no shared mutable grids/RNG.
            s.simulation_step()
        latencies.append(time.perf_counter_ns() - start)
    wall = sum(latencies) / 1e9
    cpu = time.process_time() - cpu_before
    aggregate_rate = count * samples / wall if wall else 0.0
    return {
        "world_count": count,
        "workload": "mixed_stable_25pct_fluid_75pct_solid",
        "warmup_aggregate_cycles": warmup,
        "measured_aggregate_cycles": samples,
        "allocation_ms": allocation_ns / 1e6,
        "workload_setup_ms": setup_ns / 1e6,
        "density_per_world": density,
        "aggregate_cycle_latency_ns_raw": latencies,
        "aggregate_cycle_latency_ns": {"median": int(statistics.median(latencies)),
                                        "p95": pct(latencies, 0.95), "p99": pct(latencies, 0.99)},
        "aggregate_physics_ticks_per_second": aggregate_rate,
        "aggregate_policy_macrosteps_10_ticks_per_second": aggregate_rate / 10.0,
        "average_physics_ticks_per_second_per_environment": aggregate_rate / count,
        "simulation_seconds_per_wall_second_per_environment": aggregate_rate / count * FIXED_DT,
        "process_cpu_core_percent_during_measurement": cpu / wall * 100.0 if wall else 0,
        "steady_process_memory_mib": proc_memory_mib(),
        "process_peak_rss_mib": peak_rss_mib(),
        "grid_arrays_bytes_all_worlds": sum(s.blocks.nbytes + s.fluids.nbytes for s in worlds),
        "execution": "sequential worlds in one Python process; NumPy/CPU; no Isaac runtime",
    }


def run_worker_subprocess(count, warmup, samples):
    command = [sys.executable, str(Path(__file__).resolve()), "--multi-worker", str(count),
               "--warmup", str(warmup), "--samples", str(samples)]
    result = subprocess.run(command, check=True, text=True, capture_output=True,
                           cwd=str(ROOT), env=os.environ.copy())
    for line in reversed(result.stdout.splitlines()):
        if line.startswith("MULTI_WORKER_JSON "):
            return json.loads(line[len("MULTI_WORKER_JSON "):])
    raise RuntimeError(f"worker {count} did not return JSON: {result.stdout[-2000:]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=101)
    parser.add_argument("--multi-samples", type=int, default=51)
    parser.add_argument("--multi-worker", type=int)
    parser.add_argument("--skip-multi", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "casu_grid/results/grid_scaling_20261012.json")
    args = parser.parse_args()

    if args.multi_worker is not None:
        row = multi_worker(args.multi_worker, args.warmup, args.samples)
        print("MULTI_WORKER_JSON " + json.dumps(row, separators=(",", ":")), flush=True)
        return

    gpu_before = gpu_snapshot()
    workloads = ["empty", "static_terrain_25pct", "sparse_fluid_0p1pct",
                 "moderate_fluid_11pct_dynamic", "dense_fluid_100pct",
                 "mixed_stable_25pct_fluid_75pct_solid",
                 "repeated_damage_partial", "repeated_destruction"]
    single = []
    for name in workloads:
        print(f"single benchmark: {name}", flush=True)
        single.append(benchmark_single(name, args.warmup, args.samples))
    print("snapshot benchmark", flush=True)
    snapshot = benchmark_snapshot(min(args.samples, 51))

    multi = []
    if not args.skip_multi:
        for count in (1, 2, 4, 8, 16, 32, 64):
            raw_bytes = count * 3 * WORLD_CELLS
            if raw_bytes > 512 * 1024 * 1024:
                print(f"stopping multi-world scaling before {count}: raw arrays exceed 512 MiB", flush=True)
                break
            print(f"sequential multi-world benchmark: {count}", flush=True)
            multi.append(run_worker_subprocess(count, min(args.warmup, 10), args.multi_samples))

    result = {
        "schema": "casu-grid-scaling-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "host": {"platform": platform.platform(), "processor": platform.processor(),
                 "logical_cpus": os.cpu_count(), "python": sys.version,
                 "numpy": np.__version__, "cpu_threads": "single Python thread per benchmark process"},
        "measurement_contract": {
            "world_dimensions": [1024, 1024], "chunk_dimensions": [64, 64],
            "fixed_delta_seconds_assumed": FIXED_DT,
            "fluid_schedule": "actual SimulationStep camera-relative 128x16 band, in-place x-then-y loops, seven phases",
            "array_dtype_and_order": "x-first uint16 blocks + uint8 fluids",
            "world_fluid_density": "fluid cells / 1,048,576 world cells",
            "initial_band_fluid_density": "fluid cells / initially scanned 128x16=2,048 cells",
            "warmup_excluded_from_rate": args.warmup,
            "single_sample_count": args.samples,
            "multi_warmup_cycles": min(args.warmup, 10),
            "multi_sample_count": args.multi_samples,
            "cpu_gpu": "NumPy arrays and Python transition loop run on CPU; no Isaac Kit, Torch, or CUDA grid execution",
            "gpu_sampling": "nvidia-smi point samples for context only; not used by workload",
            "fixed_step_conversion": "policy macrostep = 10 physics-equivalent ticks; simulated-time factor uses 0.02 seconds/tick",
            "multi_strategy": "independent states advanced sequentially in one worker process per count",
            "warmup_definition": "state is prepared before timing; warmup ticks/cycles are untimed and retained",
        },
        "gpu_before": gpu_before,
        "single_world_workloads": single,
        "snapshot_restore": snapshot,
        "multi_world_scaling": multi,
        "gpu_after": gpu_snapshot(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {args.output}", flush=True)
    for row in single:
        print(f"SINGLE {row['workload']} {row['physics_ticks_per_second']:.2f} ticks/s "
              f"median={row['step_latency_ns']['median']}ns p95={row['step_latency_ns']['p95']}ns")
    for row in multi:
        print(f"MULTI {row['world_count']} {row['aggregate_physics_ticks_per_second']:.2f} aggregate ticks/s "
              f"{row['average_physics_ticks_per_second_per_environment']:.2f}/env "
              f"RSS={row['steady_process_memory_mib'].get('VmRSS', 0):.1f}MiB")


if __name__ == "__main__":
    main()
