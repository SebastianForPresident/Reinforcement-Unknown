"""Summarise worker cProfile dumps (CASU_WORKER_PROFILE=<prefix> [CASU_WORKER_PROFILE_SKIP=1]) into the report's breakdown table.

    python casu_grid/perf/worker_profile_breakdown.py PREFIX            # reads PREFIX.<pid>.cumulative.txt for every worker

Categories are cumulative times of the named call trees (the dumps list the 60 most expensive functions, which covers all of them);
"other" is the remainder of ``LocalRollout.run``.
"""
from __future__ import annotations
import re, sys
from pathlib import Path

CATEGORIES = {
    "PhysX step (sim.step: simulate + fetch_results)": [r"isaaclab/sim/simulation_context\.py:\d+\(step\)"],
    "crouch-shape switching (USD writes, PhysX shape toggles)": [r"isaac_player_body\.py:\d+\(set_levels\)"],
    "compiled controller (pre/post tick, macrostep latch, unpack)": [r"fast_runtime\.py:\d+\((pre_tick|post_tick|begin_macrostep|end_macrostep|relatch)\)"],
    "PhysX tensor I/O (transforms, velocities, forces)": [r"tensors/impl/api\.py:\d+\((get_transforms|set_transforms|get_velocities|set_velocities|apply_forces_and_torques_at_position)\)"],
    "independent terrain rebuild": [r"isaac_vector_env\.py:\d+\(_rebuild_dirty\)"],
    "observation (compact id windows)": [r"isaac_vector_env\.py:\d+\(compact_observation\)"],
    "policy inference (graph replay + readback)": [],      # _PolicyStep.run: the smaller of the two local_rollout run() entries
    "world reset": [r"isaac_vector_env\.py:\d+\(reset_done\)"],
}
TOTAL = r"workbench/local_rollout\.py:\d+\(run\)"


def parse(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        m = re.match(r"\s*([\d/]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(.*)$", line)
        if m:
            rows.append((int(m.group(1).split("/")[0]), float(m.group(2)), float(m.group(4)), m.group(6)))
    return rows


def main():
    files = sorted(Path(sys.argv[1]).parent.glob(Path(sys.argv[1]).name + ".*.cumulative.txt"))
    totals, sums = [], {k: 0.0 for k in CATEGORIES}
    for f in files:
        rows = parse(f)
        runs = sorted((c for n, t, c, name in rows if re.search(TOTAL, name)), reverse=True)
        totals.append(runs[0])
        sums["policy inference (graph replay + readback)"] += runs[1]
        for cat, patterns in CATEGORIES.items():
            for n, t, c, name in rows:
                if any(re.search(p, name) for p in patterns):
                    sums[cat] += c
    total = sum(totals)
    print(f"{len(files)} workers, {total / len(files):.2f} s of profiled rollout time each")
    covered = 0.0
    for cat, v in sums.items():
        covered += v
        print(f"{cat:62s} {v / total * 100:5.1f} %   ({v / len(files):.3f} s)")
    print(f"{'other (python glue, reward/termination bookkeeping, ...)':62s} {(total - covered) / total * 100:5.1f} %")


main()
