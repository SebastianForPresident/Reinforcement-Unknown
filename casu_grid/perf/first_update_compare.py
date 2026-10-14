"""Paired comparison of the first PPO update's action statistics between two implementations (same seeds, same initial policy).

    python casu_grid/perf/first_update_compare.py RUNS_DIR --a base --b opt --seeds 1 2 3 ... [--update 0]

Both implementations start from the same initial weights and world seeds, so the action fractions of the first rollout
estimate the same quantities; their difference should scatter like the sampling noise of two independent binomial samples of
N = envs x rollout decisions (a conservative yardstick: autocorrelation inflates the real variance). Prints the distribution
of z = (fB - fA) / sqrt(2 p (1 - p) / N) over all (seed, head, choice) cells, and the mean signed z per cell over seeds.
"""
from __future__ import annotations
import argparse, json, math, statistics
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("runs", type=Path)
ap.add_argument("--a", default="base"); ap.add_argument("--b", default="opt")
ap.add_argument("--seeds", type=int, nargs="+", required=True)
ap.add_argument("--update", type=int, default=0, help="row index of metrics.jsonl (0 = first update)")
ap.add_argument("--decisions", type=int, default=1024)
args = ap.parse_args()
zs, per_cell = [], {}
for seed in args.seeds:
    ra = json.loads((args.runs / f"{args.a}_{seed}" / "metrics.jsonl").read_text().splitlines()[args.update])
    rb = json.loads((args.runs / f"{args.b}_{seed}" / "metrics.jsonl").read_text().splitlines()[args.update])
    for key in ra:
        if key.startswith("action/") and key.endswith("_fraction"):
            p = (ra[key] + rb[key]) / 2
            if 0.02 < p < 0.98:
                z = (rb[key] - ra[key]) / math.sqrt(2 * p * (1 - p) / args.decisions)
                zs.append(z); per_cell.setdefault(key, []).append(z)
n = len(zs)
print(f"cells: {n}   mean z {statistics.fmean(zs):+.3f}   sd z {statistics.pstdev(zs):.3f}   max |z| {max(abs(z) for z in zs):.2f}   "
      f"fraction |z|>2: {sum(abs(z) > 2 for z in zs) / n:.3f} (normal: 0.046)   |z|>3: {sum(abs(z) > 3 for z in zs)}")
m = [(statistics.fmean(v) * math.sqrt(len(v)), k) for k, v in per_cell.items()]
m.sort()
print("cells with the largest seed-averaged signed bias (z*sqrt(seeds)):", ", ".join(f"{k.split('/',1)[1]} {v:+.2f}" for v, k in (m[:3] + m[-3:])))
