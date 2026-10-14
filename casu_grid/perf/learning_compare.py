"""Compare per-update training statistics of two implementations over several seeds.

    python casu_grid/perf/learning_compare.py --a base_111 base_222 base_333 --b opt_111 opt_222 opt_333 [--tail 20]

Each argument is a run directory with ``metrics.jsonl``. For every statistic the per-seed mean over the first ``--tail``
and over all updates is formed, then the across-seed mean and standard deviation of both groups are printed together with
the difference in units of the pooled across-seed standard deviation (|z| below ~2 means indistinguishable at this sample size).
"""
from __future__ import annotations
import argparse, json, math, statistics
from pathlib import Path

KEYS = ("entropy", "approx_kl", "policy_loss", "value_loss", "grad_norm", "reward_term/distance_improvement",
        "reward_term/arrival_bonus", "epochs_completed", "optimizer_batches")


def load(path):
    rows = [json.loads(l) for l in (Path(path) / "metrics.jsonl").read_text().splitlines() if l.strip()]
    return rows


def per_seed(dirs, key, first=None):
    out = []
    for d in dirs:
        rows = load(d)
        rows = rows[:first] if first else rows
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        out.append(statistics.fmean(vals) if vals else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--tail", type=int, default=20)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    keys = list(KEYS)
    sample = load(args.a[0])[0]
    keys += [k for k in sample if k.startswith("action/") and k.endswith("_fraction")]
    result = {}
    print(f"{'statistic':46s} {'window':>7s} {'A mean':>11s} {'A sd':>9s} {'B mean':>11s} {'B sd':>9s} {'z':>6s}")
    worst = 0.0
    for key in keys:
        for window, first in (("first%d" % args.tail, args.tail), ("all", None)):
            a, b = per_seed(args.a, key, first), per_seed(args.b, key, first)
            if any(math.isnan(v) for v in a + b):
                continue
            sa, sb = (statistics.stdev(a) if len(a) > 1 else 0.0), (statistics.stdev(b) if len(b) > 1 else 0.0)
            pooled = math.sqrt((sa ** 2 + sb ** 2) / 2) or 1e-12
            z = (statistics.fmean(b) - statistics.fmean(a)) / (pooled * math.sqrt(2 / max(len(a), 1)))
            worst = max(worst, abs(z)) if not key.startswith("action/") else worst
            result[f"{key}|{window}"] = {"a_mean": statistics.fmean(a), "a_sd": sa, "b_mean": statistics.fmean(b), "b_sd": sb, "z": z}
            if not key.startswith("action/") or abs(z) > 2.5:
                print(f"{key:46s} {window:>7s} {statistics.fmean(a):11.5f} {sa:9.5f} {statistics.fmean(b):11.5f} {sb:9.5f} {z:6.2f}")
    print(f"largest |z| over non-action statistics: {worst:.2f}")
    if args.json:
        args.json.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
