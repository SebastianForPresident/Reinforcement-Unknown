"""Compare two ``casu.isaac-throughput.v2`` files and flag throughput regressions.

    python casu_grid/tools/compare_throughput.py BASELINE.json CANDIDATE.json [--margin 0.10] [--fail-on-regression]

Rows are matched by environment count. A row *regresses* when the candidate's mean
ticks/s is below ``baseline_mean * (1 - margin)`` and the gap exceeds both runs' observed
spread; wall-clock noise below the margin never counts. Uses only the standard library.
"""
import argparse, json, sys

p = argparse.ArgumentParser()
p.add_argument("baseline"); p.add_argument("candidate")
p.add_argument("--margin", type=float, default=0.10, help="relative noise margin (default 10%%)")
p.add_argument("--fail-on-regression", action="store_true")
a = p.parse_args()


def _b(d):
    b = d["system"]["backend"]
    return b.get("active") if isinstance(b, dict) else b

base, cand = (json.load(open(f)) for f in (a.baseline, a.candidate))
for key in ("terrain_mode", "workload"):
    if base.get(key) != cand.get(key):
        print(f"WARNING: {key} differs ({base.get(key)} vs {cand.get(key)}); rows are not like-for-like")
print(f"baseline  : {base.get('label','')} backend={_b(base)} rev={str(base['system']['git_revision'])[:8]}")
print(f"candidate : {cand.get('label','')} backend={_b(cand)} rev={str(cand['system']['git_revision'])[:8]}")
rows = {r["num_envs"]: r for r in base["rows"]}
regressed = False
print(f"{'envs':>5} {'base ticks/s':>13} {'cand ticks/s':>13} {'speedup':>8}  verdict")
for r in cand["rows"]:
    b = rows.get(r["num_envs"])
    if b is None:
        continue
    bm, cm = b["aggregate_physics_ticks_per_second"], r["aggregate_physics_ticks_per_second"]
    ratio = cm["mean"] / bm["mean"]
    noise = max(bm["std"], cm["std"])
    bad = cm["mean"] < bm["mean"] * (1 - a.margin) and (bm["mean"] - cm["mean"]) > noise
    regressed |= bad
    print(f"{r['num_envs']:>5} {bm['mean']:>13.1f} {cm['mean']:>13.1f} {ratio:>7.2f}x  "
          f"{'REGRESSION' if bad else 'ok' if ratio >= 1 - a.margin else 'within noise'}")
sys.exit(1 if regressed and a.fail_on_regression else 0)
