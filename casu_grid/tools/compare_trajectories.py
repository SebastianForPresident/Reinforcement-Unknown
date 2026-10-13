"""Compare two ``casu.trajectory-dump.v1`` files: exact equality plus maximum deviations.

    python casu_grid/tools/compare_trajectories.py A.json B.json [--tolerance 1e-4]

Exit status 0 when every position/velocity differs by at most ``--tolerance`` and all discrete
fields (grounded flags, terminations, reasons, fluid digests) are identical.
"""
import argparse, json, sys

p = argparse.ArgumentParser()
p.add_argument("a"); p.add_argument("b")
p.add_argument("--tolerance", type=float, default=1e-4)
a = p.parse_args()
A, B = json.load(open(a.a)), json.load(open(a.b))
if (A["envs"], A["macrosteps"], A["workload"], A["seed"]) != (B["envs"], B["macrosteps"], B["workload"], B["seed"]):
    sys.exit("dumps are not the same workload")
worst = {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "crouch": 0.0, "reward": 0.0}
discrete_diffs = 0
for fa, fb in zip(A["frames"], B["frames"]):
    for pa, pb in zip(fa["players"], fb["players"]):
        for k in ("x", "y", "vx", "vy"):
            worst[k] = max(worst[k], abs(pa[k] - pb[k]))
    worst["crouch"] = max([worst["crouch"]] + [abs(x - y) for x, y in zip(fa["crouch"], fb["crouch"])])
    worst["reward"] = max([worst["reward"]] + [abs(x - y) for x, y in zip(fa["reward"], fb["reward"])])
    for k in ("terminated", "truncated", "reasons", "grounded", "fluid_ticks", "fluid_digest"):
        discrete_diffs += int(fa[k] != fb[k])
exact = A["sha256"] == B["sha256"]
print(f"{a.a} vs {a.b}: bit-identical={exact} max-dev={ {k: float(f'{v:.3g}') for k, v in worst.items()} } discrete-diffs={discrete_diffs}")
ok = discrete_diffs == 0 and max(worst.values()) <= a.tolerance
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
