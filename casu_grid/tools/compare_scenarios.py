"""Compare scenario-suite outputs of two backends per decision (error as a function of simulated time).

    python casu_grid/tools/compare_scenarios.py REF_DIR CAND_DIR
"""
import json, sys
from pathlib import Path

ref, cand = Path(sys.argv[1]), Path(sys.argv[2])
for f in sorted(ref.glob("*.json")):
    a = json.loads(f.read_text())
    cf = cand / f.name
    if not cf.exists():
        print(f.stem, "candidate missing"); continue
    b = json.loads(cf.read_text())
    (name, ea), (_, eb) = next(iter(a["scenarios"].items())), next(iter(b["scenarios"].items()))
    fa, fb = ea["frames"], eb["frames"]
    print(f"\n{name}: ref {ea['status']} ({len(fa)} decisions), cand {eb['status']} ({len(fb)} decisions)")
    worst = 0.0
    for i, (x, y) in enumerate(zip(fa, fb)):
        d = {k: abs(x[k] - y[k]) for k in ("x", "y", "vx", "vy")}
        flag = "" if x["grounded"] == y["grounded"] and x["level"] == y["level"] else "  DISCRETE DIFF (grounded/level)"
        worst = max(worst, max(d.values()))
        print(f"  t={i * 0.2:4.1f}s  dx={d['x']:.4f} dy={d['y']:.4f} dvx={d['vx']:.3f} dvy={d['vy']:.3f}  ref=({x['x']:.2f},{x['y']:.2f}) cand=({y['x']:.2f},{y['y']:.2f}){flag}")
    print(f"  max deviation {worst:.4f}")
