"""Compare two ``isaac_player_ab.py`` result files (e.g. reference vs experimental backend) side by side.

    python casu_grid/tools/compare_ab.py REF.json CAND.json
"""
import json, sys

a, b = (json.load(open(f)) for f in sys.argv[1:3])
for mode in sorted(set(a["modes"]) & set(b["modes"])):
    ra, rb = a["modes"][mode], b["modes"][mode]
    print(f"== {mode}")
    if mode == "teacher":
        for cls in sorted(set(ra["classes"]) | set(rb["classes"])):
            x, y = ra["classes"].get(cls), rb["classes"].get(cls)
            if not x or not y:
                continue
            print(f"  {cls:14s} n={x['dx']['n']:4d}  " + "  ".join(
                f"{m}: mean {x[m]['mean_abs']:.4f}->{y[m]['mean_abs']:.4f} max {x[m]['max_abs']:.2f}->{y[m]['max_abs']:.2f}" for m in ("dx", "dy", "dvx", "dvy")))
    elif mode == "segment":
        for cls in sorted(k for k in ra if k != "worst" and k in rb and ra[k] and rb[k]):
            x, y = ra[cls], rb[cls]
            print(f"  {cls:16s} n={x['n']:3d}  " + "  ".join(
                f"{m}: mean {x[m]['mean_abs']:.3f}->{y[m]['mean_abs']:.3f} max {x[m]['max_abs']:.2f}->{y[m]['max_abs']:.2f}" for m in ("dx", "dy", "dvx", "dvy")))
    else:
        for key in ("position_x", "position_y", "velocity_x", "velocity_y"):
            print(f"  {key:11s} mean {ra[key]['mean_abs']:.3f}->{rb[key]['mean_abs']:.3f} max {ra[key]['max_abs']:.2f}->{rb[key]['max_abs']:.2f}")
        for key in ("first_x_error_over_0p5", "first_y_error_over_0p5"):
            print(f"  {key}: {ra[key] and ra[key]['action']} -> {rb[key] and rb[key]['action']}")
