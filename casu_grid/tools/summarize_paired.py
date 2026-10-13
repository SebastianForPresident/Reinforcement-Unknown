"""Aggregate several ``compare_ab_paired.py --json`` reports into per-class totals (n-weighted means, better/worse counts).

    python casu_grid/tools/summarize_paired.py REPORT.json [REPORT.json ...]
"""
import json, sys

tot = {}
for path in sys.argv[1:]:
    report = json.load(open(path))
    for mode, groups in report["modes"].items():
        if mode == "teacher_by_seam":
            continue
        for label, g in groups.items():
            if label == "ALL" or (mode == "free" and label == "all"):
                continue
            t = tot.setdefault((mode, label), {"n": 0, "pos_ref": 0.0, "pos_cand": 0.0, "vel_ref": 0.0, "vel_cand": 0.0,
                                               "pb": 0, "pw": 0, "vb": 0, "vw": 0})
            n = g["n"]
            t["n"] += n
            for k, s in (("pos_ref", "pos_ref"), ("pos_cand", "pos_cand"), ("vel_ref", "vel_ref"), ("vel_cand", "vel_cand")):
                t[k] += g[s] * n
            t["pb"] += g["pos_better"]; t["pw"] += g["pos_worse"]; t["vb"] += g["vel_better"]; t["vw"] += g["vel_worse"]
print(f"{'mode / class':28s} {'n':>5s} {'pos ref->cand (m)':>20s} {'vel ref->cand (m/s)':>22s} {'pos b/w':>8s} {'vel b/w':>8s}")
for (mode, label), t in sorted(tot.items()):
    n = t["n"]
    print(f"{mode + ' / ' + label:28s} {n:5d} {t['pos_ref'] / n:9.4f} -> {t['pos_cand'] / n:8.4f} {t['vel_ref'] / n:10.3f} -> {t['vel_cand'] / n:8.3f} {t['pb']:4d}/{t['pw']:<3d} {t['vb']:4d}/{t['vw']:<3d}")
