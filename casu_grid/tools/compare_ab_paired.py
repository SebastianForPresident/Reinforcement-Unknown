"""Paired per-tick / per-decision comparison of two ``isaac_player_ab.py --detail`` result files.

    python casu_grid/tools/compare_ab_paired.py COMPOUND.json HULL.json [--seams] [--json OUT.json]

Aggregate means hide whether a difference is systematic or one event, so every row of the two runs is paired
(same tick, same source state) and counted: ``better`` = candidate error smaller than reference by more than the
tolerance, ``worse`` = larger, ``same`` otherwise. Position error is hypot(dx, dy) in metres, velocity error hypot(dvx, dvy) in m/s.
``--seams`` additionally splits the teacher-forced ticks by the source position's distance to a 64-tile chunk seam.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

POS_TOL, VEL_TOL = 5e-4, 0.01
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def err(row, pos=True):
    return math.hypot(row["dx"], row["dy"]) if pos else math.hypot(row["dvx"], row["dvy"])


def pair(ref_rows, cand_rows, key, label_key=None):
    out = {}
    for a, b in zip(ref_rows, cand_rows):
        assert a[key] == b[key], (a[key], b[key])
        label = a[label_key] if label_key else "all"
        for lab in {label, "ALL"}:
            g = out.setdefault(lab, {"n": 0, "pos_ref": 0.0, "pos_cand": 0.0, "vel_ref": 0.0, "vel_cand": 0.0,
                                     "pos_better": 0, "pos_worse": 0, "vel_better": 0, "vel_worse": 0, "regress": []})
            pr, pc, vr, vc = err(a), err(b), err(a, False), err(b, False)
            g["n"] += 1; g["pos_ref"] += pr; g["pos_cand"] += pc; g["vel_ref"] += vr; g["vel_cand"] += vc
            g["pos_better"] += pc < pr - POS_TOL; g["pos_worse"] += pc > pr + POS_TOL
            g["vel_better"] += vc < vr - VEL_TOL; g["vel_worse"] += vc > vr + VEL_TOL
            g["regress"].append((pc - pr, a[key], pr, pc))
    for g in out.values():
        n = g["n"]
        for k in ("pos_ref", "pos_cand", "vel_ref", "vel_cand"):
            g[k] /= n
        g["regress"] = [(round(d, 4), k, round(r, 4), round(c, 4)) for d, k, r, c in sorted(g["regress"], reverse=True)[:3]]
    return out


def seam_label(trace, i):
    path = FIXTURES / "casu-layer1-seed-391728405-v1" / "source_traces" / trace / "post_physics.jsonl.gz"
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--seams", action="store_true")
    p.add_argument("--json", type=Path)
    args = p.parse_args()
    a, b = json.loads(args.reference.read_text()), json.loads(args.candidate.read_text())
    report = {"trace": a["trace"], "reference": str(args.reference), "candidate": str(args.candidate), "modes": {}}
    source_x = None
    if args.seams and not a["trace"].startswith("probe:"):
        path = seam_label(a["trace"], 0)
        with gzip.open(path, "rt") as stream:
            stream.readline()
            source_x = [json.loads(line)["extra"]["rb_position"][0] for line in stream]
    for mode in sorted(set(a["modes"]) & set(b["modes"])):
        ra, rb = a["modes"][mode], b["modes"][mode]
        if "rows" not in ra or "rows" not in rb:
            continue
        if mode == "teacher":
            report["modes"][mode] = pair(ra["rows"], rb["rows"], "i", "class")
            if source_x is not None:
                for r in ra["rows"]:
                    d = abs(((source_x[r["i"]] + 512.0) % 64.0))
                    r["seam"] = "within 1 m of chunk seam" if min(d, 64.0 - d) < 1.0 else "elsewhere"
                report["modes"]["teacher_by_seam"] = pair(ra["rows"], rb["rows"], "i", "seam")
        elif mode == "segment":
            report["modes"][mode] = pair(ra["rows"], rb["rows"], "action", "kind")
        else:
            report["modes"][mode] = pair(ra["rows"], rb["rows"], "i")
    for mode, groups in report["modes"].items():
        print(f"== {a['trace']} / {mode}  (pos tol {POS_TOL * 1e3:.1f} mm, vel tol {VEL_TOL} m/s)")
        for lab in sorted(groups, key=lambda k: (k != "ALL", k)):
            g = groups[lab]
            print(f"  {lab:26s} n={g['n']:4d} pos {g['pos_ref']:.4f}->{g['pos_cand']:.4f}  vel {g['vel_ref']:.3f}->{g['vel_cand']:.3f}  "
                  f"pos better/worse {g['pos_better']}/{g['pos_worse']}  vel better/worse {g['vel_better']}/{g['vel_worse']}  "
                  f"largest hull-worse {g['regress'][0][:2] + g['regress'][0][2:]}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
