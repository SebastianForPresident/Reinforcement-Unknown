"""Compare Isaac replays (compound vs convex hull) with real CasU random-tape recordings.

    python casu_grid/validation/analyze_random_tapes.py --traces casu_grid/validation/casu_random_tapes \
        --results DIR_WITH_rand_<seed>_<collider>.json --seeds 2001 ... 2010 [--json OUT.json]

Inputs: packed CasU traces + ``outcomes.json`` (recorded by validation/record_casu_random_tapes.sh) and
``isaac_player_ab.py --detail`` results (free, teacher, segment) for both colliders on the same tapes. Pure standard library.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import random
from pathlib import Path

COLLIDERS = ("compound", "convex_hull")
CAP_Y = 511.9          # highest rigid-body y reached by CasU in any recording (world-top limit)


def read_result(directory: Path, stem: str):
    """Load ``stem.json`` or ``stem.json.gz`` (compressed archives of --detail results)."""
    plain = directory / f"{stem}.json"
    if plain.exists():
        return json.loads(plain.read_text())
    with gzip.open(directory / f"{stem}.json.gz", "rt") as stream:
        return json.load(stream)


def load_post_physics(folder: Path):
    with gzip.open(folder / "post_physics.jsonl.gz", "rt") as stream:
        stream.readline()
        return [json.loads(line) for line in stream]


def casu_events(folder: Path):
    out = json.loads((folder / "outcomes.json").read_text())
    steps = out["steps"]
    rag_calls = [c for c in out["ragdoll_calls"] if (c.get("activeActionSequence") or 0) >= 1]
    first_rag = next((i + 1 for i, s in enumerate(steps) if s["outcome"]["TimeRagdolled"] > 0), None)
    dead = next((i + 1 for i, s in enumerate(steps) if s["outcome"]["PlayerDead"]), None)
    post = load_post_physics(folder)
    ys = [r["extra"]["rb_position"][1] for r in post]
    return {"ragdoll_calls_in_tape": rag_calls, "first_ragdoll_decision": first_rag, "dead_decision": dead,
            "max_rb_y": max(ys), "min_rb_y": min(ys), "ticks": len(post),
            "first_decision_at_world_top": next((i // 10 + 1 for i, y in enumerate(ys) if y > 511.0), None),
            "max_pain": max(s["outcome"]["AveragePain"] for s in steps),
            "min_blood": min(s["outcome"]["BloodVolume"] for s in steps),
            "min_consciousness": min(s["outcome"]["Consciousness"] for s in steps),
            "block_health_sum_range": [min(s["outcome"]["block_health_sum"] for s in steps),
                                       max(s["outcome"]["block_health_sum"] for s in steps)],
            "attack_swings": max(s["outcome"]["AttackSwingsExecuted"] for s in steps)}, post


def first_over(series, key, limit):
    for row in series:
        if abs(row[key]) > limit:
            return row["action"]
    return None


def isaac_summary(result, src_post):
    free = result["modes"]["free"]["rows"]
    pos = [math.hypot(r["dx"], r["dy"]) for r in free]
    terminated = next((r for r in free if r["terminated"]), None)
    teacher = {r["i"]: r for r in result["modes"]["teacher"]["rows"]}
    first_div = next((r for r in free if math.hypot(r["dx"], r["dy"]) > 0.05), None)
    stop = next((i for i, r in enumerate(free) if r["terminated"]), len(free))
    speeds = [abs(src_post[r["i"]]["velocity"][1] + r["dvy"]) for r in free[:stop]]
    speeds_x = [abs(src_post[r["i"]]["velocity"][0] + r["dvx"]) for r in free[:stop]]
    ejections = [(r["action"], r["tick"], teacher.get(r["i"], {}).get("class"), round(src_post[r["i"]]["velocity"][1] + r["dvy"], 1))
                 for r in free[:stop] if abs(src_post[r["i"]]["velocity"][1] + r["dvy"]) > 40 or abs(src_post[r["i"]]["velocity"][0] + r["dvx"]) > 40]
    summary = {
        "decisions_observed_before_first_termination": stop // 10 if terminated else len(free) // 10,
        "ticks_before_first_termination": stop,
        "ticks_with_speed_over_40_before_termination": len(ejections),
        "first_speed_over_40": ejections[0] if ejections else None,
        "max_abs_vy_before_termination": max(speeds) if speeds else 0.0,
        "first_termination": None if terminated is None else {"decision": terminated["action"], "reason": terminated["reason"]},
        "first_error_over_5cm": None if first_div is None else {
            "decision": first_div["action"], "tick": first_div["tick"],
            "casu_contact_class": teacher.get(first_div["i"], {}).get("class")},
        "first_error_over_0p5m": first_over([{"action": r["action"], "e": math.hypot(r["dx"], r["dy"])} for r in free], "e", 0.5),
        "first_error_over_2m": first_over([{"action": r["action"], "e": math.hypot(r["dx"], r["dy"])} for r in free], "e", 2.0),
        "position_error_at_decision": {str(d): pos[min(len(pos) - 1, d * 10 - 1)] for d in (10, 25, 50, 100, 200)},
        "mean_position_error": sum(pos) / len(pos),
        "teacher": {"n": len(teacher),
                    "mean_position_error": sum(math.hypot(r["dx"], r["dy"]) for r in teacher.values()) / len(teacher),
                    "mean_velocity_error": sum(math.hypot(r["dvx"], r["dvy"]) for r in teacher.values()) / len(teacher)},
        "segment_mean_position_error": sum(math.hypot(r["dx"], r["dy"]) for r in result["modes"]["segment"]["rows"])
        / len(result["modes"]["segment"]["rows"]),
        "ticks_with_isaac_above_casu_cap": sum(1 for r, s in zip(free, src_post) if s["extra"]["rb_position"][1] + r["dy"] > CAP_Y + 0.05),
    }
    return summary


def poisson_cdf(k, mu):
    term = total = math.exp(-mu)
    for i in range(1, k + 1):
        term *= mu / i
        total += term
    return total


def poisson_ci(k, exposure, alpha=0.05):
    """Exact (Garwood) confidence interval of an event rate k / exposure."""
    def solve(target, upper):
        lo, hi = 0.0, 1000.0 + 10.0 * k
        for _ in range(200):
            mid = (lo + hi) / 2
            value = poisson_cdf(k, mid) if upper else 1.0 - poisson_cdf(k - 1, mid)
            if (value > target) == upper:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    upper = solve(alpha / 2, True) / exposure
    lower = 0.0 if k == 0 else solve(alpha / 2, False) / exposure
    if k == 0:
        upper = -math.log(alpha / 2) / exposure       # two-sided 95 % bound for zero events
    return [lower, upper]


def bootstrap(diffs, rounds=20000, seed=7):
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(rounds))
    return {"mean": sum(diffs) / len(diffs), "ci95": [means[int(0.025 * rounds)], means[int(0.975 * rounds)]]}


def sign_test(diffs):
    wins = sum(d < 0 for d in diffs); losses = sum(d > 0 for d in diffs); n = wins + losses
    if n == 0:
        return {"hull_better": wins, "hull_worse": losses, "p_two_sided": 1.0}
    k = min(wins, losses)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"hull_better": wins, "hull_worse": losses, "ties": len(diffs) - n, "p_two_sided": p}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--json", type=Path)
    args = p.parse_args()
    report = {"schema": "casu.random-tape-comparison.v1", "tapes": {}, "aggregate": {}}
    for seed in args.seeds:
        name = f"rand_{seed}"
        events, post = casu_events(args.traces / name)
        entry = {"casu": events}
        for collider in COLLIDERS:
            result = read_result(args.results, f"{name}_{collider}")
            entry[collider] = isaac_summary(result, post)
        report["tapes"][name] = entry
    agg = report["aggregate"]
    tapes = list(report["tapes"].values())
    for metric, getter in (("teacher_mean_position_error", lambda t, c: t[c]["teacher"]["mean_position_error"]),
                           ("teacher_mean_velocity_error", lambda t, c: t[c]["teacher"]["mean_velocity_error"]),
                           ("segment_mean_position_error", lambda t, c: t[c]["segment_mean_position_error"]),
                           ("free_mean_position_error", lambda t, c: t[c]["mean_position_error"]),
                           ("free_error_at_decision_50", lambda t, c: t[c]["position_error_at_decision"]["50"]),
                           ("free_error_at_decision_200", lambda t, c: t[c]["position_error_at_decision"]["200"])):
        diffs = [getter(t, "convex_hull") - getter(t, "compound") for t in tapes]
        agg[metric] = {"compound_mean": sum(getter(t, "compound") for t in tapes) / len(tapes),
                       "hull_mean": sum(getter(t, "convex_hull") for t in tapes) / len(tapes),
                       "paired_difference_hull_minus_compound": bootstrap(diffs), "sign_test": sign_test(diffs)}
    for collider in COLLIDERS:
        first = [t[collider]["first_error_over_0p5m"] or 201 for t in tapes]
        agg[f"{collider}_median_decision_first_error_over_0p5m"] = sorted(first)[len(first) // 2]
        agg[f"{collider}_terminations"] = {}
        for t in tapes:
            ft = t[collider]["first_termination"]
            key = ft["reason"] if ft else "none within 200 decisions"
            agg[f"{collider}_terminations"][key] = agg[f"{collider}_terminations"].get(key, 0) + 1
    casu_decisions = sum(len(json.loads((args.traces / f"rand_{sd}" / "outcomes.json").read_text())["steps"]) for sd in args.seeds)
    casu_post_speed = 0
    agg["casu"] = {"decisions": casu_decisions, "hard_landing_events": 0, "deaths": 0,
                   "hard_landing_rate_per_decision_ci95": poisson_ci(0, casu_decisions),
                   "ragdoll_events_any_cause": sum(len(t["casu"]["ragdoll_calls_in_tape"]) for t in tapes),
                   "ragdoll_rate_per_decision_ci95": poisson_ci(sum(len(t["casu"]["ragdoll_calls_in_tape"]) for t in tapes), casu_decisions)}
    for collider in COLLIDERS:
        exposure = sum(t[collider]["decisions_observed_before_first_termination"] for t in tapes)
        hard = sum(1 for t in tapes if t[collider]["first_termination"] and t[collider]["first_termination"]["reason"] == "ragdoll:hit_ground_hard")
        top = sum(1 for t in tapes if t[collider]["first_termination"] and t[collider]["first_termination"]["reason"] == "world_bounds")
        agg[f"{collider}_hazard"] = {"decisions_at_risk": exposure, "hard_landing_events": hard, "world_top_exits": top,
                                     "hard_landing_rate_per_decision": hard / exposure, "hard_landing_rate_ci95": poisson_ci(hard, exposure),
                                     "world_top_exit_rate_per_decision": top / exposure, "world_top_exit_rate_ci95": poisson_ci(top, exposure),
                                     "ticks_speed_over_40": sum(t[collider]["ticks_with_speed_over_40_before_termination"] for t in tapes),
                                     "ticks_observed": sum(t[collider]["ticks_before_first_termination"] for t in tapes)}
        classes = {}
        for t in tapes:
            fd = t[collider]["first_error_over_5cm"]
            if fd:
                classes[fd["casu_contact_class"]] = classes.get(fd["casu_contact_class"], 0) + 1
        agg[f"{collider}_first_5cm_divergence_by_casu_contact_class"] = classes
    agg["casu_tapes_with_ragdoll"] = sum(1 for t in tapes if t["casu"]["first_ragdoll_decision"])
    agg["casu_tapes_dead"] = sum(1 for t in tapes if t["casu"]["dead_decision"])
    agg["casu_tapes_reaching_world_top"] = sum(1 for t in tapes if t["casu"]["max_rb_y"] > 511.0)
    if args.json:
        args.json.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["aggregate"], indent=1))
    print()
    print(f"{'tape':10s} {'CasU ragdoll':>12s} {'top y':>7s} | compound: first term / >0.5m / mean err | hull: first term / >0.5m / mean err")
    for name, t in report["tapes"].items():
        def f(c):
            ft = t[c]["first_termination"]
            return f"{(ft['reason'].split(':')[-1][:10] + '@' + str(ft['decision'])) if ft else '-':16s} {str(t[c]['first_error_over_0p5m']):>4s} {t[c]['mean_position_error']:8.3f}"
        print(f"{name:10s} {str(t['casu']['first_ragdoll_decision']):>12s} {t['casu']['max_rb_y']:7.2f} | {f('compound')} | {f('convex_hull')}")


if __name__ == "__main__":
    main()
