"""Analysis of the XY1 v2 three-seed validation (reads casu_grid/results/xy1_v2_validation/seed*/, writes analysis.json and analysis.md next to them).

Pre-registered decision rule (written before any seed finished; stochastic mode is primary because it is the behaviour policy PPO trains):
  ENCOURAGING  : stochastic paired improvement > 0 in all completed seeds AND its cluster-bootstrap 95 % interval excludes 0 in at least 2 of 3 seeds
                 AND deterministic paired improvement >= 0 in at least 2 seeds AND no instability (all losses finite, no run failure) AND layer exits do not dominate finished episodes
  CONCERNING   : stochastic paired improvement <= 0 in at least 2 seeds, OR instability (non-finite values, failed run), OR layer exits dominate (> 50 % of finished training episodes)
  INCONCLUSIVE : everything else (mixed or weak effects, intervals that include 0, fewer than 3 completed seeds).
Paired design: every held-out case (terrain world, goal-draw round) is played by the initial and by the final policy with identical terrain and goals. The 8 goal draws of a world share
its terrain, so episodes are not independent: the primary interval is a cluster bootstrap over the 16 terrains (10,000 resamples, seed 1); an exact McNemar test on the discordant episodes
is reported as a secondary, optimistic figure.
"""
from __future__ import annotations
import json, math, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "casu_grid/results/xy1_v2_validation"


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return min(1.0, p)


def paired(initial: dict, final: dict) -> dict:
    keys = sorted(set(initial) & set(final))
    both = sum(1 for k in keys if initial[k]["ok"] and final[k]["ok"])
    only_final = sum(1 for k in keys if final[k]["ok"] and not initial[k]["ok"])
    only_initial = sum(1 for k in keys if initial[k]["ok"] and not final[k]["ok"])
    n = len(keys)
    worlds = sorted({k[1] for k in keys})
    by_world = {w: [(k) for k in keys if k[1] == w] for w in worlds}
    def diff(sample):
        pairs = [k for w in sample for k in by_world[w]]
        return (sum(1 for k in pairs if final[k]["ok"]) - sum(1 for k in pairs if initial[k]["ok"])) / len(pairs)
    rng = random.Random(1)
    boots = sorted(diff([rng.choice(worlds) for _ in worlds]) for _ in range(10000))
    return {"cases": n, "terrains": len(worlds), "initial_successes": both + only_initial, "final_successes": both + only_final, "both": both, "only_final": only_final, "only_initial": only_initial,
            "initial_rate": (both + only_initial) / n, "final_rate": (both + only_final) / n, "paired_improvement": (only_final - only_initial) / n,
            "cluster_bootstrap_95": [boots[249], boots[9749]], "mcnemar_exact_p": mcnemar_exact(only_final, only_initial)}


def episodes_by_case(result: dict) -> dict:
    return {(e["round"], e["world"]): {"ok": e["outcome"] == "arrival", **e} for e in result["episodes"]}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def training_summary(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    col = lambda k: [r.get(k) for r in rows]
    finite = all(isinstance(v, (int, float)) and math.isfinite(v) for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "grad_norm") for v in col(k))
    steps = rows[-1]["environment_steps"]
    e2e = steps / sum(r["collection_seconds"] + r["optimization_seconds"] for r in rows)
    third = lambda k, a, b: mean(col(k)[a:b])
    n = len(rows)
    finished = sum(r.get("nav/completed_episodes") or 0 for r in rows)
    return {"updates": n, "decisions": steps, "end_to_end_decisions_per_second": e2e, "all_losses_finite": finite,
            "arrivals": sum(r.get("nav/arrivals") or 0 for r in rows), "layer_exits": sum(r.get("nav/layer_exits") or 0 for r in rows), "timeouts": sum(r.get("nav/timeouts") or 0 for r in rows),
            "finished_episodes": finished, "layer_exit_share_of_finished": (sum(r.get("nav/layer_exits") or 0 for r in rows) / finished) if finished else None,
            "arrivals_per_1k": [third("nav/arrivals_per_1k_decisions", i, i + 50) for i in range(0, n, 50)],
            "layer_exits_per_1k": [third("nav/layer_exits_per_1k_decisions", i, i + 50) for i in range(0, n, 50)],
            "progress": [third("nav/active_progress_mean", i, i + 50) for i in range(0, n, 50)],
            "entropy_first10": mean(col("entropy")[:10]), "entropy_last10": mean(col("entropy")[-10:]),
            "kl_mean": mean(col("approx_kl")), "kl_max": max(col("approx_kl")), "updates_with_kl_above_target_0.03": sum(1 for v in col("approx_kl") if v > 0.03),
            "clip_fraction_mean": mean(col("clip_fraction")), "grad_norm_max": max(col("grad_norm")), "explained_variance_first50": third("explained_variance", 0, 50), "explained_variance_last50": third("explained_variance", n - 50, n),
            "value_loss_first10": mean(col("value_loss")[:10]), "value_loss_last10": mean(col("value_loss")[-10:]),
            "attack_fraction_first10": mean(col("action/attack/choice_1_fraction")[:10]), "attack_fraction_last10": mean(col("action/attack/choice_1_fraction")[-10:]),
            "blocks_destroyed_per_1k_first50": third("nav/blocks_destroyed_per_1k_decisions", 0, 50), "blocks_destroyed_per_1k_last50": third("nav/blocks_destroyed_per_1k_decisions", n - 50, n),
            "outside_world_fraction_max": max(v or 0 for v in col("nav/active_outside_world_fraction")),
            "mean_return_first": mean([v for v in col("mean_completed_return")[:50]]), "mean_return_last": mean([v for v in col("mean_completed_return")[-50:]])}


def main():
    out = {"seeds": {}}
    md = []
    for d in sorted(ROOT.glob("seed*")):
        seed = int(d.name[4:])
        rec = {"training_done": (d / "TRAINING_DONE").exists(), "evaluation_done": (d / "EVAL_DONE").exists()}
        if rec["training_done"]:
            manifest = json.loads((d / "training/config.json").read_text())
            launch = json.loads((d / "training/studio_launch.json").read_text())
            rec["manifest"] = {"termination_contract": manifest["environment"].get("termination_contract"), "bottom_exit_terminates": manifest["environment"].get("bottom_exit_terminates"),
                               "seed": manifest["seed"], "num_envs": manifest["environment"]["num_envs"], "episode_steps": manifest["environment"]["episode_steps"], "legacy_spawn": launch["legacy_spawn"],
                               "workers": launch["workers"], "physx_threads": launch["physx_threads"], "device": launch["device"], "revision": manifest.get("git_revision"), "dirty_files": launch["dirty_files"],
                               "experiment": manifest["experiment"].get("id")}
            rec["training"] = training_summary(d / "training/metrics.jsonl")
        if rec["evaluation_done"]:
            ev = json.loads((d / "evaluation.json").read_text())
            rec["evaluation_protocol"] = ev["protocol"] | {"termination_contract": ev.get("termination_contract")}
            cells = {}
            for mode in ("stochastic", "deterministic"):
                res = {r["policy"]: r for r in ev["results"] if r["mode"] == mode}
                ini, fin = res["initial"], next(r for p, r in res.items() if p != "initial")
                a, b = episodes_by_case(ini), episodes_by_case(fin)
                cell = paired(a, b)
                for label, eps in (("initial", ini["episodes"]), ("final", fin["episodes"])):
                    cell[label + "_outcomes"] = {k: sum(1 for e in eps if e["outcome"] == k) for k in sorted({e["outcome"] for e in eps})}
                    cell[label + "_mean_progress"] = mean([1 - e["best_distance"] / e["initial_distance"] for e in eps if e["initial_distance"] > 0])
                    cell[label + "_mean_best_distance"] = mean([e["best_distance"] for e in eps])
                    wins = [e["decisions"] for e in eps if e["outcome"] == "arrival"]
                    cell[label + "_median_decisions_to_arrival"] = sorted(wins)[len(wins) // 2] if wins else None
                    cell[label + "_mean_blocks_destroyed"] = mean([e["blocks_destroyed"] for e in eps])
                cell["final_checkpoint_update"] = fin["checkpoint"]["update_count"]
                cells[mode] = cell
            rec["evaluation"] = cells
        out["seeds"][seed] = rec
    # decision rule
    done = {s: r for s, r in out["seeds"].items() if r.get("evaluation")}
    verdict = "INCONCLUSIVE"
    reasons = []
    if done:
        sto = {s: r["evaluation"]["stochastic"] for s, r in done.items()}
        det = {s: r["evaluation"]["deterministic"] for s, r in done.items()}
        unstable = any(not r["training"]["all_losses_finite"] for r in done.values())
        dominated = any((r["training"]["layer_exit_share_of_finished"] or 0) > 0.5 for r in done.values())
        pos_all = all(c["paired_improvement"] > 0 for c in sto.values())
        sig = sum(1 for c in sto.values() if c["cluster_bootstrap_95"][0] > 0)
        det_ok = sum(1 for c in det.values() if c["paired_improvement"] >= 0)
        nonpos = sum(1 for c in sto.values() if c["paired_improvement"] <= 0)
        if len(done) == 3 and pos_all and sig >= 2 and det_ok >= 2 and not unstable and not dominated:
            verdict = "ENCOURAGING"
        elif nonpos >= 2 or unstable or dominated:
            verdict = "CONCERNING"
        reasons = [f"completed seeds with evaluation: {sorted(done)}", f"stochastic improvement positive in all: {pos_all}", f"stochastic intervals excluding 0: {sig}",
                   f"deterministic improvement >= 0 in {det_ok}", f"non-positive stochastic improvements: {nonpos}", f"unstable: {unstable}", f"layer exits dominate: {dominated}"]
    out["verdict"] = verdict
    out["verdict_reasons"] = reasons
    (ROOT / "analysis.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({"verdict": verdict, "reasons": reasons}, indent=1))
    for s, r in sorted(out["seeds"].items()):
        for mode, c in (r.get("evaluation") or {}).items():
            print(f"seed {s} {mode}: initial {c['initial_successes']}/{c['cases']} -> final {c['final_successes']}/{c['cases']} paired {100 * c['paired_improvement']:+.1f} pts "
                  f"cluster95 [{100 * c['cluster_bootstrap_95'][0]:+.1f},{100 * c['cluster_bootstrap_95'][1]:+.1f}] mcnemar p={c['mcnemar_exact_p']:.3f} outcomes initial {c['initial_outcomes']} final {c['final_outcomes']}")


if __name__ == "__main__":
    main()
