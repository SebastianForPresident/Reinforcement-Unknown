"""Print the markdown tables of ``OPTIMIZATION_ACCEPTANCE_REPORT.md`` from the result files (no hand copying)."""
import json
from pathlib import Path

R = Path(__file__).resolve().parent / "results"


def load(name):
    return json.loads((R / name).read_text())


def rows(name):
    return {r["num_envs"]: r for r in load(name)["rows"]}


def t(r):
    return r["aggregate_physics_ticks_per_second"]["mean"]


def sd(r):
    return r["aggregate_physics_ticks_per_second"]["std"]


def table(title, header, body):
    print(f"\n**{title}**\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|")
    for row in body:
        print("| " + " | ".join(str(c) for c in row) + " |")


# 1. in-process scaling, before/after, both workloads
for workload, base_file, new_file in (("random (crouch-heavy)", "baseline_shared.json", "final_shared_random.json"),
                                      ("run (hold right, periodic jump)", "baseline_shared_run.json", "final_shared_run.json")):
    base, new = rows(base_file), rows(new_file)
    body = []
    for n in sorted(new):
        b = base.get(n)
        body.append([n, f"{t(b):.0f} ± {sd(b):.0f}" if b else "—", f"{t(new[n]):.0f} ± {sd(new[n]):.0f}",
                     f"{t(new[n]) / t(b):.2f}x" if b else "—",
                     f"{new[n]['aggregate_policy_decisions_per_second']['mean']:.0f}",
                     f"{new[n]['peak_rss_gib']:.2f}"])
    table(f"In-process, shared terrain, workload `{workload}` (aggregate physics ticks/s, mean ± std of 3 windows)",
          ["worlds", "baseline", "optimized", "speedup", "decisions/s (opt.)", "peak RSS GiB"], body)

# 2. independent terrain
for workload, base_file, new_file in (("random", "baseline_independent.json", "final_independent_random.json"),
                                      ("run", None, "final_independent_run.json")):
    new = rows(new_file)
    base = rows(base_file) if base_file else {}
    body = [[n, f"{t(base[n]):.0f}" if n in base else "—", f"{t(new[n]):.0f} ± {sd(new[n]):.0f}",
             f"{t(new[n]) / t(base[n]):.2f}x" if n in base else "—", f"{new[n]['peak_rss_gib']:.2f}",
             f"{new[n]['construct_seconds']:.0f}"] for n in sorted(new)]
    table(f"In-process, independent terrain, workload `{workload}`",
          ["worlds", "baseline", "optimized", "speedup", "peak RSS GiB", "construct s (before copy-build)"], body)

# 3. sharded
for workload, base_n16 in (("random", t(rows("baseline_shared.json")[16])), ("run", t(rows("baseline_shared_run.json")[16]))):
    body = []
    for w in (2, 3, 4, 6):
        for n, r in sorted(rows(f"final_shared_{workload}_w{w}.json").items()):
            body.append([w, n, f"{t(r):.0f} ± {sd(r):.0f}", f"{t(r) / base_n16:.1f}x", f"{r['peak_rss_gib']:.1f}",
                         f"{r['aggregate_policy_decisions_per_second']['mean']:.0f}"])
    table(f"Multi-process (`CASU_NUM_WORKERS`), shared terrain, workload `{workload}` (speedup vs baseline 16-world in-process)",
          ["workers", "worlds", "ticks/s", "vs baseline@16", "total RSS GiB", "decisions/s"], body)

# 4. training
base = {n: load(f"train_baseline_e{n}.json") for n in (1, 4, 16, 64)}
opt_files = [(1, 1), (4, 1), (16, 1), (16, 2), (16, 4), (32, 4), (64, 1), (64, 4), (96, 6)]
body = []
for n, w in opt_files:
    r = load(f"train_opt_e{n}_w{w}.json")
    b = base.get(n)
    body.append([n, w, f"{b['end_to_end_decisions_per_second']:.0f}" if b else "—", f"{r['simulation_only_decisions_per_second']:.0f}",
                 f"{r['rollout_decisions_per_second']:.0f}", f"{r['end_to_end_decisions_per_second']:.0f}",
                 f"{r['end_to_end_decisions_per_second'] / b['end_to_end_decisions_per_second']:.1f}x" if b else "—",
                 f"{100 * r['optimizer_overhead_fraction']:.0f}%"])
table("Actual PPO training (engineering-only smoke configuration), decisions/s",
      ["worlds", "workers", "baseline end-to-end", "sim-only", "rollout", "end-to-end", "speedup", "optimizer share"], body)
body = []
for label, f in (("baseline code, 16 worlds", "train_synth_baseline_e16.json"), ("optimized, 16 worlds, 1 process", "train_synth_opt_e16_w1.json"),
                 ("baseline code, 64 worlds", "train_synth_baseline_e64.json"), ("optimized, 64 worlds, 4 workers", "train_synth_opt_e64_w4.json")):
    r = load(f)
    body.append([label, f"{r['simulation_only_decisions_per_second']:.0f}", f"{r['rollout_decisions_per_second']:.0f}",
                 f"{r['end_to_end_decisions_per_second']:.0f}", f"{100 * r['optimizer_overhead_fraction']:.0f}%",
                 f"{100 * r['policy_inference_fraction_of_rollout']:.0f}%"])
table("SYNTHETIC large policy network (512x512x512 + 512x512), decisions/s",
      ["run", "sim-only", "rollout", "end-to-end", "optimizer share", "inference share of rollout"], body)

# 5. latency
m = load("final_mutation_latency.json")
print("\nmutation latency (independent, 4 worlds): set_block %.3f s, restore after edit %.1f s, reset one world %.4f s" % (
    m["set_block_seconds"]["median"], m["restore_after_edit_seconds"]["median"], m["reset_one_world_seconds"]["median"]))
b = rows("baseline_shared.json")
for n in (1, 16):
    r = rows("final_shared_random.json")[n]
    print(f"shared n={n}: reset all {1e3 * r['reset_all_seconds']:.0f} ms (was {1e3 * b[n]['reset_all_seconds']:.0f}), reset one {1e3 * r['reset_one_world_seconds']:.1f} ms, "
          f"snapshot {1e3 * r['snapshot_seconds']:.1f} ms (was {1e3 * b[n]['snapshot_seconds']:.1f}), restore {r['restore_seconds']:.2f} s (was {b[n]['restore_seconds']:.2f})")
