"""Build the Optimization IV comparison tables from the xy1_bench JSON files of results/opt4/final.

    python3 casu_grid/perf/make_opt4_table.py casu_grid/results/opt4/final
"""
import collections, glob, json, statistics, sys
from pathlib import Path

runs = collections.defaultdict(list)
for f in sorted(glob.glob(str(Path(sys.argv[1]) / "*.json"))):
    d = json.load(open(f)); runs[(d["label"], d["envs"])].append(d)
mean = lambda v, key: statistics.fmean(x[key] for x in runs[v]) if runs.get(v) else float("nan")
mean_sec = lambda v, key: statistics.fmean(x["seconds_per_update"][key] for x in runs[v]) if runs.get(v) else float("nan")
each = lambda v: ", ".join(f"{x['end_to_end_decisions_per_second']:,.0f}" for x in runs[v]) if runs.get(v) else "–"
sizes = sorted({k[1] for k in runs})

print("**Legacy workload (worlds ≥ 1 mis-placed, as measured in Optimization III)** — Optimization III revision `e78cfe2` vs Optimization IV with `CASU_LEGACY_MINI_SPAWN=1`\n")
print("| worlds | workers | Opt III end-to-end (runs) | Opt IV end-to-end (runs) | speed-up | PSS GiB (III → IV) | peak RSS GiB (III → IV) | GPU MiB (III → IV) |")
print("|---:|---:|---|---|---:|---|---|---|")
for n in sizes:
    a, b = mean(("opt3_legacy", n), "end_to_end_decisions_per_second"), mean(("opt4_legacy", n), "end_to_end_decisions_per_second")
    print(f"| {n} | 8 | {a:,.0f} ({each(('opt3_legacy', n))}) | **{b:,.0f}** ({each(('opt4_legacy', n))}) | **{b / a:.3f}×** | "
          f"{mean(('opt3_legacy', n), 'peak_pss_gib'):.1f} → {mean(('opt4_legacy', n), 'peak_pss_gib'):.1f} | "
          f"{mean(('opt3_legacy', n), 'peak_rss_gib'):.1f} → {mean(('opt4_legacy', n), 'peak_rss_gib'):.1f} | "
          f"{max(x['peak_gpu_memory_mib'] for x in runs[('opt3_legacy', n)])} → {max(x['peak_gpu_memory_mib'] for x in runs[('opt4_legacy', n)])} |")
print("\n**Corrected workload (every world at its own spawn)** — Optimization III code + spawn fix (`a7ebcab`) vs Optimization IV HEAD\n")
print("| worlds | workers | Opt III code (runs) | Opt IV (runs) | speed-up | collection s/update (III → IV) | optimiser s/update | PSS GiB (III → IV) | peak RSS GiB (III → IV) |")
print("|---:|---:|---|---|---:|---|---|---|---|")
for n in sizes:
    a, b = mean(("opt3_fixed", n), "end_to_end_decisions_per_second"), mean(("opt4_fixed", n), "end_to_end_decisions_per_second")
    print(f"| {n} | 8 | {a:,.0f} ({each(('opt3_fixed', n))}) | **{b:,.0f}** ({each(('opt4_fixed', n))}) | **{b / a:.3f}×** | "
          f"{mean_sec(('opt3_fixed', n), 'collection'):.2f} → {mean_sec(('opt4_fixed', n), 'collection'):.2f} | "
          f"{mean_sec(('opt3_fixed', n), 'optimization'):.2f} → {mean_sec(('opt4_fixed', n), 'optimization'):.2f} | "
          f"{mean(('opt3_fixed', n), 'peak_pss_gib'):.1f} → {mean(('opt4_fixed', n), 'peak_pss_gib'):.1f} | "
          f"{mean(('opt3_fixed', n), 'peak_rss_gib'):.1f} → {mean(('opt4_fixed', n), 'peak_rss_gib'):.1f} |")
print("\n**Effect of the workload fix alone (same code)**\n")
print("| worlds | Opt III code legacy → fixed | Opt IV code legacy → fixed |")
print("|---:|---|---|")
for n in sizes:
    f = lambda v: mean((v, n), "end_to_end_decisions_per_second")
    print(f"| {n} | {f('opt3_legacy'):,.0f} → {f('opt3_fixed'):,.0f} ({f('opt3_fixed') / f('opt3_legacy') - 1:+.1%}) | {f('opt4_legacy'):,.0f} → {f('opt4_fixed'):,.0f} ({f('opt4_fixed') / f('opt4_legacy') - 1:+.1%}) |")
