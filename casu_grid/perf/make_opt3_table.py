"""Build the report's benchmark table from xy1_bench JSON files.

    python casu_grid/perf/make_opt3_table.py BASELINE_DIR OPTIMISED_DIR
"""
import collections, glob, json, statistics, sys
from pathlib import Path

base = {}
for f in sorted(glob.glob(str(Path(sys.argv[1]) / "*.json"))):
    d = json.load(open(f)); base[(d["envs"], d["workers"])] = d
opt = collections.defaultdict(list)
for f in sorted(glob.glob(str(Path(sys.argv[2]) / "*.json"))):
    d = json.load(open(f)); opt[(d["envs"], d["workers"])].append(d)
base4 = {k[0]: v for k, v in base.items() if k[1] == 4}
print("| worlds | workers | baseline end-to-end | optimised end-to-end | speed-up | rollout (base → opt) | optimiser s/update (base → opt) | peak RSS GiB (base → opt) | GPU MiB (base → opt) | repeats |")
print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
for k in sorted(opt):
    r = opt[k]; m = lambda key: statistics.fmean(x[key] for x in r)
    e2e = m("end_to_end_decisions_per_second")
    b = base.get(k)
    ref = b if b else base4.get(k[0])
    sp = e2e / ref["end_to_end_decisions_per_second"]
    note = "" if b else " (vs 4-worker baseline)"
    bt = lambda key, fmt: (fmt % ref[key]) if b else "–"
    print(f"| {k[0]} | {k[1]} | {('%.1f' % b['end_to_end_decisions_per_second']) if b else '–'} | **{e2e:,.0f}** | **{sp:.1f}×**{note} | "
          f"{('%.0f' % b['rollout_decisions_per_second']) if b else '–'} → {m('rollout_decisions_per_second'):,.0f} | "
          f"{('%.2f' % b['seconds_per_update']['optimization']) if b else '–'} → {statistics.fmean(x['seconds_per_update']['optimization'] for x in r):.2f} | "
          f"{('%.1f' % b['peak_rss_gib']) if b else '–'} → {max(x['peak_rss_gib'] for x in r):.1f} | "
          f"{b['peak_gpu_memory_mib'] if b else '–'} → {max(x['peak_gpu_memory_mib'] for x in r)} | {len(r)}: " + ", ".join(f"{x['end_to_end_decisions_per_second']:,.0f}" for x in r) + " |")
