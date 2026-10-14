"""Performance-regression workflow: correctness gates first, then (noisy) throughput comparison.

    python casu_grid/perf/run_perf_check.py            # unit tests + trajectory gate + throughput vs reference
    python casu_grid/perf/run_perf_check.py --quick    # fewer windows, skip the unit tests
    python casu_grid/perf/run_perf_check.py --update-reference   # record the current machine/code as the reference

Deterministic gates (never flaky): the unit suite and the batched-trajectory gate, which reruns
seeded workloads (shared and independent terrain) and requires the result to equal the stored pre-optimisation baseline
(``perf/baselines/trajectories``, recorded from revision f54f505) -- bit-identical by default.
Throughput is compared to ``perf/baselines/throughput_reference_*.json`` with a relative noise margin
(``--margin``, default 15%); only drops beyond the margin *and* beyond both runs' observed spread are
reported as regressions. Reference numbers are machine specific: re-record with ``--update-reference``
after hardware changes. Exit status is non-zero when a correctness gate fails (always) or a
throughput regression is found (with ``--fail-on-regression``).
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "casu_grid/perf/baselines"
OUT = ROOT / "casu_grid/perf/results"
GATE_CONFIGS = (("e16_random_fixture", 16, "random", "fixture", "shared", 80),
                ("e4_run_fixture", 4, "run", "fixture", "shared", 80),
                ("e4_random_canonical", 4, "random", "canonical", "shared", 80),
                ("ind4_random_fixture", 4, "random", "fixture", "independent", 60),
                ("ind4_run_fixture", 4, "run", "fixture", "independent", 60),
                # experimental convex-hull player collider: its own baselines (not equivalent to the compound ones)
                ("hull_e16_random_fixture", 16, "random", "fixture", "shared", 80),
                ("hull_e4_run_fixture", 4, "run", "fixture", "shared", 80),
                ("hull_ind4_random_fixture", 4, "random", "fixture", "independent", 60))

p = argparse.ArgumentParser()
p.add_argument("--quick", action="store_true")
p.add_argument("--margin", type=float, default=0.15)
p.add_argument("--tolerance", type=float, default=0.0, help="allowed trajectory deviation (0 = bit-identical)")
p.add_argument("--workers", type=int, default=1, help="also check this many worker processes (throughput only)")
p.add_argument("--update-reference", action="store_true")
p.add_argument("--fail-on-regression", action="store_true")
args = p.parse_args()
env = dict(os.environ)
env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
env.setdefault("CASU_SHAPE_BATCH", "0")      # the bit-exact gates test the historical sequential shape-switch order
summary = {"schema": "casu.perf-check.v1", "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "gates": {}, "throughput": {}}
failed = False


def run(cmd, **kw):
    return subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env, capture_output=True, text=True, **kw)


if not args.quick and not args.update_reference:
    r = run(["-m", "unittest", "discover", "-s", "casu_grid/tests", "-p", "test_*.py"])
    ok = r.returncode == 0
    summary["gates"]["unit_tests"] = {"status": "passed" if ok else "failed", "tail": r.stderr.strip().splitlines()[-3:]}
    failed |= not ok
    print("unit tests:", "passed" if ok else "FAILED", flush=True)

if not args.update_reference:
    for name, envs, workload, spawn, terrain, steps in GATE_CONFIGS:
        collider = ["--player-collider", "convex_hull"] if name.startswith("hull_") else []
        out = OUT / "check_tmp" / f"{name}.json"
        r = run(["casu_grid/perf/trajectory_dump.py", "--envs", str(envs), "--workload", workload, "--spawn", spawn,
                 "--terrain-mode", terrain, "--macrosteps", str(steps), "--output", str(out), *collider])
        if r.returncode != 0 or not out.exists():
            summary["gates"][f"trajectory_{name}"] = {"status": "failed", "error": r.stdout[-400:] + r.stderr[-400:]}
            failed = True
            print(f"trajectory {name}: dump FAILED", flush=True)
            continue
        c = run(["casu_grid/tools/compare_trajectories.py", str(BASE / "trajectories" / f"{name}.json"), str(out),
                 "--tolerance", str(args.tolerance)])
        ok = c.returncode == 0
        summary["gates"][f"trajectory_{name}"] = {"status": "passed" if ok else "failed", "detail": c.stdout.strip().splitlines()[:2]}
        failed |= not ok
        print(f"trajectory {name}:", "passed" if ok else "FAILED", "|", c.stdout.strip().splitlines()[0][-160:], flush=True)

regression = False
for workload in ("random", "run"):
    out = OUT / "check_tmp" / f"throughput_{workload}.json"
    r = run(["casu_grid/isaac_throughput_benchmark.py", "--terrain-mode", "shared", "--num-envs", "1", "4", "16",
             "--windows", "2" if args.quick else "3", "--workload", workload, "--workers", "1",
             "--label", f"perf-check {workload}", "--output", str(out)])
    if r.returncode != 0:
        summary["throughput"][workload] = {"status": "failed", "error": r.stderr[-400:]}
        failed = True
        continue
    ref = BASE / f"throughput_reference_{workload}.json"
    if args.update_reference:
        ref.write_text(out.read_text())
        print("reference updated:", ref)
        continue
    if not ref.exists():
        summary["throughput"][workload] = {"status": "no-reference"}
        continue
    c = run(["casu_grid/tools/compare_throughput.py", str(ref), str(out), "--margin", str(args.margin)] +
            (["--fail-on-regression"] if args.fail_on_regression else []))
    summary["throughput"][workload] = {"status": "regression" if c.returncode else "ok", "table": c.stdout.strip().splitlines()}
    regression |= bool(c.returncode)
    print(c.stdout, flush=True)

OUT.mkdir(parents=True, exist_ok=True)
(OUT / f"check_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(summary, indent=2) + "\n")
print("correctness gates:", "FAILED" if failed else "passed", "| throughput:", "REGRESSION" if regression else "ok")
sys.exit(1 if failed or (regression and args.fail_on_regression) else 0)
