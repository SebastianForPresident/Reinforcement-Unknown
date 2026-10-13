# Performance tooling (`casu_grid/perf`)

Everything needed to measure, compare and protect the simulator's throughput. All commands use the
Isaac bundled Python (`"$ISAAC_SIM/python.sh"`, set up as in `ISAAC_LAB_SETUP.md`) from the repository root.

| Tool | Purpose |
|---|---|
| `isaac_throughput_benchmark.py` | simulation-only throughput per batch size: ticks/s, decisions/s, repeated windows (mean ± std), phase breakdown, CPU utilisation, RSS, VRAM, build/reset/snapshot/restore latency, backend + system identity. `--workers N` benchmarks the multi-process layout; `--workload random|run` |
| `isaac_training_benchmark.py` | end-to-end PPO training throughput through `workbench.train`: simulation-only, rollout (simulation + policy inference) and end-to-end (rollout + PPO update) decisions/s, optimizer overhead |
| `perf/run_perf_check.py` | **regression workflow**: unit tests, then the bit-exact trajectory gate against the stored pre-optimisation baseline, then throughput vs `perf/baselines/throughput_reference_*.json` with a noise margin |
| `perf/trajectory_dump.py` + `tools/compare_trajectories.py` | record / compare seeded batched trajectories (works on any checkout via `CASU_SRC_ROOT`) |
| `tools/compare_throughput.py` | compare two benchmark files with a relative margin; flags only drops beyond the margin and the runs' spread |
| `perf/controller_harness.py` | Isaac-free controller/fluid/query harness with a kinematic stand-in engine, for profiling (`--profile`) and reference-vs-compiled equivalence (`--compiled`) |
| `perf/mutation_latency.py` | terrain edit / restore / reset latency (independent terrain) |
| `perf/synthetic_large_model.py` | **synthetic** large policy network for inference-bound training measurements (not a Timmy architecture) |
| `perf/OPTIMIZATION_LOG.md` | every candidate tried, kept or rejected, with evidence |
| `casu_grid/OPTIMIZATION_ACCEPTANCE_REPORT.md` | consolidated results of the campaign |

## Backends and execution layout

| Setting | Values | Default | Meaning |
|---|---|---|---|
| `CASU_BACKEND` | `auto`, `numba`, `reference-python` | `auto` | compiled kernels (fluid, queries, controller) vs the readable reference implementation; `numba` fails loudly if unavailable |
| `CASU_COMPILED_CONTROLLER` | `1`, `0` | `1` | with the numba backend, `0` keeps compiled fluid/queries but runs the reference controller (attribution/debugging) |
| `CASU_NUM_WORKERS` | integer or `auto` | `1` | worker processes for batched environments (`1` = in-process). Same simulated experiment; world partition changes only low-order bits (see below) |

The active backend is recorded by `backend.describe()` and in every benchmark file (`system.backend`);
`env.performance_info()` reports the layout of a live environment (backend, workers, worlds per worker).
numba 0.59.1 ships inside Isaac Sim 4.5's Python; kernels compile on first use (a few seconds) and are cached in
`casu_grid/__pycache__`. `reference-python` needs nothing extra and is the correctness oracle.

## Reproducing the headline numbers

```bash
# in-process, both workloads, scaling with world count
"$ISAAC_SIM/python.sh" casu_grid/isaac_throughput_benchmark.py --terrain-mode shared \
    --num-envs 1 2 4 8 16 32 64 --workload random --output /tmp/shared_random.json
# multi-process (4 workers x 16 worlds)
"$ISAAC_SIM/python.sh" casu_grid/isaac_throughput_benchmark.py --workers 4 --num-envs 64 --output /tmp/w4.json
# actual PPO training (engineering-only config), 64 worlds on 4 workers
"$ISAAC_SIM/python.sh" casu_grid/isaac_training_benchmark.py --envs 64 --workers 4 --output /tmp/train.json
# regression check (run on an idle machine)
"$ISAAC_SIM/python.sh" casu_grid/perf/run_perf_check.py --quick
```

Baseline (pre-optimisation, revision `f54f505`) results are in `perf/results/baseline_*.json`; compare any new
file with `tools/compare_throughput.py`.

## Determinism

* Same code, same `num_envs`, same `num_workers` ⇒ bit-identical trajectories (the gate relies on this).
* Different world partitions (`num_envs`, `num_workers`) change PhysX scene composition and therefore results at the
  1e-9 level; chaotic corner contacts amplify that. Replay a recorded run with the layout it was recorded with.
* Compiled vs reference backend ⇒ bit-identical (tested: fuzzed controller states, fluid oracle, three batched workloads).
