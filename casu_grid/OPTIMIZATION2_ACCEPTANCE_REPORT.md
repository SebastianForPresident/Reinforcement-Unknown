# Optimization II — acceptance report

Campaign file `optimization2.txt`. Starting point: `a075acc` (accepted CPU-PhysX result of campaign I, `OPTIMIZATION_ACCEPTANCE_REPORT.md`).
Machine: i7-11700K (8 cores / 16 threads), RTX 3070 8 GB, 32 GB RAM, Fedora 44, Isaac Sim 4.5.0 / Isaac Lab 2.1.1 (pinned, not upgraded).
Experiment log with every candidate and its evidence: [`perf/OPTIMIZATION2_LOG.md`](perf/OPTIMIZATION2_LOG.md). Raw results: `perf/results/` (`prof2_*`,
`o1_*`, `o2_*`, `gpu_probe/`, `fall_*`, `hull_shared16_*`, `ppo_update_*`, `soak_*`) and `results/isaac_*_convex_hull_*.json`.

## 1. Executive summary

| question | answer |
|---|---|
| Is the validated physics (CPU PhysX, compound collider) unchanged? | **Yes.** All 5 compound trajectory gates stay bit-identical to the `f54f505` baselines; unit, vector, sharded and A/B checks pass (section 12). |
| End-to-end PPO training, 64 worlds / 4 workers, validated physics | **1,420 → 1,570 decisions/s (+11 %)** vs `a075acc` re-measured the same day (reference PPO execution: 1,526, simulator changes alone +7 %). The same code was 202 decisions/s at the original baseline `f54f505` (**7.8×**). |
| Fastest configuration (EXPERIMENTAL hull collider) | 64/4: **1,939** (×1.37 over validated); 96/6: **2,266**; simulation-only up to **59,855 ticks/s** (6 workers, 144 worlds, `run` tape) versus 35,461 for the compound collider. Different physics model, opt-in only. |
| GPU PhysX | **Rejected for production.** Not faster at the batch sizes that matter, crashes on crouch/jump tapes, tunnels through thin walls at run speed. Kept as an isolated, documented-unsupported switch. |
| PPO update execution | 1.10× faster update (323 → 294 ms for 64 worlds), **bit-identical** parameters, optimizer state, metrics and RNG; no researcher setting touched. |
| Pipelining simulation and inference | Analysed and measured; **not implemented** — policy inference is 3–6 % of a rollout and on-policy causality forbids overlapping the update. |
| Defect found and fixed in the campaign-I code | numba leaked ≈ 600 B per call when a numpy `Generator` was passed to the compiled kernels: a 500-update 64-world soak grew from 13.3 to 18.7 GiB and lost 10 % throughput. Fixed with a bit-exact PCG64 state-array port; the same soak is now flat. |
| Remaining ceiling | Validated path: PhysX-USD shape switching (47 % of a crouch-heavy tick) and 8 cores. Hull path: PhysX step plus the update phase of PPO (32–39 % of an update cycle). |

## 2. Final commit and branch state

Branch `isaac-lab`. Campaign II commits (all pushed): `2368656` leak fix, `67d951c` PPO update execution, `e627fdc` hull collider + diagnostics,
`a147692` GPU feasibility tools and profile, plus the final documentation/results commit recorded in the completion summary
(see `git log a075acc..HEAD`). The working tree keeps the unrelated pre-existing deletions (`PPO_Python/matched_pairs*`, `Instructions.txt`)
and untracked briefing files untouched; the workbench/GUI/integration files belong to the concurrent integration work and were not staged by this campaign.

## 3. Updated bottleneck analysis (workstream A)

Per world-tick wall time, one process, full world (`perf/results/prof2_*.json`, sub-timers `diag_*`):

| tape | µs / world-tick | pre-phase (fluid + fixed update) | tensor write/read | PhysX step | compiled post | crouch-shape switching |
|---|---:|---:|---:|---:|---:|---:|
| 16 worlds, random (crouch-heavy) | 259 | 24 | 14 / 6 | 73 | 16 | **122 (47 %)** |
| 16 worlds, landing | 191 | 23 | 11 / 5 | 71 | 15 | 62 |
| 16 worlds, run (no crouch) | 122 | 22 | 11 / 5 | 59 | 17 | 3 |
| 64 worlds, random | 233 | 23 | 6 / 2 | 62 | 16 | 120 |
| 4 independent-terrain worlds | 494 | 28 | 41 / 21 | **259 (52 %)** | 20 | 117 |
| 4 workers × 16 worlds (wall), random | 86 | 8 | 5 / 2 | 22 | 5 | 36 |

Ranking on the crouch-heavy tape: (1) crouch-shape switching, 12 `collisionEnabled` USD writes per level change at ≈ 15 µs each (USD + physics notice
cost, independent of shape type), (2) PhysX step, (3) fluid/FixedUpdate pre-phase, (4) compiled post-phase, (5) tensor staging. In PPO training the update
is 27–39 % of a cycle once the simulator is fast; inference is 3–6 % of the rollout; worker straggler wait is 4–7 %; pipe overhead 1 %.

## 4. CPU results (workstream C)

| candidate | result | verdict |
|---|---|---|
| pause PhysX change tracking around the writes | no effect (3.58 vs 3.75 ms per 16 changes) | rejected |
| Sdf layer-level writes instead of `Usd` `Set` | 12.7 vs 15.1 µs per write | not worth it |
| `Sdf.ChangeBlock` batching (campaign I, re-tested) | 0.1 m trajectory change | rejected |
| **single convex-hull player collider** (2 writes per level) | random 3,862 → 10,672 ticks/s (2.8×; 8,851 in an earlier, busier session), landing 5,225 → 13,300, run 8,223 → 14,384 at 16 worlds; PhysX step 73 → 29 µs | **experimental opt-in** (`CASU_PLAYER_COLLIDER=convex_hull`) |
| PCG64 state-array port (leak fix) | bit-identical, soak flat, +7 % end-to-end training | **accepted, default** |
| PhysX threads, Fabric off | no effect | closed |

The hull collider is a different physics model and is isolated accordingly: separate config switch (never the default), own trajectory baselines
(`perf/baselines/trajectories/hull_*.json`, three gates in `run_perf_check.py`), own validation files, and results/benchmarks label it "convex hull".
Startup and memory are not worse than the compound collider: world construction 7.6 s vs 8.3 s (16 worlds), 10.1 s vs 13.1 s (64), 17.8 s vs 18.3 s
(4 independent worlds); peak RSS 2.80 vs 2.86 GiB (16), 3.29 vs 3.54 GiB (64); snapshot restore 1.6–1.9 s vs 1.8–2.3 s.

## 5. GPU feasibility and prototype (workstream B)

Implemented: `use_cuda` config switch (documented unsupported), host-staged tensor I/O, `CASU_GPU_PHYSX` buffer overrides, `perf/gpu_probe.py`,
`perf/gpu_scenarios.py` + `run_gpu_scenarios.sh`, `perf/fall_test.py`, `--cuda` in the benchmark and A/B tools.

| finding | evidence |
|---|---|
| builds, steps, crouch-shape switching and terrain edits work | scenario runs match CPU within 0.07 for crouch tunnel, junction, landing |
| faster only for very large single-process batches without crouching | `run` tape, 16 / 64 / 256 worlds: GPU 4,545 / 11,142 / 16,727 vs CPU 7,366 / 8,318 / 4,568 ticks/s |
| crash on crouch/jump tapes | 16, 64, 256 worlds: CUDA illegal memory access ("Failed to get rigid body transforms"), context unusable afterwards |
| tunnelling | the player passes through 1- and 4-cell walls at 16–25 m/s; CPU stops at the same speed even without CCD |
| contact offset 0.2 repairs the scenarios | but changes a protected setting; seams unvalidated; crashes persist |
| CPU multi-process is faster | 4 CPU workers 23.7k (64 worlds, `run`) vs best GPU single process 16.7k |

Verdict: not production; not promoted. 8× larger GPU buffers did not move the first crash, so it is not a capacity problem.

## 6. GPU-vs-CPU fidelity

Where the GPU runs at all it is compared with CPU PhysX on the seven scenario suite (`tools/compare_scenarios.py`): matches on tunnels/landing (≤ 0.07 m)
with enlarged contact offsets, **fails** walls at default offsets (passes through), crashes on any random tape. It is therefore not validated against CasU
and no GPU performance number is presented as production performance. Falls from 20 / 50 m land identically on CPU (compound and hull), no tunnelling.

Hull vs compound (both CPU PhysX) against recorded CasU traces, `isaac_player_ab.py --player-collider convex_hull`, same options as the accepted runs:

| trace / measure | compound | hull |
|---|---:|---:|
| course_v1 free-run mean / max |dx| | 0.71 / 8.0 m | 0.009 / 0.36 m |
| course_v1 landing decisions mean |dy| | 0.275 m | 0.011 m |
| ice_run free-run mean |dy| | 0.204 m | 0.014 m |
| ice_run decision-level mean |dvx| | 0.595 | 0.298 m/s |
| flat_master wall/ceiling mean |dx| | 0.022 | 0.007 m |
| flat_master landing mean |dy| (worse) | 0.072 | 0.152 m |
| immersed_standing free-run mean |dx| (marginally worse) | 0.060 | 0.078 m |
| destroy_support (delay 31) landing decisions mean |dx| / |dy| | 0.244 / 0.359 m | 0.013 / 0.000 m |
| destroy_support free-run mean |dx| / |dy| | 0.192 / 0.207 m | 0.192 / 0.207 m (identical) |

Scenario suite hull vs compound: ≤ 0.0007 (crouch tunnel), 0.0000 (junction), 0.0003 (landing), 0.0225 (standing tunnel), 0.061 (walls). The hull is closer
to CasU on most traces but not uniformly; it must not be mixed with compound runs in one comparison.

## 7. PPO execution results (workstream D)

`perf/ppo_update_bench.py` (real `PPOTrainer`, engineering contract, fake vector env, identical start state): per minibatch of 16 samples forward 261 µs,
forward+loss+backward 689 µs, grad-clip 100 µs, Adam 250 µs (foreach 200, fused 105). 64 worlds = 256 minibatches per update.

| change | gain | bit-identical |
|---|---:|---|
| one gather per epoch; statistics accumulated on tensors, one host conversion per update | ≈ 4 % | yes |
| `Adam(foreach=True)` | ≈ 5 % | yes |
| distribution argument validation off | ≈ 2 % | yes |
| fused Adam | 105 µs/step | **no** (1 ulp) — not adopted |
| torch threads 1/2/4/8 | ±3 % | n/a |
| total (small net) | 323 → 294 ms per update (1.10×) | parameters, Adam moments, metrics, RNG state |
| total (synthetic 512-wide net) | 1,034 → 996 ms (1.01–1.04×) | same |

Selected with `CASU_PPO_EXECUTION=optimized|reference` (default optimized). `tests/test_ppo_execution.py`: default contract, `target_kl` early stop,
uneven minibatches, larger network, recurrent dispatch, foreach equality. Floor: PyTorch per-op overhead; the number of minibatches is researcher-owned.
Pre-existing, unchanged finding: the recurrent reference update path fails for non-matching rollout/minibatch layouts (owner: trainer).

## 8. Pipelining (workstream E)

Per decision, actions(t) need observations(t) which need actions(t−1): a world cannot run ahead of its policy. Independent work is only (a) other worlds — already
parallel across workers, (b) inference for already finished worlds (3 % of rollout, 6 % with the 512-wide network), (c) the PPO update, which must complete
before the next rollout (on-policy; overlap = stale policy = algorithm change). Perfect overlap would bound the gain at 3–6 %; splitting the batch to hide stragglers
(4–7 %) needs twice the worker processes, which memory does not allow beyond 6. **Not implemented; no safe overlap exists.**

## 9. Worker and environment scaling (workstream F)

Aggregate physics ticks/s, shared terrain, full world, three windows (full table in the log):

| layout | compound random | compound run | hull random | hull run |
|---|---:|---:|---:|---:|
| 1 process, 16 worlds | 3,862 | 8,223 | 10,672 | 14,384 |
| 1 process, 64 worlds | 4,299 | 8,789 | 13,537 | 21,475 |
| 4 workers, 64 worlds | 11,683 | 23,425 | 29,275 | 42,417 |
| 4 workers, 128 worlds | 13,766 | 26,893 | 35,877 | 53,866 |
| 6 workers, 96 worlds | 16,240 | 31,303 | 36,582 | 51,940 |
| 6 workers, 144 worlds | 17,760 | 35,461 | 42,520 | 59,855 |
| 4 independent-terrain worlds | 2,022 | — | 2,966 | — |

Peak RSS: 13.4–13.9 GiB (4 workers, 128 worlds), 18.9–19.3 GiB (6 workers, 144 worlds). Six workers is the practical limit on 32 GB / 8 cores. More worlds per worker still helps
(128 vs 64 worlds on 4 workers: +18 % compound, +22 % hull on random).

## 10. End-to-end training comparison

Engineering-only configuration through `workbench.train` machinery (`isaac_training_benchmark.py`, 8 updates, first dropped), decisions/s:

| layout | `f54f505` (orig.) | `a075acc` re-measured | current, reference PPO exec | current (default) | hull (EXPERIMENTAL) |
|---|---:|---:|---:|---:|---:|
| 16 worlds, 1 process | 187 | 593 | 645 | **653** | 1,015 |
| 64 worlds, 1 process | 202 | — (451*) | — | — | 1,336 |
| 64 worlds, 4 workers | 202 (in-process) | 1,419 (3 runs) | 1,526 | **1,569** (3 runs) | 1,939 |
| 96 worlds, 6 workers | — | 1,627 | — | **1,818** | 2,266 |
| 64 worlds, 4 workers, SYNTHETIC 512-wide net | — | 910 | 954 | 923 | 1,046 |

\* campaign-I figure from an earlier stage of that campaign (`0d174a5`, 16 worlds: 309). Numbers in the `a075acc` column are re-measured today with the same benchmark on an extract of that commit;
the 918 decisions/s published by campaign I at 64/4 was taken at that earlier stage and understates the final campaign-I state. The `f54f505` column is the original baseline (in-process, 64 worlds). 500-update soak: 2,159 (first 100 updates) → 1,896 (last 100) with the
policy changing; the simulator itself is flat (section 11). The synthetic network is for inference-bound experiments only and is not a Timmy architecture.

## 11. Memory and startup

* Startup: 16 worlds build in 7.6–8.3 s, 64 worlds 10–13 s, 4 independent worlds 18 s; 4–6 workers start in parallel (13.6–16.7 s). Snapshot restore 1.6–2.6 s (shared), 6 s (independent terrain).
* Memory: one process 2.8 GiB (16 worlds), 3.3–3.5 GiB (64); 13.4–13.9 GiB with 4 workers, ≈ 19 GiB with 6. Hull collider uses slightly less than compound.
* Leak found and fixed (section 1). After the fix: workers 2,821 → 2,828 MiB over a 500-update training soak (max 2,945), fixed-tape drift test 2,852 → 2,854 MiB over 32,000 macrosteps,
  sharded random tape 1,709 → 1,710 MiB parent. Trainer parent plateaus near 2.2 GiB (transient peaks at checkpoint writes). The run directory grows ≈ 0.7 MiB per update.
* Interruption: SIGTERM mid-run leaves a valid checkpoint; resume works; workers die with the parent; no orphaned process after any run of the campaign.

## 12. Correctness and regression evidence

Final run on the final code (`a147692` + this commit's documentation/results; all code commits precede it), idle machine, 2026-10-13:

| check | result |
|---|---|
| `casu_grid/tests` unit suite | **122 tests OK** (includes PCG64-vs-numpy stream equality over 800k draws, numba RSS regression test, PPO execution equivalence tests, sharding tests) |
| `perf/run_perf_check.py --workers 4`: 5 compound trajectory gates (shared and independent terrain; baselines recorded from `f54f505`) | **bit-identical** (max deviation 0.0 in x, y, vx, vy, crouch, reward; 0 discrete differences) |
| same, 3 hull gates (own baselines) | bit-identical |
| throughput vs stored reference (compound, margin 15 %) | ok: random 1/4/16 worlds 1.10× / 1.06× / 1.01×; run 1.15× / 1.05× / 1.01× |
| vector validation (4 worlds, `isaac_vector_smoke.py`) | passed (`results/isaac_vector_validation_optimization2_20261013.json`); world isolation, exact restore continuation (error 0.0), partial reset, door state, independent terrain edit/restore |
| sharded validation (2 workers, `isaac_sharded_smoke.py`) | passed, 10 checks incl. exact restore (0.0), clean shutdown and worker-failure fail-fast (`results/isaac_sharded_validation_optimization2_20261013.json`) |
| hull vector and sharded validation | passed (`results/isaac_{vector,sharded}_validation_convex_hull_20261013.json`) |
| CasU A/B, compound default, re-run `flat_master` and `ice_run` (segment, teacher, free) vs the accepted results | **identical** statistics (e.g. `ice_run` free-run mean |dx| 2.145 → 2.145, first >0.5 m error at decision 9 → 9) |
| workbench compatibility | 500-update training soak and an 8-world/2-worker 3 + 3-update train and `--resume` smoke (updates 1–3 then 4–6, checkpoint continued) pass; `workbench.test_lab` and `test_components` pass (25 tests); `test_gui` cannot be imported in the Isaac interpreter (no PySide6; environment, not code) |
| PPO update equivalence | parameters, Adam state, metrics and RNG state bit-identical (default contract, early stop, uneven minibatches, larger network) |
| orphaned processes | none after the final suite (`ps` for workers, trainers, Isaac) |

CPU PhysX reference fidelity is unchanged: the validated compound path reproduces the trajectories recorded at `f54f505` bit for bit at every gate, and A/B against recorded CasU traces is unchanged.

## 13. Rejected and deferred approaches

| approach | why |
|---|---|
| GPU PhysX pipeline | unstable (CUDA illegal memory access), tunnelling at default offsets, not faster where it works at useful batch sizes |
| ChangeBlock batching of shape switches | not bit-identical (0.1 m); would change validated physics |
| pausing PhysX change tracking / Sdf-level writes | no measurable benefit / ≈ 2 µs per write |
| fused Adam | not bit-identical (1 ulp) |
| overlapping inference or PPO update with simulation | on-policy causality; bound 3–6 % |
| PhysX thread count, Fabric off, torch thread count | no effect (±3 %) |
| more than 6 workers | memory (≈ 19 GiB at 6) and 8 cores |
| replacing the compound collider with the hull by default | different physics; needs a researcher decision and a new CasU validation |
| deferred: recurrent reference-update bug | pre-existing trainer issue, out of scope |
| deferred: trajectory/TensorBoard disk growth (0.7 MiB per update) | owned by the workbench |

## 14. Recommended configurations

* **Production (validated physics):** defaults (`compound`, CPU PhysX, optimized PPO update) with `CASU_NUM_WORKERS=4` (16 worlds per worker) or 6 on a machine with ≥ 32 GB; optionally `OMP_NUM_THREADS=2` for the trainer (≈ +4 %).
  Keep `CASU_NUM_WORKERS=1` for debugging, bit-comparison with older runs and CasU A/B work.
* **Experimental (opt-in, label results):** `CASU_PLAYER_COLLIDER=convex_hull` for exploration runs where throughput matters more than bit-compatibility with the validated collider (1.4–2.8×). Re-validate against CasU before any claim.
* **Not recommended:** `use_cuda` / GPU PhysX.

## 15. Remaining ceiling

Validated path: ≈ 120 µs of 259 µs per crouch-heavy world-tick is the USD write cost of the 33-level compound collider; the rest is PhysX (≈ 28 %) and the Python/USD scaffolding.
With the hull collider the loop is bounded by the PhysX step and, in training, by the PPO update (32–39 % of a cycle: 256 minibatches of 16 samples, dominated by PyTorch per-op overhead, a researcher-owned setting).
Machine ceiling: 6 workers × 2 PhysX threads on 8 cores, ≈ 19 GiB.

## 16. Reproduction commands

```bash
source <isaac env as in ISAAC_LAB_SETUP.md>    # once per shell
PY="$ISAAC_SIM/python.sh"
# regression gates (unit tests, 8 bit-exact trajectory gates, throughput vs reference)
"$PY" casu_grid/perf/run_perf_check.py --workers 4
# simulation-only scaling
"$PY" casu_grid/isaac_throughput_benchmark.py --workers 4 --num-envs 128 --workload random --output /tmp/w4.json
"$PY" casu_grid/isaac_throughput_benchmark.py --num-envs 16 64 --workload random --player-collider convex_hull --output /tmp/hull.json   # EXPERIMENTAL
# end-to-end training
"$PY" casu_grid/isaac_training_benchmark.py --envs 64 --workers 4 --updates 8 --output /tmp/train.json
CASU_PPO_EXECUTION=reference "$PY" casu_grid/isaac_training_benchmark.py --envs 64 --workers 4 --updates 8 --output /tmp/train_ref.json
CASU_PLAYER_COLLIDER=convex_hull "$PY" casu_grid/isaac_training_benchmark.py --envs 64 --workers 4 --updates 8 --output /tmp/train_hull.json
# PPO update equivalence (no simulator)
"$PY" casu_grid/perf/ppo_update_bench.py --envs 64 --repeats 5
# baseline of campaign I on the same machine
git archive a075acc | tar -x -C /tmp/o1_src && (cd /tmp/o1_src && CASU_SRC_ROOT=$PWD PYTHONPATH=$PWD:$PYTHONPATH "$PY" casu_grid/isaac_training_benchmark.py --envs 64 --workers 4 --updates 8)
# GPU feasibility (unsupported)
"$PY" casu_grid/perf/gpu_probe.py --help ; bash casu_grid/perf/run_gpu_scenarios.sh
# CasU A/B, compound vs hull
"$PY" casu_grid/isaac_player_ab.py --trace course_v1 --modes segment,teacher,free --step-clear 0.02 [--player-collider convex_hull] --output /tmp/ab.json
python casu_grid/tools/compare_ab.py compound.json hull.json
# stability
CASU_NUM_WORKERS=4 "$PY" -m casu_grid.workbench.train --config casu_grid/workbench/engineering_smoke.py --updates 500 --num-envs 64 --run-dir /tmp/soak/run
```
