# Performance campaign — acceptance report

Branch `isaac-lab`, seed `391728405`, mandate `optimization.txt`. Machine: Intel i7-11700K (8 cores / 16 threads), RTX 3070 8 GB
(PhysX runs on the CPU, the GPU is idle), 32 GB RAM, Fedora 44, Isaac Sim 4.5.0 / Isaac Lab 2.1.1, numba 0.59.1.
Baseline = revision `f54f505` (the state at the start of the campaign). Every number below comes from a result file under
`casu_grid/perf/results/` and was produced by the scripts in `casu_grid/perf/README.md`; `perf/make_report_tables.py` regenerates the tables.

## 1. Summary

The simulated experiment is unchanged (0.02 s tick, ten ticks per decision, same observations, rewards, terminations, collision
representation, fluid rules and RNG stream). Execution is faster:

| | baseline | now | speedup |
|---|---:|---:|---:|
| Simulation-only, 16 shared worlds, one process, `random` tape | 1,686 ticks/s | 4,226 | **2.5x** |
| same, `run` tape (no crouch) | 1,975 | 9,251 | **4.7x** |
| Simulation-only, 64 worlds, 4 worker processes, `random` | — | 12,094 | **7.2x** vs baseline@16 |
| same, `run` | — | 23,672 | **12x** |
| Simulation-only, 96 worlds, 6 workers, `run` | — | 31,303 | **15.9x** |
| **Actual PPO training**, 64 worlds, end-to-end decisions/s | 202 | 451 (1 process) / **918** (4 workers) | 2.2x / **4.5x** |
| Actual PPO training, 96 worlds, 6 workers | — | 1,111 | 5.5x vs baseline@64 |
| Terrain edit (independent terrain, 4 worlds) | 6.2 s | 0.036 s | 170x |
| Independent-terrain build, 16 worlds | 98 s | 63 s | 1.6x |

The stretch target (2x the 16-world shared throughput, ~3,400 ticks/s) is exceeded by one process on both tapes, and by ~3.5x
(`random`) to ~7x (`run`) with worker processes. **No fidelity change**: batched trajectories are bit-identical to the baseline revision on five seeded
workloads and the CasU-vs-Isaac A/B results are identical (section 7).

Simulation-only and training numbers are different things and are reported separately: training adds policy inference (2-6 % of rollout for the
engineering smoke model) and the PPO update (3-22 %, set by the user-owned PPO contract: the engineering config uses minibatch 16, hence hundreds of
tiny optimizer steps per update). The stretch target in *training* decisions/s is therefore lower than in simulation-only ticks/s by construction
(918 decisions/s = 9,180 simulated ticks/s end-to-end at 64 worlds).

## 2. Baseline reproduction and profile

Reproduced on this machine before changing code (`perf/results/baseline_*.json`, three identical windows per point, std in the tables).
At 16 shared worlds (`random` tape) wall time per tick split into: controller + fluid pre-phase 35 %, controller post-phase + crouch-shape
switching 48 %, PhysX 11 %, tensor I/O 6 %; the process used 1.25 cores (PhysX threads), the GPU was idle. A `cProfile` of the batched step (300 ticks, 16
worlds) showed: crouch-shape switching (`RoundedBoxCollider.set_level`) 31 % of wall time, PhysX `_step` 15 %, then spread over ground queries,
fluid sweep (27 % of the Python controller), polygon culling and per-element torch indexing. Findings that drove the work:
* the Python controller and fluid sweep were the first frontier (confirmed), but not a single hotspot;
* per-element `torch` indexing for the PhysX writes/reads cost ~25 % of the remaining Python time;
* crouch-shape switching costs ~26 µs per USD attribute write regardless of shape type (12 writes per level change) — the USD→PhysX notice, not cooking;
* PhysX has ~0.3 ms fixed cost per step plus ~40 µs per body; its thread count is irrelevant at this scale.

## 3. What was implemented

| Layer | Change | Gate |
|---|---|---|
| Backend selection | `casu_grid/backend.py`: `reference-python` (oracle/fallback) or `numba`, `CASU_BACKEND`, recorded in results | tests |
| Fluid + tile queries | `kernels.py`: literal numba ports of the fluid sweep and tile box-cast/ray/overlap/touching | byte-identical vs `simulation_step_reference`; identical query results |
| Controller tick | `controller_kernels.py` + `fast_runtime.py`: packed per-world state, `Body.FixedUpdate`/`Update`/collision events/liquid hooks/fluid scheduler and static-object polygon queries (exact `round(x, 12)` hull port) compiled; reference fallback on a boundary case | fuzzed states (9,000 cases incl. objects) identical; batched trajectories bit-identical |
| Tensor I/O | persistent NumPy staging buffers, column-wise reads/writes | trajectories bit-identical |
| Multi-process | `isaac_sharded_env.py` + `isaac_shard_worker.py`: `BatchedIsaacLayer1Env(config)` returns a wrapper over N Isaac worker processes when `num_workers > 1` (`CASU_NUM_WORKERS=N|auto`); by-value callback transport (`callback_transport.py`); timeouts, SIGKILL-on-parent-death, fail-fast cleanup | sharded validation suite (11 checks incl. worker-kill) |
| Terrain mutation | independent-terrain chunk rebuild no longer resets the PhysX scene; independent terrain copies built by `Sdf.CopySpec` under an offset root Xform | trajectories bit-identical; edit/restore checks; wall/pit scenario equality |
| Tooling | benchmark v2 (windows, CPU/GPU/RSS, backend identity), training benchmark, trajectory dump/compare, throughput compare, `run_perf_check.py`, controller harness | — |

Rejected or deferred with evidence (details in `perf/OPTIMIZATION_LOG.md`): batching all crouch-shape writes in one `Sdf.ChangeBlock` (2x cheaper
but 0.1 m trajectory divergence within one macrostep, rejected), switching crouch levels by prim activation (invalidates the physics tensor view),
PhysX thread count (flat; 1 thread hangs PhysX), disabling Fabric (no effect), GPU PhysX pipeline (changes solver numerics; not attempted), a
leaner PPO bootstrap-value pass (inference is only 2-6 % of rollout; deferred).

## 4. Benchmarks

Aggregate physics ticks/s (10 ticks = 1 decision), full canonical world, mean ± std of three identical windows after warm-up; each window restarts
from `reset()` with the same seeded action tape, so the spread is timing noise. `random` = seeded random move/jump/crouch (crouch-heavy; the
historical tape), `run` = hold right, jump every fourth decision. 
**In-process, shared terrain, workload `random (crouch-heavy)` (aggregate physics ticks/s, mean ± std of 3 windows)**

| worlds | baseline | optimized | speedup | decisions/s (opt.) | peak RSS GiB |
|---|---:|---:|---:|---:|---:|
| 1 | 779 ± 44 | 1263 ± 34 | 1.62x | 126 | 2.65 |
| 2 | 1109 ± 8 | 1914 ± 24 | 1.73x | 191 | 2.67 |
| 4 | 1410 ± 11 | 2822 ± 61 | 2.00x | 282 | 2.69 |
| 8 | 1591 ± 15 | 3534 ± 104 | 2.22x | 353 | 2.74 |
| 16 | 1686 ± 7 | 4114 ± 37 | 2.44x | 411 | 2.87 |
| 32 | 1738 ± 17 | 4413 ± 73 | 2.54x | 441 | 3.09 |
| 64 | — | 4557 ± 31 | — | 456 | 3.55 |

**In-process, shared terrain, workload `run (hold right, periodic jump)` (aggregate physics ticks/s, mean ± std of 3 windows)**

| worlds | baseline | optimized | speedup | decisions/s (opt.) | peak RSS GiB |
|---|---:|---:|---:|---:|---:|
| 1 | 1051 ± 13 | 1719 ± 46 | 1.64x | 172 | 2.65 |
| 2 | — | 2912 ± 40 | — | 291 | 2.66 |
| 4 | 1694 ± 22 | 4955 ± 57 | 2.92x | 496 | 2.69 |
| 8 | — | 7468 ± 78 | — | 747 | 2.75 |
| 16 | 1975 ± 9 | 9739 ± 308 | 4.93x | 974 | 2.86 |
| 32 | — | 11022 ± 279 | — | 1102 | 3.09 |
| 64 | — | 10390 ± 200 | — | 1039 | 3.55 |

**In-process, independent terrain, workload `random`**

| worlds | baseline | optimized | speedup | peak RSS GiB | construct s (before copy-build) |
|---|---:|---:|---:|---:|---:|
| 1 | 887 | 1457 ± 8 | 1.64x | 2.65 | 7 |
| 4 | 1137 | 2147 ± 25 | 1.89x | 5.23 | 25 |
| 16 | 1241 | 2259 ± 25 | 1.82x | 15.51 | 98 |

**In-process, independent terrain, workload `run`**

| worlds | baseline | optimized | speedup | peak RSS GiB | construct s (before copy-build) |
|---|---:|---:|---:|---:|---:|
| 1 | — | 1983 ± 28 | — | 2.65 | 6 |
| 4 | — | 2639 ± 193 | — | 5.25 | 25 |
| 16 | — | 2712 ± 57 | — | 15.51 | 100 |

**Multi-process (`CASU_NUM_WORKERS`), shared terrain, workload `random` (speedup vs baseline 16-world in-process)**

| workers | worlds | ticks/s | vs baseline@16 | total RSS GiB | decisions/s |
|---|---:|---:|---:|---:|---:|
| 2 | 16 | 6454 ± 33 | 3.8x | 6.1 | 645 |
| 2 | 32 | 7380 ± 50 | 4.4x | 6.5 | 738 |
| 3 | 48 | 9820 ± 150 | 5.8x | 9.5 | 982 |
| 4 | 16 | 8376 ± 114 | 5.0x | 11.3 | 838 |
| 4 | 32 | 10647 ± 74 | 6.3x | 11.6 | 1065 |
| 4 | 64 | 12094 ± 11 | 7.2x | 12.4 | 1209 |
| 6 | 48 | 13483 ± 85 | 8.0x | 17.1 | 1348 |
| 6 | 96 | 16240 ± 150 | 9.6x | 18.3 | 1624 |

**Multi-process (`CASU_NUM_WORKERS`), shared terrain, workload `run` (speedup vs baseline 16-world in-process)**

| workers | worlds | ticks/s | vs baseline@16 | total RSS GiB | decisions/s |
|---|---:|---:|---:|---:|---:|
| 2 | 16 | 13242 ± 182 | 6.7x | 6.1 | 1324 |
| 2 | 32 | 15462 ± 769 | 7.8x | 6.5 | 1546 |
| 3 | 48 | 19260 ± 658 | 9.8x | 9.5 | 1926 |
| 4 | 16 | 15479 ± 330 | 7.8x | 11.3 | 1548 |
| 4 | 32 | 20713 ± 31 | 10.5x | 11.7 | 2071 |
| 4 | 64 | 23672 ± 356 | 12.0x | 12.4 | 2367 |
| 6 | 48 | 21626 ± 898 | 11.0x | 17.2 | 2163 |
| 6 | 96 | 31303 ± 142 | 15.9x | 18.3 | 3130 |


Worker count scaling is limited by CPU cores (8 physical) and memory (≈2.9 GiB per worker); 6 workers use 18 GiB. A sweep of 2/3/4/6 PhysX
threads per worker (4 workers, 64 worlds, `random`) gave 11.8k / 12.1k / 12.0k / 11.8k ticks/s: the default 2 is fine.
Diagnostics at 4 workers: waiting for the slowest worker costs 7-11 %, pipe overhead 1-2 %.

## 5. Actual PPO training throughput

Measured through the real launcher (`workbench.train`) with the engineering-only smoke contract (not an approved Timmy experiment), 6 updates, first update
dropped (JIT/caches). Simulation-only = environment step time only; rollout = simulation + policy inference + bookkeeping; end-to-end adds the
PPO update. Hyperparameters were not touched. **Actual PPO training (engineering-only smoke configuration), decisions/s**

| worlds | workers | baseline end-to-end | sim-only | rollout | end-to-end | speedup | optimizer share |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 97 | 148 | 133 | 129 | 1.3x | 3% |
| 4 | 1 | 154 | 403 | 368 | 342 | 2.2x | 7% |
| 16 | 1 | 187 | 338 | 328 | 309 | 1.7x | 6% |
| 16 | 2 | 187 | 376 | 362 | 339 | 1.8x | 6% |
| 16 | 4 | 187 | 613 | 582 | 521 | 2.8x | 10% |
| 32 | 4 | — | 643 | 622 | 557 | — | 10% |
| 64 | 1 | 202 | 498 | 490 | 451 | 2.2x | 8% |
| 64 | 4 | 202 | 1151 | 1113 | 918 | 4.5x | 18% |
| 96 | 6 | — | 1466 | 1426 | 1111 | — | 22% |

**SYNTHETIC large policy network (512x512x512 + 512x512), decisions/s**

| run | sim-only | rollout | end-to-end | optimizer share | inference share of rollout |
|---|---:|---:|---:|---:|---:|
| baseline code, 16 worlds | 196 | 192 | 173 | 10% | 2% |
| optimized, 16 worlds, 1 process | 440 | 417 | 337 | 19% | 5% |
| baseline code, 64 worlds | 208 | 206 | 185 | 10% | 1% |
| optimized, 64 worlds, 4 workers | 1334 | 1250 | 736 | 41% | 6% |


Reading the tables: with the small smoke model training stays simulation-bound up to ~16 worlds; at 64+ worlds the PPO update (a user-owned,
config-determined number of small optimizer steps) becomes the new bottleneck (17-22 %), and with the synthetic large network it reaches 41 %.
The remaining fix is in the trainer/hyperparameters (larger minibatches, fewer epochs), which belong to the researcher. `OMP_NUM_THREADS=2` improved
the 64-world/4-worker run by ~4 % (918 → 962 decisions/s); other thread counts were neutral.

## 6. Memory, startup and latency

* RSS: one process 2.65 GiB + ~13 MiB per extra shared world; 64 worlds 3.55 GiB. Each worker process is ~2.9 GiB (Kit + terrain colliders): 4 workers 12.4 GiB, 6 workers 18.3 GiB.
  Independent terrain costs ~0.85 GiB per world (16 worlds: 15.5 GiB). VRAM: ~750 MiB per Kit process; the GPU stays at 0-6 % utilisation (CPU PhysX).
* Cold start: kernels compile in ~11 s once and are cached in `casu_grid/__pycache__` (0.2 s afterwards); world build 5 s shared; 4 workers start in ~13 s in parallel.
* Lifecycle (shared): reset one world 4-8 ms, reset 16 worlds 39 ms, snapshot 16 ms, restore 1.6-1.8 s (includes the physics-view reset needed for exact replays).
* Terrain mutation (independent, 4 worlds): `set_block` 0.036 s (was 6.2 s); restore after an edit 6.0 s (was 12.6 s), dominated by the scene reset that guarantees exact branch replay.

## 7. Correctness and fidelity evidence (gates A-G)

| Gate | Evidence | Result |
|---|---|---|
| A discrete state | fluid sweep vs `simulation_step_reference` on random fixtures and the canonical world (state, counters, RNG position and state); tile/polygon queries vs reference on thousands of random queries incl. boundary cases | identical |
| B controller semantics | compiled tick vs `StandingBodyModel`/`player_runtime` on 3 x 3,000 fuzzed states (fixed update, post phase incl. collision events and object contacts, latch), pack/unpack round trip, full macrostep loops | bit-identical |
| C physics behaviour | batched env trajectories (16 shared random, 4 shared run, 4 canonical-spawn random, 4 independent random, 4 independent run) vs the stored `f54f505` baselines | **bit-identical** (positions, velocities, crouch, rewards, terminations, fluid digests) |
| C vs CasU | `isaac_player_ab.py` teacher/segment/free on flat_master, course_v1, ice_run, immersed_standing, destroy_support (collider delay 31) re-run on the optimized code | every summary equals the stored pre-optimization result (`ab_equivalence_20261013.json`) — documented source-vs-Isaac residuals are unchanged, not enlarged |
| D lifecycle | vector validation (isolation, exact restore, partial reset, doors, independent edit/restore, edits change contacts) and sharded validation across workers | all passed |
| E training | smoke through the real launcher at 1-96 worlds and 1-6 workers: finite losses, updates, checkpoints, metrics | passed |
| F replay | 114 unit tests incl. the workbench replay tests pass; replay must use the recorded `num_envs`/`CASU_NUM_WORKERS` layout | passed |
| G usability compatibility | the workbench launcher, Luna's callback wrappers (`checked_callbacks`) and `test_lab`/`test_gui` run unchanged; sharded training works through `workbench.train` | passed |

Determinism note: bit-identity holds for the same world partition. Changing `num_envs` or `num_workers` changes PhysX scene composition and therefore results at the
1e-9 level, which chaotic corner contacts amplify (this was already true for different `num_envs`). Record and replay with the same layout.

## 8. Backends and setup

`CASU_BACKEND=auto|numba|reference-python` (default `auto` = numba when importable), `CASU_COMPILED_CONTROLLER=0` (compiled queries/fluid but reference controller),
`CASU_NUM_WORKERS=N|auto` (default 1; `auto` = up to 4 workers with >= 4 worlds each). numba 0.59.1 is bundled with Isaac Sim 4.5; no installation is needed.
`reference-python` needs nothing and is always available; the active choice is printed by `backend.describe()`, stored in every benchmark file and returned by
`env.performance_info()`. Kernel source files keep the readable reference beside them (`state.py`, `ground_query.py`, `body_model.py`).

## 9. Regression prevention

`python casu_grid/perf/run_perf_check.py` (Isaac Python, idle machine): unit tests, five bit-exact trajectory gates against `perf/baselines/trajectories`, then throughput
on both tapes against `perf/baselines/throughput_reference_*.json` with a 15 % margin that must also exceed both runs' spread. Correctness gates are deterministic
and fail the run; timing only warns unless `--fail-on-regression`. `--update-reference` re-records the reference after hardware changes. A full run just before this report
passed all gates with throughput within ±3 % of the reference.

## 10. Which performance mode to use

| Situation | Setting |
|---|---|
| Debugging, A/B against CasU, anything needing a single process or bit-comparison with older runs | defaults (`CASU_NUM_WORKERS=1`); numba is already on |
| Training with >= 16 worlds | `CASU_NUM_WORKERS=4` (or `auto`), 16 worlds per worker; 6 workers if memory allows (18 GiB) — roughly +30-50 % over 4 |
| Terrain-editing tasks | `--terrain-mode independent` (edits are now instant; one terrain copy per world costs ~0.85 GiB); with workers the copies are spread across processes |
| Suspect a numerical issue | `CASU_BACKEND=reference-python` — slower, bit-identical results |
| Replaying a recorded run | the same `--num-envs` and `CASU_NUM_WORKERS` as the recording |
| Large networks or big minibatches | no change needed; simulation is no longer the bottleneck beyond ~64 worlds |

## 11. Limits, warnings and open items

* Crouch-shape switching (12 USD attribute writes per level change, ~100 µs per world per tick on the crouch-heavy tape) is now the largest remaining cost; it is intrinsic to the validated
  33-level composite collider. Batching the writes would double its speed but changes PhysX results (rejected).
* `worker_physx_threads` must be >= 2 (1 hangs PhysX; the environment validates this). A hung worker is declared failed after `worker_timeout_seconds` (900 s) and all workers are killed.
* Terrain edits with `restore()` still reset the PhysX scene (exact branch replay), 6 s with 4 independent worlds.
* The launcher swallows exceptions raised while `app.close()` terminates Kit, so a failing environment construction in `workbench.train` can end silently; the sharded
  environment's own errors name the failing worker and carry its log tail (`env.log_dir`). Suggested follow-up for the workbench owner: add `env.performance_info()` (backend, workers) to `environment.json`.
* Two orphaned Kit processes from killed experiments were found and removed by exact PID during the campaign; worker processes now die with their parent (SIGKILL parent-death signal).
* Not done: GPU PhysX, a compiled PPO update, double-buffered pipelining of policy inference with simulation, and workers beyond 6 (memory).

## 12. Commits

`f70a9cb` numba backend + staging, `0d174a5` compiled controller / sharded environment / perf workflow, `62cd86b` callback transport and final results,
`ee44030` terrain edits and independent terrain build, `d884f08` references/tests, plus the commit containing this report. Luna-owned workbench files were not modified.
