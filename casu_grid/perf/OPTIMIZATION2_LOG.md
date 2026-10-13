# Optimization II — experiment log

Campaign file `optimization2.txt`; starting point `a075acc` (accepted CPU-PhysX baseline of campaign I).
Machine: i7-11700K (8 cores / 16 threads), RTX 3070 8 GB, 32 GB RAM, Fedora 44, Isaac Sim 4.5.0 / Isaac Lab 2.1.1.
"Compound" = the validated player collider (two cross cuboids + four corner cylinders per crouch level);
"hull" = the experimental single convex-hull collider introduced here. Result files live in `casu_grid/perf/results/`.

## Gate status of the validated default (compound, CPU PhysX)

After every change below the default path was re-run through `perf/run_perf_check.py`: unit tests (122), five bit-exact
trajectory fixtures recorded from `f54f505` (shared + independent terrain) plus three hull fixtures with their own baselines, throughput
within the noise margin of the stored reference. All passed on the final code; the default is unchanged by this campaign.

## Workstream A — profile of the optimized system

Method: `isaac_throughput_benchmark.py`, three identical windows (30 decisions, 3 warm-up) per point, full canonical world,
fixture spawn (483,1010) unless noted; diagnostic sub-timers added to the batched tick (`diag_*` keys: they are
sub-intervals of `controller_post_and_shapes`). µs are wall microseconds per world-tick.
Files: `prof2_*.json`.

| layout / tape | ticks/s | µs/world-tick | fluid+fixed (pre) | tensor wr/rd | PhysX step | post kernels | crouch-shape switching | level changes per world-tick |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 worlds, random (crouch-heavy) | 3,862 | 259 | 24 | 14 / 6 | 73 | 16 | **122** | 0.61 |
| 16 worlds, landing-heavy | 5,225 | 191 | 23 | 11 / 5 | 71 | 15 | 62 | 0.28 |
| 16 worlds, no-crouch run | 8,223 | 122 | 22 | 11 / 5 | 59 | 17 | 3 | 0.01 |
| 16 worlds, fluid (immersed fixture) | 7,246 | 138 | 22 | 10 / 5 | 67 | 17 | 12 | 0.05 |
| 64 worlds, random | 4,299 | 233 | 23 | 6 / 2 | 62 | 16 | **120** | 0.62 |
| 64 worlds, run | 8,789 | 114 | 23 | 4 / 2 | 64 | 16 | 2 | 0.01 |
| 4 independent worlds, random | 2,022 | 494 | 28 | 41 / 21 | **259** | 20 | 117 | 0.54 |
| 4 workers x 16 worlds, random | 11,683 | 86 (wall) | 8 | 5 / 2 | 22 | 5 | 36 | — |
| 4 workers x 16 worlds, run | 23,425 | 43 (wall) | 8 | 4 / 2 | 20 | 5 | 1 | — |

Ranking (crouch-heavy tape, one process): 1 crouch-shape switching (47 %), 2 PhysX step (28 %), 3 fluid/FixedUpdate pre-phase (9 %),
4 compiled post-phase (6 %), 5 tensor staging (5 %). On a no-crouch tape: PhysX 48 %, pre-phase 18 %, post kernels 14 %,
staging 13 %. With independent terrain PhysX dominates (52 %: 650k static shapes). Sharded runs: waiting for the slowest worker 4–7 %, pipe
overhead 1 %. Python dispatch of the compiled kernels is inside the pre/post figures (≈ 0.6 µs and 2.5 µs per call in the harness).
Each crouch-level change costs ≈ 200 µs (12 USD attribute writes ≈ 15 µs each); even an unrelated custom attribute write costs
8.5 µs (USD + notice listeners), a ChangeBlock does not reduce the cost per write (45 µs for 10 writes), and pausing PhysX change
tracking made no difference (3.58 vs 3.75 ms per 16 changes).

## Workstream B — GPU PhysX

Hypothesis: the GPU pipeline removes the per-body PhysX cost. Implementation: host-staged tensor I/O (`_np`/`_dev`), `use_cuda`
config switch, probes `perf/gpu_probe.py`, scenario suite `perf/gpu_scenarios.py`, A/B flag `--cuda`.

| finding | evidence |
|---|---|
| builds, steps, crouch-shape switching and terrain edits work | `scenario` runs (crouch tunnel, standing tunnel, support removal) match CPU within 0.07 |
| faster only at large single-process world counts on no-crouch tapes | probe, `run` tape 16/64/256 worlds: GPU 4,545 / 11,142 / 16,727 vs CPU 7,366 / 8,318 / 4,568 ticks/s (`results/gpu_probe/`) |
| random (crouch/jump) tape fails | 16, 64, 256 worlds: "Failed to get rigid body transforms" (CUDA illegal memory access) |
| qualitatively wrong at default offsets | walls: the player passes through 1- and 4-cell walls at 16–25 m/s (CPU stops; CPU without CCD also stops, so not CCD) |
| not fixed in a validated way by larger contact offsets | contact offset 0.2 makes all 7 scenarios match (≤ 0.07) but changes a protected setting, seams unvalidated, crashes persist |
| crashes | step-up teleports into overlap, second `reset()` after a measurement window, A/B free-run first jump |
| multi-process CPU is faster anyway | 4 CPU workers: 23.7k (64 worlds, run) vs best GPU single process 16.7k (256 worlds) |

Verdict: **rejected for production, kept as an isolated opt-in switch** (`use_cuda`, documented unsupported/unstable).
Not pursued: GPU + worker processes (shared GPU contexts) and GPU with enlarged buffers — the crashes are triggered by
deep pose writes and tunnelling, not by buffer capacity (8x larger buffers did not change the first crash).

## Workstream C — CPU candidates

| # | candidate | result | verdict |
|---|---|---|---|
| C1 | pause omni.physx change tracking around the 12N writes | no change (3.58 vs 3.75 ms) | rejected |
| C2 | Sdf layer-level writes instead of `Usd` attribute `Set` | 12.7 vs 15.1 µs per write; physics notices unchanged | not worth it |
| C3 | **single convex-hull player collider** (2 writes per level change) | random 3,862 → 8,851 ticks/s (2.3x; 10,672 = 2.8x in the final idle-machine suite), landing 2.1x (2.5x final), run 1.5x (1.75x final) at 16 worlds; PhysX step 73 → 29 µs; better CasU fidelity (below) | **accepted as experimental opt-in** |
| C4 | PhysX threads / Fabric off (campaign I) | no effect | closed |
| C5 | **numba memory leak**: a numpy `Generator` passed into the compiled fluid kernels leaks ≈ 600 B per call (numba 0.59.1). Replaced by a port of PCG64 XSL-RR 128/64 + `Generator.integers(0, 2)` that advances a 6-word state array (exported/imported around the kernel calls) | a075acc: 500-update 64-world training soak grew 13.3 → 18.7 GiB and lost 10 % throughput; after the fix workers stay at 2.82 → 2.83 GiB and the fixed-tape drift test is flat. Stream equality with numpy over 800k draws; all bit-exact gates unchanged | **accepted** (bug fix of the campaign-I code; also +11 % simulation-only decisions/s in training, no per-call Generator unboxing) |

### C3 fidelity (hull vs compound, both CPU PhysX, against recorded CasU traces)
`isaac_player_ab.py --player-collider convex_hull`, same options as the accepted runs (`--step-clear 0.02`); compare with
`tools/compare_ab.py`. Selected results (compound → hull):

| trace / measure | compound | hull |
|---|---:|---:|
| course_v1 free-run mean |dx| / max |dx| | 0.71 / 8.0 m | 0.009 / 0.36 m |
| course_v1 first |dx| > 0.5 m | decision 13 | none in 730 ticks |
| course_v1 decision-level landing mean |dy| | 0.275 m | 0.011 m |
| ice_run free-run mean |dy| | 0.204 m | 0.014 m |
| ice_run decision-level mean |dvx| | 0.595 m/s | 0.298 m/s |
| flat_master wall/ceiling decisions mean |dx| | 0.022 m | 0.007 m |
| flat_master landing decisions mean |dy| (worse) | 0.072 m | 0.152 m |
| immersed_standing free-run mean |dx| (marginally worse) | 0.060 m | 0.078 m |

Scenario suite (7 scenarios, hull vs compound): max deviation 0.0007 (crouch tunnel), 0.0000 (junction run), 0.0003 (landing),
0.0225 (standing tunnel), 0.061 (walls); falls from 20 / 50 m (≈ 47 / 74 m/s) land identically (no tunnelling in either).
Lifecycle: vector validation, sharded validation (incl. worker kill) and a 4-worker train + resume all pass with the hull;
runs are bit-repeatable (own baselines `perf/baselines/trajectories/hull_*.json`, 3 new gates in `run_perf_check.py`).

## Workstream D — PPO execution

`perf/ppo_update_bench.py` (real `PPOTrainer`, engineering contract, fake vector environment, identical model/optimizer/RNG
state for both implementations). Per minibatch (16 samples): forward 261 µs, forward + loss + backward 689 µs, gradient clip
100 µs, Adam 250 µs (foreach 200 µs, fused 105 µs). 64 worlds = 2048 samples = 256 minibatches per update.

| change | update time | bit-identical |
|---|---:|---|
| one gather per epoch instead of per minibatch; stats accumulated on tensors, one conversion per update | ≈ 4 % | yes |
| `Adam(foreach=True)` | ≈ 5 % | yes (300-step check) |
| distribution argument validation off | ≈ 2 % | yes (outputs identical; invalid inputs still error in the sampler/loss) |
| fused Adam | 105 µs/step | **no** (1 ulp differences) — not adopted |
| torch thread count 1/2/4/8 | ±3 % | n/a |
| total | 323 → 294 ms per update (1.10x), threads 2; 326 → 297 ms, threads 8 | parameters, Adam moments, metrics, RNG state |

Floor: PyTorch per-op overhead on a 467-parameter network; update time scales with the config-owned number of minibatches.
`tests/test_ppo_execution.py` covers default, early-stop on `target_kl`, uneven minibatches, larger network, recurrent dispatch
and Adam foreach equality. Pre-existing finding (not changed, owner: trainer): the recurrent reference path computes
`approx_kl` from `old_log_prob[idx]` without reshaping and fails whenever the rollout length differs from the minibatch layout
(shape mismatch), so recurrent updates are only exercised in the narrow shapes of the existing test.

## Workstream E — pipelining (analysis)

Causal chain per decision: actions(t) depend on observations(t) which depend on actions(t-1) of the same world, so a world cannot
be stepped ahead of its policy output. The only independent work is (a) other worlds — already parallel across worker processes,
(b) policy inference for worlds that finished early, (c) the PPO update, which must complete before the next rollout begins
(on-policy; overlapping would use a stale policy = algorithm change). Measured headroom:
policy inference + bookkeeping = 3 % of rollout at 64 worlds (6 % with the 512-wide synthetic network); waiting for the slowest
worker = 4–7 %; pipe overhead 1 %. Even perfect overlap of inference with simulation would give ≤ 3–6 %; splitting the batch
into two groups to hide straggler waits needs twice the worker processes (memory-limited at 6) or halves the batch per worker.
**Conclusion: not implemented; the dependency structure and measured shares leave no useful overlap.**

## Workstream F — scaling, resources, long-run stability

Result files `o2_*` (suite of 2026-10-13 15:19–15:34, idle machine, 3 windows each), `o1_*` (campaign-I final `a075acc`,
extracted with `git archive` and run through the same benchmark on the same day), `soak_train_500_w4_e64.json`.

### Simulation-only scaling (aggregate physics ticks/s, shared terrain, full world)

| layout | compound `random` | compound `run` | hull `random` | hull `run` |
|---|---:|---:|---:|---:|
| 1 process, 16 worlds | 3,862 | 8,223 | 10,672 | 14,384 |
| 1 process, 64 worlds | 4,299 | 8,789 | 13,537 | 21,475 |
| 4 workers, 64 worlds | 11,683 | 23,425 | 29,275 | 42,417 |
| 4 workers, 128 worlds | 13,766 | 26,893 | 35,877 | 53,866 |
| 6 workers, 96 worlds | 16,240 (campaign I run) | 31,303 (campaign I run) | 36,582 | 51,940 |
| 6 workers, 144 worlds | 17,760 | 35,461 | 42,520 | 59,855 |
| 4 independent-terrain worlds, 1 process | 2,022 | — | 2,966 | — |

Hull landing tape: 13,300 (16 worlds) / 18,471 (64 worlds), one process. Peak RSS: 4 workers 13.4–13.9 GiB (128 worlds),
6 workers 18.9–19.3 GiB (144 worlds), 1 process 2.8–3.5 GiB. 8 cores/16 threads and 32 GB are the ceiling: beyond 6 workers
(2 PhysX threads each plus the trainer) memory and cores are exhausted, so 6 is the supported maximum on this machine.

### End-to-end PPO training (engineering-only configuration; 8 updates, first dropped; decisions/s)

| layout | a075acc (campaign I final) | current, PPO reference execution | current, PPO optimized | hull + optimized (EXPERIMENTAL) |
|---|---:|---:|---:|---:|
| 16 worlds, 1 process | 593 | 645 | 653 | 1,015 |
| 64 worlds, 1 process | — | — | — | 1,336 |
| 64 worlds, 4 workers (3 repeats) | 1,416 / 1,428 / 1,413 | 1,526 / 1,534 / 1,519 | 1,583 / 1,538 / 1,585 | 1,939 |
| 96 worlds, 6 workers | 1,627 | — | 1,818 | 2,266 |
| 64 worlds, 4 workers, SYNTHETIC large net | 910 | 954 | 923 | 1,046 |

Notes: the figures published in the campaign-I report (918 decisions/s at 64/4) were taken at an earlier stage of that campaign;
`a075acc` re-measured today gives ≈ 1,420, which is the baseline used here. Gains over `a075acc` at 64/4: +7 % with this campaign's simulator changes (reference PPO execution column; simulation-only decisions/s
2,030 → 2,265, attributed to the compiled PCG64 path replacing per-call Generator unboxing), +11 % with the optimized PPO update, ×1.37 with the hull collider. For the synthetic
512-wide network the update is dominated by matrix products (≈ 1 s per update, 256 minibatches of 16 samples) and execution changes
are within noise (1.01–1.04× in the microbenchmark; torch threads 1/2/4/8: 1.32/1.30/1.09/0.98 s).

### Long-run stability

* **Soak** (500 updates, 64 worlds, 4 workers, compound, 16 min): worker RSS 2,821 → 2,828 MiB (max 2,945), no growth;
  parent RSS 2,002 → plateau ≈ 2,190 MiB after 100 updates (transient peaks at checkpoint/trajectory writes, max 2,961); trainer
  loop with a fake environment: 616.4 → 616.5 MiB over 120 updates; sharded environment with a random tape (64 worlds, 4 workers,
  1,600 macrosteps): parent 1,709 → 1,710 MiB, throughput flat (11.6–12.0 k ticks/s).
  Decisions/s drifted 2,159 → 1,896 across the soak while the fixed-tape in-process test is flat (4.0–4.2 k ticks/s, RSS 2,852 → 2,854 MiB
  over 32,000 macrosteps): the drift is attributed to the changing policy behaviour (different action mix and episode lengths as training proceeds), not to simulator degradation.
* The run directory grows ≈ 0.7 MiB per update (343 MiB for 500 updates, `trajectories.jsonl` and TensorBoard); no cleanup is done
  by the trainer (owner: workbench).
* **Interruption/cleanup**: SIGTERM of the trainer mid-run leaves a valid `latest.pt`, resume continues in a new run directory,
  workers exit with the parent (PDEATHSIG SIGKILL), no orphaned processes after any run in this campaign (checked with `ps` after each suite).
* Failure modes found and fixed: orphaned workers after `worker_physx_threads` 0/1 (PhysX hangs) → validated `>= 2`; GPU crashes
  leaving unusable contexts (GPU kept opt-in).
