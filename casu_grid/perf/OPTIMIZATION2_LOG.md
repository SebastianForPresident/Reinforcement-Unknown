# Optimization II — experiment log

Campaign file `optimization2.txt`; starting point `a075acc` (accepted CPU-PhysX baseline of campaign I).
Machine: i7-11700K (8 cores / 16 threads), RTX 3070 8 GB, 32 GB RAM, Fedora 44, Isaac Sim 4.5.0 / Isaac Lab 2.1.1.
"Compound" = the validated player collider (two cross cuboids + four corner cylinders per crouch level);
"hull" = the experimental single convex-hull collider introduced here. Result files live in `casu_grid/perf/results/`.

## Gate status of the validated default (compound, CPU PhysX)

After every change below the default path was re-run through `perf/run_perf_check.py`: unit tests, five bit-exact
trajectory fixtures recorded from `f54f505` (shared + independent terrain), throughput within the noise margin of the
stored reference. All passed; the default is unchanged by this campaign.

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
| C3 | **single convex-hull player collider** (2 writes per level change) | random 3,862 → 8,851 ticks/s (2.3x), landing 2.1x, run 1.5x at 16 worlds; PhysX step 73 → 29 µs; better CasU fidelity (below) | **accepted as experimental opt-in** |
| C4 | PhysX threads / Fabric off (campaign I) | no effect | closed |

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

## Workstream F — scaling and stability

(see report; raw files `o2_*`, `soak_*`)
