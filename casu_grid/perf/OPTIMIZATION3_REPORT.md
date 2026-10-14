# Optimization III — final report

Branch `optimization3` (worktree `Reinforcement-Unknown-Isaac-opt3`), starting commit `0d46520`. Machine: i7-11700K (8 cores / 16 threads), RTX 3070 8 GB,
31 GB RAM, Fedora 44, Isaac Sim 4.5.0 / Isaac Lab 2.1.1, PhysX on CPU. Experiment: XY1 point navigation in the 128 × 128 mini-world
(independent mutable terrain, compound collider, 16 × 64 × 64 observation + 2 goal coordinates, seven categorical heads, ten physics ticks per decision,
PPO rollout 64 / minibatch 32 / 2 epochs, 16,385 parameters). The experiment is unchanged; every number below is for it.
The step-by-step ledger (hypothesis → change → evidence → result → decision) is `OPTIMIZATION3_LOG.md`; raw logs are under `casu_grid/results/opt3/`.

Accounting used everywhere: `end-to-end = decisions / (collection_seconds + optimization_seconds)` over the steady updates (first update dropped), from the
trainer's own per-update timers; *rollout* = decisions / collection_seconds; *sim-only* = decisions / physics-step seconds. The mission's stated baselines
(128.6 / 139.6 / 150.0 decisions/s) used a different load and accounting; the baseline was therefore **re-measured under the identical protocol on the
starting commit** (CPU PPO, central policy evaluation, 4 workers × 2 PhysX threads), and speed-ups are quoted against that.

## A. Executive summary

* **End-to-end throughput of the unchanged experiment rose 5.5–7.0×** (clean-machine re-measurement, identical protocol): 16 worlds 199 → 1,275 decisions/s (6.4×), 32 worlds 227 → 1,536 (6.8×),
  64 worlds 243 → 1,772 (7.3× against the 4-worker baseline; 7.0× against the baseline's own 8-worker run of 253). At the baseline's own configuration (4 workers) the gains are 5.9× / 6.2× / 5.5×.
  A 41-minute soak (1,000 updates, 4.09 M decisions) at 64 worlds × 8 workers sustained **1,688 decisions/s end-to-end (1,663 wall-clock including logging and checkpoints)** with flat GPU memory (3.5 GiB),
  summed RSS 24.5 → 24.9 GiB, no error and finite losses; throughput settles at ≈ 1,675 after the first 300 updates as the random policy digs the terrain.
* What the time was: the CPU ran the 16 × 64 × 64 convolution network for inference and for PPO (PPO alone was 43 % of the wall clock at 16 worlds, 52 % at 64 worlds with 8 workers), the parent evaluated the policy after
  every decision while all workers waited for the slowest one, 262 KB of float observations per world and decision crossed a pipe, 35 % of the worker time was the
  reference Python controller, crouch-shape switching issued one USD notice per attribute write, and every excavation re-authored a 300-cuboid chunk.
* What was done (all behind the same interfaces, most switchable by an environment variable): PPO model, buffers and update on the GPU, the minibatch step
  captured as one CUDA graph; **worker-local rollouts** (policy copy and CUDA-graph sampler inside each worker, weights broadcast once per update, compact 8 KB
  id-window observations expanded to channels on the GPU bit-exactly); the controller compiled for the mini-world (bit-identical to the reference); direct PhysX
  tensor-view I/O; batched, spec-level crouch-shape writes; diff-based terrain collider updates.
* **Fidelity**: the compiled controller, direct tensor I/O, spec-level writes, compact observations and the local-rollout machinery are bit-identical to the
  starting commit on seeded trajectories (SHA-256 of 4 worlds × 300 random decisions incl. excavation identical; local vs central rollouts bit-identical in observations,
  actions, rewards and terminations including resets and arrivals). Three changes are *not* bit-identical and were validated statistically against predeclared
  tolerances: GPU/TF32/CUDA-graph PPO numerics, batched crouch writes and incremental terrain colliders (PhysX sees shapes in another internal order), and the
  sampler's random stream (Gumbel-max on a per-update generator). Learning statistics over 3 seeds × 40 updates and 12 paired first-update comparisons show no
  detectable difference. See D.
* Two defects were found and fixed along the way: XY1 checkpoints **could not be resumed** (the goal, best-distance reward state and spawn stream were not part of the
  slot snapshot — the baseline fails identically — and loading a checkpoint onto CUDA failed on the RNG state), and out-of-range `EnvSlot.edit_block` coordinates wrote a wrapped
  cell and marked a nonexistent `Chunk_-1_*` chunk (the source of the negative-chunk-name warnings).
* What is left is native: ~43 % of a worker's time is the PhysX step (about half of it CCD, which cannot be removed without changing the physics) and ~27 % is
  the crouch-shape toggling that the six-shape compound requires (a USD/PhysX notice per attribute). Eight workers saturate the 8 physical cores and the memory
  (1.9 GiB per Kit process); 10 workers are slower, 12 do not fit.

## B. Benchmarks (worlds 16 / 32 / 64)

Identical protocol for baseline and optimised (`perf/xy1_bench.py`, `workbench.train` as a research launch would run it, `--terrain-mode independent`, 2 PhysX threads per
worker, 5 (baseline) / 9 (optimised) updates, first dropped; optimised figures are the mean of two repeats that agree within 1 %). **These are clean-machine re-measurements** (`results/opt3/base_clean/`, `results/opt3/final3/`, final revision). An earlier
series (`results/opt3/final/`, `final2/`, `bench/base_e*`, and the one-hour soak `bench/soak60_e64_w8.json`) was taken while a stale diagnostic process — a hung profiling run that I had not killed — occupied one logical core from 01:57 to 05:55;
it lowered the baseline by ≈ 11–14 % and the optimised runs by ≈ 2–7 %, so speed-ups there were 5–10 % higher than the clean ones. The clean series is the one to quote.
Peak RSS is the sum over the process tree (shared pages counted once per process, so an overestimate); GPU memory is the whole-GPU high-water mark of the process
tree (the desktop uses ≈ 380 MiB).

| worlds | workers | baseline end-to-end | **optimised end-to-end** | speed-up | baseline rollout → optimised | optimiser s/update (base → opt) | peak RSS GiB (base → opt) | GPU MiB (base → opt) | repeats |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | 2 | – | **902** | **4.5×** (vs 4-worker baseline) | – → 1,059 | – → 0.17 | – → 8.4 | – → 1504 | 2: 905, 899 |
| 16 | 4 | 198.6 | **1,173** | **5.9×** | 351 → 1,451 | 2.24 → 0.17 | 10.3 → 13.2 | 1670 → 2140 | 2: 1,174, 1,173 |
| 16 | 8 | – | **1,275** | **6.4×** (vs 4-worker baseline) | – → 1,612 | – → 0.17 | – → 23.0 | – → 3428 | 2: 1,279, 1,272 |
| 32 | 4 | 226.6 | **1,394** | **6.2×** | 435 → 1,796 | 4.33 → 0.33 | 11.4 → 13.7 | 1670 → 2160 | 2: 1,390, 1,398 |
| 32 | 8 | – | **1,536** | **6.8×** (vs 4-worker baseline) | – → 2,039 | – → 0.33 | – → 23.5 | – → 3432 | 2: 1,537, 1,535 |
| 64 | 4 | 243.1 | **1,344** | **5.5×** | 480 → 1,718 | 8.32 → 0.66 | 13.9 → 14.8 | 1670 → 2264 | 2: 1,339, 1,348 |
| 64 | 6 | – | **1,631** | **6.7×** (vs 4-worker baseline) | – → 2,217 | – → 0.66 | – → 19.6 | – → 2820 | 2: 1,634, 1,628 |
| 64 | 8 | 253.4 | **1,772** | **7.0×** | 534 → 2,469 | 8.49 → 0.65 | 21.4 → 24.5 | 2694 → 3456 | 2: 1,778, 1,767 |

Capacity checks (clean machine, 7 updates each, bounded by a memory guard that kills the run below 2.5 GiB available or +1.5 GiB swap):

| worlds | workers | end-to-end | optimiser s/update | peak RSS | GPU MiB | note |
|---:|---:|---:|---:|---:|---:|---|
| 96 | 8 | 1,744 | 0.99 | 25.5 GiB | 3,490 | fits; −2 % vs 64 (CPU-bound, larger PPO batch) |
| 128 | 8 | 1,789 | 1.31 | 26.5 GiB | 3,682 | fits; +1 % vs 64 |
| 96 | 12 | – | – | – | – | guard stopped the run at 2.3 GiB available: 12 Kit workers do not fit in 31 GB (earlier series) |
| 64 | 10 | 1,519 (earlier series) | – | 29.1 GiB | 4,063 | slower than 8 workers (7/6-world shards, SMT) |

Soak, clean machine (`results/opt3/bench/soakclean_e64_w8.json`): 64 worlds × 8 workers, 1,000 updates, 4,091,904 decisions in 41 minutes, **1,688 decisions/s end-to-end, 1,663 wall-clock**
(the 1.5 % difference is logging, the 160-byte-per-decision trajectory file and the per-update checkpoint). Per-100-update means: 1,711 → 1,717 → 1,716 → 1,690 → 1,671 → 1,677 → 1,678 → 1,673 → 1,679 → 1,676: the first
~300 updates lose 2–3 % as the random policy digs and the workers' physics time rises 1.46 → 1.48 s per rollout, then throughput plateaus. Summed RSS 24.5 → 24.9 GiB (≈ +0.4 GiB/h), GPU 3,458 MiB flat, minimum available memory 7.3 GiB, all losses finite.
An earlier one-hour soak (`bench/soak60_e64_w8.json`, 6.14 M decisions) taken while the stale process occupied a core gave 1,580 / 1,558 with the same shape; per-process RSS probe: workers +42 MiB/h each, trainer flat (`bench/rssprobe_e64_w8.json`).
`trajectories.jsonl` grows ≈ 16 MB/min (≈ 7.7 GB per 8 h).

How the speed-up accumulated (16 worlds × 4 workers, measured in sequence on the same machine; the baseline in that session was 189): GPU PPO and inference 373 →
worker-local rollouts 505 → CUDA-graph sampler + compact observations 604 → batched crouch writes 666 → CUDA-graph PPO minibatch 783 → compiled mini-world controller +
direct tensor I/O 1,069 → spec writes + cudnn benchmark 1,140 → direct `simulate`/`fetch_results` 1,151, and 1,173 in the clean final table (the intermediate figures were measured in the first hours, partly with the stale process present, so only their order of magnitude is comparable). Gains do not multiply: after each step a different part is the critical path.

## C. Where the time goes

Baseline, 16 worlds × 4 workers, per update of 1,024 decisions: collection 2.92 s (physics 2.08 s, the rest policy evaluation, observation transfer and straggler wait),
PPO 2.24 s (35 ms per minibatch on the CPU) — 43 % of the wall clock was PPO (52 % at 64 worlds × 8 workers: collection 7.68 s, PPO 8.49 s). Within the workers (first-session profile): pre-phase 0.13 s, tensor write 0.12, PhysX 0.33, tensor read 0.06,
controller + crouch shapes 0.50, observation construction 0.25, straggler wait 0.45, IPC 0.085 (per 64-decision update, 4 worlds each).

Optimised, one worker's `LocalRollout.run`, profiled with cProfile after the warm-up rollout (`perf/worker_profile_breakdown.py`; cProfile inflates the Python parts a little):

| component | 64 worlds × 8 workers (8 per worker) | 16 worlds × 4 workers (4 per worker) |
|---|---:|---:|
| PhysX step (`simulate` + `fetch_results`, CCD included) | 43.0 % | 38.6 % |
| crouch-shape switching (USD writes + PhysX shape toggles) | 26.8 % | 23.9 % |
| compiled controller (pre/post tick, latch, unpack) | 10.0 % | 9.4 % |
| PhysX tensor-view I/O | 3.1 % | 4.3 % |
| independent terrain rebuild (diff-based) | 0.8 % | 2.0 % |
| observation (compact id windows) | 3.2 % | 3.3 % |
| policy inference (CUDA-graph replay + readback, 8 processes sharing one GPU) | 2.5 % | 4.5 % |
| world reset | 0.0 % | 0.0 % |
| other Python bookkeeping | 10.6 % | 14.1 % |

Trainer side, 64 worlds per update (4,096 decisions): collection 1.72 s, of which the slowest worker's physics 1.57 s, inference ≈ 0.08 s and orchestration (weight
broadcast, result merge, upload) ≈ 0.07 s; optimisation 0.66 s = 256 minibatches × 2.6 ms (GPU kernel time 2.4–3.0 ms per minibatch: convolution forward/weight-gradient
≈ 1.5 ms, elementwise and reductions ≈ 1.2 ms; eager execution would take 4.6 ms). The PPO update is 28 % of the wall clock at 64 worlds, 13 % at 16.
Per-worker straggler imbalance is small once policy evaluation is local: mean vs max worker physics time 3.08 vs 3.20 s per rollout at 128 worlds (3.7 %).

Per physics tick of an 8-world worker (isolated, random policy): PhysX 1,250 µs (156 µs per world, linear in worlds: 1 world 169 µs, 4 worlds 536, 8 worlds 1,086),
crouch switching 430 µs (860 µs when eight workers contend for the caches), controller 150, tensor I/O 85, other ≈ 170. CCD is more than half of the PhysX step
(diagnostic scene with CCD off: 592 vs 1,246 µs); disabled crouch shapes cost 0.125 µs each per tick (198 per world ≈ 16 % of the step).

## D. Scientific validation

Contracts preserved: mini-world and procedural terrain/seeding, independent mutable terrain with immediate consequences, compound collider (no hull), locomotion/crouch/jump/aim/
excavation physics, ten ticks per decision, episode initialisation/termination, observation (16 × 64 × 64 + goal), seven-head action contract, reward and goal rules,
PPO hyper-parameters (rollout 64, minibatch 32, 2 epochs, lr, γ, λ, target KL, clip), network topology (16,385 parameters), optimizer, 100,000-decision ceiling.

1. **Controller, tensor I/O, spec writes — bit-identical.** `perf/xy1_trajectory_dump.py` (4 mini-worlds × 300 decisions, all seven heads random incl. excavation; per decision position,
   velocity, crouch, grounded, reward terms, termination flags, logical and physical terrain hashes): SHA-256 `2ceaaae1…` for the starting commit's reference controller and for the
   optimised tree with `CASU_SHAPE_BATCH=0 CASU_INCREMENTAL_TERRAIN=0` (`results/opt3/trajectory_gates.txt`); `UsdAttribute.Set` and spec writes give identical SHA-256 as well.
   `perf/run_perf_check.py`: 7 of 8 stored trajectory gates bit-identical (shared/independent, fixture, run/random, hull); the 8th, `e4_random_canonical`, fails
   identically on the unmodified starting commit in the shared worktree (its stored baseline from revision f54f505 is stale) and our tree is bit-identical to the starting commit for it.
   Unit suite: 188 tests OK (1 skipped = the Kit-only test, run with `perf/run_kit_tests.py`: OK).
2. **Observation transport — bit-identical.** `tests/test_xy1_compact.py`: the GPU expansion of the compact id windows equals `encode_xy1_observation` bit for bit (incl. windows
   outside the world and the cursor). `perf/local_rollout_equivalence.py` on the real simulator (fresh workers, argmax actions): observations, actions, rewards and
   termination flags identical for 40 decisions × 4 worlds, with and without forced truncations (12-step episodes) and arrivals (goal radius 2 × 1, +10.44 bonus observed, 13–17 resets in the window);
   values differ by ≤ 6e-8 (`results/opt3/equivalence/`).
3. **PPO numerics.** CPU vs CUDA update on identical rollouts/minibatch order (`tests/test_ppo_device.py`): every logged statistic within the predeclared 5e-3 relative (observed ≤ 1e-4),
   parameters within 3e-3 absolute (observed 5.4e-4 ≈ 1.8 lr after one 128-step update, Adam-noise on near-zero-gradient weights); CUDA-graph vs eager: statistics ≤ 1e-4 relative, parameters
   ≤ 1e-4 absolute after three updates. TF32 convolutions and `cudnn.benchmark` are on (algorithm choice may differ at the 1e-6 level).
4. **Sampling.** Gumbel-max on external uniform noise = categorical sampling (`tests/test_local_rollout_sampling.py`: frequencies within 5σ of the softmax for every head on CPU and CUDA graph; log-probabilities equal `Categorical.log_prob`).
   End to end: 12 seeds, first update (same initial weights and worlds, baseline vs optimised): 384 paired action-fraction cells, mean z +0.001, sd 0.98, max |z| 2.95, 3.9 % beyond |z| > 2 (normal 4.6 %) (`results/opt3/first_update/`).
5. **Learning statistics.** 3 seeds × 40 updates × 16 worlds, baseline vs optimised (`perf/learning_compare.py`, `results/opt3/learning/`): entropy, KL, policy/value loss, gradient norm, reward terms differ by |z| ≤ 0.8 across seeds;
   the largest of 60 action-fraction statistics is |z| = 2.9 (expected by chance among 60). Nothing systematic.
6. **Terrain and collision.** Incremental collider updates produce the identical prim set and attributes as a full rebuild after 60 random edit rounds (Kit test); an edit 15 or 30 tiles from the player leaves its trajectory bit-identical;
   adjacent edits change PhysX's internal shape order, so trajectories then differ — 24 paired seeds: speed, position, grounded, crouch and edit rates all within noise (`results/opt3/incremental_terrain_equivalence.txt`).
   Batched crouch writes against CasU (tapes `rand_2001…2010` and four recorded traces): teacher-forced position error +1.3 %, velocity +1.3 %, within the predeclared ±3 % on means (ledger entry 5).
   Collision probes and shape-switch geometry are unchanged (the same 33 × 6 pre-authored shapes; only the write mechanism differs).
7. **Resume.** An optimised run resumes from its own checkpoint on CUDA (updates 4–6 continue from 2,048 steps with entropy/KL/value-loss continuity). Baseline XY1 checkpoints were never resumable (the baseline fails identically on its own checkpoint); fixed for new checkpoints.
   Limitation: a checkpoint written by the worker-local path stores a placeholder observation, so it resumes in the worker-local (CUDA) trainer, not in `CASU_LOCAL_ROLLOUT=0` or CPU mode.

Remaining uncertainty: (i) incremental terrain and batched shape writes are statistically, not bit-wise, equivalent to the old write order (the old order remains available: `CASU_INCREMENTAL_TERRAIN=0`, `CASU_SHAPE_BATCH=0`);
(ii) learning comparisons cover 40 updates, not a full campaign — early statistics are matched, long-horizon training dynamics are not separately verified;
(iii) the sampler's random stream differs from the baseline's, so identical seeds no longer reproduce identical runs (distributions are identical); (iv) cuDNN algorithm selection may differ between processes at the 1e-6 level.

## E. Recommended training configuration

* **Maximum throughput:** 64 worlds, 8 workers × 2 PhysX threads (the default thread count; 1 thread hangs PhysX in a worker, 3–4 do not help), CUDA PPO (`--train-device auto`, the new default), worker-local rollouts and CUDA graphs (all default on),
  `CASU_NUM_WORKERS=8`. Burst **1.77 k**, sustained over 41 minutes **≈ 1.69 k decisions/s end-to-end (1.66 k wall-clock)**. Budget for ~8 h (28,800 s): ≈ 48 M decisions = ≈ 11,700 PPO updates of 4,096 decisions (each ≈ 2.5 s: 1.8 s collection + 0.66 s optimisation + ≈ 1.5 % logging/checkpoint).
  Resources: ≈ 24.5–25 GiB summed RSS (≈ 18 GiB PSS; growth ≈ 0.4 GiB/h → ≈ 28 GiB after 8 h), ≈ 7 GiB system memory left, 3.5 GiB GPU, ≈ 7.7 GB of `trajectories.jsonl` per 8 h — fine for a dedicated run, tight beside other heavy jobs (the GUI). The CPU runs at its thermal limit (82–84 °C package), so the clock is the practical limit.
* **Memory-lean alternative:** 64 worlds × 6 workers: 1.63 k burst (≈ 1.55 k sustained, not soak-tested), 19.6 GiB RSS, 2.8 GiB GPU; budget ≈ 45 M decisions ≈ 11,000 updates in 8 h.
* **Smaller experiments:** 32 worlds × 8 workers 1.54 k burst (≈ 42 M decisions in 8 h ≈ 20,500 updates of 2,048; 32 × 4 gives 1.39 k if memory must be shared); 16 worlds × 8 workers 1.28 k (4 workers 1.17 k, ≈ 35 M decisions ≈ 34,000 updates of 1,024).
* Do not exceed 8 workers on this machine (10 is slower, 12 does not fit); 96–128 worlds fit but add ≤ 1 % over 64.
* Keep `CASU_SHAPE_BATCH`, `CASU_INCREMENTAL_TERRAIN`, `CASU_SHAPE_SPEC` at their defaults for production; set the first two to `0` (and `CASU_COMPILED_CONTROLLER` as is) for bit-exact regression gates.
* Resume with `--resume <run>/latest.pt` (works for checkpoints written by this branch). The 8-hour campaign was **not** started.

## F. Outstanding work

* **Crouch-shape toggling (27 % of a worker).** Each level change is 12 `collisionEnabled` writes; the cost is linear in writes (≈ 5.5 µs each) and dominated by the USD stage/notice machinery and PhysX's shape attach/detach. One `CollisionAPI` per level Xform
  changes the physics; the tensor API cannot enable shapes. Candidates: lazy authoring of only the levels near the current crouch (must be shown not to invalidate the simulation view), a native PhysX shape toggle if one can be exposed.
* **PhysX step (43 %).** CCD is half of it; disabled shapes ≈ 16 %. **Overlap is possible but not integrated:** `perf/physx_async_probe.py` shows that `IPhysxSimulation.simulate()` returns in ≈ 20 µs and `fetch_results()` blocks for the rest of the step, and that 500 µs of main-thread work placed between them is hidden completely
  (fetch 2,148 → 1,582 µs). Within one scene every world is stepped together, so nothing is independent of the step; splitting a worker's worlds into two PhysX scenes (`physics:simulationOwner` on each world's bodies and colliders, `simulate_scene` / `fetch_results_scene` exist in `omni.physx`) would let one half's Python/USD work (crouch writes, controller, tensor I/O ≈ 55 % of a worker) run while the other half's PhysX step is in flight.
  Estimated +15–30 % end-to-end; needs per-scene tensor views, scene-owner authoring for the terrain colliders, and a bit-exactness check against the single-scene trajectories — not attempted within this campaign.
  A first feasibility probe (`perf/two_scene_probe.py`: a second `UsdPhysics.Scene` plus `physics:simulationOwner` on two groups of bodies and colliders, one tensor view per group, `simulate_scene` / `fetch_results_scene`) hung with 100 % CPU before printing its first progress line
  (undiagnosed: probe authoring or omni.physx multi-scene parsing), so the idea is untested beyond the single-scene overlap measurement above.
* **Workers are memory- and core-bound.** A Kit process costs ≈ 0.95 GiB before any world plus ≈ 0.6 GiB for the CUDA context used by the local policy; eight workers fill the 8 physical cores. More workers would need a leaner Kit configuration (fewer extensions) or more RAM.
* **PPO update (28 % at 64 worlds).** The graph is launch-free; the remaining 2.4–3.0 ms per 32-sample minibatch is small-batch convolution and elementwise kernels. `torch.compile` needs a C++ compiler (`g++` is not installed); lower precision would change numerics and was not attempted.
* **Inference contention.** Eight workers time-slice one GPU (≈ 2.5–4.5 % of the rollout); CUDA MPS might halve it (not enabled: system-wide daemon).
* **Observed flake.** One of ~70 benchmark launches hung in the *trainer's* Kit start-up before any world was built (no worker had started; killed by the benchmark's 10-minute timeout; an identical relaunch worked). Long unattended runs should be launched with a start-up watchdog.
* **Housekeeping.** The `e4_random_canonical` stored trajectory is stale (fails on the starting commit too) and should be re-recorded; `casu_grid/workbench/test_gui.py` does not import on this machine (not touched). Checkpoints written by the worker-local path do not resume in the central path.
* **Not done:** a full-length learning comparison, a multi-hour soak (300 updates / 13 minutes were run), and any PhysX-GPU pipeline experiment (would change the physics).

## G. Switches (all defaults are the optimised behaviour; `0` restores the historical one)

| switch | default | effect |
|---|---|---|
| `train.py --train-device` / `CASU_TRAIN_DEVICE` | `auto` (CUDA when present) | device of the PPO model, buffers and update (was hard-wired CPU) |
| `CASU_LOCAL_ROLLOUT` | auto (XY1 stock observation + CUDA trainer) | worker-local policy copies and rollouts; `0` = parent evaluates the policy after every decision |
| `CASU_PPO_GRAPH` | on (CUDA) | CUDA-graph minibatch step with `Adam(capturable=True)`; `0` = eager |
| `CASU_COMPILED_CONTROLLER` | on | compiled controller (now also for the mini-world) |
| `CASU_DIRECT_TENSOR_IO` | on | PhysX tensor-view calls without Isaac Lab wrappers |
| `CASU_DIRECT_STEP` | on | `simulate` + `fetch_results` without `sim.step` wrappers |
| `CASU_SHAPE_BATCH` | on | one `Sdf.ChangeBlock` per tick for all crouch-level writes (not bit-identical to sequential order) |
| `CASU_SHAPE_SPEC` | on | `Sdf.AttributeSpec` writes instead of `UsdAttribute.Set` (bit-identical) |
| `CASU_INCREMENTAL_TERRAIN` | on | diff-based chunk collider updates (same prim set; PhysX shape order differs after adjacent edits) |
| `CASU_NUM_WORKERS` / `CASU_WORKER_PHYSX_THREADS` | workbench default / 2 | worker processes and PhysX threads per worker (≥ 2) |
| `CASU_WORKER_PROFILE=<prefix>` (+ `CASU_WORKER_PROFILE_SKIP=n`) | off | cProfile dump of every worker's rollouts (skipping the first *n*) |

Bit-exact regression gates run with `CASU_SHAPE_BATCH=0 CASU_INCREMENTAL_TERRAIN=0` (`perf/run_perf_check.py` sets the first itself).
Tools added under `casu_grid/perf/`: `xy1_bench.py` (end-to-end benchmark with process-tree RSS / GPU / progress monitoring and `--series`), `xy1_env_profile.py`, `xy1_train_profile.py`,
`xy1_trajectory_dump.py`, `local_rollout_equivalence.py`, `ppo_device_equivalence.py`, `terrain_edit_locality.py`, `learning_compare.py`, `first_update_compare.py`,
`worker_profile_breakdown.py`, `crouch_switch_bench.py`, `crouch_loop.py`, `resume_test.sh`, `run_kit_tests.py`.
