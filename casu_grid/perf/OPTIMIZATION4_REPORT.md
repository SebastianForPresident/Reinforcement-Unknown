# Optimization IV — report

Branch `optimization4`, worktree `Reinforcement-Unknown-Isaac-opt4`, starting revision `e78cfe2` (Optimization III, branch `optimization3`, untouched).
Machine: i7-11700K (8 cores / 16 threads), RTX 3070 8 GB, 31 GB RAM, Fedora 44, Isaac Sim 4.5.0 / Isaac Lab 2.1.1.
Task: the unchanged XY1 mini-world experiment (128 × 128 world, independent destructible terrain per world, compound collider, 16 × 64 × 64 observation + 2 goal coordinates,
heads [3, 2, 3, 2, 9, 11, 2], ten physics ticks per decision, PPO rollout 64 / minibatch 32 / 2 epochs, 16,385 parameters). Nothing about the task, the reward, the physics
fidelity or the PPO configuration was changed. Experiment ledger: `OPTIMIZATION4_LOG.md`; raw data `casu_grid/results/opt4/`.

## A. Summary (read this first)

1. **Optimization III (and the baseline before it) measured a broken workload.** In the XY1 mini-world every world except world 0 was mis-placed: the spawn and the goal were
   computed in the world's *local* frame while `reset()` subtracts the world offset again. With more than one world, each world ≥ 1 therefore started at a slot-local
   x of −offset: its body was written at world 0's position (all bodies stacked on top of one another, standing on world 0's collider copy), its observation window lay
   outside the world (no non-empty cell, fluid id 7 everywhere) and its goal was ≈ 10³ tiles away. Single-world runs were unaffected. **Any throughput or learning number
   taken with more than one world before commit `a7ebcab` describes that broken workload**; the speed-ups of Optimization III are real for what was measured, but the
   learning behaviour of those runs was not that of the intended experiment. Fixed in `a7ebcab`; `CASU_LEGACY_MINI_SPAWN=1` restores the old behaviour for references.
   **This fix changes what the experiment does and needs your review before any learning campaign** (it makes every world a proper, independent, correctly-placed copy).
2. **Throughput gains of Optimization IV are small: +2.2 % at 64 worlds, +5.2 % at 32, +1.8 % at 16** (corrected workload, same code base and worker layout, clean machine,
   two repeats each). The remaining cost is native PhysX/USD work in eight CPU-saturated processes; the cheap Python/NumPy wins were already taken in Optimization III. The one
   measurable code change is the vectorised terrain rectangle decomposition (+4 % on the corrected workload, where worlds actually dig).
3. **Soak and memory are healthy**: a 27-minute 700-update and a 36-minute 899-update soak at 64 × 8 (1,752 decisions/s end-to-end, 1,716 wall-clock) completed without failure; unique memory (PSS/USS, not summed RSS)
   is ≈ 17 GiB and grows ≈ 0.35–0.4 GiB/h in total. Summed RSS (25 GiB) overstates real use by shared pages.
4. **Two latent bugs found and fixed**: a crash when a world's boundary cell was dug (`dafc767`) — only reachable after the spawn fix, because before it worlds ≥ 1 never
   dug in their own terrain — and the "stale" `e4_random_canonical` regression gate, which turned out to be a real behaviour change in the Layer-1 environment introduced by
   the XY1 commit `dd6b66f` (collider level at reset); restored for non-XY1 environments (`e6e3838`), all 8 of 8 trajectory gates are now bit-identical.
5. **Checkpoint / resume works on the corrected workload**, including intentional interrupts, a killed worker, and clear failures for incompatible resumes (section E).
6. **Rejected with data**: two PhysX scenes per worker (−12 %), fp16/bf16 PPO (unstable / outside tolerance), more or fewer workers, THP malloc tunable.

## B. Benchmarks

Protocol as in Optimization III (`perf/xy1_bench.py`, 11 updates, first dropped, end-to-end = decisions / (collection + optimisation seconds), 8 workers × 2 PhysX threads,
CUDA trainer, worker-local rollouts), but interleaved: for every world count and repeat the four variants run back to back on an idle machine, and the memory columns are
**PSS/USS** from `/proc/PID/smaps_rollup` (summed RSS double-counts shared pages). "Legacy" = the mis-placed workload of Optimization III; "corrected" = every world at its own spawn.
Reference code: `e78cfe2` for the legacy columns, `a7ebcab` (Optimization III + only the spawn fix) for the corrected columns. Data: `results/opt4/final/`, table `final/TABLE.md`.

Legacy workload — Optimization III (`e78cfe2`) vs Optimization IV with `CASU_LEGACY_MINI_SPAWN=1` (decisions/s):

| worlds | workers | Opt III (runs) | Opt IV (runs) | speed-up | peak PSS GiB | peak RSS GiB |
|---:|---:|---|---|---:|---:|---:|
| 16 | 8 | 1,259 (1,251, 1,267) | **1,273** (1,266, 1,279) | 1.011× | 15.4 | 23.0 |
| 32 | 8 | 1,471 (1,407, 1,535) | **1,537** (1,537, 1,537) | 1.045× | 15.9 | 23.5 |
| 64 | 8 | 1,762 (1,768, 1,756) | **1,771** (1,761, 1,782) | 1.005× | 16.9 | 24.5 |

Corrected workload — Optimization III code + spawn fix (`a7ebcab`) vs Optimization IV HEAD:

| worlds | workers | Opt III code (runs) | Opt IV (runs) | speed-up | collection s/update | optimiser s/update | peak PSS GiB |
|---:|---:|---|---|---:|---|---|---:|
| 16 | 8 | 1,245 (1,236, 1,254) | **1,267** (1,263, 1,272) | 1.018× | 0.65 → 0.64 | 0.17 | 15.4 |
| 32 | 8 | 1,468 (1,420, 1,515) | **1,545** (1,546, 1,543) | 1.052× | 1.06 → 1.00 | 0.34 → 0.33 | 15.9 |
| 64 | 8 | 1,702 (1,699, 1,704) | **1,740** (1,740, 1,739) | 1.022× | 1.75 → 1.70 | 0.66 → 0.65 | 16.9 |

The workload fix itself costs 0–3 % (64 worlds: 1,762 → 1,702 with Optimization III code): the corrected workload does less PhysX work (bodies no longer stack: ≈ 530 µs
instead of ≈ 1,250 µs per 8-world tick) but now digs in every world, so terrain collider rebuilds and the Python share of a tick grow.
Optimization III's published figures (1,275 / 1,536 / 1,772) are reproduced within 1–2 % by the legacy column (1,259 / 1,471 / 1,762): same machine, same protocol, no drift.
The spread between runs of the same variant is ≈ ±1 % at 64 worlds and up to ±4 % at 32 worlds (the 32-world Optimization III runs differ by 9 %), so the 16- and 64-world
gains are at the edge of what can be resolved with two repeats and only the 32-/64-world ones are outside noise on the corrected workload.

Peak memory and GPU are identical between Optimization III and IV (no code that allocates changed): GPU 3.4–3.5 GiB, PSS 15.4 / 15.9 / 16.9 GiB at 16 / 32 / 64 worlds.

## C. What was done

| # | change / finding | evidence | decision |
|---|---|---|---|
| C0 | **Mini-world spawn / goal offset bug** (section A.1) | `perf/scene_state_probe.py` (XFORMS / PHYSXTR / OBS lines), `OPTIMIZATION4_LOG.md` C0 | fixed `a7ebcab`, legacy flag |
| D2 | Terrain rectangle decomposition vectorised (`geometry._chunk_row_runs`): identical rectangles in identical order, 1.93 → 0.64 ms per chunk (8.3 % → ≈ 3 % of worker time on the corrected workload) | `tests/test_geometry_runs.py`; 64 × 8: 1,670 → 1,741 | **accepted** `1f5d68b` (Class I, bit-identical) |
| F1 | Boundary dig crash (`chunk lies outside terrain bounds`, plus a wrapped write into the top row): the hit cell is clamped like the grid clamps it | 30-minute run died at update 35 before the fix, completed 700 updates after | **accepted** `dafc767` (in-range digs unchanged: gate `284e8dde…`) |
| H1 | `e4_random_canonical` gate: first bad commit `dd6b66f`; `_write_slot_states` used `level_for_crouch(crouch)` instead of `set_level(0)` for all modes | bisect + per-hunk patches | **accepted** `e6e3838` (Layer-1 only; XY1 hashes unchanged) — please review, it touches non-XY1 code |
| B1–B3 | Two PhysX scenes per worker: root cause of the Optimization III hang (numThreads = 2 with two scenes deadlocks `_warm_start`; 0/3/4 work) and a working software-pipelined prototype | 64 × 8 × 4 threads: 1,461 vs 1,669 (−12 %) | **rejected**, patch kept (`results/opt4/two_scene_prototype.patch`) |
| E1 | PPO minibatch under autocast | fp16: update error 4.8 % vs 2.6 % for the accepted TF32, +6.4 % speed, but non-finite loss after 15–42 updates in 6/6 seeds for loss scales 1/64/1024; bf16 outside tolerance (14 % update error) | **rejected**, patch kept |
| D1 | Worker count on the corrected workload | 6 / 8 / 9 / 10 workers = 1,544 / 1,740 / 1,625 / 1,579 | 8 stays optimal |
| D3 | `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` | Kit start-up hung | rejected |
| F2 | Kit persists `/persistent/physics/numThreads` in `user.config.json`: a `--physx-threads 1` probe made later default-thread runs hang until the key was removed | `faulthandler` stack | tools got hard deadlines; workers always set the value explicitly |

No change to PPO, the model, the optimizer, the controller, the PhysX configuration (CCD untouched), the crouch implementation or the observation path was accepted.
`git diff e78cfe2..HEAD` over production code touches four files: `geometry.py`, `isaac_vector_env.py`, `test_xy1.py`, `tests/test_geometry_runs.py`.

### Where the time goes now (corrected workload, 64 worlds × 8 workers)

Per update (4,096 decisions): collection 1.70 s (slowest worker's physics ≈ 1.53 s), PPO 0.65 s (256 minibatches, 2.4–2.6 ms each, CUDA graph) = 28 % of the wall clock.
Inside a worker (cProfile + isolated per-tick timers): PhysX step plus tick overhead ≈ 39 % (its in-situ cost doubles when eight workers contend for caches and memory
bandwidth), crouch-shape switching ≈ 27 %, compiled controller and glue ≈ 10 %, terrain rebuilds ≈ 3 % (was ≈ 8 % with the unvectorised decomposition), tensor-view I/O ≈ 3 %,
policy inference ≈ 2.5 %, other Python ≈ 10 %. The ideas that remain (hoisting the pre-step transform read, trimming controller glue) are worth 1–3 % each and carry
cache-invalidation risk; they were not taken. Larger steps would need native changes (a PhysX-side crouch toggle, CCD cost) or a different physics model, which are out of scope.

## D. Sustained run and memory

* 700-update soak after the boundary fix (`results/opt4/mem30_e64_w8.json`): no failure; 1,722 decisions/s end-to-end (1,686 wall-clock); first / last 100 updates 1,665 → 1,758;
  PSS 16.9 → 17.0 GiB, USS 16.08 → 16.23 GiB; USS slope after 300 s ≈ 34 MiB/h per worker, ≈ 55 MiB/h trainer, ≈ 344 MiB/h total; peak system used 23.3 GiB, minimum available 7.9 GiB; no swapping.
* Final soak: 899 measured updates (3,682,304 decisions, 36 minutes) at 64 × 8 on the corrected workload: **1,752 decisions/s end-to-end, 1,716 wall-clock** (logging, trajectory file and per-update checkpoint cost 2 %); per-100-update means 1,708 → 1,740 → 1,752 → 1,742 → 1,773 → 1,800 → 1,738 → 1,747 → 1,779 (no degradation); all losses finite; peak PSS 17.25 GiB, USS 16.44 GiB (USS slope ≈ 0.4 GiB/h), summed RSS 24.9 GiB, GPU 3.6 GiB, minimum available memory 6.8 GiB, no swapping (`results/opt4/soak_final_e64_w8.json`).
* Summed RSS (≈ 25 GiB) is not the footprint: ≈ 8 GiB of it is pages shared between the Kit processes. Use PSS/USS when sizing; the machine stays 7–8 GiB below its limit at 64 × 8,
  which still rules out 12 workers (each Kit worker has ≈ 1.8 GiB of unique memory plus shared pages; the 10- and 9-worker runs were slower anyway).

## E. Checkpoint and resume (16 worlds × 2 workers, CUDA trainer, worker-local rollouts; `results/opt4/resume/`)

| test | result |
|---|---|
| fresh 4 updates → `latest.pt` | update 4, 4,096 steps, finite weights, 24 Adam states at step 256, observation and sharded environment snapshot present |
| resume + 4 updates | update 8, 8,192 steps, Adam step 512 — the same counters as an uninterrupted 8-update run |
| fresh vs fresh | updates 1–4 of two separate runs are bit-identical in every metric |
| resumed vs uninterrupted | not bit-identical from update 5 (restore resets PhysX contact caches and warm starts); statistics stay in distribution (entropy 8.76 vs 8.77, KL 0.004–0.008) |
| resumed vs resumed | same trajectory and PPO statistics to ≈ 1e-6 relative |
| SIGINT to the process group (what the GUI does) | exits in 0.5 s, checkpoint of the last completed update is valid, resume continues (3 → 5). Kit's handler wins, so no `INTERRUPTED` marker and exit status 0; `lab._status` classifies metrics without `COMPLETE` as *interrupted* |
| `kill -9` of a shard worker | `WorkerFailure` and `failure.json` within 10.6 s, no leftover processes, valid checkpoint, resume continues (3 → 5) |
| incompatible resume: other world partition / shared terrain / other experiment | clear `ValueError`s ("snapshot has wrong sharded schema or world partition", "XY1 navigation requires independent terrain…", "PPO checkpoint learning contract is incompatible with this run"), `failure.json` written |

The `python.sh` wrapper exits with status 0 even on failure (pre-existing); `failure.json` is authoritative, as `lab.training_result_failed` assumes.

## F. Equivalence and validation

| change | class | evidence |
|---|---|---|
| rectangle decomposition | I (bit-identical) | `tests/test_geometry_runs.py` compares with the previous per-row implementation on random chunks; XY1 trajectory gate unchanged |
| boundary clamp | I for in-range digs; a crash fix otherwise | trajectory gate `284e8dde…` unchanged |
| spawn / goal offset | **workload change** (not an optimisation) | per-world probes; reference hashes `284e8dde…` (corrected) and `2ceaaae1…` (`CASU_LEGACY_MINI_SPAWN=1`, equal to the Optimization III starting commit) |
| Layer-1 reset collider level | restores bit-identity of the stored gate | `perf/run_perf_check.py --quick`: 8 of 8 trajectory gates bit-identical (was 7 of 8); XY1 hashes unchanged |

Tests on HEAD: `casu_grid/tests` 147 tests OK (1 skipped = the Kit-only test, run with `perf/run_kit_tests.py casu_grid.tests.test_incremental_terrain`: OK);
`casu_grid` top level 46 tests OK except `workbench/test_gui.py`, which cannot import PySide6 in this environment (pre-existing). The XY1 goal-sampling test needed
the per-world offset and was updated; a new test checks the stage-frame placement of worlds ≥ 1. `perf/run_perf_check.py --quick` throughput: 1.2–1.7× over the stored pre-optimisation reference.
PPO and the model are byte-identical to Optimization III, so no new numerical or statistical PPO validation was required; the rejected autocast variant was validated and failed
(`results/opt4/amp_fp16_learning_compare.txt`).
Not done: a statistical learning comparison of the corrected against the legacy workload (a learning run is outside this mission); expect learning curves to differ, since the
legacy runs trained on stacked bodies with off-world observations.

## G. Recommendation

* Configuration unchanged: **64 worlds × 8 workers × 2 PhysX threads, CUDA trainer**, corrected workload. Do not use 10+ workers.
* Expect ≈ 1,700–1,750 decisions/s sustained (≈ 6.2 M per hour). 8-hour forecast: ≈ 1,716 × 28,800 s ≈ **49 M decisions** (range 48–50 M), unique memory ≈ 17 → 20 GiB.
* Memory: ≈ 17 GiB unique, +0.35 GiB/h; keep 7+ GiB free; `trajectories.jsonl` grows ≈ 16 MB/min (≈ 7.7 GB per 8 h).
* Before the learning campaign: review commit `a7ebcab` (workload fix) and `e6e3838` (Layer-1 reset), then re-run any learning baselines that were collected with more than one world.
* Best validated commit: the commit that contains this report (`git log -1 -- casu_grid/perf/OPTIMIZATION4_REPORT.md`; code identical to `92022c2`); worktree `Reinforcement-Unknown-Isaac-opt4`, branch `optimization4` (not merged; `optimization3` untouched).

## H. Tools added

`perf/xy1_bench.py` (PSS/USS, GPU utilisation, per-process series), `perf/make_opt4_table.py`, `perf/checkpoint_inspect.py`, `perf/resume_fault_test.py`,
`perf/two_scene_probe.py`, `two_scene_overlap.py`, `two_scene_terrain_probe.py`, `scene_state_probe.py`, `ppo_amp_equivalence.py`; deadlines (`PROFILE_DEADLINE`, `DUMP_DEADLINE`) in the profiling and trajectory tools.
