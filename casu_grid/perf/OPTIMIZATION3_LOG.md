# Optimization III — experiment ledger

* Start: **2026-10-13 23:39 EDT** (epoch 1791949175); deadline 2026-10-14 07:39 EDT. Branch `optimization3` (worktree `Reinforcement-Unknown-Isaac-opt3`), starting commit `0d46520`.
* Machine: i7-11700K (8C/16T), RTX 3070 8 GB, 32 GB, Fedora 44; Isaac Sim 4.5.0 / Isaac Lab 2.1.1, PhysX on CPU, 2 PhysX threads per worker (min 2).
* Experiment: XY1 mini-world (128 × 128, independent terrain, compound collider, 16 × 64 × 64 spatial observation, 7 action heads, 10 ticks/decision, PPO rollout 64 / minibatch 32 / 2 epochs, 16,385 parameters).
* Throughput accounting (all benchmarks): `end-to-end = decisions / (collection_seconds + optimization_seconds)` over steady updates (first update dropped), taken from the trainer's own per-update timers;
  rollout = decisions / collection_seconds. Benchmark: `perf/../isaac_training_benchmark.py --terrain-mode independent --config workbench/experiments/xy1_point_navigation_mini.py`, runner `bench.sh` (scratch).
  `random-policy` regime: PPO has not learned; the near-uniform policy digs and crouches constantly, i.e. the worst case for the simulator.

## Baseline (commit 0d46520, CPU PPO, 6 updates, 2nd–6th measured)
| worlds × workers | sim-only | rollout | end-to-end | optimizer share |
|---|---:|---:|---:|---:|
| 16 × 4 | 464 | 330 | **189.3** | 42.6 % |

Per update (16 worlds, 1,024 decisions): collection 3.05 s (env step 2.15 s + policy inference/bookkeeping 0.9 s), PPO 2.32 s (36 ms per minibatch). The mission's 128.6 figure used a different accounting/load; this ledger's numbers are re-measured baselines.

## Profile (A) — where the time went (16 × 4)
* Trainer process (cProfile, 3 updates, 15.2 s): waiting for workers 5.6 s; `run_backward` 3.7 s; `conv2d` forward 2.75 s; `relu` 0.7 s — **the CPU runs the 16 × 64 × 64 convolution network for inference and PPO** (`train.py` hardcoded `device='cpu'`; the RTX 3070 was idle).
* Workers (averaged per 64-decision update): pre-phase 0.13 s, tensor write 0.12, PhysX 0.33, tensor read 0.06, controller-post + crouch-shape switching 0.50, observation construction ≈ 0.25, straggler wait 0.45 (24 % of the env step), IPC 0.085.
* Single-process 4-world cProfile: PhysX 27 %, crouch-shape switching (`set_level`) 27 %, reference Python controller ≈ 35 %, XY1 observation encoder 6 %. The mini-world does **not** use the compiled controller (`_build_fast_runtime` returns `None` for it).

## Ledger
| # | hypothesis | change | evidence | result | decision |
|---|---|---|---|---|---|
| 1 | CPU conv network dominates; GPU is idle | `--train-device` / `CASU_TRAIN_DEVICE` (cpu\|cuda\|auto) in `workbench/train.py`; model, rollout buffers, update on CUDA | `perf/ppo_device_equivalence.py` (rel. diff of every logged statistic ≤ 1e-4, max |Δparam| 5.4e-4 ≈ 1.8 lr after one 128-step update, Adam-noise on near-zero-gradient weights), `tests/test_ppo_device.py` (tolerances predeclared: 5e-3 relative on losses/grad-norm, 3e-3 absolute on parameters) | 16 × 4: optimizer 2.32 → 0.39 s, inference 0.9 → ~0.3 s; **189 → 373 decisions/s (1.97×)** | keep |
