# XY1 128 × 128 mini-world

## Design

`xy1_point_navigation_mini.py` selects a separate `xy1_mini_128x128` environment variant. Canonical XY1 remains the 1024 × 1024 CasU Layer 1 world with its existing 85 × 49 observation. Mini XY1 uses 128 × 128 randomized terrain and a provisional 64 × 64 spatial observation. The action branches, ten-physics-tick decision cadence, 0.1 best-distance reward, +10 arrival reward, arrival radius, compound player collider, PPO architecture and settings, gamma 0.99, GAE lambda 0.98, and 100,000-decision ceiling are inherited from the canonical experiment.

Each environment receives a deterministic terrain seed based on its global environment index. Terrain is a smoothed random occupancy field populated with CasU Layer 1 material IDs sampled at measured source frequencies. A level two-cell solid floor is always present. Side boundaries are noisy Infinirock (ID 14), with a 2–5 cell inward profile; the top remains open. No pod, spawn fixture, or recurring structure is created. Only the 3 × 6 tile footprint needed to clear the standing body at the center spawn is cleared. The center body begins with zero velocity. Mini terrain is authored independently for each environment and excavation only updates that environment's collision chunks.

The goal is sampled uniformly from the 64 × 64 viewport centered on the initial spawn, then stored in world coordinates until reset. Relative offsets remain signed and unbounded after the goal leaves view. This first mini-world implementation tests 64 × 64 only. Later 32/48/64 FOV comparisons should draw goals from the same fixed common region so that visibility changes do not also change goal-distance distributions.

## Bounded correctness and PPO evidence

The focused regression suite passed 38 tests covering canonical geometry coordinates, mini partial chunks, body clearance, independent terrain edits, center spawn, open top and boundaries, goal sampling and persistence, seven action branches, observation dimensions, arrival, reward, and existing controller/runtime/reference behavior. The mini experiment config passed the workbench contract preflight with `(16,64,64)` spatial input, two goal-relative scalars, the canonical seven action branches, unchanged rewards and PPO parameters.

Matched-density engineering-smoke runs used 16 worlds, three PPO updates, 64 rollout decisions per world/update, independent terrain, seed 391728405, and an explicitly labeled `--engineering-smoke` launch. Each completed profile therefore performed 3,072 vectorized environment decisions and three optimizer updates; no substantive campaign was run. Throughput below is aggregate across 16 worlds and is the mean of updates 2 and 3 to reduce first-update warm-up effects.

| Variant | Workers | Environment startup | Terrain geometry build | Terrain box colliders (16 worlds) | Aggregate decisions/s | Physics step/update | Collection/update | PPO optimization/update | Peak / steady RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Canonical 1024² | 1 | 68.98 s | 40.39 s | 635,936 | 112.9 | 7.63 s | 9.07 s | 3.72 s | 16.12 / 13.70 GiB |
| Mini 128² | 1 | 5.63 s | 2.32 s | 19,597 | 199.8 | 3.71 s | 5.13 s | 4.32 s | 3.14 / 2.58 GiB |
| Canonical 1024² | 4 × 4 worlds | 28.77 s | 14.60 s | 635,936 | 178.7 | 4.29 s | 5.73 s | 4.23 s | 23.18 / 22.35 GiB |
| Mini 128² | 4 × 4 worlds | 8.41 s | 0.68 s | 19,597 | 270.4 | 2.36 s | 3.80 s | 4.16 s | 10.27 / 9.53 GiB |

The mini has 96.9% fewer terrain box colliders than canonical. At one worker it reaches about 1.8× canonical throughput; at four workers it reaches about 1.5× canonical throughput. With four workers, startup and terrain build were about 3.4× and 21.6× shorter, respectively. PPO optimization time stayed similar, so the throughput improvement comes from environment construction/physics/collection rather than changing PPO. Mini average occupancy was 4,777 solid cells/world (29.2%); canonical occupancy was 320,647 (30.6%), so the collider reduction reflects both the 64× smaller raster and comparable density rather than making the mini terrain artificially sparse.

RSS is the sum of the launcher process and descendants sampled every 0.25 seconds with `psutil`; steady RSS is the median of samples after the first 10 seconds. It is host resident memory, not GPU allocation. Run artifacts are under `experiments/xy1_canonical_bench_16w_1p_retry`, `experiments/xy1_canonical_bench_16w_4p`, and `experiments/xy1_mini128_matched_density_16w_1p`; the final corrected 4-worker mini run is `experiments/xy1_mini128_matched_density_16w_4p_final`.

## Fidelity limits and next step

This generator is a deterministic CasU-material procedural approximation, not CasU's original game RNG, geological topology, or seed reproduction. It uses one exact occupancy rectangle collider per merged region within each 64 × 64 source chunk; raster occupancy is exact, while PhysX contact behavior remains an engine approximation shared with the accepted compound-collider baseline. Infinirock uses the source simulator's very large finite health value rather than mathematical infinity (420,133,760 health); it is effectively indestructible within the 100,000-decision episode ceiling. The floor is two cells thick, and noisy side walls extend only their profile depth; no world-top cap is present.

Recommended next step: review the 128 × 128 scaling evidence and approve a bounded, non-learning capacity check at the desired environment count before choosing a substantive campaign. Keep 64 × 64 as the provisional mini FOV. Run the 32/48/64 controlled FOV ablation later with a common goal-sampling region and unchanged XY1 rewards, action semantics, PPO settings, and ceiling.
