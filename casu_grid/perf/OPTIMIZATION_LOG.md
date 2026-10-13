# Optimization experiment log

Chronological record of every candidate tried in the performance campaign (`optimization.txt`), with
the evidence for keeping or rejecting it. Machine: i7-11700K (8 cores / 16 threads), RTX 3070, 32 GB,
Isaac Sim 4.5.0 / Isaac Lab 2.1.1, PhysX on CPU. Workloads: `random` (seeded random move/jump/crouch,
crouch-heavy, the historical benchmark tape) and `run` (hold right, jump every 4th decision). All
throughput figures are aggregate physics ticks/s, shared terrain, full canonical world, mean of three
identical measurement windows unless stated. Baseline = revision `f54f505`.

Correctness gate used for every accepted change ("Gate C"): the batched environment is replayed on three
seeded workloads (`perf/trajectory_dump.py`) and must reproduce the stored baseline trajectories
(`perf/baselines/trajectories/`, recorded from `f54f505`) **bit for bit** (positions, velocities, crouch,
rewards, terminations, fluid digests). All accepted items below passed it.

| # | Candidate | Result | Verdict |
|---|---|---|---|
| 0 | Baseline reproduction | 16 worlds random: 1686 ± 7 ticks/s (run: 1975). Phase split: controller+fluid pre 35 %, controller post + crouch-shape switching 48 %, PhysX 11 %, tensor I/O 6 % | reference |
| 1 | Numba kernels for the fluid sweep and tile queries (`kernels.py`), polygons stay in Python | 2622 (+56 %) | **accepted** |
| 2 | Persistent NumPy staging buffers for the per-tick tensor writes/reads instead of per-element torch indexing | 3257 together with 1 (+24 % over 1) | **accepted** |
| 3 | Batch all 12 crouch-shape `collisionEnabled` writes of every world in one `Sdf.ChangeBlock` | 3818 (+17 % over 2) but trajectories differ from the baseline by 0.1 m within the first macrostep (PhysX processes the changes in another order) | **rejected** (not bit-identical; would change the validated physics) |
| 4 | Compiled controller tick (`controller_kernels.py`) with reference fallback for tiles only, polygon queries in Python | random 3777 (+16 % over 2) but *slower* on the `run` workload (static objects near the spawn force frequent fallbacks) | superseded by 5 |
| 5 | 4 plus a compiled polygon path (exact port of the Minkowski-hull sweep incl. `round(x, 12)`, ray, SAT overlap) | random 4160 (2.47x baseline), run 9036 (4.6x baseline) in one process; bit-identical on fuzzed states and on all gate workloads | **accepted** |
| 6 | PhysX CPU thread count (`/persistent/physics/numThreads`) 2 / 4 / 8 at 16 worlds | 3238 / 3225 / 3257: no effect (0/1 threads hang) | rejected |
| 7 | `use_fabric=False` | crouch-shape switch 187 µs vs 173 µs per change | rejected |
| 8 | Switch crouch levels by activating/deactivating `Level_k` prims instead of toggling 12 `collisionEnabled` attributes | invalidates the physics-tensor view ("prim deleted while used by a shape") | rejected |
| 9 | Cheaper shape switch by shape type | every attribute write costs ~26 µs regardless of shape type (boxes 27, cylinders 26): it is the USD-to-PhysX notice, not cooking | no lever without changing the representation |
| 10 | Independent Isaac processes, 2 / 4 concurrent benchmark processes x 16 worlds | 5982 / 10389 aggregate ticks/s (1.84x / 3.2x of one process) | led to 11 |
| 11 | `ShardedBatchedEnv`: worlds spread over N worker processes behind the same API (`CASU_NUM_WORKERS`) | 64 worlds: 4 workers 12.1k (`random`) / 23.7k (`run`) ticks/s; 6 workers 96 worlds 16.2k / 31.3k; training 64 worlds 918 decisions/s (4.5x baseline); isolation, restore, doors, edits, failure handling validated | **accepted** (opt-in) |
| 13 | PhysX threads per worker 2 / 3 / 4 / 6 (4 workers, 64 worlds) | 11.8k / 12.1k / 12.0k / 11.8k ticks/s; 1 thread hangs PhysX | default 2 kept |
| 14 | Skip the PhysX scene reset after independent-terrain chunk rebuilds | `set_block` 6.2 s -> 0.036 s; wall/pit scenarios give equal trajectories; bit-identical gates unchanged | **accepted** (`terrain_edit_scene_reset=True` restores the old behaviour) |
| 15 | Independent terrain copies via `Sdf.CopySpec` under an offset root Xform | 16 worlds build 98 s -> 63 s; trajectories bit-identical (the rest is the PhysX scene parse) | **accepted** |
| 16 | `OMP_NUM_THREADS` for the trainer process (64 worlds, 4 workers) | 1 / 2 / 4 / 8 threads: 962 / 966 / 949 / 928 vs 918 decisions/s unset | optional (+4 %), documented |
| 17 | Reuse the previous step's value for the PPO bootstrap value (drops one forward pass per decision) | not implemented: inference is 2-6 % of rollout even with a 512-wide network, so the gain is below 3 % | deferred |
| 18 | Bucketed polygon culling + additive bucket painting on door updates | reset latency back to baseline (39 ms for 16 worlds; a naive per-update rebuild had made it 108 ms) | **accepted** |
| 12 | GPU PhysX pipeline | not attempted: changes solver numerics/contact behaviour (fidelity invariant), and the CPU controller is the limit at small body counts | deferred |

Notes that matter for future work
* Changing the *composition* of a PhysX scene changes results at the 1e-9 level and chaotic corner
  contacts amplify that: trajectories are bit-identical only for the same world partition (same
  `num_envs` and `num_workers`). The same property already held for different `num_envs`.
* `set_level` (crouch-shape switching) is the largest remaining single cost on crouch-heavy tapes
  (about 100 µs per world per tick on `random`); it is intrinsic to the validated 33-level
  composite-collider representation.
* Never `source isaac_env.sh` twice in one shell: it rewrites `HOME` and the second source silently
  produces an empty interpreter path.
