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
| 11 | `ShardedBatchedEnv`: worlds spread over N worker processes behind the same API (`CASU_NUM_WORKERS`) | see the acceptance report tables; isolation, restore, doors, edits, failure handling validated | **accepted** (opt-in) |
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
