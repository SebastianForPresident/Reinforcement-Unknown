# Isaac Lab Layer 1 — final engineering acceptance report

Updated 2026-10-13. Branch `isaac-lab`, seed `391728405`.

**Status in one paragraph.** The lab is operational and training-ready: the
canonical Layer 1 world loads in about 5 s, a source-validated standing
controller drives a PhysX player through it, worlds reset, snapshot and restore
exactly, many independent worlds step in one process, and the user-owned
workbench runs a real PPO loop with metrics, checkpoints, resume and replay.
It is **not** a bit-exact CasU replica. Per-decision motion matches CasU closely
on ground, in the air and against walls; landings, rare tile-junction bumps and
long free-running trajectories diverge (section 6). Do not describe it as fully
CasU-equivalent.

## 1. Runnable environment and exact startup commands

All commands run from the repository root with Isaac Sim 4.5.0's bundled Python
(set up as in [`../ISAAC_LAB_SETUP.md`](../ISAAC_LAB_SETUP.md)):

```bash
ISAAC_ROOT="$HOME/Applications/IsaacLab-2.1.1-IsaacSim-4.5.0"
ISAAC_SIM="$ISAAC_ROOT/isaac-sim-4.5.0"

# regression suite (85 tests, pure Python/NumPy, ~5 s)
PYTHONPATH=$PWD "$ISAAC_SIM/python.sh" -m unittest discover -s casu_grid/tests -p "test_*.py"

# real PPO run (engineering-only config shown; replace with your own)
"$ISAAC_SIM/python.sh" -m casu_grid.workbench.train \
  --config casu_grid/workbench/engineering_smoke.py \
  --updates 100 --num-envs 8 --run-dir experiments/my-run

# monitor and inspect
"$ISAAC_SIM/python.sh" -m tensorboard.main --logdir experiments/my-run/tensorboard
"$ISAAC_SIM/python.sh" -m casu_grid.workbench.replay \
  --config casu_grid/workbench/engineering_smoke.py --run-dir experiments/my-run
```

Validation entry points: `casu_grid/isaac_player_ab.py` (CasU-vs-Isaac player
A/B), `casu_grid/isaac_vector_smoke.py` (batched isolation/restore/doors/edits),
`casu_grid/isaac_throughput_benchmark.py` (throughput). Section 13 is the
short user guide.

## 2. Canonical Layer 1 fixture verification

- Fixture `casu_grid/fixtures/casu-layer1-seed-391728405-v1/`, content ID
  `sha256:0f1413f47d255bbab25fbc43aaa433efdbbb8c15a260ec48268cd02429495ba8`.
- Full 1024×1024 terrain and fluid arrays (x-first), 320,647 occupied cells, 256
  logical 64×64 chunks. The importer tests assert array equality with the
  fixture payload and checksums; the discrete terrain/fluid update rules
  (`state.py`) are transcribed from the game source and checked by
  `test_state`/`test_reference`. A fast fluid sweep is tested for exact
  equivalence to the retained oracle `simulation_step_reference`.
- 601 non-tile static colliders from world generation (shuttle doors, crates,
  pods, jump pads…) are measured from the source (`physics_audit/object_colliders.json`)
  and authored in Isaac; movable bodies and layers that do not collide with the
  player are excluded and listed in the audit manifest. The ShuttleStartOpen
  doors are modelled per world (trigger, 2 s delay, rise, 10 s destroy).

## 3. Collision reconstruction evidence and CasU A/B

- Terrain: one cuboid per exact occupied raster rectangle, chunk-local
  (39,746 cuboids). Authored as raw Sdf specs: ~4.6 s for the full world in
  shared mode (was ~200 s with the USD helper API).
- Source Physics2D occupancy audit of all 1024×1024 cells
  (`results/casu_physics_occupancy_full_20261013.bin`; tests in
  `test_world_objects`): every occupied terrain cell is covered by a source
  collider at its centre and four corner probes, and every non-tile solid cell is
  explained by a measured static object collider or the footprint of an excluded
  dynamic one.
- Player collider: rounded box (edge radius 0.1) as two cross-expanded cuboids
  plus four corner cylinders, pre-authored in 32 crouch levels; rest offset
  0.0155 reproduces the source's 0.1155 resting gap. Terrain contact offset
  0.001 / player 0.04 eliminates "seam ghost" braking on coplanar tile boundaries
  while keeping speculative contacts for ~8 m/s landings (found and fixed in
  this session; see commit `fe273b4`).
- Contact A/B at the engine level (`isaac_player_ab.py`, tables in section 4):
  isolated tile, adjacent tiles, chunk seam, wall, glass wall (Mission 12
  report), a natural flat run, a multi-feature course, an ice run, immersed
  standing and a support-destruction fixture.
- Terrain mutation: `destroy_support` removes three support tiles while the
  player stands on them. CasU keeps the old colliders for ~31 render-frame-delayed
  ticks. Isaac reproduces this with `terrain_collider_delay_ticks=31`: the first
  vertical divergence (>0.05 m) is at action 8 tick 2, i.e. the fall starts within
  about one tick of the source's; with delay 0 it diverges at action 5. Peak vertical error over the free run: 1.5 m
  with delay 31 vs 9.0 m with delay 0. The default is 0 (immediate); the delay
  is a config variable because CasU's value is a frame-timing artifact (30–38
  ticks measured).

## 4. Player motion A/B results

Source repeatability (A/A): `results/source_aa_repeatability_20261013.json`.
Two staged CB1 runs of the same tape: 700/700 (flat) and 720/720 (course) ticks
identical in velocity, grounded flag, crouch and contact count; rigidbody
position identical (4 ticks differ by ≤1.5e-5 m). The interpolated Transform
position (render-frame dependent) differs on 12–16 ticks by up to 0.39 m; Isaac
uses the rigidbody position. The earlier 70/70 natural-flat-ledge A/A is in
[`PLAYER_PHYSICS_AB_REPORT.md`](PLAYER_PHYSICS_AB_REPORT.md).

The controller (`body_model.py`, `player_runtime.py`) is a tick-for-tick port of
the source `Body` (FixedUpdate forces, grounded BoxCasts, step-up, forced crouch,
landing squash, Update ×1/×2 scheduling and held-action reapplication, 12 Updates
per macrostep) validated against source traces in
`validation/controller_replay.py` and `test_controller_replay`.

Engine A/B, current configuration (`results/isaac_player_ab_*_20261013.json`).
*Teacher-forced*: every tick starts from the recorded CasU state; *decision-level*:
every 10-tick action starts from the CasU state; *free-run*: Isaac runs open-loop.
Values are |Isaac − CasU|, mean (max).

| trace | teacher ground dx / dy (m) | decision-level dx / dy (m) | decision ground-only dx | decision landing dx / dy | free-run first |dx|>0.5 m |
|---|---|---|---|---|---|
| flat_master (700 ticks) | 0.002 (0.39) / 0.002 (0.08) | 0.047 (0.59) / 0.017 (0.39) | 0.028 (0.39) | 0.25 (0.59) / 0.07 (0.39) | action 8 |
| course_v1 (730) | 0.004 (0.29) / 0.002 (0.16) | 0.010 (0.32) / 0.036 (1.64) | 0.000 (0.00) | 0.08 (0.32) / 0.28 (1.64) | action 13 |
| ice_run (420) | 0.001 (0.20) / 0.001 (0.07) | 0.040 (0.57) / 0.022 (0.20) | 0.014 (0.25) | — | action 9 |
| immersed_standing (450) | 0.001 (0.02) / 0.0004 (0.03) | 0.006 (0.22) / 0.004 (0.06) | 0.000 | 0.06 (0.22) / 0.03 (0.06) | not reached |

Airborne-only and air-to-ground decisions match to ≤0.02 m (flat_master: 11 + 5
decisions at 0.000 m horizontal). Wall/ceiling decisions: course_v1 0.005 m mean
(54 decisions), flat_master 0.022 m. Slippery (ice) grounded decisions match to
0.014 m mean; the former mid-run seam stall (−10 m/s at x≈63.2) is gone.
Free-run results are reported because they are what training sees, but they are
chaotic by nature; use the decision-level numbers as the fidelity measure.

## 5. Isaac internal parity and reproducibility

`results/isaac_vector_validation_20261013.json` (4 worlds, full terrain), all passed:

- `world_isolation`: world 0's trajectory is unchanged (≤1e-4) when the other
  worlds act differently; worlds do differ from each other.
- `restore_continuation_exact`: snapshot → run → restore → run reproduces every
  world exactly (error 0.0).
- `partial_reset`: resetting one world leaves the others unchanged (≤1e-6) and puts it
  at the spawn.
- `door_state_per_world`: only the world whose player triggered the shuttle door
  has it activated.
- `independent_terrain_edit_and_restore`: a block edit rebuilds only that world's
  chunk colliders; restore returns the exact collider set; other worlds untouched.
- `shared_terrain_edit_rejected`: shared-terrain mode refuses edits (it would
  corrupt every world).
- Informational (not pass/fail): a body tumbling off a lone tile's corner is
  chaotic in PhysX (branch spread 0.28 m depending on other worlds' history).
  Flat-support cases are exact.
- Single-env reset/replay and the PPO resume path are exact (section 8/9).

## 6. Remaining physics differences and their practical significance

| difference | size / frequency | practical significance |
|---|---|---|
| Box2D tile-junction micro-bumps (a ≈1.7 m/s upward kick when the rounded box crosses some tile joints) are not reproduced | 1 per ~200 settled grounded ticks on flat runs; 7 per 253 on the fast ice run | position drifts of 0.05–0.5 m after such a joint; the main reason free-run trajectories diverge within ~1.6–2.6 s of game time |
| Landing response (squash and contact resolution) is approximate | decision-level landing dx 0.08–0.25 m mean, one outlier −1.6 m vertical | landing spots/speeds differ by tenths of a metre; fall-ragdoll (>52 m/s impact) detection uses the same rule |
| Per-limb liquid forces are not modelled (torso body only) | immersed standing matches (0.006 m) but swimming/limb dynamics are untested | do not use for aquatic navigation conclusions |
| Collider refresh after terrain edits is immediate by default | CasU delays 30–38 ticks | set `terrain_collider_delay_ticks=31` for source-like behaviour |
| Crouch collider quantised to 32 levels | ≤ 1/64 of the height change | below the A/B noise |
| Fluid RNG is a seeded PCG64 stream, not Unity's RNG | repeatable, not seed-identical to Unity | statistical, not trajectory, equivalence of fluid evolution |
| Liquid callback order (Body vs LiquidAffect FixedUpdate) not pinned | exposed as `liquid_affect_phase` | small effect in measured immersed traces |
| Corner contacts in PhysX can be chaotic and history-dependent in batched mode | informational spread 0.28 m | avoid relying on bit-reproducibility at corner falls across batch compositions |
| Out of scope | — | climbing/ropes, items, combat, ragdoll recovery, enemies, jump pads (terminates as `ragdoll:jumppad`) |

Terrain edits in the batched env require `terrain_mode="independent"` (≈ N× the
geometry cost, see section 10).

## 7. Researcher-facing interfaces

`casu_grid/workbench/contracts.py` exposes: action heads and timing, spatial and
scalar observation fields with normalisation and masks, reward terms, encoder and
model layers (optional GRU), PPO settings. A research config is one Python file
exporting `ACTION`, `OBSERVATION`, `REWARD`, `MODEL`, `PPO` and optionally
`callbacks()` returning `(action_decoder, observation_encoder, reward_function)`;
checkpoints reject a load whose contracts differ. The action cadence is pinned to
ten 20 ms physics ticks per decision. `engineering_smoke.py` is a temporary,
clearly labelled plumbing example, not an approved Timmy experiment. See
[`workbench/README.md`](workbench/README.md).

## 8. PPO smoke results (engineering-only config)

Run on 2026-10-13 with the temporary `engineering_smoke.py` contract:

| run | command | result |
|---|---|---|
| single env | `--updates 3` | 3 updates, finite losses/grads, checkpoint written |
| 4 worlds | `--updates 3 --num-envs 4` | 3 updates, 384 environment steps |
| resume | `--updates 2 --num-envs 4 --resume …/latest.pt` | continued at 512 → 640 steps (update counter and step count preserved) |
| episodes | `--updates 6 --num-envs 4 --episode-steps 40` | 18 episodes ended: `time_limit` and `ragdoll:hit_ground_hard` counted per world; returns/lengths logged; resets worked |

Metrics rows include policy/value loss, entropy, approx-KL, grad-norm, per-term
rewards, action frequencies, completed episodes/returns/lengths, termination
reasons, and timing. This establishes that the training stack works; it says
nothing about whether Timmy can learn the task.

## 9. Training metrics and replay inspection workflow

A run directory contains `config.json` (config hash, git revision, working-tree
status, environment settings), `environment.json`, `metrics.jsonl`,
`trajectories.jsonl`, `latest.pt`, and `tensorboard/`. TensorBoard shows every
metric. `python -m casu_grid.workbench.replay` re-executes the recorded actions
in a fresh environment (mirroring the trainer's reset rules), writes
`replay/replay.csv` (per step and world: pose, velocity, reward, termination
reason), `replay/paths.png` (x/y paths per world) and `replay/summary.json`.
Verified on the 6-update 4-world run: 192 steps and 18 episode ends replayed with a
maximum pose/velocity deviation of **0.0** from the training log.

## 10. Measured throughput and resource usage

Full canonical terrain, one RTX 3070, `results/isaac_throughput_{shared,independent}_20261013.json`
(30 macrosteps per point; physics step dt = 0.02 s, 10 ticks per decision):

| terrain mode | worlds | physics ticks/s | decisions/s | per-world ticks/s | construct (s) | peak RSS (GiB) |
|---|---|---|---|---|---|---|
| shared | 1 | 875 | 87 | 875 | 6.4 | 2.56 |
| shared | 4 | 1,441 | 144 | 360 | 6.7 | 2.60 |
| shared | 16 | 1,693 | 169 | 106 | 7.7 | 2.76 |
| independent | 1 | 887 | 89 | 887 | 6.4 | 2.56 |
| independent | 4 | 1,087 | 109 | 272 | 24.8 | 5.15 |
| independent | 16 | 1,230 | 123 | 77 | 101 | 15.5 |

- Latency: reset one world 3–4 ms; reset all 16 worlds 34 ms; snapshot 1–15 ms;
  restore 1.5–1.7 s (shared; includes a physics-view reset and contact-history flush),
  5.6 s (4 independent), 22.6 s (16 independent).
- Where time goes (shared, 16 worlds): controller and fluid pre-phase 35%,
  controller post-phase and collider shape switching 47%, PhysX step 11%,
  tensor read/write 6%. The Python controller, not PhysX, is the bottleneck at
  ≥4 worlds; PhysX scales well. A short profile of the batched step shows no single
  dominant hot spot (fluid sweep ~27%, crouch-level toggles ~17% with random crouch
  actions); the remaining headroom is micro-optimisation or multi-process scaling,
  not attempted here.
- PPO overhead: in the smoke runs the rollout (simulation) took 0.34 s vs 0.007 s
  of optimisation for a single env and 0.75 s vs 0.03 s for 4 worlds; policy
  inference is inside the rollout time. These tiny models are simulation-bound;
  larger user models change the balance.
- A training budget of 1M decisions at ~170 decisions/s is about 1.6 hours.

## 11. Tests, failures and unresolved limitations

- Regression suite: **85 tests passing** (terrain/fluid reference, full-world
  fixture, controller replay, runtime scheduling, shuttle doors, world objects,
  workbench/PPO including checkpoint compatibility, termination counting).
- Failures found and fixed this session: source-harness probe hang (Vector3
  serialisation), orphan Unity processes (owned-tree cleanup with bounded SIGKILL
  escalation), sampling-alignment artefact in the A/B, PhysX seam braking, a
  stale shape-name reference after the corner-shape option, termination counts that
  included unfinished worlds.
- Unresolved/limits: everything in section 6; a Kit process supports one
  `SimulationContext` (validation phases and benchmark run as subprocesses); the
  Isaac build prints a benign "external wrench frame" warning; the engineering-smoke
  objective is temporary; source A/A for the ice/immersed/destroy traces was
  not repeated (flat and course were).

## 12. Commits and repository state

Branch `isaac-lab` (pushed to `origin`). Focused commits in order: importer and
grid (`3e21655`…), collision geometry (`73e078c`), navigation env (`6746061`),
workbench (`f0a9514`), reports (`8168c04`), raw-Sdf colliders (`ffca6e2`), object
colliders (`c23b821`), harness sync (`eaa56a3`), Body port and replay (`4d4f6a5`),
collision events (`c19cf8b`), env on Body model (`4c096a8`), A/B tool (`7b3bacb`),
EnvSlot (`a984732`), batched env (`a375134`), seam fix and scenarios (`fe273b4`),
results (`878c4d2`), workbench tools (`0450483`) and the final documentation
commit. The pre-existing `PPO_Python/matched_pairs*` deletions were left
unstaged. Experiment output (`experiments/`) is git-ignored. The installed game
and historical RL artifacts were not modified. The vendored harness copy
(`source_harness/`) mirrors uncommitted edits in the historical checkout and is
re-syncable with `tools/sync_source_harness.sh`.

## 13. How to configure Timmy, launch training, and inspect results

1. **Copy the template.** `cp casu_grid/workbench/engineering_smoke.py casu_grid/workbench/my_experiment.py`
   (it is a temporary plumbing example — not a recommended design).
2. **Edit the research choices in that file** (no simulator changes needed):
   - `ACTION`: heads (names, size, categorical/continuous) and the mapping to
     controls (`move_x`, `jump`, `crouch`…).
   - `OBSERVATION`: scalar/spatial fields, their shapes, normalisation and masks.
   - `REWARD`: terms and weights; the reward function lives in `callbacks()`.
   - `MODEL`: encoder/fusion layer sizes, `recurrent=True` for a GRU.
   - `PPO`: learning rate, rollout length, minibatch, epochs, KL stop, etc.
3. **Launch** (headless, canonical full terrain):
   `"$ISAAC_SIM/python.sh" -m casu_grid.workbench.train --config casu_grid/workbench/my_experiment.py --num-envs 8 --updates 500 --run-dir experiments/my-run`
   Add `--episode-steps N` to shorten episodes, `--terrain-mode independent` only
   if your task edits terrain, `--resume experiments/my-run/latest.pt` to continue.
4. **Monitor**: `"$ISAAC_SIM/python.sh" -m tensorboard.main --logdir experiments/my-run/tensorboard`.
   Watch `reward_term/*`, `termination/*`, `mean_completed_*`, `entropy`,
   `approx_kl`, `*_decisions_per_second`.
5. **Inspect behaviour**: `"$ISAAC_SIM/python.sh" -m casu_grid.workbench.replay --config casu_grid/workbench/my_experiment.py --run-dir experiments/my-run`
   then open `experiments/my-run/replay/paths.png` and `replay.csv`.
6. **Reproduce**: `config.json` in the run folder holds the config hash, git
   revision and environment settings; checkpoints refuse to load under a changed
   contract.

Fidelity caveats to keep in mind when reading results: section 6.
