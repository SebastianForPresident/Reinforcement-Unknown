# Autonomous engineering checkpoint

Updated: 2026-10-13

## Objective and state

The goal is to complete `Instructions.txt`: a training-ready Isaac Lab
replica of canonical Casualties Unknown Layer 1, seed 391728405, with source
collision/movement A/B, repeatable lifecycle, configurable RL workbench, real
PPO smoke, monitoring, throughput and final acceptance report. The goal is
paused at the user's request to stop this session because usage is expiring.
The latest user instruction waives the previously requested final Gmail
handoff; no email is required or was sent.

## Latest verified implementation

- Full-world Isaac navigation adapter in `casu_grid/isaac_navigation_env.py`:
  0.02 s timestep, ten physics ticks per action, source-sized 49.28 kg player,
  canonical full terrain box geometry, configurable callbacks, reset/snapshot/
  restore and explicit terrain mutation API.
- Full canonical geometry built and physically exercised after fluid scheduler
  integration: 256 chunks, 320,647 occupied cells, 39,746 static cuboid
  colliders. Latest integrated build measured 209.19 s, 3.26 GiB peak RSS.
- `CasUGridState.simulation_step()` now runs once per physics tick; environment
  passes current player cell to the grid camera scheduler each tick. Player
  liquid impulses use SearchBase `WaterInfo`: IDs 1/5 .6/.915, 2 2/.8, 3
  .5/.75, 4 1/.8, 6 3/.7. Immersion permits jump as source `Body.CanJump`
  requires. Source-vs-Isaac Unity callback ordering remains unmeasured and is
  exposed via `liquid_affect_phase`.
- Fluid random decisions use compact seeded NumPy PCG64 with exact state
  snapshots. This is repeatable, not Unity RNG parity.
- Chunk dirty tracking is snapshotted. `set_block()` rebuilds only changed
  chunk collision; reset and restore compare occupancy and rebuild chunks to
  restore collider state. The full-world USD mutation rollback test passed:
  removing [0,0] changed one chunk from 205 to 206 box colliders; restoring
  returned it to 205. Isaac emitted a nonfatal Fabric `removePath called on
  non-existent path` warning during refresh; post-mutation contact behavior
  still needs validation.
- Workbench includes configurable action/observation/reward/model/PPO
  contracts, actor-critic and PPO; PPO now validates observation shapes and
  action bounds, preserves rollout state and episode statistics across updates,
  saves/restores simulator snapshot when supported, reports action frequencies,
  episode lengths, terminal reasons, simulation/update timing and physics
  throughput, and records trajectories as JSONL. TensorBoard writer added.
- `casu_grid/workbench/train.py` is the researcher launch CLI. It captures
  config, source revision, checkpoints, JSONL metrics, trajectory logs and
  TensorBoard events. One-env config only; it rejects action timing other than
  the established ten ticks.
- Latest portable suite: **61 tests passed** with Isaac bundled Python after
  adding a PPO GAE regression for time-limit bootstrapping. TensorBoard's
  earlier NumPy ABI overlay was repaired.
- Latest full-world PPO/lifecycle smoke:
  `casu_grid/results/isaac_navigation_training_resume_smoke_20261012.json`.
  Status passed. Reset/replay max absolute errors were 0.0. Two actual PPO
  updates; each had optimizer batches and finite losses/gradients; completed
  returns were 17, episode lengths 8, reward terms nonzero; checkpoint restored
  and continued; TensorBoard event files emitted; peak RSS about 3.26 GiB.
- Researcher CLI smoke passed once at diagnostic spawn-chunk scope in
  `experiments/engineering-cli-smoke-retry-20261012/`: config, checkpoint,
  metric row, TensorBoard event and 32 trajectory rows were created. One PPO
  update reported 320 physics ticks at about 436 ticks/s. Relative-import bugs
  in the sample config and error-path masking were fixed.
- The default body remains the deterministic, unexpanded cuboid proxy. The
  source A/B probe on a canonical 22-cell flat run recorded a repeatable
  CasU tape and compared 70 physics boundaries. With the unexpanded proxy,
  maximum horizontal position/velocity errors were 0.0873 m/0.439 m/s; the
  first-contact vertical velocity residual reached 2.20 m/s. An experimental
  expanded rounded proxy reduced max/mean vertical position error from
  0.1618/0.1172 m to 0.0618/0.0202 m, but PhysX logged a triangle-mesh
  fallback warning and the new shape has not passed branch restore testing.
  See `casu_grid/PLAYER_PHYSICS_AB_REPORT.md` and the result JSONs.
- The staged source-game harness has an opt-in `flat_soil` terrain fixture that
  uses source `UpdateChunk` and `ChunkUpdated`. It changed only the disposable
  runtime and exposed intermittent contacts after chunk rewrite. It is not a
  validated flat-floor baseline. The original staged plugin is backed up at
  `/var/tmp/reinforcement-unknown-cb1-runtime-mission6-20261012/game/BepInEx/plugins/PPO_Harness.dll.before-grid-fixture-20261012`;
  the current staged build hash is `b8231a1394a10cef775e4a325469d93cc049a68f9d59b8453db7cc1068c3b109`.
- Earlier full fluid smoke also passed: `isaac_navigation_full_fluid_smoke_20261012.json`.
- TensorBoard CLI verified: `python.sh -m tensorboard.main --version` returns
  2.21.0.
- TensorBoard install briefly overlaid NumPy 2.2.6; this broke OpenCV ABI and
  was caught and repaired by installing NumPy 1.26.4 in the machine-local
  `isaaclab-deps` path. Current bundled imports report NumPy 1.26.4,
  OpenCV 4.9.0, Torch 2.7.0+cu128, TensorBoard 2.21.0.

## Source parity evidence and limits

- Two isolated staged CasU runs of the same seed/action tape matched exactly
  across 70/70 selected physical samples on canonical terrain. Their traces,
  smoke records and A/A report are under `casu_grid/results/` as
  `casu_player_natural_flat_ledge_*_20261012.*`.
- CasU-versus-Isaac replay on that natural flat support run matched horizontal
  controller motion closely but showed an initial-contact vertical velocity
  residual. The expanded rounded collider improved vertical placement in an
  experimental comparison but still needs deterministic restore validation.
- A separate irregular-terrain replay at `[538,990]` exposed a large geometry
  discrepancy (max 5.89 m horizontal position and 25.53 m/s horizontal
  velocity error). It does not isolate controller behavior.
- Paired isolated tile, adjacent tile, wall, chunk seam, Glass, destruction
  refresh, jump/air control, slippery terrain and immersion experiments remain
  incomplete. Exact Unity collider-refresh timing remains unresolved.
- Isaac reset/replay and block-mutation collider-list rollback on canonical
  terrain pass. Player contact behavior after canonical terrain mutation is
  unmeasured.

## Continuation update (2026-10-13)

- Re-read `Instructions.txt` and verified the existing goal remains active.
  The final Gmail requirement is recorded above: only send to
  the user when the goal is completed or paused.
- Re-ran the Isaac bundled-Python regression suite: **67 tests passed** in
  1.526 seconds.
- Added opt-in read-only torso/limb liquid samples to the historical
  `PPO_Harness/Counterfactual/CollisionProbe.cs`. Pre- and post-Physics samples
  include fluid type, source buoyancy/drag, cell, position and velocity.
  `dotnet build PPO_Harness/PPO_Harness.csproj -c Debug --no-restore` succeeded
  with zero warnings/errors. The staged plugin was backed up as
  `PPO_Harness.dll.before-liquid-trace-20261013`; copied only into the isolated
  `/var/tmp/reinforcement-unknown-cb1-runtime-mission12-20261012` runtime.
  Staged plugin SHA-256: `28d510a35c8e8fc1a49986a9a6ad1beae1aaa017a09c77d3c5e8d89b9a480e39`.
- Discarded tentative seam/wall coordinates after raw-array clearance checks.
  Valid source fixtures: isolated ID 14 at `[7,372]`; adjacent ID 1 at
  `[300,100]`; seam support at `[63,376]`; wall support `[212,100]` with wall
  column `x=214`; Glass wall ID 7 at `x=921`, support `[919,429]`; immersed
  standing pose `[958,178]`.
- Successful source observations: isolated, adjacent, static seam, moving seam,
  wall, Glass wall and immersed body. Seam-crossing smoke ended at observed
  cell `x=64`. Post-Physics contacts: isolated two upward; adjacent and static
  seam six upward; wall sixteen including left-facing normal; Glass wall
  fourteen including left-facing normal. Two isolated-tile source runs matched
  exactly across ten aligned common fields.
- Paired Isaac nominal-proxy max errors (X/Y position, VX/VY velocity):
  isolated `(0, .1155 m, .0004, 2.20 m/s)`, adjacent `(0, .1155 m, 0,
  2.20 m/s)`, static seam `(0, .1155 m, .0003, 2.20 m/s)`, moving seam
  `(.0192, .1366 m, .239, 2.20 m/s)`, wall two-step `(.1164, .1155 m, 9.740,
  .0535 m/s)`, Glass wall `(.1155, .1156 m, 9.740, .0010 m/s)`. Grounded flags
  matched in sampled rows. Expanded rounded proxy settles about 0.015 m from
  the source wall pose but still has one-sample first-contact delay and a PhysX
  mesh-to-convexHull fallback warning.
- Source fluid probe reads type 1 at torso and all 15 limbs. `in_water` becomes
  true after the first fixed tick; limbs reach ~4.16 m/s vertical speed after
  ten ticks while the torso remains grounded. The current Isaac player proxy
  is a single rigid body and does not represent separate limb liquid impulses.
  See `casu_grid/MISSION12_COLLISION_REPORT.md` and
  `casu_grid/PLAYER_PHYSICS_AB_REPORT.md` for artifacts and limitations.
- `CF_STEP` resumption remains intermittent: some runs time out after action
  acknowledgement; the listed valid one- and two-action runs pass. The
  `flat_soil` fixture still hangs during synchronous source chunk-collider
  rebuilding. Failed attempts were excluded. Installed game and canonical
  arrays remain untouched.
- Acceptance and A/B reports include these results. **The end-to-end mandate
  remains incomplete.**

- At the user's 2026-10-13 request, work was stopped due to usage limits and the
  goal was paused, not completed. The user explicitly waived Gmail. On resume,
  continue the remaining technical work below; no email is part of the current
  handoff.

## Immediate next steps

1. Diagnose intermittent `CF_STEP` callback resumption and repeat A/A for wall,
   seam, Glass and fluid. `flat_soil` mutation currently hangs during synchronous
   chunk collider rebuilding; safely characterize source collider refresh in
   the staged runtime.
2. Validate actual PhysX contacts after terrain mutation/removal/restore; USD
   child-list equality alone is insufficient. Investigate the Fabric path
   removal warning.
3. Implement and validate immersed A/B with source per-limb `LiquidAffect`,
   characterize callback phase, and improve source camera-follow fidelity.
4. Capture source/Isaac fall/landing, jump/release, coyote, air steering and a
   native slippery surface if available (ice IDs 27/28 are absent in this
   captured seed). Quantify continuous crouch interpolation residual.
5. Re-verify the researcher CLI and rerun full PPO smoke after related
   lifecycle/training edits. The existing two-env path and full-world PPO smoke
   passed, with metrics and TensorBoard artifacts already present.
6. Finish final report, repository hygiene and focused commits. Preserve the
   historical `matched_pairs*` deletions; do not broadly stage or modify the
   installed canonical game.
7. Do not mark the paused goal complete until every required acceptance item in
   `Instructions.txt` is actually satisfied.

## Commands

Set up the bundled Isaac environment as in `ISAAC_LAB_SETUP.md`. Regression:

```bash
ISAAC_ROOT="$HOME/Applications/IsaacLab-2.1.1-IsaacSim-4.5.0"
export LD_LIBRARY_PATH="$ISAAC_ROOT/compat-libs/usr/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ISAAC_ROOT/isaaclab-deps:$ISAAC_ROOT/IsaacLab-2.1.1/source/isaaclab:$ISAAC_ROOT/IsaacLab-2.1.1/source/isaaclab_tasks:$ISAAC_ROOT/IsaacLab-2.1.1/source/isaaclab_assets${PYTHONPATH:+:$PYTHONPATH}"
export OMNI_KIT_ACCEPT_EULA=YES HEADLESS=1 HOME="$ISAAC_ROOT/user-home"
export XDG_CACHE_HOME="$ISAAC_ROOT/cache" XDG_CONFIG_HOME="$ISAAC_ROOT/config"
"$ISAAC_ROOT/isaac-sim-4.5.0/python.sh" -m unittest discover -s casu_grid/tests -v
```

Full smoke (about 4 minutes):

```bash
"$ISAAC_ROOT/isaac-sim-4.5.0/python.sh" casu_grid/isaac_navigation_smoke.py \
  --spawn-mode canonical --geometry-scope full --episode-steps 8 \
  --output casu_grid/results/isaac_navigation_training_resume_smoke_20261012.json
```

Preserve the historical deleted `PPO_Python/matched_pairs*` user state and
all existing untracked artifacts. Do not modify the installed canonical game.
