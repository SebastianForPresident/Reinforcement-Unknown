# Isaac Lab Layer 1 engineering acceptance report

Updated 2026-10-13. This report distinguishes verified engineering from
remaining CasU fidelity work; the implementation is not represented as a
near-identical, fully source-validated CasU replica.

## Run commands

Use Isaac Sim 4.5.0's bundled Python, not the host interpreter. From the project
root, initialize the environment exactly as in [`../ISAAC_LAB_SETUP.md`](../ISAAC_LAB_SETUP.md),
then run an editable research config:

```bash
ISAAC_ROOT="$HOME/Applications/IsaacLab-2.1.1-IsaacSim-4.5.0"
ISAAC_SIM="$ISAAC_ROOT/isaac-sim-4.5.0"
"$ISAAC_SIM/python.sh" -m casu_grid.workbench.train \
  --config casu_grid/workbench/engineering_smoke.py \
  --updates 100 --num-envs 1 \
  --geometry-scope full --run-dir experiments/timmy-run
```

Replace the engineering placeholder with a user-authored/approved contract
before interpreting results as Timmy research. Resume compatible runs with
`--resume experiments/timmy-run/latest.pt`. View scalars with:

```bash
"$ISAAC_SIM/python.sh" -m tensorboard.main \
  --logdir experiments/timmy-run/tensorboard
```

Inspect `config.json`, `environment.json`, `metrics.jsonl`,
`trajectories.jsonl`, TensorBoard events, and `latest.pt` under the run
folder. Checkpoints fingerprint the action, observation, model and PPO
contracts and include optimizer/RNG/observation/episode/simulator snapshot
state. The config source file SHA-256 is stored in the run manifest.

## Canonical world and collision

- Fixture: `casu_grid/fixtures/casu-layer1-seed-391728405-v1/`
- Content ID: `sha256:0f1413f47d255bbab25fbc43aaa433efdbbb8c15a260ec48268cd02429495ba8`
- Full arrays: 1024×1024, x-first, 320,647 occupied terrain cells and 256
  logical 64×64 chunks. The exact source arrays and canonical checksums are
  verified by the importer tests.
- Captured Unity metadata reports 256 enabled layer-6 TilemapCollider2D
  components with `usedByComposite=false`; sibling CompositeCollider2D
  components have no effective paths in the export. All 16 occupied tile IDs
  including Glass ID 7 report Grid collider type with complete unit-cell
  overlap samples. Source audit found no terrain-specific IgnoreCollision
  pairs. Friction/material response and actual contact parity remain
  unmeasured.
- Isaac geometry is one merged-rectangle cuboid per exact occupied raster
  rectangle, chunk-local and unmerged across source chunk boundaries. Latest
  full construction: 39,746 static cuboid colliders, 209.78 s geometry build,
  3.24 GiB peak process RSS (3,400,292 KiB). Canonical full geometry supports
  block edit and collider list restoration. PhysX contact refresh after
  mutation has not been proven; a nonfatal Fabric `removePath called on
  non-existent path` warning remains.

## Movement evidence

The source A/A trace at the canonical flat 22-cell run matched across repeated
staged CasU executions on all 70 selected fixed-tick samples. Mission 12 added
controlled source/Isaac probes for an isolated tile, adjacent tiles, a static
chunk seam, movement across the seam, a vertical wall, a Glass wall and one
immersed standing pose. See [`MISSION12_COLLISION_REPORT.md`](MISSION12_COLLISION_REPORT.md)
for fixture validation, raw artifact names and per-case error measurements.
A same-tape
CasU/Isaac comparison using the nominal cuboid reported max errors of 0.0873 m
horizontal position, 0.1618 m vertical position, 0.439 m/s horizontal velocity,
and 2.20 m/s vertical velocity. An experimental edge-expanded rounded proxy
reduced mean vertical position error from 0.1172 m to 0.0202 m; its max vertical
position/velocity errors were 0.0618 m/1.708 m/s. It is not the default because
its triangle-mesh convex fallback and branch restore have not been cleared.

The full standing controller exposes measured source parameters and includes
ground/air force, horizontal cap and reversal, jump edges/release gravity,
coyote timer, slippery force, and liquid drag/buoyancy. The optional rounded
body dimension approximation can be enabled in config. Crouch uses a
pre-authored half-height PhysX cuboid selected when the source controller's
smoothed height crosses the standing threshold; the transition is not a
continuous PhysX collider interpolation. The refreshed full-world smoke
passed shape switch, reset and exact branch replay.

The new short stationary/moving collision probes match grounded flags but show
about 0.115 m vertical resting offset from the nominal cuboid. Wall response is
one sampled tick late in Isaac; the first-contact horizontal velocity residual
reaches 9.74 m/s. The experimental edge-expanded proxy reduces settled wall
offset but retains a one-sample first-contact delay and emits a PhysX convex
fallback warning. The immersed source trace records separate responses from
the torso and 15 limbs; the Isaac one-body proxy does not model those independent
limb impulses. Free-fall/landing, jump/air steering, slippery terrain and
post-destruction source-vs-Isaac fixtures remain unvalidated. Irregular terrain
replay showed a large collision approximation discrepancy (up to 5.89 m
horizontal position and 25.53 m/s horizontal velocity error). Do not treat
nominal flat-platform agreement as general navigation fidelity.

## Lifecycle, batching and PPO

Latest full smoke: `casu_grid/results/isaac_navigation_final_full_smoke_20261012.json`.
Canonical reset and checkpoint branch replay max state errors were 0.0. It
exercised full terrain construction, mutation and collider-list restoration,
standing/crouched collider switch, two PPO updates, checkpoint restore and
continued training, reward terms, JSONL trajectories and TensorBoard events.
This is a temporary engineering-only progress-to-x objective, not an approved
Timmy contract and not evidence of learning quality.

A full-geometry two-world batched run is recorded at
`casu_grid/results/isaac_vector_smoke.json`: two canonical worlds, 256 triangle
meshes/320,647 cells each, 2.106 GiB peak RSS, replay error 0, partial-reset
neighbor movement within 1 mm / 0.01 m/s bounds. Action-phase rates in that
short probe were 506 aggregate physics ticks/s, 50.6 aggregate decisions/s,
253 ticks/s per environment. A full-world two-env PPO run and checkpoint
resume also passed (recorded in experiments/engineering-vector-full-20261012
and experiments/engineering-vector-resume-full-20261012). Full geometry mode
has both cuboid (navigation default) and triangle-mesh (vector smoke) paths;
contact parity of the two representations has not been established.

Latest one-world full-terrain PPO update rates were 450–503 physics ticks/s
and 43–47 policy decisions/s. Per-update rollout collection took 0.253–0.278 s;
optimization took 0.004–0.020 s. Peak RSS was 3.24 GiB. Two updates and
checkpoint continuation establish interface operation only.

## Research interfaces and verification

`casu_grid/workbench/contracts.py` exposes action heads/timing, spatial and
scalar observation fields with normalization/masks, reward terms, encoder and
model layer sizes/recurrence, and PPO settings. `workbench/model.py` builds
configurable spatial/scalar/fusion, actor/value, and optional GRU modules.
`workbench/ppo.py` validates shapes/bounds, collects rollouts, computes GAE,
trains PPO, records reward/action/episode/termination/timing metrics and
trajectories, and saves compatible resumable checkpoints. See
[`workbench/README.md`](workbench/README.md).

Regression command:

```bash
"$ISAAC_SIM/python.sh" -m unittest discover -s casu_grid/tests -v
```

Latest result: **67 tests passed**. The full canonical physics/PPO smoke and
diagnostic crouch-shape smoke passed. Mission 12's staged source harness
intermittently failed to resume fixed ticks after acknowledging `CF_STEP`, but
the corrected isolated, adjacent, seam, wall, Glass and immersed fixtures all
produced successful source observations and paired Isaac replays. Failed
timeouts were excluded; see the mission report. The staged launcher stopped
all failed runs through its exact-owned process cleanup. The source terrain
rewrite fixture still hangs during synchronous chunk collider rebuilding.

## Unresolved acceptance items

1. Repair/bypass the intermittent staged source protocol callback stall.
2. Measure Unity collider rebuild-to-Physics2D effective timing and material
   coefficients; verify that Isaac live terrain mutation changes actual PhysX
   contacts, not only the USD hierarchy.
3. Add source/Isaac free-fall, landing, jump, air-steering and fluid-response
   comparisons. The current immersed A/B omits source limb impulses.
4. Characterize continuous crouch collider interpolation, liquid callback
   order/camera scheduling, and residual contact differences that change
   navigation decisions. No canonical slippery ice tile exists in this seed.
5. Rerun/review user-facing CLI after the latest metadata/fail-fast changes.
6. Final repository hygiene and focused commits remain outstanding. Historical
   `PPO_Python/matched_pairs*` deletions were already present and are user-owned;
   preserve them. Do not touch the canonical installed game.

Therefore the project has an operational, configurable engineering lab with
canonical full terrain, batching, lifecycle and PPO plumbing, but the full
mandate's near-identical CasU A/B and verified mutation-contact acceptance
criteria are not yet met.
