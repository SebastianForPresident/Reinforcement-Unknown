# Player physics A/B evidence (2026-10-13)

## Source repeatability

Two isolated CB1 runs used seed `391728405`, the same staged game/plugin, the
same standing-temperature initialization, and the same action tape at canonical
grid position `[483,1010]`. Terrain was unchanged. That cell lies on the
canonical 22-cell horizontal run `[473,494]` in row 1010; the player moved only
across block ID 11 during this tape. The tape was neutral, right, left, neutral,
right, left, neutral, with each policy action held for ten 20 ms physics ticks.

The first 10 tick samples from each of the seven action sequences matched
exactly across both CasU runs for position, velocity, grounded state, movement
and jump controller fields, collider configuration, ground BoxCast, contact
count and contact records: **70/70 identical samples**. Absolute fixed index,
frame and fixed-time counters differed and were excluded. The smoke harness
keeps recording fixed updates while it waits for shutdown after the final
command; those post-horizon rows were excluded from the A/A comparison.

The raw source traces, smoke records and comparison summary are:

- `results/casu_player_natural_flat_ledge_ticks_a_20261012.jsonl`
- `results/casu_player_natural_flat_ledge_ticks_b_20261012.jsonl`
- `results/casu_player_natural_flat_ledge_aa_20261012.json`
- `results/casu_player_natural_flat_ledge_smoke_a_20261012.json`
- `results/casu_player_natural_flat_ledge_smoke_b_20261012.json`

A third isolated run was completed on 2026-10-13 with the same staged binary,
temperature-cache fixture, player relocation, and tape. The original two runs
and this third run match exactly across the same **70/70** aligned action
boundary samples for all fields above; see
`results/casu_player_natural_flat_ledge_aaa_20261012.json` and the third raw
trace `results/casu_player_natural_flat_ledge_ticks_c_20261012.jsonl`. This
confirms deterministic source repeatability for the chosen natural flat-run
fixture, not general collision or motion parity.

## CasU-versus-Isaac controller replay

Isaac replayed the same 70 source fixed-tick boundaries, canonical terrain
chunk, start pose, measured 49.28 kg mass, 1.6 × 5.0 box dimensions, 0.02 s
timestep, 4,000 N movement force, and action tape. Isaac used zero contact
friction because the source tile material coefficients are not yet measured.
Controller commands were compared at the same pre-physics boundary.

With the unexpanded cuboid proxy, the maximum absolute errors were 0.0873 m in
horizontal position, 0.1618 m in vertical position, 0.4390 m/s in horizontal
velocity and 2.20 m/s in vertical velocity. Grounded-state proxy matched on
all 70 samples. The vertical discrepancy was concentrated at first contact:
Unity's `edgeRadius=0.1` produces an expanded, rounded box, while the baseline
Isaac cuboid had the unexpanded nominal box dimensions. Isaac took additional
ticks to settle onto the floor.

Unity documents `BoxCollider2D.edgeRadius` as extending around the box and
making a larger rounded 2D collider ([Unity API reference](https://docs.unity3d.com/es/current/ScriptReference/BoxCollider2D-edgeRadius.html)). An **experimental** expanded rounded
convex proxy was therefore tested separately. Its errors were 0.0873 m
horizontal position, 0.0618 m vertical position, 0.4390 m/s horizontal
velocity and 1.708 m/s vertical velocity; mean vertical position error fell
from 0.1172 m to 0.0202 m. PhysX emitted a triangle-mesh-to-convexHull fallback
warning. This probe has not yet passed environment snapshot/restore parity, so
the proxy remains experimental and is not the default.

Results:

- `results/player_motion_ab_natural_flat_ledge_20261012.json`
- `results/player_motion_ab_natural_flat_ledge_rounded_edge_20261012.json`

The third source repeatability run was also replayed in Isaac with both the
nominal cuboid and the optional expanded rounded approximation. The 70-sample
max errors matched the earlier comparisons exactly: nominal max errors
0.0873 m horizontal position, 0.1618 m vertical position, 0.4390 m/s
horizontal velocity and 2.20 m/s vertical velocity; expanded rounded max
errors 0.0873 m, 0.0618 m, 0.4390 m/s and 1.708 m/s. Results are
`results/player_motion_ab_natural_flat_ledge_c_nominal_20261012.json` and
`results/player_motion_ab_natural_flat_ledge_c_20261012.json`. PhysX logged the
same triangle-mesh-to-convexHull fallback for the rounded proxy.

These results establish a repeatable source baseline and close horizontal
controller agreement in one canonical standing-motion fixture. They do not
establish full navigation fidelity. The chosen tape does not validate jump,
fall/landing, wall/seam contact, slippery ground, fluids or tile deletion.
Source contact normals and material response remain incomplete; most per-tick
source contact buffers were empty even when the source grounded flag and
ground BoxCast showed support.

## Terrain movement and mutation observations

Mission 12 added controlled solid-contact cases on the canonical layer. A
validated isolated ID 14 tile at `[7,372]`, adjacent ID 1 support at
`[300,100]`, and support crossing chunk boundary `x=64` each produced stable
grounding and upward Unity contacts. A rightward macrostep from the seam site
ended in observed grid cell `x=64`; Isaac stayed grounded through that crossing.
The paired nominal-cuboid replay's maximum position errors were 0.019 m
horizontally and 0.137 m vertically, with 0.239 m/s horizontal and 2.20 m/s
vertical velocity error. Static isolated/adjacent/seam probes had zero
horizontal position error and roughly 0.1155 m maximum vertical position error.

A wall fixture at `[212,100]` captured 16 contacts at rest, including upward
floor and left-facing wall normals. With two rightward actions, CasU first
resolved the wall on sample 6. The nominal Isaac proxy resolved it one sample
later; maximum errors reached 0.116 m horizontally and 9.74 m/s horizontally.
The optional expanded rounded proxy reduced settled wall-position error from
0.116 m to about 0.015 m, but still showed the first-contact sample delay and
logged a triangle-mesh-to-convexHull fallback. A two-cell Glass ID 7 wall at
`[921,430:432]` also stopped CasU and produced left-facing wall contacts. Glass
is confirmed solid in this canonical world even though no exposed Glass floor
with full standing clearance was found.

The paired case matrix and specific trace names are recorded in
[`MISSION12_COLLISION_REPORT.md`](MISSION12_COLLISION_REPORT.md). These are
single source runs per fixture (apart from a repeated isolated-tile run), so
they establish concrete contact behavior but not per-case run-to-run variance.

### Immersed-body trace

At `[958,178]`, the captured water type is 1 for the standing torso and all 15
limb sample points. `inWater` becomes true after the first fixed tick. The torso
remains held by its support tile while limb vertical velocities rise to about
4.16 m/s over the ten-tick action. Source `LiquidAffect` applies velocity drag
and buoyancy per rigidbody; Isaac's current navigation proxy represents only a
single rigid body. Its paired one-action comparison remains close in horizontal
pose but has the familiar approximately 0.115 m vertical contact offset and
2.20 m/s initial vertical velocity difference. It does not validate the
source's 16-body immersed response or callback ordering.

The current staged harness can intermittently acknowledge `CF_STEP` without
advancing to a new observation. Failed attempts were excluded. Successful
single-action tests and the two-action wall test used the same isolated runtime
and left the installed game untouched. A source `flat_soil` rewrite still
hangs during synchronous chunk regeneration, so collider-refresh timing after
terrain destruction is unresolved.

## Canonical collision metadata cross-check

The seed-specific Unity detail export sampled all 17 terrain IDs, including
glass (ID 7). Every nonzero tile type reports `Tile.ColliderType.Grid`, and its
layer-6 overlap sample is a complete 1-by-1 cell footprint. Glass has 677
occupied cells in the canonical fixture; it is therefore solid terrain for
navigation collision in this world. The source audit found no terrain-specific
`Physics2D.IgnoreCollision` pairs; the only direct pair calls concern limb
colliders. Layer 6 is not ignored by the captured global layer matrix. The
world uses 256 per-chunk TilemapCollider2D components with
`usedByComposite=false`; sibling composite components have no effective paths
in the captured configuration. This supports the exact occupancy footprint
used by Isaac geometry, but does not identify Unity contact-material response
or mutation-to-physics refresh timing.

The 2026-10-12 mission-12 diagnostic source attempt once timed out after
observation 1. A rebuilt staged assembly added explicit queued-action and
latch-attempt traces plus persisted the action-sequence ack in the client
artifact. A retry passed end-to-end (sequence 1, observation 2 after ten
source ticks); a later natural-flat seven-action tape also passed all eight
observations with source sequences 1–7. Therefore the transport failure was
intermittent, not a persistent protocol blocker. The terrain-mutating
`flat_soil` fixture did independently hang before publishing its fixture record
or advancing physics, consistent with its synchronous `UpdateChunk` /
`ChunkUpdated` collider rebuild being an unsafe operation at the policy
boundary. No terrain was changed in the game outside the disposable staged
process. These failed flat-soil rows are idle and excluded from A/B evidence.

The newly wired Isaac crouch smoke succeeded in the diagnostic spawn-chunk
scene: standing and half-height collision shapes switch at the controller's
source-derived crouch threshold, then restore; reset and snapshot-branch
replay errors were both zero. The current PhysX representation switches a
full-height and pre-authored half-height cuboid, while the source smoothly
interpolates collider scale and vertical offset. That is a known crouch
transition residual; it is not claimed as exact collider animation parity.
The optional edge-radius expansion also remains an approximation because the
experimental rounded convex mesh has not been validated across full-world
contact and restore behavior.

A separate replay started at `[538,990]` on irregular natural terrain. CasU
rose into nearby terrain and continued moving while Isaac's chunk cuboids
stopped it; maximum horizontal position error reached 5.89 m and horizontal
velocity error 25.53 m/s. This is recorded as a terrain-geometry discrepancy,
not as a controller comparison (`results/player_motion_ab_clear_ledge_20261012.json`).

An opt-in staged-only `flat_soil` mutation hook also exercised a batched row
and air-clear followed by `UpdateChunk` and `ChunkUpdated` for the affected
source chunks. Its trace showed intermittent post-refresh contacts and
vertical impulses, so it is not accepted as a flat-floor baseline. The
diagnostic file is
`/var/tmp/reinforcement-unknown-cb1-runtime-mission6-20261012/user-home/diagnostics/terrain-fixture-cb1-smoke-1-1-1791845277.json`.
The changes existed only in the staged Unity process and were discarded when
that process exited. Exact source mutation-to-collider physics timing remains
unresolved.
