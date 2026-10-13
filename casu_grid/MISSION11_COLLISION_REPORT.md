# Mission 11 — terrain collision characterization

## Result

The source collider configuration and a bounded Isaac contact prototype are
captured. The source runtime evidence does **not** yet provide controlled
CasU contact traces, effective outlines for filled tiles, or collider-refresh
timing. Therefore the prototype is a geometry/contact smoke test, not a
CasU–Isaac A/B validation, and the full Mission 11 completion criteria are not
met. The next source-side milestone is to run the controlled fixtures in a
disposable CasU world with contact/query capture enabled.

## CasU evidence

The isolated runtime was launched through the existing guarded CB1 harness.
Detailed collision output is opt-in (`PPO_DIAG_COLLISION_DETAIL=1`) and is
written separately from the original protocol response and state hash. The
instrumented plugin was built successfully and the staged plugin restored to
its baseline hash after the smoke capture. The raw capture is
[`results/mission11_casu_collision_detail_seed391728405.json`](results/mission11_casu_collision_detail_seed391728405.json).
The opt-in hook is implemented in the historical CasU checkout at
`PPO_Harness/Counterfactual/CollisionProbe.cs` and called from
`PPO_Harness/Counterfactual/SemanticSnapshot.cs`. Those files remain in the
already-dirty historical worktree and were not committed as part of the Isaac
worktree changes.

The captured live runtime reports:

- 256 enabled `TilemapCollider2D` components, all on layer 6, non-trigger,
  density 1, zero offset, `usedByComposite=false`, `usedByEffector=false`,
  extrusion factor 0.0005, and `useDelaunayMesh=false`.
- 256 sibling `CompositeCollider2D` components in `Outlines` geometry mode
  with synchronous generation, but zero paths. Since the tilemap colliders
  are not configured to use them, these components do not establish merged
  terrain outlines.
- The observed non-air assets in seed 391728405 use `Tile.ColliderType.Grid`.
  The capture includes representative used block IDs, per-ID world counts,
  and 25×25 interior `OverlapPoint` patterns for nine representative block
  assets. The sampled interior points were all solid. No boundary contour or
  per-tile material was extracted; the default shared PhysicsMaterial2D is
  null.
- Physics2D gravity `[0,-55]`, default contact offset `0.01`, velocity
  iterations 8, position iterations 3, queries-hit-triggers false,
  queries-start-in-colliders true, and auto-sync-transforms false. Layer 6 is
  ignored against layer 10 in the captured layer matrix. The source audit
  found a body-limb ignore call, not a terrain-specific exception.

Relevant source code in `SearchBase/WorldGeneration.cs` creates each chunk
object on layer 6 with a static Rigidbody2D, TilemapCollider2D and
CompositeCollider2D. `SetBlock` clamps and writes `worldBlocks`, calls
`UpdateChunkClosest`, and that calls `UpdateChunk`. `UpdateChunk` rewrites all
4096 cells of the owning 64×64 Tilemap and invokes `ChunkUpdated` immediately
after the rewrite. This establishes source call ordering only. The capture did
not mutate a tile in CasU or observe collider paths/queries/contacts over
subsequent Unity callbacks and physics ticks, so the first physics boundary
where a changed collider is observable remains unknown. The seam refresh path
is likewise not measured.

The existing one-step diagnostic contact capture contains incidental contacts
among generated object rigidbodies. It is not a controlled terrain fixture
and is excluded from A/B conclusions. Unity overlap samples establish solid
interior coverage only, not exact tile outlines, edge normals, solver
response, or deletion behavior.

## Fixture layout and Isaac prototype

[`collision.py`](collision.py) defines nine small fixtures: isolated rock,
adjacent soil, 2×2 concrete, an eight-cell platform, a platform spanning grid
cells 63/64, an eight-cell vertical steel wall, glass, and before/after tile
destruction layouts. Grid coordinates map to cell centers `(x-511.5,
y-511.5)` in a 1024×1024 world. Isaac embeds source `(x,y)` as `(X,Z)` and uses
Y as a 0.2-unit extrusion axis. Static cells are individual 1×0.2×1 cuboids
under grid-indexed 64×64 chunk parents. The Isaac smoke creates only 29 static
cell colliders, not the complete world.

`isaac_collision_smoke.py` drops 1 kg, 0.25×0.2×0.25 dynamic probe boxes on
the isolated Rock, Soil and Glass fixtures. At `dt=1/60`, gravity `(0,0,-55)`,
and 60 ticks, each probe settled at Z `-410.875`, matching the expected top
face of the local unit cell plus probe half-height. Source interior samples
for all three types were solid. The numeric positions and traces are in
[`results/mission11_isaac_collision_trace.json`](results/mission11_isaac_collision_trace.json).
These are PhysX contact-smoke results against the cuboid proxy. They are not
paired with equivalent CasU bodies or source contact samples. The cuboid
thickness, default Isaac material, layer filtering and contact solver remain
provisional because source contact geometry/material measurements are
incomplete. A direct Isaac ray query was not used because it returned no hit
on the smoke geometry; dynamic contacts supplied the working contact check.

## Validation and limits

- Mission 10 canonical regression suite: 40 tests passed before M11 changes.
- New pure fixture tests cover fixture inventory, cell-center conversion,
  64-cell chunk indexing, and the seam's split chunk ownership.
- Final full regression run: all 43 tests passed (the 40 existing tests plus
  three new collision fixture tests).
- Isaac Kit smoke: 29 fixture cuboids built; three probes settled at the
  expected geometric height after 60 ticks.
- Source C# harness build succeeded with zero warnings/errors. A guarded
  one-action runtime smoke passed; the canonical initial state hash matched
  the M9 value. The staged runtime plugin was restored and its original hash
  verified.
- No canonical Layer 1 payload or source CB1 response was changed by the
  collision detail capture.

Remaining required source work is a disposable-world fixture driver that
places the explicit block IDs, records reliable contact normals/points and
per-tick collider/query outcomes, then removes a tile and measures refresh
across Unity callback/physics boundaries. Once those traces exist, rerun the
same layouts and probes in Isaac and compare positions, outlines, seam
contacts and destruction behavior with tolerances derived from repeated CasU
measurements. Until then, no friction/restitution equivalence or solver
tolerance should be claimed.

## Files and next step

- `casu_grid/collision.py` — bounded fixture layouts and coordinate mapping.
- `casu_grid/isaac_collision_smoke.py` — headless PhysX prototype.
- `casu_grid/tests/test_collision.py` — pure layout/coordinate tests.
- `casu_grid/results/mission11_casu_collision_detail_seed391728405.json` — raw
  opt-in source component/tile sampling capture.
- `casu_grid/results/mission11_isaac_collision_trace.json` — raw Isaac probe
  trace.

Recommended next mission: add the smallest opt-in CasU test-world fixture
control and collision-timeline recorder, characterize delete/refresh timing
and chunk seams, then perform controlled CasU–Isaac A/B. Do not build all 256
chunk colliders or add the standing player until that comparison passes.

## Follow-on canonical geometry experiment (2026-10-12)

The initial M11 conclusion above remains valid for CasU-vs-Isaac parity: no
controlled source-game contact trace or deletion-refresh timing was collected.
Later engineering work separately implemented and exercised full canonical
occupancy geometry; that does not retroactively fill the source A/B gap.

`geometry.py` decomposes every occupied cell into same-ID-independent,
chunk-local row rectangles. `collision_usd.py` authored these as 39,746 static
cuboids across all 256 nonempty 64×64 chunks, preserving the exact 320,647-cell
occupancy. The full Kit/PhysX build completed in 196.96 s on the RTX 3070 host.
At its peak, the process used about 2.9 GB RSS. This is a real full-world
construction result; it is not yet a construction/cleanup lifecycle benchmark.

The first player-sized probe attempts used terrain extrusion depth 0.2 and
settled at Z=43.251, 1.249 units below the expected root Z=44.5. Per-tick
telemetry showed contact followed by Y-axis drift toward the edge of the
artificial 3-D terrain slab. Increasing terrain extrusion to 2.0 units, while
keeping the source body dimensions (1.6×5.0), mass (49.28 kg), gravity
(-55 m/s²), and timestep (1/60 s), produced a stable full-world resting pose:
root Z=44.500095, error 0.000095 units after 180 ticks. The complete trace is
`results/layer1_collision_boxes_player.json`. This establishes the current
Isaac static contact proxy and its extrusion requirement, but it does not
establish Unity-equivalent contact normals, movement, seam contacts, or tile
mutation timing. The 2-unit slab is an Isaac embedding choice for 2-D contact,
not a measured Unity collider depth.

After geometry and collision tests were added, the current reference suite has
47 passing tests. The next source-side work remains controlled CasU fixture
instrumentation and paired traces. The Python source standing-controller
translation is in progress and is not yet validated by movement A/B.
