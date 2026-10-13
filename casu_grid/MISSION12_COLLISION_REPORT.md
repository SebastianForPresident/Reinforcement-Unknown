# Mission 12 — Canonical contact and fluid probes

Updated 2026-10-13. Probes used the staged, disposable CB1 runtime and the
frozen seed `391728405` arrays. They did not mutate the historical Unity data
or installed game. Unity and Isaac samples were aligned at the pre-physics
controller boundary; Unity also recorded the immediately post-Physics2D state.

## Validated CasU fixtures

Array checks validated the support cell and the complete standing-body
clearance volume before each relocation. This avoids confusing a geometrically
nearby tile with an accessible fixture.

| Case | Canonical fixture | CasU result |
|---|---|---|
| Isolated occupied tile | ID 14 at `(7,372)` | Standing, two upward contacts, stable pose |
| Adjacent occupied tiles | IDs 1 at `(300,100)` and `(301,100)` | Standing, six upward contacts, stable pose |
| Chunk seam | ID 1 support across `x=64`, player fixture `(63,376)` | Standing, six upward contacts, stable pose |
| Moving chunk seam | Same seam fixture, one right action | Player observation ends in grid cell `x=64`; remains grounded with four upward contacts |
| Vertical wall | Support `(212,100)`, wall at `x=214` | Six-tick approach then stopped; post-physics contacts include upward and left-facing normals |
| Glass wall | ID 7 wall at `x=921`, support `(919,429)` | Stops at wall; contacts include upward and left-facing normals. Glass is solid in the canonical build |
| Immersed standing body | Support `(958,178)`, liquid type 1 | Body and all 15 limb samples query type 1; `in_water` becomes true after the first tick |

Raw source records are in `results/casu_collision_*_20261012.*` and
`results/casu_fluid_body_fixture_*_20261013.*`. The source trace now includes
read-only per-tick liquid type, buoyancy, drag, position, and velocity for the
torso and limbs at both measured boundaries. The staged harness rebuild had
zero warnings and errors; its plugin SHA-256 is
`28d510a35c8e8fc1a49986a9a6ad1beae1aaa017a09c77d3c5e8d89b9a480e39`.

Two isolated runs on the single-tile fixture matched exactly across their ten
aligned pre-physics samples for the fields present in both traces. They used
different diagnostic plugin versions, so the second run's new liquid sample
fields were not included in this repeatability comparison. See the raw
`casu_collision_isolated_tile_ticks*` files.

## Isaac paired replays

These use the existing zero-friction, nominal cuboid player proxy over the
canonical source chunk, with the same measured action tape and `0.02 s` step.
The measured maximum absolute errors are:

| Case | Samples | X position (m) | Y position (m) | X velocity (m/s) | Y velocity (m/s) | Grounded mismatches |
|---|---:|---:|---:|---:|---:|---:|
| Isolated tile, neutral | 10 | 0.0000 | 0.1155 | 0.0004 | 2.200 | 0 |
| Adjacent tiles, neutral | 10 | 0.0000 | 0.1155 | 0.0000 | 2.200 | 0 |
| Chunk seam, neutral | 10 | 0.0000 | 0.1155 | 0.0003 | 2.200 | 0 |
| Moving across chunk seam | 10 | 0.0192 | 0.1366 | 0.2390 | 2.200 | 0 |
| Wall approach, two right actions | 20 | 0.1164 | 0.1155 | 9.740 | 0.0535 | 0 |
| Glass-wall approach, one right action | 10 | 0.1155 | 0.1156 | 9.740 | 0.0010 | 0 |
| Immersed standing, neutral | 10 | 0.0000 | 0.1154 | 0.0003 | 2.200 | 0 |

Per-case numerical traces are `results/player_motion_ab_{isolated_tile,
adjacent_tiles,chunk_seam,chunk_seam_cross,wall,glass_wall,
fluid_body_fixture}_*.json`. The steady horizontal pose and grounded state match
in these short samples. The roughly 0.115 m resting-height offset is consistent
with omitting Unity's measured `BoxCollider2D.edgeRadius=0.1` in the default
Isaac cuboid. At wall contact, Isaac resolves the side collision one sampled
tick later and leaves an approximately 0.116 m horizontal offset. The optional
expanded rounded proxy reduced the settled horizontal offset to about 0.015 m,
but it still resolved the first contact on the following sample and PhysX
reported a mesh-to-convex-hull fallback. It remains an approximation rather
than an accepted default.

The fluid A/B script does not apply Unity's per-body `LiquidAffect` sampling;
its one-rigid-body proxy was intentionally a controller/collision comparison.
The source-only immersed trace shows the torso remains supported while limb
vertical velocities rise to approximately 4.16 m/s over the first ten ticks
under water type 1. The current Isaac environment applies one source-coefficient
impulse to its single rigid body per tick, so it cannot reproduce these
15 independent limb responses. Immersion and limb-mediated standing behavior
remain a meaningful navigation-fidelity gap.

## Harness and remaining cases

Several source attempts timed out after `CF_STEP` was acknowledged but before a
new observation arrived. Successful runs above used one-step source probes,
except for the two-step wall tape. Diagnostics confirm this is intermittent
fixed-update/protocol resumption, not an invalid fixture result; timed-out runs
were excluded. The source terrain-rewrite fixture still hangs inside synchronous
chunk collision rebuilding, so mutation-to-Physics2D refresh timing remains
unmeasured.

Canonical Glass overhead clearance was not found, but a Glass wall approach
confirmed physical contact. No slippery IDs 27/28 occur in this captured seed;
that controller modifier therefore has no native canonical test site. Jump,
fall/landing, air steering, diverse fluid types, and repeated-contact A/A
campaigns remain incomplete. These probes expand the evidence but do not
establish a near-identical navigation replica.
