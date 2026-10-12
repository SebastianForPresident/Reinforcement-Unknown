# Mission 10 validation record

Canonical fixture: `casu-layer1-seed-391728405-v1`

Content ID: `sha256:0f1413f47d255bbab25fbc43aaa433efdbbb8c15a260ec48268cd02429495ba8`

## Provenance and import

The preserved Mission 9 exports `mission9-seed-391728405-final-a` and
`mission9-seed-391728405-final-b` were each accepted by the historical
`VerifyLayer1Reference.py`. The verifier confirmed both against their own
source snapshots and confirmed full payload byte equality between independent
exports. Both use seed 391728405, the same generation settings, Unity/game/
harness versions and hashes, world, spawn, runtime state, payload hashes, and
collision summary. Their source capture paths differ, as expected for the two
independent captures; both captured state hashes are
`5caad9017268a9745707e0779b1b075014e94b8207507c4a0c62d701f907072d`.

| Payload | Shape / dtype / order | SHA-256 | Full-array result |
| --- | --- | --- | --- |
| Terrain | 1024×1024 / uint16 little-endian / x-first C order | `9c98fb3d9d80b693b7733e6157d8d295f4ab9026e7054b4d86fb839207b4d5c6` | Imported values equal both original exports at all 1,048,576 cells |
| Fluid | 1024×1024 / uint8 / x-first C order | `451f5e991debcb30cc88d3bf0a7e56aaf7b6c01b5555ee9902a98dacae1b6dc7` | Imported values equal both original exports at all 1,048,576 cells |

Historical verifier output: `status=passed`, 192 representative random, edge,
and seam cells; repeated payloads byte-identical; spawn world `(0,508)` maps to
grid `(512,1020)`. The portable project test reopens the fixture payload and
compares every imported cell, while enforcing the recorded hashes.

Coordinate checks confirm the recorded center transform, spawn mapping,
camera grid `(512,1020)`, zero initial fluid `simIndex` and `tileCooldown`,
empty damage list, 64-cell chunk addressing (16×16 chunks), and exclusive
transition `y < -508.9`. `checkTime=0` is retained in the manifest. V0 has no
wall-clock accumulator, so the importer does not synthesize one.

## Simulation and Isaac smoke

The fixture creates separate mutable states with independent arrays. Eight
ticks from the captured initial state were repeated after snapshot restore and
produced identical snapshots. Mutating one instance left the other instance
and canonical payload unchanged. This is replay self-consistency for existing
Isaac grid mechanics; it is not a measurement of CasU-versus-Isaac transition
parity for this seed.

Commands, run from the worktree root using the documented environment in
`ISAAC_LAB_SETUP.md`:

```bash
"$ISAAC_SIM/python.sh" -m unittest discover -s casu_grid/tests -v
"$ISAAC_SIM/python.sh" "$PWD/casu_grid/isaac_reference_smoke.py" --headless
```

Results: 40 tests passed. The headless AppLauncher smoke used Isaac Lab 2.1.1,
Isaac Sim 4.5.0 and `cuda:0` (RTX 3070), printed
`ISAAC_CANONICAL_LAYER1_SMOKE_OK ... 8 16`, restored and replayed all eight
grid ticks, and exited through `app.close()`. It loaded arrays and stepped
Python grid logic inside Kit; it did not build terrain geometry or instantiate
PhysX colliders.

## Remaining fidelity boundary

The manifest carries 256 TilemapCollider2D and 256 CompositeCollider2D
component counts and captured configuration. Per-tile shapes, effective
composite paths, ignore-collision pairs, contact behavior at chunk seams, and
collider refresh after tile destruction remain open. The proposed bounded
measurement set is in `COLLISION_HANDOFF.md`.
