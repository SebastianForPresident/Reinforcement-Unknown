# CasU grid-state V0

`state.py` is a NumPy-backed, x-first discrete terrain/fluid testbed transcribed
from `SearchBase/WorldGeneration.cs` and `SearchBase/FluidManager.cs`. It keeps
grid transitions separate from Unity rendering/colliders and Isaac physics.
Logical dimensions are configurable, including 1024×1024; the default fixture
is 65×64 to include cells on both sides of x=64.

## Source mapping

- Terrain IDs, health, metallic/slippery/toxicity metadata: `WorldGeneration.GetBlockInfo`.
- Damage list accumulation/cap, bonus, threshold, ice water write and replacement:
  `WorldGeneration.DamageBlock`.
- Clamped block reads, writes and chunk selection: `WorldGeneration.GetBlock`,
  `SetBlock`; chunks are 64 cells (`CHUNKSIZE`).
- Fluid empty rule, range clamps, band scheduler and in-place movement:
  `FluidManager.Empty`, `SimulationRangeIndex`, `SimulationStep`.

The small map keeps the source's world-border clamps. Consequently its camera
must be positioned deliberately: `camera_y=64` exposes the lower portion of
the first 16-row band. The source's seven-band schedule assumes a much taller
world; near this fixture's edges, clamping causes bands to overlap or collapse.
This is preserved and documented rather than normalized.

`DecisionStream` provides caller-supplied decisions for state-changing random
branches. It makes replay exact but does not reproduce Unity's RNG algorithm or
the RNG draws consumed by audio/other loot. Ice's water drop happens only in the
original `!ignoreLoot` loot switch; by default it consumes a stream decision,
or a measured branch result can be supplied as `ice_drop`. The integrated
Isaac adapter supplies an independently seeded replayable PCG64 decision tape,
steps this grid once per 20 ms physics tick, and applies `LiquidAffect` drag and
buoyancy for the selected liquid ID. It is explicit that this RNG stream and
Unity's callback ordering are not yet source-parity validated. See
`isaac_navigation_env.py` and `workbench/README.md`.

## Verification

Use the bundled Isaac Python environment described in `ISAAC_LAB_SETUP.md`
(system Python on this device does not include NumPy). From the repository
root, run the pure reference fixtures with:

```bash
"$ISAAC_SIM/python.sh" -m unittest discover -s casu_grid/tests -v
```

Run the headless Isaac Kit smoke using the locally installed stack as described
in `ISAAC_LAB_SETUP.md` (set its environment variables, then invoke):

```bash
"$ISAAC_SIM/python.sh" "$PWD/casu_grid/isaac_smoke.py" --headless
```

The smoke initializes Kit, advances the grid testbed alongside Kit updates,
and checks its tick count. It does not instantiate a DirectRLEnv or physics
scene because no player/action contract exists in this milestone.

## Full-world validation and benchmark

Run the full-world correctness fixtures with the same bundled interpreter:

```bash
"$ISAAC_SIM/python.sh" -m unittest discover -s casu_grid/tests -v
```

The standalone benchmark does **not** start Isaac Kit or use CUDA for grid
updates. It runs the Python/NumPy core on CPU, measures the source-equivalent
128×16 scan bands and sequential multi-world stepping, and writes raw per-step
measurements to `casu_grid/results/grid_scaling_20261012.json`:

```bash
"$ISAAC_SIM/python.sh" casu_grid/benchmark.py
```

Options `--warmup`, `--samples`, `--multi-samples`, and `--skip-multi` control
run length; see `benchmark.py --help`. Multi-world scaling is isolated in a
fresh process for each count to report independent resident/peak memory.

## Canonical Layer 1 reference

The approved full-world fixture is
`fixtures/casu-layer1-seed-391728405-v1/`. It contains the verified binary
arrays, a versioned canonical manifest, and the captured collision-component
metadata. `reference.py` verifies manifest identity and payload checksums,
then loads arrays in CLR-compatible `[x,y]` order. `new_state()` produces an
independent mutable simulation initialized with seed-world arrays, camera,
fluid scheduler, and empty damage list. The captured `checkTime` value remains
metadata because V0 advances by explicit integer ticks.

Run the complete portable regression suite, including full-array equality
between the importer and fixture payloads, with:

```bash
"$ISAAC_SIM/python.sh" -m unittest discover -s casu_grid/tests -v
```

Run the bounded grid and snapshot-replay smoke under the pinned Isaac Kit
runtime with the environment setup above:

```bash
"$ISAAC_SIM/python.sh" "$PWD/casu_grid/isaac_reference_smoke.py" --headless
```

This proves array loading and discrete stepping inside the Isaac-launched
process. It does not create terrain geometry, rendering, player physics, or
PhysX colliders. See `COLLISION_HANDOFF.md` for the next measurement campaign.

## Mission 11 collision fixtures

The bounded terrain collision prototype is in `collision.py` and
`isaac_collision_smoke.py`. It creates a few cell cuboids and passive probes;
it does not instantiate canonical-world terrain. See
`MISSION11_COLLISION_REPORT.md` for source runtime evidence, fixture geometry,
the Isaac contact trace, and the remaining CasU A/B work.

## Canonical collision geometry prototype

`geometry.py` builds exact occupancy rectangles within the source 64×64 chunk
boundaries. `collision_usd.py` can author full canonical geometry as static
boxes or closed triangle meshes. Reproduce the current full-world standing
contact result with the headless Isaac environment from `ISAAC_LAB_SETUP.md`:

```bash
"$ISAAC_SIM/python.sh" casu_grid/isaac_layer1_collision_smoke.py \
  --geometry boxes --probe-profile player --enable-ccd \
  --terrain-extrusion-depth 2
```

The run constructs 39,746 box colliders for the 320,647 occupied cells across
256 chunks, then drops a 49.28 kg, 1.6×5.0 source-sized probe. On the measured
host the build took about 197 s; the resting root was within 0.0001 units of
the expected support height. The depth of 2 units is an Isaac 3-D extrusion
used to keep the planar body inside the terrain collision volume. It is not a
Unity depth measurement. The raw trace is
`results/layer1_collision_boxes_player.json`. The integrated full-world env
now passes reset, snapshot replay and block-mutation collider rollback. This
validates Isaac's own lifecycle; it does not establish CasU contact parity.
The rounded source tile/body contact, crouch collider, controlled source A/B,
and multi-environment throughput remain unresolved.

## Isaac environment and research workbench

The integrated environment (`isaac_navigation_env.py`, batched
`isaac_vector_env.py`) runs a source-validated standing controller
(`body_model.py`, `player_runtime.py`) on PhysX against the full canonical terrain
and the measured static objects. Validation and measurements are summarised in
[`ACCEPTANCE_REPORT.md`](ACCEPTANCE_REPORT.md) (player A/B in
[`PLAYER_PHYSICS_AB_REPORT.md`](PLAYER_PHYSICS_AB_REPORT.md)); the source-game
instrumentation is vendored in `source_harness/` and the recorded source traces
are under `fixtures/*/source_traces/`.

`workbench/` contains the editable action, observation, reward, model and PPO
contracts, a configurable actor-critic, the resumable PPO trainer, the launch CLI
(`workbench.train`) and a deterministic replay tool (`workbench.replay`). The
contract in `engineering_smoke.py` is a temporary plumbing example only. See
[`workbench/README.md`](workbench/README.md) for commands.

Useful entry points (all run with the Isaac bundled Python):

```bash
"$ISAAC_SIM/python.sh" casu_grid/isaac_player_ab.py --trace flat_master --modes teacher,segment,free --output /tmp/ab.json
"$ISAAC_SIM/python.sh" casu_grid/isaac_vector_smoke.py --num-envs 4 --output casu_grid/results/isaac_vector_validation.json
"$ISAAC_SIM/python.sh" casu_grid/isaac_throughput_benchmark.py --terrain-mode shared --output /tmp/thr.json
```
