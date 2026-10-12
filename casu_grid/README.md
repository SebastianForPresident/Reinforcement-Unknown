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
or a measured branch result can be supplied as `ice_drop`. Collision geometry,
rendering, loot, audio, `LiquidAffect`, and `WaterPusher` are outside V0.

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
