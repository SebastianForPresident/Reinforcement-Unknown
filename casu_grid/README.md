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
