"""Load, tick, and restore the canonical grid alongside headless Isaac Kit."""
import argparse
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
launcher = AppLauncher(args)
app = launcher.app

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from casu_grid.reference import load_layer1_reference

try:
    reference = load_layer1_reference()
    state = reference.new_state(decisions=[False, True] * 10000)
    assert state.blocks.shape == (1024, 1024)
    assert state.fluids.shape == (1024, 1024)
    before = state.snapshot()
    for _ in range(8):
        app.update()
        state.simulation_step()
    advanced = state.snapshot()
    state.restore(before)
    for _ in range(8):
        app.update()
        state.simulation_step()
    assert state.snapshots_equal(advanced, state.snapshot())
    print("ISAAC_CANONICAL_LAYER1_SMOKE_OK",
          reference.manifest["content_id"], state.tick, state.sim_index,
          flush=True)
finally:
    app.close()
