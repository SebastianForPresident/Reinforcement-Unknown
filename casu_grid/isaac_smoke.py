"""Headless Isaac Lab smoke: advance CasU grid state alongside Kit updates."""
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
from casu_grid import CasUGridState

state = CasUGridState(65, 64, camera_x=32, camera_y=64, decisions=[False] * 1000)
state.set_liquid(20, 20, 1)
state.set_block(20, 19, 1)
state.set_block(19, 20, 1)
state.set_block(21, 20, 1)
for _ in range(8):
    app.update()
    state.simulation_step()
assert state.tick == 8
assert state.get_liquid(20, 20) == 1
print("ISAAC_GRID_SMOKE_OK", state.tick, state.sim_index, flush=True)
app.close()
