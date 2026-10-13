"""High-speed landing / tunnelling check: drop the player from ``--height`` units onto the flat fixture."""
import argparse, json, sys
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--height", type=float, required=True)
p.add_argument("--player-collider", default="compound")
p.add_argument("--fixture", type=int, nargs=2, default=[68, 211], help="support cell with a tall free shaft above")
p.add_argument("--output", type=Path, required=True)
args, _ = p.parse_known_args()
sys.stdout.reconfigure(line_buffering=True)
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from casu_grid.isaac_navigation_env import IsaacLayer1EnvConfig
from casu_grid.isaac_vector_env import BatchedIsaacLayer1Env
env = BatchedIsaacLayer1Env(IsaacLayer1EnvConfig(num_envs=1, spawn_mode="fixture", fixture_grid_xy=tuple(args.fixture),
                                                 num_workers=1, player_collider=args.player_collider), terrain_mode="shared")
env.reset()
env.slots[0].model.s.y += args.height
env._write_slot_states([0])
frames = []
for k in range(14):
    env.step({"move_x": np.array([1]), "jump": np.array([0]), "crouch": np.array([0])})
    s = env._body_state(0)
    frames.append({**s, "ragdoll": env.slots[0].model.s.ragdoll_event})
floor = frames[0]['y'] - args.height if False else env.slots[0].spawn_world_xy()[1]
args.output.write_text(json.dumps({"height": args.height, "collider": args.player_collider, "frames": frames,
                                   "final_y": frames[-1]["y"], "tunnelled": frames[-1]["y"] < floor - 0.5}, indent=1))
print("FALL", args.height, args.player_collider, "final y", round(frames[-1]["y"], 3), "tunnelled", frames[-1]["y"] < floor - 0.5, flush=True)
import os; os._exit(0)
