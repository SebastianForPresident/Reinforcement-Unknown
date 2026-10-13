"""Compare Isaac contact for the M8B source body dimensions and small probes."""
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = __import__("argparse").ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
launcher = AppLauncher(args)
app = launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
    dt=1 / 60, device="cpu", gravity=(0.0, 0.0, -55.0)))
try:
    cases = {
        "small": ((0.25, 0.2, 0.25), 1.0),
        "standing_height_1": ((1.6, 0.2, 1.0), 49.28),
        "standing_source": ((1.6, 0.2, 5.0), 49.28),
    }
    bodies = {}
    for index, (name, (size, mass)) in enumerate(cases.items()):
        x = index * 5.0
        ground = sim_utils.CuboidCfg(
            size=(4.0, 0.2, 1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True))
        ground.func(f"/World/Ground_{name}", ground, translation=(x, 0.0, 0.5))
        spawn = sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False, disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True))
        cfg = RigidObjectCfg(
            prim_path=f"/World/Body_{name}", spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(x, 0.0, 1 + size[2] / 2 + 3)))
        bodies[name] = RigidObject(cfg=cfg)
    import isaacsim.core.utils.stage as stage_utils
    stage_utils.update_stage()
    sim.reset(); sim.play()
    for _ in range(120):
        sim.step(render=False)
        for body in bodies.values():
            body.update(sim.get_physics_dt())
    result = {}
    for name, body in bodies.items():
        result[name] = {
            "position": body.data.root_pos_w[0].tolist(),
            "velocity": body.data.root_lin_vel_w[0].tolist(),
        }
    output = Path(__file__).resolve().parent / "results" / "body_collision_probe.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("BODY_COLLISION_PROBE", json.dumps(result), flush=True)
finally:
    sim.clear_all_callbacks(); sim.clear_instance(); app.close()
