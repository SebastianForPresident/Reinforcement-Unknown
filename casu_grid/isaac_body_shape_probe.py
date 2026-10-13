"""Probe: rounded compound player collider rest height, rest-offset calibration, runtime resize."""
import argparse, json, sys
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--rest-offset", type=float, default=0.0155)
parser.add_argument("--contact-offset", type=float, default=0.04)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import isaaclab.sim as sim_utils
import isaacsim.core.utils.stage as stage_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from pxr import Gf, UsdGeom, UsdPhysics
from casu_grid.isaac_player_body import RoundedBoxCollider

W, H, R, D = 1.6, 5.0, 0.1, 0.2
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.02, device="cpu", gravity=(0, 0, -55.0),
                                                           physx=sim_utils.PhysxCfg(enable_ccd=True)))
stage = stage_utils.get_current_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
mat = sim_utils.RigidBodyMaterialCfg(static_friction=0.0, dynamic_friction=0.0, restitution=0.0)
mat.func("/World/Mat", mat)
floor = sim_utils.CuboidCfg(size=(100.0, 2.0, 1.0), physics_material=mat, physics_material_path="/World/Mat",
                            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True))
floor.func("/World/Floor", floor, translation=(0.0, 0.0, -0.5))   # top at z = 0
wall = sim_utils.CuboidCfg(size=(1.0, 2.0, 20.0), physics_material=mat, physics_material_path="/World/Mat",
                           collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True))
wall.func("/World/Wall", wall, translation=(10.5, 0.0, 10.0))      # face at x = 10
spawn = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=8, solver_velocity_iteration_count=8),
    mass_props=sim_utils.MassPropertiesCfg(mass=49.28),
    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False))
player = RigidObject(RigidObjectCfg(prim_path="/World/CasuPlayer", spawn=spawn,
                                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 6.0))))
body = RoundedBoxCollider(stage, "/World/CasuPlayer", width=W, standing_height=H, edge_radius=R, depth=D,
                          material_path="/World/Mat", levels=8, rest_offset=args.rest_offset, contact_offset=args.contact_offset)
anchor = UsdGeom.Xform.Define(stage, "/World/Anchor")
UsdGeom.Xformable(anchor).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
UsdPhysics.RigidBodyAPI.Apply(anchor.GetPrim()).CreateKinematicEnabledAttr(True)
joint = UsdPhysics.Joint.Define(stage, "/World/Joint")
joint.CreateBody0Rel().SetTargets(["/World/Anchor"]); joint.CreateBody1Rel().SetTargets(["/World/CasuPlayer"])
for attr, val in ((joint.CreateLocalPos0Attr, Gf.Vec3f(0.0)), (joint.CreateLocalPos1Attr, Gf.Vec3f(0.0))): attr().Set(val)
joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0)); joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
for axis in (UsdPhysics.Tokens.transY, UsdPhysics.Tokens.rotX, UsdPhysics.Tokens.rotY, UsdPhysics.Tokens.rotZ):
    limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis); limit.CreateLowAttr(1.0); limit.CreateHighAttr(-1.0)
stage_utils.update_stage()
sim.reset(); sim.play()

def settle(ticks=90):
    for _ in range(ticks):
        sim.step(render=False); player.update(0.02)
    return float(player.data.root_pos_w[0, 2]), float(player.data.root_lin_vel_w[0, 2])

def place(x, z, vx=0.0, vz=0.0):
    player.write_root_pose_to_sim(torch.tensor([[x, 0.0, z, 1, 0, 0, 0]], dtype=torch.float32))
    player.write_root_velocity_to_sim(torch.tensor([[vx, 0, vz, 0, 0, 0]], dtype=torch.float32))
    player.write_data_to_sim(); player.update(0.02)

result = {"rest_offset": args.rest_offset, "contact_offset": args.contact_offset, "standing": {}, "resize": [], "wall": {}}
# standing rest: source core bottom sits 0.1155 above the support (recorded 0.1155-0.1201)
place(0.0, 6.0); z, vz = settle()
result["standing"] = {"root_z": z, "core_bottom_above_floor": z - H / 2, "vz": vz}
# runtime resize: crouch to half height with centre offset -H/4, then back
for crouch in (0.25, 0.5, 1.0, 0.0):
    level = body.level_for_crouch(crouch); h, off = body.level_shape(level)
    body.set_level(level)
    z, vz = settle(60)
    result["resize"].append({"crouch": crouch, "height": h, "offset": off, "root_z": z,
                             "core_bottom_above_floor": z + off - h / 2, "vz": vz})
# wall rest: run at the wall (face x=10): source rest gap ~0.12 between core edge and tile
body.set_level(0)
place(5.0, 2.6 + 0.0155 + 0.1, vx=25.0)
for _ in range(60):
    player.set_external_force_and_torque(torch.tensor([[[4000.0, 0, 0]]], dtype=torch.float32), torch.zeros((1, 1, 3)), is_global=True)
    player.write_data_to_sim(); sim.step(render=False); player.update(0.02)
x = float(player.data.root_pos_w[0, 0])
result["wall"] = {"root_x": x, "core_edge_gap_to_wall": 10.0 - (x + W / 2), "vx": float(player.data.root_lin_vel_w[0, 0])}
args.output.write_text(json.dumps(result, indent=1))
print("PROBE", json.dumps(result))
sim.clear_instance(); app.close()
