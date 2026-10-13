"""Paired controlled standing-motion replay: CasU fixed ticks vs Isaac PhysX.

This is a bounded diagnostic, not the training environment. It builds the
source chunk containing a validated flat run, replays the captured CB1 action
tape at the measured 20 ms/10-tick protocol, and compares pre-physics
controller-boundary position and velocity samples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--source-trace", type=Path, required=True)
parser.add_argument("--source-smoke", type=Path, required=True)
parser.add_argument("--output", type=Path, default=Path("casu_grid/results/player_motion_ab.json"))
parser.add_argument("--fixture-x", type=int, default=483)
parser.add_argument("--fixture-y", type=int, default=1010)
parser.add_argument("--terrain-extrusion-depth", type=float, default=2.0)
parser.add_argument("--max-actions", type=int, default=8)
parser.add_argument("--model-edge-radius", action="store_true",
                     help="experimental expanded rounded convex proxy for Unity BoxCollider2D.edgeRadius")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
launcher = AppLauncher(args)
app = launcher.app

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from pxr import Gf, UsdGeom, UsdPhysics
import isaaclab.sim as sim_utils
import isaacsim.core.utils.stage as stage_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from casu_grid.collision_usd import author_layer1_box_collision
from casu_grid.player_controller import StandingAction, StandingPlayerController
from casu_grid.reference import load_layer1_reference


def read_source_tape(path: Path, max_actions: int):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    header = next(row for row in rows if row.get("record_type") == "header")
    player = [row for row in rows if row.get("record_type") == "player_tick"]
    selected = []
    sequence_ids = sorted({r["action_sequence"] for r in player if 1 <= r["action_sequence"] <= max_actions})
    for sequence in sequence_ids:
        sequence_rows = [r for r in player if r["action_sequence"] == sequence][:10]
        if len(sequence_rows) != 10:
            raise ValueError(f"source action sequence {sequence} contains {len(sequence_rows)} ticks, expected 10")
        selected.extend(sequence_rows)
    if not selected:
        raise ValueError("no bounded action tape found in CasU fixed-tick trace")
    return header, selected


def main():
    reference = load_layer1_reference()
    header, source_rows = read_source_tape(args.source_trace, args.max_actions)
    smoke = json.loads(args.source_smoke.read_text())
    actions = smoke["actions"]
    if len(actions) < args.max_actions:
        raise ValueError("source smoke action tape is shorter than --max-actions")

    chunk_x, chunk_y = args.fixture_x // 64, args.fixture_y // 64
    x0, y0 = chunk_x * 64, chunk_y * 64
    terrain = np.zeros_like(reference.terrain)
    terrain[x0:x0 + 64, y0:y0 + 64] = reference.terrain[x0:x0 + 64, y0:y0 + 64]
    sim_cfg = sim_utils.SimulationCfg(
        dt=float(header["unity_fixed_delta_time"]), device="cpu",
        gravity=(0.0, 0.0, float(header["physics2d_gravity"][1])),
        physx=sim_utils.PhysxCfg(enable_ccd=True))
    sim = sim_utils.SimulationContext(sim_cfg)
    stage = stage_utils.get_current_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    try:
        # CasU's live material coefficients remain unresolved. A zero-friction
        # trial isolates the measured 4,000 N controller force and explicit
        # source slowdown from PhysX's default 0.5 contact friction.
        contact_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.0, dynamic_friction=0.0, restitution=0.0)
        shared_material_path = "/World/CasuPhysicsMaterial"
        contact_material.func(shared_material_path, contact_material)
        geometry = author_layer1_box_collision(
            stage, terrain, half_thickness=args.terrain_extrusion_depth / 2,
            physics_material=contact_material,
            physics_material_path=shared_material_path)

        initial = source_rows[0]
        start_xyz = (float(initial["position"][0]), 0.0, float(initial["position"][1]))
        source_mass = float(initial["mass"])
        source_size = tuple(float(v) for v in initial["collider_size"])  # 2-D width, height
        edge_radius = float(initial.get("collider_edge_radius", 0.0)) if args.model_edge_radius else 0.0
        body_size = (source_size[0] + 2.0 * edge_radius, 0.2,
                     source_size[1] + 2.0 * edge_radius)
        body_cfg = sim_utils.CuboidCfg(
            size=body_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False, disable_gravity=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=8),
            mass_props=sim_utils.MassPropertiesCfg(mass=source_mass),
            physics_material_path=shared_material_path,
            physics_material=contact_material,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True))
        player_cfg = RigidObjectCfg(
            prim_path="/World/Probe", spawn=body_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=start_xyz))
        player = RigidObject(cfg=player_cfg)
        if args.model_edge_radius and edge_radius > 0:
            root = stage.GetPrimAtPath("/World/Probe")
            root_collision = UsdPhysics.CollisionAPI.Apply(root)
            root_collision.CreateCollisionEnabledAttr(False)
            radius = edge_radius
            half_x, half_z = source_size[0] / 2, source_size[1] / 2
            # Unity's BoxCollider2D edge radius expands the box boundary. The
            # quarter-circle centers stay on the measured box corners; the
            # resulting outer bounds are size + 2 * radius.
            corners = ((half_x, half_z, 0.0, 90.0),
                       (-half_x, half_z, 90.0, 180.0),
                       (-half_x, -half_z, 180.0, 270.0),
                       (half_x, -half_z, 270.0, 360.0))
            profile = []
            for cx, cz, angle0, angle1 in corners:
                for segment in range(5):
                    angle = np.deg2rad(angle0 + (angle1 - angle0) * segment / 4)
                    profile.append((cx + radius * np.cos(angle),
                                    cz + radius * np.sin(angle)))
            depth = body_size[1]
            points = [(x, y, z) for y in (-depth / 2, depth / 2) for x, z in profile]
            count = len(profile)
            faces, counts = [], []
            for index in range(count):
                nxt = (index + 1) % count
                faces.extend((index, nxt, count + nxt, count + index))
                counts.append(4)
            faces.extend(reversed(range(count))); counts.append(count)
            faces.extend(range(count, 2 * count)); counts.append(count)
            mesh = UsdGeom.Mesh.Define(stage, "/World/Probe/RoundedCollision")
            mesh.CreatePointsAttr(points)
            mesh.CreateFaceVertexCountsAttr(counts)
            mesh.CreateFaceVertexIndicesAttr(faces)
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
            mesh_collision.CreateApproximationAttr().Set("convexHull")
            sim_utils.bind_physics_material("/World/Probe/RoundedCollision",
                                            shared_material_path)

        anchor = UsdGeom.Xform.Define(stage, "/World/CasuPlanarAnchor")
        UsdGeom.Xformable(anchor).AddTranslateOp().Set(Gf.Vec3d(*start_xyz))
        anchor_body = UsdPhysics.RigidBodyAPI.Apply(anchor.GetPrim())
        anchor_body.CreateKinematicEnabledAttr(True)
        joint = UsdPhysics.Joint.Define(stage, "/World/CasuPlanarConstraint")
        joint.CreateBody0Rel().SetTargets(["/World/CasuPlanarAnchor"])
        joint.CreateBody1Rel().SetTargets(["/World/Probe"])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
        for axis in (UsdPhysics.Tokens.transY, UsdPhysics.Tokens.rotX,
                     UsdPhysics.Tokens.rotY, UsdPhysics.Tokens.rotZ):
            limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
            limit.CreateLowAttr(1.0)
            limit.CreateHighAttr(-1.0)

        stage_utils.update_stage()
        sim.reset()
        sim.play()
        # AppLauncher/PhysX can apply a startup gravity integration during
        # play(); explicitly restore the captured source pose and zero twist
        # so the first compared row is the same pre-physics boundary.
        player.reset()
        initial_pose = torch.tensor([[start_xyz[0], start_xyz[1], start_xyz[2],
                                      1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        initial_velocity = torch.zeros((1, 6), dtype=torch.float32)
        player.write_root_pose_to_sim(initial_pose)
        player.write_root_velocity_to_sim(initial_velocity)
        player.write_data_to_sim()
        player.update(sim.get_physics_dt())
        controller = StandingPlayerController()
        output_rows = []
        previous_sequence = None
        action_index = 0
        for source in source_rows:
            sequence = source["action_sequence"]
            if sequence != previous_sequence:
                action = actions[action_index]
                # CB1 MultiDiscrete move field: categories 0,1,2 map to -1,0,1.
                controller.set_action(StandingAction(
                    move_x=int(action[0]) - 1,
                    jump=bool(action[1]),
                    move_y=int(action[2]) - 1,
                    crouch=bool(action[3])))
                previous_sequence = sequence
                action_index += 1

            player.update(sim.get_physics_dt())
            root = player.data.root_pos_w[0]
            velocity = player.data.root_lin_vel_w[0]
            # Source's grounded flag is based on a BoxCast/contact window, not
            # exact foot-plane equality. This geometric proxy uses the same
            # 0.18-unit tolerance as the navigation adapter and rejects upward
            # or fast downward motion.
            foot_height = float(root[2]) - source_size[1] / 2
            support_top = args.fixture_y - 512 + 1.0
            isaac_grounded = (foot_height <= support_top + 0.18 and
                              float(velocity[2]) <= 0.25)
            command = controller.pre_physics_tick(
                velocity_x=float(velocity[0]), velocity_y=float(velocity[2]),
                grounded=isaac_grounded,
                moving_allowed=True,
                force_walk=False,
                slippery=bool(source["slippery"]),
                leg_speed_multiplier=1.0,
                measured_max_speed=float(source["max_speed"]),
                measured_move_force=float(source["move_force"]),
                can_jump=True,
                dt=sim.get_physics_dt())

            if command.horizontal_velocity_override is not None or command.vertical_velocity_override is not None:
                new_velocity = player.data.root_com_vel_w[0].clone()
                if command.horizontal_velocity_override is not None:
                    new_velocity[0] = command.horizontal_velocity_override
                if command.vertical_velocity_override is not None:
                    new_velocity[2] = command.vertical_velocity_override
                player.write_root_velocity_to_sim(new_velocity.unsqueeze(0))
                player.update(sim.get_physics_dt())
                root = player.data.root_pos_w[0]
                velocity = player.data.root_lin_vel_w[0]

            output_rows.append({
                "source_fixed_index": source["fixed_index"],
                "action_sequence": sequence,
                "position_xyz": root.tolist(),
                "velocity_xyz": velocity.tolist(),
                "grounded": isaac_grounded,
                "source_grounded": bool(source["grounded"]),
                "foot_height_error_from_support": foot_height - support_top,
                "horizontal_force_n": command.horizontal_force,
                "gravity_scale": command.gravity_scale,
                "horizontal_velocity_override": command.horizontal_velocity_override,
            })

            force_z = source_mass * sim_cfg.gravity[2] * (command.gravity_scale - 1.0)
            force = torch.tensor([[command.horizontal_force, 0.0, force_z]], dtype=torch.float32)
            torque = torch.zeros((1, 3), dtype=torch.float32)
            player.set_external_force_and_torque(force, torque, is_global=True)
            player.write_data_to_sim()
            sim.step(render=False)

        x_errors = [abs(isaac["position_xyz"][0] - casu["position"][0])
                    for isaac, casu in zip(output_rows, source_rows)]
        y_errors = [abs(isaac["position_xyz"][2] - casu["position"][1])
                    for isaac, casu in zip(output_rows, source_rows)]
        vx_errors = [abs(isaac["velocity_xyz"][0] - casu["velocity"][0])
                     for isaac, casu in zip(output_rows, source_rows)]
        vy_errors = [abs(isaac["velocity_xyz"][2] - casu["velocity"][1])
                     for isaac, casu in zip(output_rows, source_rows)]
        result = {
            "schema": "casu.player-motion-ab.v1",
            "canonical_fixture_id": reference.manifest["fixture_id"],
            "canonical_content_id": reference.manifest["content_id"],
            "source_trace_path": str(args.source_trace),
            "source_header": header,
            "source_smoke_status": smoke["status"],
            "action_tape": actions[:args.max_actions],
            "physics_dt_s": sim_cfg.dt,
            "physics_ticks_per_action": 10,
            "local_source_chunk_xy": [chunk_x, chunk_y],
            "terrain_geometry": geometry,
            "terrain_extrusion_depth": args.terrain_extrusion_depth,
            "body_mass_kg": source_mass,
            "body_size_xyz": body_size,
            "source_box_size_xy": source_size,
            "source_edge_radius": float(initial.get("collider_edge_radius", 0.0)),
            "edge_radius_modelled": bool(args.model_edge_radius),
            "source_tile_collider_material": "coefficients unmeasured; current Isaac trial isolates the measured controller with 0 static/dynamic friction",
            "sample_alignment": "controller outputs applied before PhysX tick; state compared at matching pre-physics boundaries",
            "sample_count": len(output_rows),
            "max_abs_position_x_error": max(x_errors),
            "mean_abs_position_x_error": float(np.mean(x_errors)),
            "max_abs_position_y_error": max(y_errors),
            "mean_abs_position_y_error": float(np.mean(y_errors)),
            "max_abs_velocity_x_error": max(vx_errors),
            "mean_abs_velocity_x_error": float(np.mean(vx_errors)),
            "max_abs_velocity_y_error": max(vy_errors),
            "mean_abs_velocity_y_error": float(np.mean(vy_errors)),
            "grounded_mismatch_count": sum(
                isaac["grounded"] != isaac["source_grounded"]
                for isaac in output_rows),
            "source_trace": source_rows,
            "isaac_trace": output_rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("PLAYER_MOTION_AB_RESULT", args.output, flush=True)
        print("PLAYER_MOTION_AB_ERRORS", result["max_abs_position_x_error"],
              result["max_abs_velocity_x_error"], result["max_abs_velocity_y_error"], flush=True)
    finally:
        sim.clear_instance()


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    app.close()
