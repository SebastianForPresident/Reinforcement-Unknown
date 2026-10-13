"""Build all canonical Layer 1 terrain meshes and drop one source-sized body."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path)
parser.add_argument("--geometry", choices=("primitive", "direct", "boxes", "mesh"), default="primitive")
parser.add_argument("--probe-profile", choices=("small", "source_light", "player"), default="player")
parser.add_argument("--extrusion-depth", type=float, default=0.2,
                    help="Probe depth along Isaac Y; the 2-D source has no finite depth.")
parser.add_argument("--terrain-extrusion-depth", type=float, default=2.0,
                    help="Artificial 3-D depth for 2-D source terrain collisions.")
parser.add_argument("--enable-ccd", action="store_true",
                    help="Enable PhysX continuous collision detection for this diagnostic.")
parser.add_argument("--free-rotation", action="store_true",
                    help="Do not constrain the source-sized probe to the CasU 2-D plane.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
launcher = AppLauncher(args)
app = launcher.app

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaaclab.sim as sim_utils
import isaacsim.core.utils.stage as stage_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from casu_grid.reference import load_layer1_reference
from casu_grid.collision_usd import (
    author_layer1_box_collision,
    author_layer1_collision,
)


def find_flat_support(blocks):
    """Find a long same-row solid run with clear cells above its midpoint."""
    import numpy as np
    occupied = blocks != 0
    choices = []
    for y in range(1, blocks.shape[1] - 20):
        changes = np.diff(np.pad(occupied[:, y].astype(np.int8), (1, 1)))
        for x0, x1 in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            if x1 - x0 < 5:
                continue
            x = int((x0 + x1) // 2)
            if not occupied[x, y + 1:y + 20].any():
                choices.append((int(x1 - x0), x, y))
    if not choices:
        raise RuntimeError("canonical terrain has no open flat support fixture")
    _, x, y = max(choices)
    return x, y


def main():
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    reference = load_layer1_reference()
    x, y = find_flat_support(reference.terrain)
    world_x, world_z = x - 512 + 0.5, y - 512 + 0.5
    floor_z = world_z + 0.5
    print("CANONICAL_TERRAIN_BUILD_START", x, y, flush=True)
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 60.0, device="cpu", gravity=(0.0, 0.0, -55.0),
        physx=sim_utils.PhysxCfg(enable_ccd=args.enable_ccd))
    sim = sim_utils.SimulationContext(sim_cfg)
    stage = stage_utils.get_current_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    try:
        started = perf_counter()
        if args.geometry == "primitive":
            ground_cfg = sim_utils.CuboidCfg(
                size=(10.0, args.terrain_extrusion_depth, 1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True))
            ground_cfg.func("/World/CasUPrimitiveGround", ground_cfg,
                            translation=(world_x, 0.0, floor_z - 0.5))
            geometry = {"representation": "single IsaacLab CuboidCfg static collider",
                        "support_size_xyz": [10.0, args.terrain_extrusion_depth, 1.0]}
            geometry["build_seconds"] = perf_counter() - started
            geometry["stage_authoring_seconds"] = 0.0
            count = 1
        elif args.geometry == "direct":
            cube = UsdGeom.Cube.Define(stage, "/World/DirectGround")
            cube.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(cube)
            xf.AddTranslateOp().Set(Gf.Vec3d(world_x, 0.0, floor_z - 0.5))
            xf.AddScaleOp().Set(Gf.Vec3f(10.0, args.terrain_extrusion_depth, 1.0))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            sim_utils.define_collision_properties(
                "/World/DirectGround",
                sim_utils.CollisionPropertiesCfg(collision_enabled=True), stage=stage)
            geometry = {"representation": "direct USD cube; configurable extrusion support"}
            geometry["build_seconds"] = perf_counter() - started
            geometry["stage_authoring_seconds"] = 0.0
            count = 1
        elif args.geometry == "boxes":
            geometry = author_layer1_box_collision(
                stage, reference.terrain,
                half_thickness=args.terrain_extrusion_depth / 2)
            geometry["stage_authoring_seconds"] = perf_counter() - started - geometry["build_seconds"]
            count = geometry["box_collider_count"]
        else:
            geometry = author_layer1_collision(
                stage, reference.terrain,
                half_thickness=args.terrain_extrusion_depth / 2)
            geometry["stage_authoring_seconds"] = perf_counter() - started - geometry["build_seconds"]
            count = geometry["mesh_quad_count"]

        if args.probe_profile in ("source_light", "player"):
            body_size = (1.6, args.extrusion_depth, 5.0)
            body_mass = 1.0 if args.probe_profile == "source_light" else 49.28
            start_clearance = 6.0
            expected_root_z = floor_z + body_size[2] / 2
        else:
            body_size = (0.25, args.extrusion_depth, 0.25)
            body_mass = 1.0
            start_clearance = 2.0
            expected_root_z = floor_z + body_size[2] / 2
        print("CANONICAL_TERRAIN_GEOMETRY_READY", args.geometry, count, flush=True)

        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False, disable_gravity=False,
            solver_position_iteration_count=3,
            solver_velocity_iteration_count=8)
        mass_props = sim_utils.MassPropertiesCfg(mass=body_mass)
        collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        small_probe = args.probe_profile == "small"
        body_cfg = sim_utils.CuboidCfg(
            size=body_size, rigid_props=rigid_props,
            mass_props=mass_props, collision_props=collision_props)
        player = RigidObjectCfg(
            prim_path="/World/Probe",
            spawn=body_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(world_x, 0.0, expected_root_z + start_clearance)),
        )
        body = RigidObject(cfg=player)
        if not small_probe and not args.free_rotation:
            # Rigidbody2D stays in its X/Y plane: lock Isaac Y translation
            # and all rotations while retaining dynamic X/Z translation.
            anchor_path = "/World/CasuPlanarAnchor"
            anchor = UsdGeom.Xform.Define(stage, anchor_path)
            UsdGeom.Xformable(anchor).AddTranslateOp().Set(
                Gf.Vec3d(world_x, 0.0, expected_root_z + start_clearance))
            anchor_body = UsdPhysics.RigidBodyAPI.Apply(anchor.GetPrim())
            anchor_body.CreateKinematicEnabledAttr(True)
            joint = UsdPhysics.Joint.Define(stage, "/World/CasuPlanarConstraint")
            joint.CreateBody0Rel().SetTargets([anchor_path])
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
        bounds = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
        ).ComputeWorldBound(stage.GetPrimAtPath("/World/Probe/geometry/mesh")).ComputeAlignedBox()
        body_bounds = list(bounds.GetSize())
        body_min, body_max = list(bounds.GetMin()), list(bounds.GetMax())
        print("PLAYER_COLLISION_BOUNDS", body_min, body_max, flush=True)
        if args.geometry == "primitive":
            ground_bounds = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
            ).ComputeWorldBound(
                stage.GetPrimAtPath("/World/CasUPrimitiveGround/geometry/mesh")
            ).ComputeAlignedBox()
            ground_min = list(ground_bounds.GetMin())
            ground_max = list(ground_bounds.GetMax())
            print("SUPPORT_BOUNDS", ground_min, ground_max, flush=True)
        stage_utils.update_stage()
        sim.reset()
        sim.play()
        trace = []
        for tick in range(180):
            sim.step(render=False)
            body.update(sim.get_physics_dt())
            trace.append({"tick": tick + 1,
                          "root_xyz": body.data.root_pos_w[0].tolist(),
                          "velocity_xyz": body.data.root_lin_vel_w[0].tolist()})
        final_z = float(body.data.root_pos_w[0, 2].item())
        passed = abs(final_z - expected_root_z) <= 0.02
        result = {
            "schema": "casu.isaac-layer1-collision-smoke.v1",
            "canonical_fixture_id": reference.manifest["fixture_id"],
            "canonical_content_id": reference.manifest["content_id"],
            "terrain_geometry": geometry,
            "geometry_kind": args.geometry,
            "probe_profile": args.probe_profile,
            "probe_extrusion_depth": args.extrusion_depth,
            "terrain_extrusion_depth": args.terrain_extrusion_depth,
            "probe_planar_constraint": not small_probe and not args.free_rotation,
            "physx_ccd": args.enable_ccd,
            "support_cell_xy": [x, y],
            "support_cell_id": int(reference.terrain[x, y]),
            "support_cell_world_center_xz": [world_x, world_z],
            "expected_support_top_z": floor_z,
            "body_mass_kg": body_mass,
            "body_size_xyz": list(body_size),
            "authored_body_bounds_xyz": body_bounds,
            "authored_body_min_xyz": body_min,
            "authored_body_max_xyz": body_max,
            "physics_dt": sim_cfg.dt,
            "gravity_xyz": [0.0, 0.0, -55.0],
            "ticks": 180,
            "trace": trace,
            "final_root_z": final_z,
            "rest_root_z_error": final_z - expected_root_z,
            "contact_validation_pass": passed,
        }
        output = args.output or Path(__file__).resolve().parent / "results" / (
            f"layer1_collision_{args.geometry}_{args.probe_profile}.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        if abs(final_z - expected_root_z) > 0.02:
            print("CANONICAL_TERRAIN_CONTACT_MISMATCH", final_z,
                  expected_root_z, body.data.root_lin_vel_w[0].tolist(),
                  "diagnostic_saved_to", output, flush=True)
            raise RuntimeError(
                f"canonical terrain contact failed: final z={final_z}, expected {expected_root_z}"
            )
        print("CANONICAL_TERRAIN_COLLISION_OK", output, flush=True)
    finally:
        sim.clear_all_callbacks()
        sim.clear_instance()


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    app.close()
