"""Two PhysX scenes with the real mini-world terrain colliders: does a body owned by scene 2 find its (scene-2-owned) terrain?

    PROBE_VARIANT=... python casu_grid/perf/two_scene_terrain_probe.py
"""
import argparse, faulthandler, os, sys
faulthandler.dump_traceback_later(80, exit=True)
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p); args = p.parse_args(); args.headless = True
app = AppLauncher(args).app
import numpy as np
import isaaclab.sim as sim_utils
from pxr import Gf, PhysxSchema, PhysicsSchemaTools, Sdf, UsdGeom, UsdPhysics
import omni.physx
sys.path.insert(0, os.getcwd())
from casu_grid.collision_usd import author_layer1_box_collision
from casu_grid.xy1_mini_world import load_mini_world_reference

V = set(filter(None, os.environ.get("PROBE_VARIANT", "").split(",")))


def main():
    import carb.settings
    carb.settings.get_settings().set("/persistent/physics/numThreads", 4)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.02, device="cpu", gravity=(0.0, 0.0, -55.0), physx=sim_utils.PhysxCfg(enable_ccd=True)))
    stage = sim.stage
    material = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
    material.func("/World/Mat", material)
    if "latescene" not in V:
        Sdf.CopySpec(stage.GetEditTarget().GetLayer(), Sdf.Path("/physicsScene"), stage.GetEditTarget().GetLayer(), Sdf.Path("/physicsScene2"))
    NW = int(os.environ.get("PROBE_WORLDS", "2"))
    owners = {w: (None if w < NW // 2 else Sdf.Path("/physicsScene2")) for w in range(NW)}
    if "alldefault" in V: owners = {w: None for w in range(NW)}
    if "allsame" in V: owners = {w: Sdf.Path("/physicsScene2") for w in range(NW)}
    for w in range(NW):
        root = f"/World/Env_{w:05d}/CasUTerrain"
        ref = load_mini_world_reference(391728405 + w)
        author_layer1_box_collision(stage, ref.terrain, half_thickness=0.5, physics_material=material, physics_material_path="/World/Mat",
                                    root_path=root, contact_offset=0.001, simulation_owner=(str(owners[w]) if owners[w] else None) if "authortime" in V else None)
        if w:
            UsdGeom.Xformable(stage.GetPrimAtPath(root)).AddTranslateOp().Set(Gf.Vec3d(1040.0 * w, 0, 0))
        if "player" in V:
            from casu_grid.isaac_player_body import RoundedBoxCollider
            from casu_grid.body_model import BodyConfig
            cfgb = BodyConfig()
            rigid = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False, max_depenetration_velocity=100.0,
                                                     solver_position_iteration_count=8, solver_velocity_iteration_count=8)
            body_cfg = sim_utils.CuboidCfg(size=(0.1, 0.2, 0.1), rigid_props=rigid, mass_props=sim_utils.MassPropertiesCfg(mass=cfgb.mass),
                                           collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                                           physics_material=material, physics_material_path="/World/Mat")
            body_cfg.func(f"/World/Env_{w:05d}/Body", body_cfg, translation=(1040.0 * w + 0.5, 0.0, 1.5))
            PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(f"/World/Env_{w:05d}/Body")).CreateEnableCCDAttr().Set(True)
            RoundedBoxCollider(stage, f"/World/Env_{w:05d}/Body", width=cfgb.orig_col_size[0], standing_height=cfgb.orig_col_size[1],
                               edge_radius=cfgb.edge_radius, depth=0.2, material_path="/World/Mat", levels=32,
                               rest_offset=0.0155, contact_offset=0.04)
            b = stage.GetPrimAtPath(f"/World/Env_{w:05d}/Body")
        else:
            b = UsdGeom.Cube.Define(stage, f"/World/Env_{w:05d}/Body"); b.GetSizeAttr().Set(1.0)
            b.AddTranslateOp().Set(Gf.Vec3d(1040.0 * w + 0.5, 0.0, 1.5))
            UsdPhysics.RigidBodyAPI.Apply(b.GetPrim()); UsdPhysics.CollisionAPI.Apply(b.GetPrim()); UsdPhysics.MassAPI.Apply(b.GetPrim()).CreateMassAttr(1.0)
            b = b.GetPrim()
        if "joint" in V:
            apath = f"/World/Env_{w:05d}/Anchor"
            an = UsdGeom.Xform.Define(stage, apath); UsdGeom.Xformable(an).AddTranslateOp().Set(Gf.Vec3d(1040.0 * w + 0.5, 0.0, 1.5))
            UsdPhysics.RigidBodyAPI.Apply(an.GetPrim()).CreateKinematicEnabledAttr(True)
            joint = UsdPhysics.Joint.Define(stage, f"/World/Env_{w:05d}/Joint")
            joint.CreateBody0Rel().SetTargets([apath]); joint.CreateBody1Rel().SetTargets([f"/World/Env_{w:05d}/Body"])
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0)); joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
            joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0)); joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
            for axis in (UsdPhysics.Tokens.transY, UsdPhysics.Tokens.rotX, UsdPhysics.Tokens.rotY, UsdPhysics.Tokens.rotZ):
                lim = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis); lim.CreateLowAttr(1.0); lim.CreateHighAttr(-1.0)
        if owners[w] and "authortime" not in V:
            from pxr import Usd
            for prim in Usd.PrimRange(stage.GetPrimAtPath(f"/World/Env_{w:05d}")):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
                    prim.CreateRelationship("physics:simulationOwner").SetTargets([owners[w]])
        elif owners[w]:
            b.CreateRelationship("physics:simulationOwner").SetTargets([owners[w]])
    if "latescene" in V:
        Sdf.CopySpec(stage.GetEditTarget().GetLayer(), Sdf.Path("/physicsScene"), stage.GetEditTarget().GetLayer(), Sdf.Path("/physicsScene2"))
    sim.reset()
    iface = omni.physx.get_physx_simulation_interface()
    ids = [PhysicsSchemaTools.sdfPathToInt(Sdf.Path(x)) for x in ("/physicsScene", "/physicsScene2")]
    views = [sim.physics_sim_view.create_rigid_body_view([f"/World/Env_{w:05d}/Body"]) for w in range(NW)]
    for _ in range(60):
        for i in (0, 1):
            iface.simulate_scene(ids[i], 0.02, 0.0); iface.fetch_results_scene(ids[i])
    print("TPROBE", sorted(V), "rest z per world:", [round(float(v.get_transforms().numpy()[0, 2]), 3) for v in views], flush=True)

import traceback
try:
    main()
except BaseException:
    traceback.print_exc(); sys.stderr.flush()
finally:
    sys.stdout.flush(); os._exit(0)
