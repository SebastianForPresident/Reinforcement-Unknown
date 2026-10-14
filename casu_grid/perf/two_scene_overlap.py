"""Two PhysX scenes in one stage: does simulate_scene(A) overlap with main-thread work and with scene B's step?

    PROBE_THREADS=4 PROBE_N=48 python casu_grid/perf/two_scene_overlap.py

Root cause of the Optimization III hang (found in Optimization IV): with two PhysX scenes and /persistent/physics/numThreads = 2 the
warm-start ``update_simulation`` deadlocks; 0, 3 or 4 threads work.
"""
import argparse, faulthandler, os, sys, time
faulthandler.dump_traceback_later(int(os.environ.get("PROBE_DEADLINE", "70")), exit=True)
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p); args = p.parse_args(); args.headless = True
app = AppLauncher(args).app
import numpy as np
import isaaclab.sim as sim_utils
from pxr import Gf, PhysxSchema, PhysicsSchemaTools, Sdf, UsdGeom, UsdPhysics
from omni.physx import get_physx_simulation_interface

N = int(os.environ.get("PROBE_N", "48"))
THREADS = int(os.environ.get("PROBE_THREADS", "4"))


def busy(us):
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) * 1e6 < us:
        pass


def main():
    import carb.settings
    carb.settings.get_settings().set("/persistent/physics/numThreads", THREADS)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.02, device="cpu", gravity=(0.0, 0.0, -55.0),
                                                              physx=sim_utils.PhysxCfg(enable_ccd=True)))
    stage = sim.stage
    scene2 = UsdPhysics.Scene.Define(stage, "/physicsScene2")
    scene2.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1)); scene2.CreateGravityMagnitudeAttr(55.0)
    sapi = PhysxSchema.PhysxSceneAPI.Apply(scene2.GetPrim()); sapi.CreateTimeStepsPerSecondAttr(50); sapi.CreateEnableCCDAttr(True)
    rng = np.random.default_rng(0)
    shared_material = os.environ.get("PROBE_MATERIAL") == "1"
    if shared_material:
        from pxr import UsdShade
        UsdShade.Material.Define(stage, "/World/Mat"); UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath("/World/Mat")).CreateStaticFrictionAttr(0.5)
        UsdPhysics.MaterialAPI(stage.GetPrimAtPath("/World/Mat")).CreateDynamicFrictionAttr(0.5)
    for si, owner in enumerate(("/physicsScene", "/physicsScene2")):
        gpath = f"/World/G{si}"
        if os.environ.get("PROBE_NESTED") == "1":
            UsdGeom.Xform.Define(stage, f"/World/Env{si}"); UsdGeom.Xform.Define(stage, f"/World/Env{si}/T"); gpath = f"/World/Env{si}/T/G"
        ground = UsdGeom.Cube.Define(stage, gpath); ground.GetSizeAttr().Set(1.0)
        ground.AddTranslateOp().Set(Gf.Vec3d(0, si * 1000.0, -0.5)); ground.AddScaleOp().Set(Gf.Vec3f(400, 400, 0.5))
        UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
        if os.environ.get("PROBE_STATIC_OWNER", "1") == "1":
            ground.GetPrim().CreateRelationship("physics:simulationOwner").SetTargets([Sdf.Path(owner)])
        for k in range(int(os.environ.get("PROBE_STATICS", "0"))):
            sc = UsdGeom.Cube.Define(stage, f"/World/ST{si}_{k}"); sc.GetSizeAttr().Set(1.0)
            sc.AddTranslateOp().Set(Gf.Vec3d(500.0 + k, si * 1000.0, 5.0)); UsdPhysics.CollisionAPI.Apply(sc.GetPrim())
            sc.GetPrim().CreateRelationship("physics:simulationOwner").SetTargets([Sdf.Path(owner)])
        for k in range(N):
            c = UsdGeom.Cube.Define(stage, f"/World/S{si}_{k}"); c.GetSizeAttr().Set(1.0)
            c.AddTranslateOp().Set(Gf.Vec3d(float(rng.uniform(-30, 30)), si * 1000.0 + float(rng.uniform(-30, 30)), 1.0 + float(rng.uniform(0, 6))))
            for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI):
                api.Apply(c.GetPrim())
            PhysxSchema.PhysxRigidBodyAPI.Apply(c.GetPrim()).CreateEnableCCDAttr(True)
            UsdPhysics.MassAPI.Apply(c.GetPrim()).CreateMassAttr(1.0)
            c.GetPrim().CreateRelationship("physics:simulationOwner").SetTargets([Sdf.Path(owner)])
            if os.environ.get("PROBE_JOINT") == "1":
                a = UsdGeom.Xform.Define(stage, f"/World/A{si}_{k}"); a.AddTranslateOp().Set(c.GetPrim().GetAttribute("xformOp:translate").Get())
                UsdPhysics.RigidBodyAPI.Apply(a.GetPrim()).CreateKinematicEnabledAttr(True)
                a.GetPrim().CreateRelationship("physics:simulationOwner").SetTargets([Sdf.Path(owner)])
                if os.environ.get("PROBE_NOJOINTPRIM") == "1": continue
                j = UsdPhysics.Joint.Define(stage, f"/World/J{si}_{k}")
                if os.environ.get("PROBE_JOINT_WORLD") == "1":
                    j.CreateBody1Rel().SetTargets([f"/World/S{si}_{k}"])
                else:
                    j.CreateBody0Rel().SetTargets([f"/World/A{si}_{k}"]); j.CreateBody1Rel().SetTargets([f"/World/S{si}_{k}"])
                for ax in (UsdPhysics.Tokens.transY, UsdPhysics.Tokens.rotX, UsdPhysics.Tokens.rotY, UsdPhysics.Tokens.rotZ):
                    lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), ax); lim.CreateLowAttr(1.0); lim.CreateHighAttr(-1.0)
                j.GetPrim().CreateRelationship("physics:simulationOwner").SetTargets([Sdf.Path(owner)])
    if shared_material:
        from pxr import UsdShade
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(stage.GetPrimAtPath("/World/Mat")), UsdShade.Tokens.weakerThanDescendants, "physics")
    sim.reset()
    iface = get_physx_simulation_interface()
    if os.environ.get("PROBE_RUNTIME_STATIC"):
        # a static collider added after the simulation started, owned by scene 2 (and one owned by scene 1), plus a falling cube above each
        def add(si, owner, mode):
            x0 = 700.0 + si * 20
            if mode == "usd":
                g = UsdGeom.Cube.Define(stage, f"/World/RT{si}"); g.GetSizeAttr().Set(1.0)
                g.AddTranslateOp().Set(Gf.Vec3d(x0, si * 1000.0, -0.5)); g.AddScaleOp().Set(Gf.Vec3f(5, 5, 0.5))
                UsdPhysics.CollisionAPI.Apply(g.GetPrim())
                g.GetPrim().CreateRelationship("physics:simulationOwner").SetTargets([Sdf.Path(owner)])
            else:
                layer = stage.GetEditTarget().GetLayer()
                with Sdf.ChangeBlock():
                    spec = Sdf.PrimSpec(layer.GetPrimAtPath("/World"), f"RT{si}", Sdf.SpecifierDef, "Cube")
                    spec.SetInfo("apiSchemas", Sdf.TokenListOp.Create(prependedItems=["PhysicsCollisionAPI"]))
                    Sdf.AttributeSpec(spec, "size", Sdf.ValueTypeNames.Double).default = 1.0
                    Sdf.AttributeSpec(spec, "xformOp:translate", Sdf.ValueTypeNames.Double3).default = Gf.Vec3d(x0, si * 1000.0, -0.5)
                    Sdf.AttributeSpec(spec, "xformOp:scale", Sdf.ValueTypeNames.Float3).default = Gf.Vec3f(5, 5, 0.5)
                    Sdf.AttributeSpec(spec, "xformOpOrder", Sdf.ValueTypeNames.TokenArray, variability=Sdf.VariabilityUniform).default = ["xformOp:translate", "xformOp:scale"]
                    rel = Sdf.RelationshipSpec(spec, "physics:simulationOwner"); rel.targetPathList.explicitItems = [Sdf.Path(owner)]
        mode = os.environ["PROBE_RUNTIME_STATIC"]
        add(0, "/physicsScene", mode); add(1, "/physicsScene2", mode)
        # dynamic cubes created at runtime too (so the test cubes exist)
        for si, owner in ((0, "/physicsScene"), (1, "/physicsScene2")):
            c = UsdGeom.Cube.Define(stage, f"/World/RC{si}"); c.GetSizeAttr().Set(1.0)
            c.AddTranslateOp().Set(Gf.Vec3d(700.0 + si * 20, si * 1000.0, 3.0))
            UsdPhysics.RigidBodyAPI.Apply(c.GetPrim()); UsdPhysics.CollisionAPI.Apply(c.GetPrim()); UsdPhysics.MassAPI.Apply(c.GetPrim()).CreateMassAttr(1.0)
            c.GetPrim().CreateRelationship("physics:simulationOwner").SetTargets([Sdf.Path(owner)])
        import omni.usd
        for _ in range(5):
            for i in (0, 1):
                iface.simulate_scene(PhysicsSchemaTools.sdfPathToInt(Sdf.Path(("/physicsScene", "/physicsScene2")[i])), 0.02, 0.0)
                iface.fetch_results_scene(PhysicsSchemaTools.sdfPathToInt(Sdf.Path(("/physicsScene", "/physicsScene2")[i])))
        views_rt = []
        try:
            for si in (0, 1):
                v = sim.physics_sim_view.create_rigid_body_view([f"/World/RC{si}"]); views_rt.append(v)
            for _ in range(60):
                for i in (0, 1):
                    sid = PhysicsSchemaTools.sdfPathToInt(Sdf.Path(("/physicsScene", "/physicsScene2")[i])); iface.simulate_scene(sid, 0.02, 0.0); iface.fetch_results_scene(sid)
            print("OVL runtime-added static, cube z after fall (rest ~0.0 on slab top at 0):", [round(float(v.get_transforms().numpy()[0, 2]), 2) for v in views_rt], flush=True)
        except Exception as e:
            print("OVL runtime view failed", str(e)[:200], flush=True)
    ids = [PhysicsSchemaTools.sdfPathToInt(Sdf.Path(x)) for x in ("/physicsScene", "/physicsScene2")]
    t = [0.0]

    def timeit(label, fn, n=300):
        for _ in range(30): fn()
        a = time.perf_counter()
        for _ in range(n): fn()
        us = 1e6 * (time.perf_counter() - a) / n
        print(f"OVL threads={THREADS} N={N} {label:44s} {us:8.1f} us/iter", flush=True)
        return us

    def a_only():
        iface.simulate_scene(ids[0], 0.02, t[0]); iface.fetch_results_scene(ids[0]); t[0] += 0.02
    def a_busy(us):
        def f():
            iface.simulate_scene(ids[0], 0.02, t[0]); busy(us); iface.fetch_results_scene(ids[0]); t[0] += 0.02
        return f
    def ab_seq():
        for i in (0, 1):
            iface.simulate_scene(ids[i], 0.02, t[0]); iface.fetch_results_scene(ids[i])
        t[0] += 0.02
    def ab_conc():
        for i in (0, 1): iface.simulate_scene(ids[i], 0.02, t[0])
        for i in (0, 1): iface.fetch_results_scene(ids[i])
        t[0] += 0.02
    def ab_conc_busy(us):
        def f():
            for i in (0, 1): iface.simulate_scene(ids[i], 0.02, t[0])
            busy(us)
            for i in (0, 1): iface.fetch_results_scene(ids[i])
            t[0] += 0.02
        return f
    def both_default():
        iface.simulate(0.02, t[0]); iface.fetch_results(); t[0] += 0.02
    def busy_only(us):
        return lambda: busy(us)
    from pxr import UsdGeom as _G
    views = [sim.physics_sim_view.create_rigid_body_view([f"/World/S{i}_{k}" for k in range(N)]) for i in (0, 1)]
    for _ in range(200):                      # let everything fall
        for i in (0, 1): iface.simulate_scene(ids[i], 0.02, t[0])
        for i in (0, 1): iface.fetch_results_scene(ids[i])
        t[0] += 0.02
    for i in (0, 1):
        z = views[i].get_transforms().numpy()[:, 2]
        print(f"OVL resting height scene {i}: min {z.min():.2f} max {z.max():.2f} (cube half-size 0.5 -> resting 0.5)", flush=True)
    timeit("busy 500us only", busy_only(500))
    a = timeit("scene A step", a_only)
    timeit("scene A step + 500us busy between", a_busy(500))
    b = timeit("A then B sequential", ab_seq)
    timeit("A,B started together then fetched", ab_conc)
    timeit("A,B together + 500us busy between", ab_conc_busy(500))
    timeit("default simulate() (all scenes)", both_default)
    # pipeline: Python work for group B (busy) overlaps PhysX for A, and vice versa
    def pipe():
        iface.simulate_scene(ids[0], 0.02, t[0]); busy(500); iface.fetch_results_scene(ids[0])
        iface.simulate_scene(ids[1], 0.02, t[0]); busy(500); iface.fetch_results_scene(ids[1]); t[0] += 0.02
    timeit("pipeline A|busy B|busy (2 x 500us busy)", pipe)

import traceback
try:
    main()
except BaseException:
    traceback.print_exc(); sys.stderr.flush()
finally:
    sys.stdout.flush(); os._exit(0)
