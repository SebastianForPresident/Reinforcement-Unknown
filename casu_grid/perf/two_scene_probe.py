"""Feasibility probe: two PhysX scenes in one stage, stepped independently with simulate_scene / fetch_results_scene, bodies read through
separate tensor views. Prints what works; no part of the training stack depends on it.

    python casu_grid/perf/two_scene_probe.py
"""
import argparse, faulthandler, os, sys, time
MODE = os.environ.get('PROBE_MODE', 'owners')      # control | empty2 | owners | owners_noapi
faulthandler.dump_traceback_later(45, exit=True)      # hard stop with a Python stack if a physics call hangs
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p); args = p.parse_args(); args.headless = True
app = AppLauncher(args).app
import numpy as np, torch
import isaaclab.sim as sim_utils
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics
import omni.physx
from omni.physx.scripts import physicsUtils
from omni.physx import get_physx_simulation_interface


def main():
    if os.environ.get('PROBE_THREADS'):
        import carb.settings; carb.settings.get_settings().set('/persistent/physics/numThreads', int(os.environ['PROBE_THREADS']))
    for item in filter(None, os.environ.get('PROBE_CARB', '').split(';')):
        import carb.settings; k, v = item.split('='); carb.settings.get_settings().set(k, {'true': True, 'false': False}.get(v, v)); print('PROBE carb', k, v, flush=True)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.02, device="cpu", gravity=(0.0, 0.0, -9.81)))
    stage = sim.stage
    # a second physics scene next to the default one
    if MODE != "control":
        scene2 = UsdPhysics.Scene.Define(stage, "/physicsScene2")
        scene2.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1)); scene2.CreateGravityMagnitudeAttr(9.81)
        if MODE != "owners_noapi":
            PhysxSchema.PhysxSceneAPI.Apply(scene2.GetPrim()).CreateTimeStepsPerSecondAttr(50)
    n_per_scene = 4
    for scene_index, owner in enumerate(("/physicsScene", "/physicsScene2")):
        for k in range(n_per_scene):
            root = f"/World/S{scene_index}_{k}"
            ground = UsdGeom.Cube.Define(stage, root + "_ground"); ground.GetSizeAttr().Set(1.0)
            ground.AddTranslateOp().Set(Gf.Vec3d(k * 10.0, scene_index * 10.0, -0.5)); ground.AddScaleOp().Set(Gf.Vec3f(5, 5, 0.5))
            UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
            cube = UsdGeom.Cube.Define(stage, root + "_body"); cube.GetSizeAttr().Set(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(k * 10.0, scene_index * 10.0, 3.0 + k))
            UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim()); UsdPhysics.CollisionAPI.Apply(cube.GetPrim()); UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(1.0)
            for prim in (ground.GetPrim(), cube.GetPrim()):
                if MODE in ("control", "empty2"):
                    continue
                rel = prim.CreateRelationship("physics:simulationOwner"); rel.SetTargets([Sdf.Path(owner)])
    sim.reset()
    print("PROBE", MODE, "reset ok; playing", sim.is_playing(), flush=True)
    sv = sim.physics_sim_view
    views = []
    for scene_index in range(2):
        paths = [f"/World/S{scene_index}_{k}_body" for k in range(n_per_scene)]
        try:
            view = sv.create_rigid_body_view(paths)
            views.append(view)
            print("PROBE view", scene_index, "count", view.count, flush=True)
        except Exception as exc:
            print("PROBE view", scene_index, "FAILED", repr(exc)[:300], flush=True)
            views.append(None)
    try:
        both = sv.create_rigid_body_view([f"/World/S{i}_{k}_body" for i in range(2) for k in range(n_per_scene)])
        print("PROBE combined view count", both.count, flush=True)
    except Exception as exc:
        print("PROBE combined view FAILED", repr(exc)[:300], flush=True)
        both = None
    iface = get_physx_simulation_interface()
    from pxr import PhysicsSchemaTools
    ids = [PhysicsSchemaTools.sdfPathToInt(Sdf.Path(pth)) for pth in ("/physicsScene", "/physicsScene2")]
    z0 = [v.get_transforms().numpy()[:, 2].copy() if v is not None else None for v in views]
    t = 0.0
    # step only scene 2 for 20 ticks: scene 1 must stay put
    for _ in range(20):
        iface.simulate_scene(ids[1], 0.02, t); iface.fetch_results_scene(ids[1]); t += 0.02
    z1 = [v.get_transforms().numpy()[:, 2].copy() if v is not None else None for v in views]
    print("PROBE scene1 moved (should be 0):", float(np.abs(z1[0] - z0[0]).max()) if views[0] is not None else None, flush=True)
    print("PROBE scene2 moved (should be > 0):", float(np.abs(z1[1] - z0[1]).max()) if views[1] is not None else None, flush=True)
    # both scenes started back to back, then fetched: overlap?
    a = time.perf_counter()
    for _ in range(200):
        iface.simulate_scene(ids[0], 0.02, t); iface.simulate_scene(ids[1], 0.02, t)
        iface.fetch_results_scene(ids[0]); iface.fetch_results_scene(ids[1]); t += 0.02
    print("PROBE two scenes concurrent us/tick", round(1e6 * (time.perf_counter() - a) / 200, 1), flush=True)
    a = time.perf_counter()
    for _ in range(200):
        iface.simulate_scene(ids[0], 0.02, t); iface.fetch_results_scene(ids[0])
        iface.simulate_scene(ids[1], 0.02, t); iface.fetch_results_scene(ids[1]); t += 0.02
    print("PROBE two scenes sequential us/tick", round(1e6 * (time.perf_counter() - a) / 200, 1), flush=True)
    a = time.perf_counter()
    for _ in range(200):
        iface.simulate(0.02, t); iface.fetch_results(); t += 0.02
    print("PROBE default simulate() both us/tick", round(1e6 * (time.perf_counter() - a) / 200, 1), flush=True)


import traceback
try:
    main()
except BaseException:
    traceback.print_exc(); sys.stderr.flush()
finally:
    sys.stdout.flush(); app.close()
