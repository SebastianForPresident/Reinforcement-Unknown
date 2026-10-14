"""Profile the XY1 mini-world vector environment (one process) under near-uniform random policy actions.

    python casu_grid/perf/xy1_env_profile.py --envs 4 --steps 60 --cprofile --output out.json
"""
from __future__ import annotations
import argparse, cProfile, io, json, os, pstats, sys, time
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--envs", type=int, default=4)
p.add_argument("--steps", type=int, default=60)
p.add_argument("--warmup", type=int, default=5)
p.add_argument("--seed", type=int, default=3)
p.add_argument("--cprofile", action="store_true")
p.add_argument("--top", type=int, default=35)
p.add_argument("--player-collider", default="compound")
p.add_argument("--empty-terrain", action="store_true", help="DIAGNOSTIC ONLY: floor-only worlds, to size terrain-collider cost")
p.add_argument("--crouch-levels", type=int, default=None, help="DIAGNOSTIC ONLY: changes physics")
p.add_argument("--carb", action="append", default=[], help="NAME=VALUE carb setting applied after construction (diagnostic)")
p.add_argument("--no-ccd", action="store_true", help="DIAGNOSTIC ONLY (changes physics)")
p.add_argument("--physx-threads", type=int, default=None)
p.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); args.headless = True
app = AppLauncher(args).app
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from casu_grid.isaac_navigation_env import IsaacLayer1EnvConfig
from casu_grid.isaac_vector_env import BatchedIsaacLayer1Env
from casu_grid.workbench.experiments import xy1_point_navigation_mini as cfgmod

SIZES = [h.size for h in cfgmod.ACTION.heads]
NAMES = [h.name for h in cfgmod.ACTION.heads]


def main():
    if args.empty_terrain:
        import casu_grid.xy1_mini_world as mw
        orig = mw.generate_mini_terrain
        def floor_only(seed):
            t = orig(seed); keep = np.zeros_like(t); keep[:, :2] = t[:, :2]; keep[:7, :] = t[:7, :]; keep[-7:, :] = t[-7:, :]
            c = mw.MINI_WIDTH // 2; keep[c-1:c+2, 2:] = 0; return keep * (np.arange(t.shape[1])[None, :] < 3)
        mw.generate_mini_terrain = floor_only
    cb = cfgmod.callbacks()
    env_cfg = IsaacLayer1EnvConfig(seed=391728405, num_envs=args.envs, spawn_mode="canonical", navigation_mode=True,
                                   world_variant="xy1_mini_128x128", observation_window_tiles=(64, 64),
                                   player_collider=args.player_collider, episode_length_steps=100000, num_workers=1, physx_ccd=not args.no_ccd, **({'crouch_levels': args.crouch_levels} if args.crouch_levels else {}))
    if args.physx_threads:
        import carb.settings; carb.settings.get_settings().set('/persistent/physics/numThreads', int(args.physx_threads))
    started = time.perf_counter()
    env = BatchedIsaacLayer1Env(env_cfg, action_decoder=cb[0], observation_encoder=cb[1], reward_function=cb[2], terrain_mode="independent")
    construct = time.perf_counter() - started
    import carb.settings
    st = carb.settings.get_settings()
    for item in args.carb:
        k, v = item.split("=", 1)
        v = {"true": True, "false": False}.get(v.lower(), v)
        if isinstance(v, str):
            try: v = int(v)
            except ValueError: v = float(v)
        print("CARB", k, st.get(k), "->", v, flush=True); st.set(k, v)
    for k in ("/physics/updateToUsd", "/physics/updateVelocitiesToUsd", "/physics/fabricEnabled", "/physics/suppressReadback", "/physics/cudaDevice", "/physics/numThreads", "/physics/outputVelocitiesLocalSpace"):
        print("CARBSETTING", k, st.get(k), flush=True)
    _orig = env._rebuild_dirty
    def _wrap(idx):
        print("REBUILD", list(idx), [sorted(env.slots[i].physical_dirty) for i in idx], flush=True)
        return _orig(idx)
    env._rebuild_dirty = _wrap
    env.reset(seed=args.seed)
    import omni.physx
    _pi = omni.physx.get_physx_interface()
    for w in range(args.envs):
        try:
            print("PHYSXTR world", w, _pi.get_rigidbody_transformation(f"/World/Env_{w:05d}/CasuPlayer"), flush=True)
        except Exception as e:
            print("PHYSXTR err", str(e)[:120], flush=True)
    print("ROOTPOSW", [[round(float(v), 2) for v in row] for row in env.player.data.root_pos_w.cpu().numpy()], flush=True)
    from pxr import UsdGeom, Usd
    cache = UsdGeom.XformCache()
    for w in range(min(n_env := args.envs, 4)):
        pl = env.stage.GetPrimAtPath(f"/World/Env_{w:05d}/CasuPlayer"); tr = env.stage.GetPrimAtPath(f"/World/Env_{w:05d}/CasUTerrain")
        ch = [c for c in tr.GetChildren()][0]; rc = [c for c in ch.GetChildren()][0]
        print("XFORMS world", w, "player", [round(v, 2) for v in cache.GetLocalToWorldTransform(pl).ExtractTranslation()],
              "terrain root", [round(v, 2) for v in cache.GetLocalToWorldTransform(tr).ExtractTranslation()], "terrain type", tr.GetTypeName() or "(untyped)",
              "first rect world", [round(v, 2) for v in cache.GetLocalToWorldTransform(rc).ExtractTranslation()], flush=True)
    n = args.envs
    if os.environ.get("PROBE_OBS"):
        comp = env.compact_observation()
        for w in range(n):
            sl = env.slots[w]
            px, py = sl.grid.world_to_cell(sl.model.s.x, sl.model.s.y, sl.grid.width, sl.grid.height)
            print("OBS world", w, "slot x,y", round(sl.model.s.x, 2), round(sl.model.s.y, 2), "cell", (px, py), "goal_cell", sl.goal_cell,
                  "nonzero block ids in window", int((comp["block_ids"][w] != 0).sum()), "fluid!=7", int((comp["fluid_ids"][w] != 7).sum()),
                  "goal rel", comp["goal"][w].tolist(), flush=True)
    import omni.physx
    if os.environ.get("PROBE_OWNERS"):
        from pxr import Usd, UsdPhysics
        for w in range(n):
            tot = own = 0; missing = []
            for prim in Usd.PrimRange(env.stage.GetPrimAtPath(f"/World/Env_{w:05d}")):
                if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    tot += 1
                    r = prim.GetRelationship("physics:simulationOwner")
                    if r and r.GetTargets(): own += 1
                    elif len(missing) < 3: missing.append(str(prim.GetPath()))
            print("OWNERS world", w, "physics prims", tot, "with owner", own, "missing sample", missing, flush=True)
    if os.environ.get("PROBE_TELEPORT"):
        import torch
        from pxr import PhysicsSchemaTools, Sdf
        iface = omni.physx.get_physx_simulation_interface()
        ids = [PhysicsSchemaTools.sdfPathToInt(Sdf.Path(x)) for x in ("/physicsScene", "/physicsScene2")]
        view = env.player.root_physx_view
        w = n - 1
        lx, lz = env.slots[w].spawn_world_xy()
        off = env.offsets[w]
        print("SPAWN local", lx, lz, "offset", [float(v) for v in off], flush=True)
        for label, (px, pz) in (("offset-applied", (lx + float(off[0]), lz + float(off[2]))), ("unshifted-local", (lx, lz))):
            pose = torch.zeros((n, 7)); pose[:, 6] = 1.0
            tr = view.get_transforms().clone(); tr[w, 0] = px; tr[w, 1] = 0.0; tr[w, 2] = pz + 0.5; tr[w, 3:] = torch.tensor([0, 0, 0, 1.0])
            view.set_transforms(tr, indices=torch.tensor([w], dtype=torch.int32))
            view.set_velocities(torch.zeros((n, 6)), indices=torch.arange(n, dtype=torch.int32))
            trace = []
            for _ in range(60):
                iface.simulate_scene(ids[1], 0.02, 0.0); iface.fetch_results_scene(ids[1])
                trace.append(round(float(view.get_transforms().numpy()[w, 2]), 2))
            print("TRACE", label, trace[::3], flush=True)
            print("TELEPORT", label, "start z", round(pz + 0.5, 2), "-> z after 30 ticks", round(float(view.get_transforms().numpy()[w, 2]), 3), flush=True)
    if os.environ.get("PROBE_RAY"):
        sq = omni.physx.get_physx_scene_query_interface()
        for w in range(n):
            off = float(env.offsets[w][0]); lx, lz = env.slots[w].spawn_world_xy()
            out = []
            for dx in (-6, -3, 0, 3, 6):
                hit = sq.raycast_closest((lx + off + dx, 0.0, lz + 40.0), (0.0, 0.0, -1.0), 200.0)
                out.append((dx, round(40.0 - float(hit["distance"]), 2) if hit and hit.get("hit") else None, str(hit.get("collision", ""))[-28:] if hit and hit.get("hit") else None))
            print("RAY world", w, "spawn", (lx, lz), out, flush=True)
    zero = {nm: np.zeros(n, dtype=np.int64) for nm in NAMES}
    if os.environ.get("PROBE_SCENES"):
        from pxr import PhysicsSchemaTools, Sdf
        iface = omni.physx.get_physx_simulation_interface()
        view = env.player.root_physx_view
        print("VIEWPATHS", list(view.prim_paths), flush=True)
        print("MAXSHAPES", view.max_shapes, flush=True)
        co = view.get_contact_offsets().numpy(); ro = view.get_rest_offsets().numpy()
        for b in range(co.shape[0]):
            print("OFFSETS body", b, "nonzero contact", int((co[b] != 0).sum()), "first", [round(float(x), 4) for x in co[b][:4]], "rest nonzero", int((ro[b] != 0).sum()), flush=True)
        ids = [PhysicsSchemaTools.sdfPathToInt(Sdf.Path(x)) for x in ("/physicsScene", "/physicsScene2")]
        def pos(): return [round(float(v), 3) for v in view.get_transforms().numpy()[:, 2]]
        print("Z0", pos(), flush=True)
        for which in (0, 1):
            for _ in range(5):
                iface.simulate_scene(ids[which], 0.02, 0.0); iface.fetch_results_scene(ids[which])
            print("Z after stepping scene", which, pos(), flush=True)
    for d in range(4):
        env.step(zero)
        _t = env.player.root_physx_view.get_transforms().numpy()
        print("DEPTH_Y", d, [round(float(v), 4) for v in _t[:, 1]], "quat", [[round(float(q), 3) for q in row[3:]] for row in _t[:2]], flush=True)
        print("WORLDX", d, [round(float(v), 2) for v in _t[:, 0]], "offsets", [float(o) for o in env.offsets[:, 0]], flush=True)
        print("STATE", d, [(round(float(s.model.s.x), 4), round(float(s.model.s.y), 4), round(float(s.model.s.vy), 4)) for s in env.slots], flush=True)
    env.close()

import traceback
try:
    main()
except BaseException:
    traceback.print_exc(); sys.stderr.flush(); raise
finally:
    sys.stdout.flush(); os._exit(0)
