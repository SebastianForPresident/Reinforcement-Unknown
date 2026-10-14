"""Profile the XY1 mini-world vector environment (one process) under near-uniform random policy actions.

    python casu_grid/perf/xy1_env_profile.py --envs 4 --steps 60 --cprofile --output out.json
"""
from __future__ import annotations
import argparse, cProfile, io, json, pstats, sys, time
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


def run_k(env, rng, args, set_levels):
    import statistics
    for k in (1, 8):
        t_set, t_step = [], []
        for it in range(150):
            who = rng.permutation(args.envs)[:k]
            levels = [c.active_level for c in env.colliders]
            for i in who:
                levels[i] = int((levels[i] + rng.integers(1, 4)) % 33)
            a = time.perf_counter(); set_levels(env.colliders, levels); b = time.perf_counter()
            env.sim.step(render=False); c = time.perf_counter()
            if it >= 20: t_set.append(b - a); t_step.append(c - b)
        print("K", k, "set_levels us", round(1e6 * statistics.median(t_set), 1), "sim.step us", round(1e6 * statistics.median(t_step), 1), flush=True)


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
    rng = np.random.default_rng(args.seed)
    env.reset(seed=args.seed)
    from casu_grid.isaac_player_body import set_levels
    for _ in range(5):
        env.step({n: rng.integers(0, s, args.envs) for n, s in zip(NAMES, SIZES)})
    import statistics
    out = {}
    for rep in range(6):
        a = time.perf_counter()
        for _ in range(200):
            env.sim.step(render=False)
        print("RESTSTEP us", round(1e6 * (time.perf_counter() - a) / 200, 1), flush=True)
    from pxr import Sdf
    layer = env.stage.GetEditTarget().GetLayer() if hasattr(env, "stage") else None
    print("LAYER", layer is not None, flush=True)
    import casu_grid.isaac_player_body as ipb
    def variant_sdf(self, level):
        if level == self.active_level:
            return
        L = self.stage.GetEditTarget().GetLayer()
        for name in self.shape_names:
            L.GetAttributeAtPath(f"{self.root_path}/Level_{self.active_level}/{name}.physics:collisionEnabled").default = False
        for name in self.shape_names:
            L.GetAttributeAtPath(f"{self.root_path}/Level_{level}/{name}.physics:collisionEnabled").default = True
        self.active_level = level
    def variant_cached(self, level):
        if level == self.active_level:
            return
        c = self.__dict__.setdefault("_specs", {})
        L = self.stage.GetEditTarget().GetLayer()
        for lv, val in ((self.active_level, False), (level, True)):
            if lv not in c:
                c[lv] = [L.GetAttributeAtPath(f"{self.root_path}/Level_{lv}/{n}.physics:collisionEnabled") for n in self.shape_names]
            for spec in c[lv]:
                spec.default = val
        self.active_level = level
    orig = ipb.RoundedBoxCollider.set_level
    from pxr import Sdf as _S
    def variant_dummy(self, level):
        # same number of Sdf writes, but to a custom attribute no listener cares about
        if level == self.active_level:
            return
        c = self.__dict__.setdefault("_dummy", None)
        if c is None:
            L = self.stage.GetEditTarget().GetLayer()
            prim = L.GetPrimAtPath(self.root_path)
            c = [_S.AttributeSpec(prim, f"dummy{i}", _S.ValueTypeNames.Bool) for i in range(12)]
            self._dummy = c
        v = bool(level & 1)
        for spec in c:
            spec.default = v
        self.active_level = level
    def variant_one(self, level):
        if level == self.active_level:
            return
        c = self.__dict__.setdefault("_specs1", {})
        L = self.stage.GetEditTarget().GetLayer()
        for lv, val in ((self.active_level, False), (level, True)):
            if lv not in c:
                c[lv] = L.GetAttributeAtPath(f"{self.root_path}/Level_{lv}/CrossX.physics:collisionEnabled")
            c[lv].default = val
        self.active_level = level
    def variant_three(self, level):
        if level == self.active_level:
            return
        c = self.__dict__.setdefault("_specs3", {})
        L = self.stage.GetEditTarget().GetLayer()
        for lv, val in ((self.active_level, False), (level, True)):
            if lv not in c:
                c[lv] = [L.GetAttributeAtPath(f"{self.root_path}/Level_{lv}/{n}.physics:collisionEnabled") for n in ("CrossX", "CrossZ", "CornerNE")]
            for spec in c[lv]:
                spec.default = val
        self.active_level = level
    for label, fn in (("sdf_cached", variant_cached), ("one_shape", variant_one), ("three_shapes", variant_three), ("sdf_cached", variant_cached)):
        ipb.RoundedBoxCollider.set_level = fn
        print("VARIANT", label, flush=True)
        run_k(env, rng, args, ipb.set_levels)
    env.close()
    return
    for k in (0, 1, 4, args.envs):
        t_set, t_step = [], []
        for it in range(150):
            who = rng.permutation(args.envs)[:k]
            levels = [c.active_level for c in env.colliders]
            for i in who:
                levels[i] = int((levels[i] + rng.integers(1, 4)) % 33)
            a = time.perf_counter(); set_levels(env.colliders, levels); b = time.perf_counter()
            env.sim.step(render=False); c = time.perf_counter()
            if it >= 20: t_set.append(b - a); t_step.append(c - b)
        out[k] = (1e6 * statistics.median(t_set), 1e6 * statistics.median(t_step))
        print("K", k, "set_levels us", round(out[k][0], 1), "sim.step us", round(out[k][1], 1), flush=True)
    env.close()

import traceback
try:
    main()
except BaseException:
    traceback.print_exc(); sys.stderr.flush(); raise
finally:
    sys.stdout.flush(); app.close()
