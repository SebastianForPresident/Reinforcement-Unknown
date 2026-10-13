"""Isaac-free harness for the Python-side simulator hot path (controller, ground queries, fluid).

PhysX is replaced by a small deterministic kinematic stand-in so the *same* controller,
query and fluid code that runs inside the vector environment can be profiled and compared
(reference backend vs optimised backend) in seconds without launching Kit.

    python -m casu_grid.perf.controller_harness --envs 16 --macrosteps 30 --profile

It is a profiling/equivalence tool, not a physics validator: engine fidelity is covered by
``isaac_player_ab.py``.
"""
from __future__ import annotations

import argparse, hashlib, time

import numpy as np

from ..body_model import BodyConfig, PolicyAction
from ..env_slot import EnvSlot
from ..ground_query import GroundWorld
from ..isaac_navigation_env import IsaacLayer1EnvConfig
from ..player_runtime import post_physics_phase, pre_physics_phase
from ..reference import load_layer1_reference
from ..world_objects import ground_polygons, load_physics_audit

DT = 0.02
GRAVITY = -55.0
MASS = 49.28
REST_GAP = 0.1155


class KinematicEngine:
    """Semi-implicit Euler with axis-aligned stops against the tile/object query world."""

    def __init__(self, world: GroundWorld, size=(1.6, 5.0)):
        self.world, self.size = world, size
        self.s = [0.0, 0.0, 0.0, 0.0]

    def integrate(self, model, fx, fy):
        s = model.s
        x, y, vx, vy = s.x, s.y, s.vx + fx / MASS * DT, s.vy + (fy / MASS + GRAVITY) * DT
        size = s.col_size
        center = (x + s.col_offset[0], y + s.col_offset[1])
        if vy < 0.0:
            hit = self.world.box_cast(center, size, (0.0, -1.0), -vy * DT + REST_GAP)
            if hit is not None:
                y += -(max(hit.distance - REST_GAP, 0.0)); vy = 0.0
            else:
                y += vy * DT
        else:
            hit = self.world.box_cast(center, size, (0.0, 1.0), vy * DT + 0.01)
            if hit is not None:
                vy = 0.0
            else:
                y += vy * DT
        if vx != 0.0:
            d = (1.0 if vx > 0 else -1.0, 0.0)
            hit = self.world.box_cast((x + s.col_offset[0], y + s.col_offset[1]), size, d, abs(vx) * DT + 0.01)
            if hit is not None:
                x += d[0] * max(hit.distance - 0.01, 0.0); vx = 0.0
            else:
                x += vx * DT
        return x, y, vx, vy


def build_slots(n: int, grid_xy=(483, 1010), include_objects=True):
    cfg = IsaacLayer1EnvConfig(num_envs=n, spawn_mode="fixture", fixture_grid_xy=grid_xy)
    reference = load_layer1_reference()
    polygons, doors = [], []
    if include_objects:
        objects, _ = load_physics_audit()
        doors = [c for c in objects.colliders if "shuttledoor" in c.name]
        polygons = ground_polygons([c for c in objects.colliders if "shuttledoor" not in c.name])
    body = BodyConfig(step_up_clearance=cfg.step_up_clearance)
    slots = [EnvSlot(cfg, reference, object_polygons=polygons, door_defs=doors,
                     fluid_seed=cfg.fluid_rng_seed + i, body_config=body) for i in range(n)]
    return cfg, slots


def tape(step: int, rng, n: int, workload: str):
    if workload == "random":
        return rng.integers(0, 3, n), rng.integers(0, 2, n), rng.integers(0, 2, n)
    return np.full(n, 2), np.full(n, 1 if step % 4 == 3 else 0), np.zeros(n, int)


def run(slots, macrosteps: int, workload="random", seed=0, record=None):
    """Advance every slot ``macrosteps`` decisions; returns per-phase wall seconds."""
    n = len(slots)
    engines = [KinematicEngine(slot.ground_world) for slot in slots]
    rng = np.random.default_rng(seed)
    clock = time.perf_counter
    timers = {"pre": 0.0, "engine": 0.0, "post": 0.0}
    for step in range(macrosteps):
        move, jump, crouch = tape(step, rng, n, workload)
        for i, slot in enumerate(slots):
            slot.model.apply_actions(latch=PolicyAction(move=int(move[i]) - 1, jump=int(jump[i]), crouch=int(crouch[i])))
        for tick in range(10):
            forces = []
            t0 = clock()
            for slot in slots:
                forces.append(pre_physics_phase(slot.model, pre_tick_hook=slot.pre_tick, post_fixed_hook=slot.post_fixed)[0])
            t1 = clock()
            for slot, eng, (fx, fy) in zip(slots, engines, forces):
                s = slot.model.s
                s.x, s.y, s.vx, s.vy = eng.integrate(slot.model, fx, fy)
            t2 = clock()
            for slot in slots:
                post_physics_phase(slot.model, slot.ground_world, slot.runtime, tick + 1, 10, post_tick_hook=slot.post_tick)
            t3 = clock()
            timers["pre"] += t1 - t0; timers["engine"] += t2 - t1; timers["post"] += t3 - t2
            if record is not None:
                record.append([(s.model.s.x, s.model.s.y, s.model.s.vx, s.model.s.vy) for s in slots])
    return timers


def digest(slots) -> str:
    h = hashlib.sha256()
    for slot in slots:
        s = slot.model.s
        h.update(repr((s.x, s.y, s.vx, s.vy, s.grounded, s.crouch_amount, s.jump_cooldown)).encode())
        h.update(slot.grid.fluids.tobytes())
    return h.hexdigest()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--macrosteps", type=int, default=30)
    p.add_argument("--workload", default="random")
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()
    cfg, slots = build_slots(a.envs)
    run(slots, 2, a.workload)  # warm up
    if a.profile:
        import cProfile, pstats
        pr = cProfile.Profile(); pr.enable()
    t = time.perf_counter()
    timers = run(slots, a.macrosteps, a.workload)
    wall = time.perf_counter() - t
    if a.profile:
        pr.disable(); pstats.Stats(pr).sort_stats("tottime").print_stats(25)
    ticks = a.macrosteps * 10 * a.envs
    print(f"{a.envs} envs: {ticks / wall:.0f} controller ticks/s (engine stub included)  phases:",
          {k: f"{v / wall:.2f}" for k, v in timers.items()}, "digest", digest(slots)[:16])
