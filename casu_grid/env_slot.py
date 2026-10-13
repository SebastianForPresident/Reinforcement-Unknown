"""Engine-agnostic state of one independent canonical CasU world ("slot").

A slot owns everything that is not rigid-body integration: the discrete grid and
its seeded RNG, the source-ordered body model, the macrostep runtime, scripted
spawn-shuttle doors, liquid and hazard hooks, episode counters, and its
snapshot/restore. Single and batched Isaac environments both wrap slots, so there
is no shared mutable state between worlds.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .body_model import BodyConfig, BodyState, PolicyAction, StandingBodyModel
from .ground_query import GroundWorld
from .player_runtime import RuntimeState
from .shuttle_doors import ShuttleDoors
from .state import SeededDecisionStream

LIQUID_COEFFICIENTS = {1: (0.6, 0.915), 2: (2.0, 0.8), 3: (0.5, 0.75),
                       4: (1.0, 0.8), 5: (0.6, 0.915), 6: (3.0, 0.7)}  # id -> (buoyancy, drag)
SHUTTLE_TRIGGERS = ((3.5, 506.5, 4.5, 507.5), (-4.5, 506.5, -3.5, 507.5))


class EnvSlot:
    def __init__(self, cfg, reference, *, object_polygons=(), door_defs=(), fluid_seed: int,
                 body_config: BodyConfig):
        self.cfg, self.reference = cfg, reference
        self.fluid_seed = int(fluid_seed)
        self.grid = reference.new_state()
        self.grid.rng = SeededDecisionStream(self.fluid_seed)
        self.grid_initial = self.grid.snapshot()
        self.grid_initial["dirty_chunks"] = []
        self.body_config = body_config
        polygons = list(object_polygons)
        self.door_defs = tuple(sorted(door_defs, key=lambda c: c.center[0]))
        self.door_polygon_slots: list[int] = []
        for door in self.door_defs:
            self.door_polygon_slots.append(len(polygons))
            polygons.append((door.name, self._door_polygon(door.center, door)))
        self.ground_world = GroundWorld(self.grid.blocks, polygons)
        self.model = StandingBodyModel(self.ground_world, body_config)
        self.runtime = RuntimeState()
        self.shuttle_doors = (ShuttleDoors(SHUTTLE_TRIGGERS, tuple(d.center for d in self.door_defs))
                              if self.door_defs else None)
        self.on_doors_changed = None       # engine callback: poses of the door bodies changed
        self.elapsed_steps = 0
        self.physics_ticks = 0
        self.was_water = False
        self.last_state: dict | None = None
        self.model.s = self.initial_body_state()

    # ----------------------------------------------------------------- geometry
    @staticmethod
    def _door_polygon(center, door):
        hw, hh = door.size[0] / 2, door.size[1] / 2
        cx, cy = center
        return np.array([(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)])

    def apply_door_polygons(self):
        if self.shuttle_doors is None:
            return
        for door, slot, center in zip(self.door_defs, self.door_polygon_slots, self.shuttle_doors.centers()):
            self.ground_world.polygons[slot] = (door.name, self._door_polygon(center, door))

    # --------------------------------------------------------------------- state
    def spawn_world_xy(self):
        if self.cfg.spawn_mode == "canonical":
            xy = self.reference.spawn_world_xy
            return float(xy[0]), float(xy[1])
        x, y = self.cfg.fixture_grid_xy
        if not self.grid.get_block(x, y):
            raise ValueError(f"fixture support cell {(x, y)} is empty")
        # tile-centre x, collider standing on the support top (rest gap 0.1155)
        return x - 511.5, y - 511.0 + self.body_config.orig_col_size[1] / 2 + 0.1155

    def initial_body_state(self) -> BodyState:
        x, y = self.spawn_world_xy()
        if self.cfg.spawn_mode == "canonical":
            # Measured at the first policy boundary of the canonical spawn: crouchAmount 1.0
            # and grounded; the collider is still standing-sized until the first Update.
            return BodyState(x=x, y=y, grounded=True, crouch_amount=1.0,
                             col_size=self.body_config.orig_col_size, col_offset=(0.0, 0.0))
        return BodyState(x=x, y=y, grounded=True, col_size=self.body_config.orig_col_size,
                         col_offset=(0.0, 0.0), first_wall_jump=True)

    def body_dict(self) -> dict:
        s = self.model.s
        return {"x": float(s.x), "y": float(s.y), "vx": float(s.vx), "vy": float(s.vy)}

    def support_info(self):
        s = self.model.s
        return bool(s.grounded), bool(s.slippery)

    def default_observation_vector(self) -> np.ndarray:
        body, s = self.body_dict(), self.model.s
        return np.asarray([body["x"], body["y"], body["vx"], body["vy"], float(s.grounded),
                           s.crouch_amount, max(s.jump_cooldown, 0.0),
                           float(self.elapsed_steps) / self.cfg.episode_length_steps], dtype=np.float32)

    # ------------------------------------------------------------------- lifecycle
    def reset_logical(self) -> bool:
        """Restore grid/body/doors to the canonical start. Returns True if terrain changed."""
        before = self.grid.blocks.copy()
        self.grid.restore(copy.deepcopy(self.grid_initial))
        changed = not np.array_equal(before, self.grid.blocks)
        if changed:
            self.grid.dirty_chunks = {
                (cx, cy)
                for cx in range(self.grid.width // 64) for cy in range(self.grid.height // 64)
                if not np.array_equal(before[cx*64:(cx+1)*64, cy*64:(cy+1)*64],
                                      self.grid.blocks[cx*64:(cx+1)*64, cy*64:(cy+1)*64])}
        self.grid.rng = SeededDecisionStream(self.fluid_seed)
        self.model.s = self.initial_body_state()
        self.runtime = RuntimeState()
        if self.shuttle_doors is not None:
            self.shuttle_doors.restore({"activated": False, "progress": 0.0, "destroyed": False})
            self.apply_door_polygons()
        self.elapsed_steps = 0
        self.physics_ticks = 0
        self.was_water = False
        return changed

    def snapshot_logical(self) -> dict:
        return {
            "grid": self.grid.snapshot(), "body_model": copy.deepcopy(self.model.s),
            "runtime_touching": sorted(self.runtime.touching),
            "shuttle_doors": None if self.shuttle_doors is None else self.shuttle_doors.snapshot(),
            "elapsed_steps": self.elapsed_steps, "physics_ticks": self.physics_ticks,
            "was_water": self.was_water, "grid_tick": self.grid.tick,
            "collider_revision": self.grid.block_revision, "fluid_rng_seed": self.fluid_seed,
        }

    def restore_logical(self, snap: dict) -> bool:
        """Restore a logical snapshot. Returns True if terrain changed (colliders need a rebuild)."""
        before = self.grid.blocks.copy()
        before_revision = self.grid.block_revision
        self.grid.restore(snap["grid"])
        changed = before_revision != self.grid.block_revision or not np.array_equal(before, self.grid.blocks)
        if changed:
            self.grid.dirty_chunks = {
                (cx, cy)
                for cx in range(self.grid.width // 64) for cy in range(self.grid.height // 64)
                if not np.array_equal(before[cx*64:(cx+1)*64, cy*64:(cy+1)*64],
                                      self.grid.blocks[cx*64:(cx+1)*64, cy*64:(cy+1)*64])}
        self.model.s = copy.deepcopy(snap["body_model"])
        self.runtime = RuntimeState(frozenset(tuple(t) for t in snap["runtime_touching"]))
        if self.shuttle_doors is not None and snap["shuttle_doors"] is not None:
            self.shuttle_doors.restore(snap["shuttle_doors"])
            self.apply_door_polygons()
        self.elapsed_steps = int(snap["elapsed_steps"])
        self.physics_ticks = int(snap["physics_ticks"])
        self.was_water = bool(snap.get("was_water", False))
        return changed

    # ------------------------------------------------------------------- tick hooks
    def liquid_state(self):
        s = self.model.s
        cell_x, cell_y = self.grid.world_to_cell(s.x, s.y, self.grid.width, self.grid.height)
        fluid_id = self.grid.get_liquid(cell_x, cell_y)
        buoyancy, drag = LIQUID_COEFFICIENTS.get(fluid_id, (0.0, 0.0))
        return fluid_id > 0, buoyancy, drag

    def apply_liquid(self, model):
        in_water, buoyancy, drag = self.liquid_state()
        if in_water:
            model.s.vx = model.s.vx * drag
            model.s.vy = model.s.vy * drag + buoyancy
        return in_water

    def pre_tick(self, model):
        s = model.s
        in_water, _, _ = self.liquid_state()
        self.was_water = in_water
        if in_water:
            # Body.Jump accepts bodyAffect.wasWater as an alternate support condition.
            s.time_since_grounded = 0.0
            s.first_wall_jump = True
        # PlayerCamera follows the standing body and supplies its position to
        # FluidManager.SimulationRangeIndex; its Update lerp is collapsed to the latest pose.
        self.grid.camera_x, self.grid.camera_y = self.grid.world_to_cell(
            s.x, s.y, self.grid.width, self.grid.height)
        self.grid.simulation_step()
        self.physics_ticks += 1
        if self.cfg.liquid_affect_phase == "before_controller":
            self.apply_liquid(model)

    def post_fixed(self, model):
        if self.cfg.liquid_affect_phase == "after_controller":
            self.apply_liquid(model)

    def post_tick(self, model, touched):
        s = model.s
        if self.shuttle_doors is not None:
            x0, y0 = s.x - s.col_size[0] / 2, s.y + s.col_offset[1] - s.col_size[1] / 2
            if self.shuttle_doors.tick(self.cfg.physics_dt, (x0, y0, x0 + s.col_size[0], y0 + s.col_size[1])):
                self.apply_door_polygons()
                if self.on_doors_changed is not None:
                    self.on_doors_changed()
        if s.ragdoll_event is None:
            for kind, index in (t for t in touched if t[0] == "object"):
                if "jumppad" in self.ground_world.polygons[index][0]:
                    s.ragdoll_event = "jumppad"

    # -------------------------------------------------------------------- outcome
    def finish_step(self, reward_function):
        """Episode bookkeeping after a macrostep: returns (obs-independent) outcome tuple."""
        self.elapsed_steps += 1
        current = self.body_dict()
        ragdoll = self.model.s.ragdoll_event
        out_of_bounds = bool(abs(current["x"]) > 512.0 or abs(current["y"]) > 512.0)
        terminated = bool(out_of_bounds or ragdoll)
        truncated = self.elapsed_steps >= self.cfg.episode_length_steps
        reward, terms = reward_function(self.last_state, current, terminated, truncated)
        self.last_state = current
        reason = ("ragdoll:" + ragdoll) if ragdoll else "world_bounds" if out_of_bounds \
            else "time_limit" if truncated else None
        return reward, terms, terminated, truncated, reason, current
