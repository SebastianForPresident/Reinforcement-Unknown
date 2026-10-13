"""Compiled per-world tick runtime for the batched environment (backend ``numba``).

``FastSlot`` drives one :class:`~casu_grid.env_slot.EnvSlot` through a macrostep with the packed-state
kernels of :mod:`casu_grid.controller_kernels`. The reference path stays the source of truth:

* the slot's ``BodyState`` is packed once at the start of a macrostep and unpacked once at the end,
  so everything outside the tick loop (observations, snapshots, resets, doors) keeps using the
  ordinary dataclass;
* during the tick loop the packed row ``S`` is authoritative and the engine adapter reads/writes its
  position and velocity columns directly;
* a tick phase whose queries touch a convex static-object polygon is re-run by the reference
  controller on the pre-phase state (``fallbacks`` counts them), so results never differ.

Rows of one batch live in one ``(worlds, N_STATE)`` array so the environment can stage PhysX
input/output with whole-column NumPy operations.
"""
from __future__ import annotations

import numpy as np

from . import controller_kernels as CK
from .body_model import PolicyAction
from .env_slot import LIQUID_COEFFICIENTS
from .player_runtime import CONTACT_REACH, post_physics_phase, pre_physics_phase
from .state import SeededDecisionStream

_RAGDOLL_CODES = {None: 0.0, "hit_ground_hard": 1.0, "jumppad": 2.0}
_RAGDOLL_NAMES = {v: k for k, v in _RAGDOLL_CODES.items()}


def _liquid_tables():
    buoyancy, drag = np.zeros(8), np.zeros(8)
    for fluid_id, (b, d) in LIQUID_COEFFICIENTS.items():
        buoyancy[fluid_id], drag[fluid_id] = b, d
    return buoyancy, drag


_BUOYANCY, _DRAG = _liquid_tables()


def _collider_ids(touching) -> list[int]:
    """Encode runtime.touching entries (``("chunk", cx, cy)`` / ``("object", index)``) for the kernels."""
    return [((t[1] << 16) | t[2]) if t[0] == "chunk" else CK.OBJECT_BASE + t[1] for t in touching]


def _decode_ids(ids) -> frozenset:
    return frozenset(("chunk", int(c) >> 16, int(c) & 0xFFFF) if c < CK.OBJECT_BASE
                     else ("object", int(c) - CK.OBJECT_BASE) for c in ids)


class FastSlot:
    def __init__(self, slot, row: np.ndarray, config_vector: np.ndarray, ticks: int, result_row: np.ndarray | None = None):
        self.slot, self.S, self.C = slot, row, config_vector
        self.R = np.zeros(4) if result_row is None else result_row
        self.prev = np.zeros(64, dtype=np.int64)
        self.out = np.zeros(64, dtype=np.int64)
        self.ticks = ticks
        self.G = np.zeros(CK.N_GRID, dtype=np.int64)          # grid scheduler state during a macrostep
        self.S_before_fixed = np.zeros(CK.N_STATE)
        self.fallbacks = 0
        self.liquid_before = slot.cfg.liquid_affect_phase == "before_controller"

    # ------------------------------------------------------------------ dataclass <-> packed
    def pack(self) -> None:
        s, S = self.slot.model.s, self.S
        S[CK.X], S[CK.Y], S[CK.VX], S[CK.VY] = s.x, s.y, s.vx, s.vy
        S[CK.GROUNDED], S[CK.MOVE_X], S[CK.MOVE_Y], S[CK.CROUCHING] = s.grounded, s.move_x, s.move_y, s.crouching
        S[CK.CROUCH] = s.crouch_amount
        S[CK.COL_W], S[CK.COL_H] = s.col_size
        S[CK.COFF_X], S[CK.COFF_Y] = s.col_offset
        S[CK.ENDED_JUMP], S[CK.WANTS_JUMP], S[CK.JUMP_CD] = s.ended_jump, s.wants_jump_input, s.jump_cooldown
        S[CK.T_GROUNDED], S[CK.T_SLID_L], S[CK.T_SLID_R] = (
            s.time_since_grounded, s.time_since_slid_left, s.time_since_slid_right)
        S[CK.T_SLID_FOR] = s.time_slid_for
        S[CK.SLIDING_L], S[CK.SLIDING_R] = s.sliding_left, s.sliding_right
        S[CK.FIRST_WALL], S[CK.LAST_RIGHT] = s.first_wall_jump, s.last_jumped_on_right_wall
        S[CK.GN_X], S[CK.GN_Y] = s.ground_normal
        S[CK.LAST_VX], S[CK.LAST_VY] = s.last_time_step_vx, s.last_time_step_vy
        S[CK.SLIPPERY], S[CK.STANDING_ON], S[CK.GRAV] = s.slippery, s.standing_on, s.gravity_scale
        S[CK.LOOK_X], S[CK.LOOK_Y] = s.look_x, s.look_y
        S[CK.RAGDOLL] = _RAGDOLL_CODES[s.ragdoll_event]
        la, ca = s.last_action, s.current_action
        S[CK.LAST_ACTION:CK.LAST_ACTION + 6] = (la.move, la.jump, la.vertical, la.crouch, la.look_dx, la.look_dy)
        S[CK.CUR_ACTION:CK.CUR_ACTION + 6] = (ca.move, ca.jump, ca.vertical, ca.crouch, ca.look_dx, ca.look_dy)

    def unpack(self) -> None:
        s, S = self.slot.model.s, self.S
        s.x, s.y, s.vx, s.vy = float(S[CK.X]), float(S[CK.Y]), float(S[CK.VX]), float(S[CK.VY])
        s.grounded, s.move_x, s.move_y = bool(S[CK.GROUNDED]), float(S[CK.MOVE_X]), float(S[CK.MOVE_Y])
        s.crouching, s.crouch_amount = bool(S[CK.CROUCHING]), float(S[CK.CROUCH])
        s.col_size = (float(S[CK.COL_W]), float(S[CK.COL_H]))
        s.col_offset = (float(S[CK.COFF_X]), float(S[CK.COFF_Y]))
        s.ended_jump, s.wants_jump_input, s.jump_cooldown = (
            bool(S[CK.ENDED_JUMP]), bool(S[CK.WANTS_JUMP]), float(S[CK.JUMP_CD]))
        s.time_since_grounded, s.time_since_slid_left, s.time_since_slid_right = (
            float(S[CK.T_GROUNDED]), float(S[CK.T_SLID_L]), float(S[CK.T_SLID_R]))
        s.time_slid_for = float(S[CK.T_SLID_FOR])
        s.sliding_left, s.sliding_right = bool(S[CK.SLIDING_L]), bool(S[CK.SLIDING_R])
        s.first_wall_jump, s.last_jumped_on_right_wall = bool(S[CK.FIRST_WALL]), bool(S[CK.LAST_RIGHT])
        s.ground_normal = (float(S[CK.GN_X]), float(S[CK.GN_Y]))
        s.last_time_step_vx, s.last_time_step_vy = float(S[CK.LAST_VX]), float(S[CK.LAST_VY])
        s.slippery, s.standing_on, s.gravity_scale = bool(S[CK.SLIPPERY]), int(S[CK.STANDING_ON]), float(S[CK.GRAV])
        s.look_x, s.look_y = float(S[CK.LOOK_X]), float(S[CK.LOOK_Y])
        s.ragdoll_event = _RAGDOLL_NAMES[float(S[CK.RAGDOLL])]
        la, ca = S[CK.LAST_ACTION:CK.LAST_ACTION + 6], S[CK.CUR_ACTION:CK.CUR_ACTION + 6]
        s.last_action = PolicyAction(*(int(v) for v in la))
        s.current_action = PolicyAction(*(int(v) for v in ca))

    # ----------------------------------------------------------------------- macrostep
    def _pack_grid(self) -> None:
        grid, G = self.slot.grid, self.G
        G[CK.G_TILE_COOLDOWN], G[CK.G_WATER_MOVE], G[CK.G_PUSH_EVENTS] = (
            grid.tile_cooldown, grid.water_move_count, grid.push_events)
        G[CK.G_SIM_INDEX], G[CK.G_TICK] = grid.sim_index, grid.tick
        G[CK.G_CAM_X], G[CK.G_CAM_Y] = grid.camera_x, grid.camera_y
        G[CK.G_RNG_POSITION] = 0          # draws made during this macrostep

    def _unpack_grid(self) -> None:
        grid, G = self.slot.grid, self.G
        grid.tile_cooldown, grid.water_move_count, grid.push_events = (
            int(G[CK.G_TILE_COOLDOWN]), int(G[CK.G_WATER_MOVE]), int(G[CK.G_PUSH_EVENTS]))
        grid.sim_index, grid.tick = int(G[CK.G_SIM_INDEX]), int(G[CK.G_TICK])
        grid.camera_x, grid.camera_y = int(G[CK.G_CAM_X]), int(G[CK.G_CAM_Y])
        grid.rng.position += int(G[CK.G_RNG_POSITION])
        G[CK.G_RNG_POSITION] = 0

    def begin_macrostep(self, action: PolicyAction) -> None:
        if not isinstance(self.slot.grid.rng, SeededDecisionStream):
            raise RuntimeError("the compiled runtime needs the seeded fluid decision stream")
        self._pack_grid()
        self.pack()
        CK.latch_actions(self.S, self.C, action.move, action.jump, action.vertical, action.crouch,
                         action.look_dx, action.look_dy)

    def end_macrostep(self) -> None:
        self.unpack()
        self._unpack_grid()

    # ---------------------------------------------------------------------- tick phases
    def _liquid(self):
        """``EnvSlot.liquid_state`` on the packed position."""
        grid, S = self.slot.grid, self.S
        cell_x, cell_y = grid.world_to_cell(S[CK.X], S[CK.Y], grid.width, grid.height)
        return grid.get_liquid(cell_x, cell_y)

    def _apply_liquid(self) -> None:
        from .env_slot import LIQUID_COEFFICIENTS
        fluid_id = self._liquid()
        if fluid_id > 0:
            buoyancy, drag = LIQUID_COEFFICIENTS.get(fluid_id, (0.0, 0.0))
            S = self.S
            S[CK.VX] = S[CK.VX] * drag
            S[CK.VY] = S[CK.VY] * drag + buoyancy

    def pre_tick(self) -> None:
        """``EnvSlot.pre_tick`` + ``Body.FixedUpdate`` + liquid hooks; the engine force lands in ``R[0:2]``."""
        slot, S, G = self.slot, self.S, self.G
        slot.apply_due_edits()
        grid, world = slot.grid, slot.ground_world
        flag = CK.pre_tick(S, self.S_before_fixed, self.C, G, grid.fluids, grid.blocks, world.blocks,
                           world.poly_pack, grid.rng.rng, grid.fluid_push_enabled,
                           self.liquid_before, _BUOYANCY, _DRAG, self.R)
        slot.physics_ticks += 1
        slot.was_water = bool(self.R[3])
        if flag != CK.OK:
            # polygon near the player (or hypot boundary): redo only FixedUpdate with the reference
            # controller from the state the hooks produced
            self.fallbacks += 1
            S[:] = self.S_before_fixed
            self.unpack()
            (fx, fy), _ = pre_physics_phase(slot.model)
            self.pack()
            self.R[0], self.R[1] = fx, fy
            if not self.liquid_before:
                self._apply_liquid()

    def post_tick(self, completed: int) -> None:
        """Collision events, Update/apply schedule and ``EnvSlot.post_tick`` for the engine result in ``S``."""
        slot, S = self.slot, self.S
        runtime, world = slot.runtime, slot.ground_world
        prev_ids = _collider_ids(runtime.touching)
        n_prev = len(prev_ids)
        if n_prev > len(self.prev):
            flag = CK.FALLBACK
        else:
            self.prev[:n_prev] = prev_ids
            backup = S.copy()
            flag = CK.post_physics(S, self.C, world.blocks, world.poly_pack, self.prev, n_prev, self.out,
                                   self.R, completed, self.ticks)
        if flag != CK.OK:
            self.fallbacks += 1
            if n_prev <= len(self.prev):
                S[:] = backup
            self.unpack()
            post_physics_phase(slot.model, world, runtime, completed, self.ticks, post_tick_hook=slot.post_tick)
            self.pack()
            return
        count = int(self.R[0])
        ids = self.out[:count]
        runtime.touching = _decode_ids(ids)
        doors = slot.shuttle_doors
        if doors is not None:
            x0 = S[CK.X] - S[CK.COL_W] / 2
            y0 = S[CK.Y] + S[CK.COFF_Y] - S[CK.COL_H] / 2
            if doors.tick(slot.cfg.physics_dt, (x0, y0, x0 + S[CK.COL_W], y0 + S[CK.COL_H])):
                slot.apply_door_polygons()
                if slot.on_doors_changed is not None:
                    slot.on_doors_changed()
        if S[CK.RAGDOLL] == 0.0:
            for c in ids:
                if c >= CK.OBJECT_BASE and "jumppad" in world.polygons[int(c) - CK.OBJECT_BASE][0]:
                    S[CK.RAGDOLL] = 2.0
                    break
