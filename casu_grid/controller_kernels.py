"""Numba port of one world's standing-controller tick (backend ``numba``).

Statement-for-statement port of :mod:`casu_grid.body_model` (``StandingBodyModel``) and the
per-tick schedule of :mod:`casu_grid.player_runtime`, operating on a packed float64 state vector
(layout below) so a whole tick costs two compiled calls instead of ~100 Python attribute operations.
Floating-point operation order is kept identical to the reference, and
``tests/test_controller_kernels.py`` compares the packed state with the reference bit for bit.

Scope and fallback
  * Tile queries run in the kernels (:mod:`casu_grid.kernels`). Convex static-object polygons keep
    the reference arithmetic in Python: when any query's bounding box touches a polygon's, the
    kernel returns ``FALLBACK`` and the caller discards the packed result and re-runs that phase
    with the reference controller on the untouched state. Output is therefore identical in every case.
  * ``try_step_up``'s ``math.hypot(look - position)`` is compared against -0.9; when it is within
    1e-9 of that boundary (hypot rounding could decide differently) the kernel also falls back.

State vector layout (``S``) and config vector (``C``) indices are the module constants below.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from . import kernels as K

KERNEL_VERSION = 2
OK, FALLBACK = 0, 1

# ---- state layout (float64; booleans are 0.0/1.0) -------------------------------------------
X, Y, VX, VY = 0, 1, 2, 3
GROUNDED, MOVE_X, MOVE_Y, CROUCHING, CROUCH = 4, 5, 6, 7, 8
COL_W, COL_H, COFF_X, COFF_Y = 9, 10, 11, 12
ENDED_JUMP, WANTS_JUMP, JUMP_CD = 13, 14, 15
T_GROUNDED, T_SLID_L, T_SLID_R, T_SLID_FOR = 16, 17, 18, 19
SLIDING_L, SLIDING_R, FIRST_WALL, LAST_RIGHT = 20, 21, 22, 23
GN_X, GN_Y, LAST_VX, LAST_VY = 24, 25, 26, 27
SLIPPERY, STANDING_ON, GRAV, LOOK_X, LOOK_Y, RAGDOLL = 28, 29, 30, 31, 32, 33
LAST_ACTION, CUR_ACTION = 34, 40    # six values each: move, jump, vertical, crouch, look_dx, look_dy
N_STATE = 46

# ---- config layout ---------------------------------------------------------------------------
(C_MAX_SPEED, C_MOVE_FORCE, C_JUMP_SPEED, C_SLOWDOWN, C_WALL_SLIDE, C_MASS, C_ORIG_W, C_ORIG_H,
 C_EDGE, C_GRAVITY, C_LEG, C_STAMINA, C_JUMP_CD, C_COYOTE, C_WJ_WINDOW, C_UPDATE_DT, C_AUTO_JUMP,
 C_N_SLIPPERY) = range(18)
C_SLIPPERY0 = 18            # up to 8 slippery block ids
C_CLEARANCE = 26
N_CONFIG = 27

CONTACT_REACH = 0.125
BUCKET = 8.0          # must match ground_query.BUCKET / BUCKETS
BUCKETS = 128


def pack_config(cfg) -> np.ndarray:
    out = np.zeros(N_CONFIG)
    out[C_MAX_SPEED], out[C_MOVE_FORCE], out[C_JUMP_SPEED] = cfg.max_speed, cfg.move_force, cfg.jump_speed
    out[C_SLOWDOWN], out[C_WALL_SLIDE], out[C_MASS] = cfg.slowdown_amount, cfg.wall_slide_slowdown, cfg.mass
    out[C_ORIG_W], out[C_ORIG_H], out[C_EDGE] = cfg.orig_col_size[0], cfg.orig_col_size[1], cfg.edge_radius
    out[C_GRAVITY], out[C_LEG], out[C_STAMINA] = cfg.gravity_y, cfg.leg_speed_mult, cfg.actual_stamina_strength
    out[C_JUMP_CD], out[C_COYOTE], out[C_WJ_WINDOW] = cfg.jump_cooldown, cfg.coyote_time, cfg.wall_jump_window
    out[C_UPDATE_DT], out[C_AUTO_JUMP] = cfg.update_dt, float(cfg.auto_jump)
    blocks = sorted(cfg.slippery_blocks)
    if len(blocks) > 8:
        raise ValueError("at most 8 slippery block ids are supported by the compiled controller")
    out[C_N_SLIPPERY] = len(blocks)
    out[C_SLIPPERY0:C_SLIPPERY0 + len(blocks)] = blocks
    out[C_CLEARANCE] = cfg.step_up_clearance
    return out


# ---------------------------------------------------------------------------- query helpers
@njit(cache=True)
def _poly_near(aabb, bucket, x0, y0, x1, y1):
    """True if any polygon bounding box meets the closed query box (bucket-accelerated, exact)."""
    if aabb.shape[0] == 0:
        return False
    bx0 = min(max(int(math.floor((x0 + 512.0) / BUCKET)) - 1, 0), BUCKETS - 1)
    bx1 = min(max(int(math.floor((x1 + 512.0) / BUCKET)) + 1, 0), BUCKETS - 1)
    by0 = min(max(int(math.floor((y0 + 512.0) / BUCKET)) - 1, 0), BUCKETS - 1)
    by1 = min(max(int(math.floor((y1 + 512.0) / BUCKET)) + 1, 0), BUCKETS - 1)
    flagged = False
    for i in range(bx0, bx1 + 1):
        for j in range(by0, by1 + 1):
            if bucket[i, j]:
                flagged = True
    if not flagged:
        return False
    for k in range(aabb.shape[0]):
        if aabb[k, 0] <= x1 and aabb[k, 2] >= x0 and aabb[k, 1] <= y1 and aabb[k, 3] >= y0:
            return True
    return False


@njit(cache=True)
def _box_cast(blocks, poly, flag, cx, cy, hw, hh, dx, dy, distance):
    """``GroundWorld.box_cast``: nearest of the tile hit and (in index order) polygon hits."""
    verts, offs, aabb, bucket = poly
    best = K.box_cast_tiles(blocks, cx, cy, hw, hh, dx, dy, distance)
    reach_x = abs(dx) * distance
    reach_y = abs(dy) * distance
    x0 = cx - hw - (reach_x if dx < 0 else 0.0)
    x1 = cx + hw + (reach_x if dx > 0 else 0.0)
    y0 = cy - hh - (reach_y if dy < 0 else 0.0)
    y1 = cy + hh + (reach_y if dy > 0 else 0.0)
    qx0, qy0, qx1, qy1 = (x0 - K.BOX_CAST_SKIN, y0 - K.BOX_CAST_SKIN, x1 + K.BOX_CAST_SKIN, y1 + K.BOX_CAST_SKIN)
    if _poly_near(aabb, bucket, qx0, qy0, qx1, qy1):
        for k in range(aabb.shape[0]):
            if aabb[k, 0] <= qx1 and aabb[k, 2] >= qx0 and aabb[k, 1] <= qy1 and aabb[k, 3] >= qy0:
                hit = K.sweep_polygon(verts, offs, k, cx, cy, hw, hh, dx, dy, distance)
                if hit[0] and ((not best[0]) or hit[1] < best[1]):
                    best = hit
    return best


@njit(cache=True)
def _raycast(blocks, poly, flag, ox, oy, dx, dy, distance):
    """``GroundWorld.raycast``: normalise, then nearest of tile and polygon hits."""
    verts, offs, aabb, bucket = poly
    norm = math.hypot(dx, dy)
    if norm < 1e-12 or distance <= 0:
        return (False, 0.0, 0.0, 0.0, 0.0, 0.0)
    dx, dy = dx / norm, dy / norm
    ex, ey = ox + dx * distance, oy + dy * distance
    best = K.raycast_tiles(blocks, ox, oy, dx, dy, distance)
    qx0, qy0, qx1, qy1 = min(ox, ex) - 1e-9, min(oy, ey) - 1e-9, max(ox, ex) + 1e-9, max(oy, ey) + 1e-9
    if _poly_near(aabb, bucket, qx0, qy0, qx1, qy1):
        for k in range(aabb.shape[0]):
            if aabb[k, 0] <= qx1 and aabb[k, 2] >= qx0 and aabb[k, 1] <= qy1 and aabb[k, 3] >= qy0:
                found, t = K.ray_polygon(verts, offs, k, ox, oy, dx, dy)
                if found and t <= distance and ((not best[0]) or t < best[1]):
                    best = (True, t, 0.0, 0.0, ox + dx * t, oy + dy * t)
    return best


@njit(cache=True)
def _overlap_box(blocks, poly, flag, cx, cy, w, h):
    verts, offs, aabb, bucket = poly
    hw, hh = w / 2.0, h / 2.0
    if K.overlap_box_tiles(blocks, cx, cy, hw, hh):
        return True
    qx0, qy0, qx1, qy1 = cx - hw, cy - hh, cx + hw, cy + hh
    if _poly_near(aabb, bucket, qx0, qy0, qx1, qy1):
        for k in range(aabb.shape[0]):
            if aabb[k, 0] <= qx1 and aabb[k, 2] >= qx0 and aabb[k, 1] <= qy1 and aabb[k, 3] >= qy0:
                if K.convex_overlap(verts, offs, k, cx, cy, hw, hh):
                    return True
    return False


@njit(cache=True)
def _block_at(blocks, wx, wy):
    cx = int(math.floor(wx + 512.0))
    cy = int(math.floor(wy + 512.0))
    if cx < 0 or cy < 0 or cx >= blocks.shape[0] or cy >= blocks.shape[1]:
        return 0
    return int(blocks[cx, cy])


@njit(cache=True)
def _lerp(a, b, t):
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return a + (b - a) * t


# ------------------------------------------------------------------ body_model.StandingBodyModel
@njit(cache=True)
def _max_speed(S, C):
    return min(C[C_LEG] * C[C_MAX_SPEED], _lerp(C[C_MAX_SPEED], C[C_MAX_SPEED] * 0.5, S[CROUCH]))


@njit(cache=True)
def _move_force(S, C):
    return C[C_MOVE_FORCE] * C[C_LEG] * (0.2 if S[SLIPPERY] != 0.0 else 1.0)


@njit(cache=True)
def _jump_speed(C):
    return C[C_LEG] * C[C_JUMP_SPEED]


@njit(cache=True)
def _wall_slide(S, C):
    return C[C_WALL_SLIDE] * (1.0 / (1.0 + S[T_SLID_FOR] * 0.35)) * C[C_STAMINA]


@njit(cache=True)
def _jump(S, C):
    if not (S[JUMP_CD] <= 0.0):
        return
    if S[GROUNDED] != 0.0 or S[T_GROUNDED] <= C[C_COYOTE]:
        S[JUMP_CD] = C[C_JUMP_CD]
        S[VY] = _jump_speed(C)
        S[GROUNDED] = 0.0
        S[WANTS_JUMP] = 0.0
        return
    can_left = S[T_SLID_L] < C[C_WJ_WINDOW]
    can_right = S[T_SLID_R] < C[C_WJ_WINDOW]
    first = S[FIRST_WALL] != 0.0
    last_right = S[LAST_RIGHT] != 0.0
    if (not can_left and not can_right) or (
            not first and (not can_left or not last_right) and (not can_right or last_right)):
        return
    direction = 1.0 if can_left else -1.0
    S[LAST_RIGHT] = 1.0 if can_right else 0.0
    S[FIRST_WALL] = 0.0
    S[JUMP_CD] = C[C_JUMP_CD]
    S[VX] = direction * _jump_speed(C)
    S[VY] = _jump_speed(C)
    S[SLIDING_R] = 0.0
    S[SLIDING_L] = 0.0
    S[WANTS_JUMP] = 0.0


@njit(cache=True)
def _apply_actions(S, C):
    """One ``PPOBridge.ApplyPPOActions`` application of the held action (latch handled by caller)."""
    cur_jump = S[CUR_ACTION + 1]
    last_jump = S[LAST_ACTION + 1]
    S[MOVE_X] = 0.0
    S[MOVE_Y] = 0.0
    if (last_jump == 0 and cur_jump == 1) or (C[C_AUTO_JUMP] != 0.0 and cur_jump == 1):
        S[WANTS_JUMP] = 1.0
    if last_jump == 1 and cur_jump == 0:
        S[WANTS_JUMP] = 0.0
        S[ENDED_JUMP] = 1.0
    if S[WANTS_JUMP] != 0.0:
        _jump(S, C)
        S[ENDED_JUMP] = 0.0
    S[MOVE_X] = S[CUR_ACTION]
    S[MOVE_Y] = S[CUR_ACTION + 2]
    S[CROUCHING] = 1.0 if S[CUR_ACTION + 3] == 1 else 0.0
    tile_x = int(math.floor(S[X] + 512.0)) + int(S[CUR_ACTION + 4])
    tile_y = int(math.floor(S[Y] + 512.0)) + int(S[CUR_ACTION + 5])
    S[LOOK_X] = tile_x - 511.5
    S[LOOK_Y] = tile_y - 511.5
    for k in range(6):
        S[LAST_ACTION + k] = S[CUR_ACTION + k]


@njit(cache=True)
def latch_actions(S, C, move, jump, vertical, crouch, look_dx, look_dy):
    """``apply_actions(latch=...)`` at a macrostep boundary."""
    for k in range(6):
        S[LAST_ACTION + k] = S[CUR_ACTION + k]
    S[CUR_ACTION] = move
    S[CUR_ACTION + 1] = jump
    S[CUR_ACTION + 2] = vertical
    S[CUR_ACTION + 3] = crouch
    S[CUR_ACTION + 4] = look_dx
    S[CUR_ACTION + 5] = look_dy
    _apply_actions(S, C)


@njit(cache=True)
def _try_step_up(S, C, blocks, poly, flag):
    dx, dy = S[LOOK_X] - S[X], S[LOOK_Y] - S[Y]
    norm = math.hypot(dx, dy)
    look_y = dy / norm if norm > 1e-9 else 0.0
    if abs(look_y - (-0.9)) < 1e-9:      # hypot rounding could decide this comparison differently
        flag[0] = FALLBACK
    if (not (look_y > -0.9)) and S[CROUCHING] != 0.0:
        return False
    direction = S[MOVE_X]
    if direction == 0.0:
        return False
    ox, oy = S[X], S[Y] - S[COL_H] * 0.5 + S[COFF_Y]
    ray = _raycast(blocks, poly, flag, ox, oy, direction, 0.0, S[COL_W] * 0.5 + C[C_EDGE] * 5.0)
    if not ray[0]:
        return False
    px, py = ray[4] + direction * 0.5, ray[5] + 1.0
    if _raycast(blocks, poly, flag, px, py, 0.0, 1.0, S[COL_H])[0]:
        return False
    down = _raycast(blocks, poly, flag, px, py, 0.0, -1.0, 1.0)
    if not down[0]:
        return False
    vec_y = 0.1 + down[5] - ray[5] + C[C_CLEARANCE]
    vec_x = direction * (ray[1] - S[COL_W] * 0.5)
    S[X] += vec_x
    S[Y] += vec_y
    S[VX] = S[LAST_VX]
    return True


@njit(cache=True)
def fixed_update(S, C, blocks, poly, R):
    """``Body.FixedUpdate`` + engine force. ``R`` receives (force_x, force_y, stepped)."""
    flag = np.zeros(1, dtype=np.int64)
    if S[GROUNDED] != 0.0 and S[VY] < 1.0:
        S[ENDED_JUMP] = 0.0
    S[GRAV] = 2.5 if (S[VY] > 0.0 and S[ENDED_JUMP] != 0.0 and S[MOVE_Y] != 1.0 and S[VY] <= C[C_JUMP_SPEED]) else 1.0
    max_speed = _max_speed(S, C)
    force_x, force_y = 0.0, 0.0
    if S[VX] < max_speed and S[MOVE_X] > 0.0:
        force_x += _move_force(S, C)
    if S[VX] > -max_speed and S[MOVE_X] < 0.0:
        force_x -= _move_force(S, C)
    grounded = S[GROUNDED] != 0.0
    if grounded and S[VX] > max_speed:
        S[VX] = max_speed
    if grounded and S[VX] < -max_speed:
        S[VX] = -max_speed
    if grounded and S[GN_X] != 0.0:
        force_x += -S[GN_X] * C[C_MASS] * 52.5
        force_y += abs(S[GN_X]) * C[C_MASS] * 22.5
    if (S[SLIPPERY] == 0.0) and (S[MOVE_X] == 0.0 or (S[MOVE_X] > 0.0 and S[VX] < 0.0)
                                 or (S[MOVE_X] < 0.0 and S[VX] > 0.0)):
        factor = C[C_SLOWDOWN] * (1.0 if grounded else 0.5) * C[C_LEG]
        S[VX] = S[VX] * (1.0 - factor)
    stepped = _try_step_up(S, C, blocks, poly, flag)
    if (S[SLIDING_L] != 0.0 or S[SLIDING_R] != 0.0) and S[VY] < 0.0 and S[SLIPPERY] == 0.0:
        S[VY] = S[VY] * (1.0 - _wall_slide(S, C))
    S[LAST_VX], S[LAST_VY] = S[VX], S[VY]
    R[0] = force_x
    R[1] = force_y + C[C_MASS] * C[C_GRAVITY] * (S[GRAV] - 1.0)
    R[2] = 1.0 if stepped else 0.0
    return flag[0]


@njit(cache=True)
def _grounded_state(S, C, blocks, poly, flag):
    edge = C[C_EDGE]
    cx, cy = S[X] + S[COFF_X], S[Y] + S[COFF_Y]
    hw, hh = S[COL_W] / 2.0, S[COL_H] / 2.0
    down = _box_cast(blocks, poly, flag, cx, cy, hw, hh, 0.0, -1.0, edge + 0.2)
    right = _box_cast(blocks, poly, flag, cx, cy, hw, hh, 1.0, 0.0, edge + 0.1)
    left = _box_cast(blocks, poly, flag, cx, cy, hw, hh, -1.0, 0.0, edge + 0.1)
    S[SLIDING_R] = 1.0 if (S[MOVE_X] > 0.0 and right[0]) else 0.0
    S[SLIDING_L] = 1.0 if (S[MOVE_X] < 0.0 and left[0]) else 0.0
    if S[SLIDING_R] != 0.0:
        S[T_SLID_R] = 0.0
    if S[SLIDING_L] != 0.0:
        S[T_SLID_L] = 0.0
    found = down[0]
    if found:
        S[GN_X], S[GN_Y] = down[2], down[3]
        block = _block_at(blocks, down[4], down[5] - 0.5)
        if block > 0:
            S[STANDING_ON] = block
    slippery = False
    if S[GROUNDED] != 0.0:
        for k in range(int(C[C_N_SLIPPERY])):
            if S[STANDING_ON] == C[C_SLIPPERY0 + k]:
                slippery = True
    S[SLIPPERY] = 1.0 if slippery else 0.0
    S[GROUNDED] = 1.0 if found else 0.0


@njit(cache=True)
def _handle_physics(S, C, blocks, poly, flag, dt):
    if S[GROUNDED] != 0.0:
        S[T_SLID_FOR] = 0.0
        S[FIRST_WALL] = 1.0
        S[T_GROUNDED] = 0.0
    else:
        S[T_GROUNDED] += dt
        if S[SLIDING_L] != 0.0 or S[SLIDING_R] != 0.0:
            S[T_SLID_FOR] += dt
    _try_step_up(S, C, blocks, poly, flag)
    orig_x, orig_y = C[C_ORIG_W], C[C_ORIG_H]
    if _overlap_box(blocks, poly, flag, S[X], S[Y] + orig_y * 0.25, orig_x, orig_y * 0.5):
        S[CROUCHING] = 1.0
    target = 1.0 if S[CROUCHING] != 0.0 else 0.0
    S[CROUCH] = _lerp(S[CROUCH], target, dt * 6.0)
    t = S[CROUCH]
    S[COL_W] = _lerp(orig_x, orig_x, t)
    S[COL_H] = _lerp(orig_y, orig_y * 0.5, t)
    S[COFF_X] = 0.0
    S[COFF_Y] = _lerp(0.0, -orig_y * 0.25, t)


@njit(cache=True)
def _update(S, C, blocks, poly, flag):
    dt = C[C_UPDATE_DT]
    S[T_SLID_L] += dt
    S[T_SLID_R] += dt
    S[JUMP_CD] -= dt
    _grounded_state(S, C, blocks, poly, flag)
    _handle_physics(S, C, blocks, poly, flag, dt)


@njit(cache=True)
def _on_collision_enter(S, C):
    if S[LAST_VY] < -max(_jump_speed(C) * 2.0, C[C_JUMP_SPEED] * 1.5):
        S[RAGDOLL] = 1.0
        return
    S[CROUCH] -= S[LAST_VY] * 0.03


# ------------------------------------------------------------ player_runtime.post_physics_phase
OBJECT_BASE = 1 << 32      # ids >= OBJECT_BASE are static-object polygons (index = id - OBJECT_BASE)


@njit(cache=True)
def post_physics(S, C, blocks, poly, prev, n_prev, out, R, completed, ticks):
    """Collision events, ``Update`` x1|x2 and held-action reapplication after the engine step.

    ``prev`` / ``out`` hold collider ids: ``cx * 65536 + cy`` for a chunk, ``OBJECT_BASE + index``
    for a polygon. ``R`` receives (touched count, entered).
    """
    verts, offs, aabb, bucket = poly
    flag = np.zeros(1, dtype=np.int64)
    cx, cy = S[X] + S[COFF_X], S[Y] + S[COFF_Y]
    hw, hh = S[COL_W] / 2.0, S[COL_H] / 2.0
    n = K.touching_chunks(blocks, cx, cy, hw, hh, CONTACT_REACH, out)
    if n < 0:
        return FALLBACK
    qx0, qy0 = cx - hw - CONTACT_REACH, cy - hh - CONTACT_REACH
    qx1, qy1 = cx + hw + CONTACT_REACH, cy + hh + CONTACT_REACH
    if _poly_near(aabb, bucket, qx0, qy0, qx1, qy1):
        for k in range(aabb.shape[0]):
            if aabb[k, 0] <= qx1 and aabb[k, 2] >= qx0 and aabb[k, 1] <= qy1 and aabb[k, 3] >= qy0:
                if n >= out.shape[0]:
                    return FALLBACK
                out[n] = OBJECT_BASE + k
                n += 1
    entered = 0
    for i in range(n):
        known = False
        for j in range(n_prev):
            if prev[j] == out[i]:
                known = True
        if not known:
            entered += 1
    for _ in range(entered):
        _on_collision_enter(S, C)
    for _ in range(2 if completed % 5 == 0 else 1):
        _update(S, C, blocks, poly, flag)
    if completed < ticks:
        for _ in range(2 if (completed == 4 or completed == 8) else 1):
            _apply_actions(S, C)
    R[0] = n
    R[1] = entered
    return flag[0]


# ----------------------------------------------------------------- EnvSlot.pre_tick + FixedUpdate
# grid scheduler state vector (int64) shared with the slot's CasUGridState via FastSlot sync
G_TILE_COOLDOWN, G_WATER_MOVE, G_PUSH_EVENTS, G_SIM_INDEX, G_TICK, G_CAM_X, G_CAM_Y, G_RNG_POSITION = range(8)
N_GRID = 8


@njit(cache=True)
def _clamp_range(lo, hi, size):
    lo = max(lo, 1)
    hi = max(hi, 1)
    hi = min(hi, size - 2)
    lo = min(lo, size - 2)
    return lo, hi


@njit(cache=True)
def _liquid_id(fluids, x, y, width, height):
    cx = min(max(int(x + width // 2), 0), width - 1)
    cy = min(max(int(y + height // 2), 0), height - 1)
    return int(fluids[cx, cy])


@njit(cache=True)
def _apply_liquid(S, fluids, width, height, buoyancy, drag):
    fluid_id = _liquid_id(fluids, S[X], S[Y], width, height)
    if fluid_id > 0:
        S[VX] = S[VX] * drag[fluid_id]
        S[VY] = S[VY] * drag[fluid_id] + buoyancy[fluid_id]


@njit(cache=True)
def pre_tick(S, S_before_fixed, C, G, fluids, grid_blocks, blocks, poly, rng, push_enabled,
             liquid_before, buoyancy, drag, R):
    """``EnvSlot.pre_tick`` (liquid flags, camera, one fluid sweep, liquid drag) + ``Body.FixedUpdate``.

    Returns OK/FALLBACK; on FALLBACK ``S_before_fixed`` holds the state after the hooks so the caller
    can redo only FixedUpdate with the reference controller. ``R`` = (force_x, force_y, stepped, in_water).
    """
    width, height = fluids.shape[0], fluids.shape[1]
    in_water = _liquid_id(fluids, S[X], S[Y], width, height) > 0
    if in_water:
        S[T_GROUNDED] = 0.0
        S[FIRST_WALL] = 1.0
    G[G_CAM_X] = int(S[X] + width // 2)
    G[G_CAM_Y] = int(S[Y] + height // 2)
    # CasUGridState.simulation_range_index + sweep (literal oracle loop) + tick
    xmin, xmax = _clamp_range(G[G_CAM_X] - 64, G[G_CAM_X] + 64, width)
    ymin, ymax = _clamp_range(G[G_CAM_Y] - 64 + G[G_SIM_INDEX], G[G_CAM_Y] - 48 + G[G_SIM_INDEX], height)
    G[G_SIM_INDEX] += 16
    if G[G_SIM_INDEX] >= 112:
        G[G_SIM_INDEX] = 0
    counters = G[0:3]
    G[G_RNG_POSITION] += K.fluid_sweep(fluids, grid_blocks, xmin, xmax, ymin, ymax, width, height,
                                       counters, push_enabled, rng)
    G[G_TICK] += 1
    if liquid_before:
        _apply_liquid(S, fluids, width, height, buoyancy, drag)
    for k in range(N_STATE):
        S_before_fixed[k] = S[k]
    flag = fixed_update(S, C, blocks, poly, R)
    if not liquid_before:
        _apply_liquid(S, fluids, width, height, buoyancy, drag)
    R[3] = 1.0 if in_water else 0.0
    return flag
