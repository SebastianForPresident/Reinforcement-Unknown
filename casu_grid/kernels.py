"""Numba CPU kernels for the simulator's hot paths (backend ``numba``; see :mod:`casu_grid.backend`).

Every kernel is a statement-for-statement port of a readable reference routine and keeps its
iteration order, tie-breaking and floating-point operation order, so results are identical
(``tests/test_kernels.py`` compares against the references):

* :func:`fluid_sweep`        <- ``CasUGridState.simulation_step_reference``
* :func:`box_cast_tiles`     <- ``GroundWorld.box_cast`` (tile part, ``_sweep_rect``)
* :func:`raycast_tiles`      <- ``GroundWorld.raycast`` (tile part, ``_ray_rect``)
* :func:`overlap_box_tiles`  <- ``GroundWorld.overlap_box`` (tile part)
* :func:`touching_chunks`    <- ``GroundWorld.touching_colliders`` (tile part)
* :func:`polygons_near`      <- ``GroundWorld._polygons_near``

Convex static-object polygons stay in the Python reference code; the kernels only report which
polygons are near a query so the rare polygon case reuses the reference arithmetic exactly.
The fluid kernel draws from the same ``numpy.random.Generator`` (numba shares its bit-generator
state in place), consuming exactly one ``integers(0, 2)`` draw per two-sided move, like the oracle.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

KERNEL_VERSION = 1
HALF_WORLD = 512.0
BOX_CAST_SKIN = 0.0055
_NO_HIT = (False, 0.0, 0.0, 0.0, 0.0, 0.0)


# ------------------------------------------------------------------------ fluid
@njit(cache=True)
def _empty(fluids, blocks, x, y, width, height):
    xx = min(max(x, 0), width - 1)
    yy = min(max(y, 0), height - 1)
    return fluids[xx, yy] == 0 and blocks[xx, yy] == 0


@njit(cache=True)
def _liquid(fluids, x, y, width, height):
    return fluids[min(max(x, 0), width - 1), min(max(y, 0), height - 1)]


@njit(cache=True)
def _incr_move(counters, push_enabled):
    if push_enabled:
        counters[1] += 1
        if counters[1] > 10:
            counters[1] = 0
            counters[2] += 1


@njit(cache=True)
def fluid_sweep(fluids, blocks, xmin, xmax, ymin, ymax, width, height, counters, push_enabled, rng):
    """One sweep over ``[xmin,xmax) x [ymin,ymax)``.

    ``counters`` = int64[tile_cooldown, water_move_count, push_events]. Returns the number of
    decisions drawn from ``rng``.
    """
    draws = 0
    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            if fluids[x, y] == 0:
                continue
            if _empty(fluids, blocks, x, y - 1, width, height):
                if y <= 2:
                    fluids[x, y] = 0
                tmp = fluids[x, y]
                fluids[x, y] = fluids[x, y - 1]
                fluids[x, y - 1] = tmp
                counters[0] = (counters[0] + 1) % 256
                _incr_move(counters, push_enabled)
                if counters[0] > 16:
                    counters[0] = 0
                continue
            below = _liquid(fluids, x, y - 1, width, height)
            if (fluids[x, y] == 2 and below == 1) or (fluids[x, y] == 1 and below == 2):
                fluids[x, y - 1] = 2
                fluids[x, y] = 2
            right = _empty(fluids, blocks, x + 1, y, width, height)
            left = _empty(fluids, blocks, x - 1, y, width, height)
            if right and left:
                direction = 1 if rng.integers(0, 2) else -1
                draws += 1
                tmp = fluids[x, y]
                fluids[x, y] = fluids[x + direction, y]
                fluids[x + direction, y] = tmp
            elif right:
                tmp = fluids[x, y]
                fluids[x, y] = fluids[x + 1, y]
                fluids[x + 1, y] = tmp
                _incr_move(counters, push_enabled)
            elif left:
                tmp = fluids[x, y]
                fluids[x, y] = fluids[x - 1, y]
                fluids[x - 1, y] = tmp
                _incr_move(counters, push_enabled)
    return draws


# ------------------------------------------------------------------ ground queries
@njit(cache=True)
def _cell_range(blocks, x0, x1, y0, y1):
    cx0 = max(int(math.floor(x0 + HALF_WORLD)), 0)
    cx1 = min(int(math.floor(x1 + HALF_WORLD)), blocks.shape[0] - 1)
    cy0 = max(int(math.floor(y0 + HALF_WORLD)), 0)
    cy1 = min(int(math.floor(y1 + HALF_WORLD)), blocks.shape[1] - 1)
    return cx0, cx1, cy0, cy1


@njit(cache=True)
def _sweep_rect(cx, cy, hw, hh, dx, dy, distance, rx0, ry0, rx1, ry1):
    bx0, bx1, by0, by1 = cx - hw, cx + hw, cy - hh, cy + hh
    ox = min(bx1, rx1) - max(bx0, rx0)
    oy = min(by1, ry1) - max(by0, ry0)
    if ox > 1e-9 and oy > 1e-9:
        return True, 0.0, -dx, -dy, cx, cy
    if dy != 0:
        if ox <= 1e-9:
            return _NO_HIT
        gap = (by0 - ry1) if dy < 0 else (ry0 - by1)
        nx, ny = 0.0, (1.0 if dy < 0 else -1.0)
        px, py = max(bx0, rx0), (ry1 if dy < 0 else ry0)
    else:
        if oy <= 1e-9:
            return _NO_HIT
        gap = (bx0 - rx1) if dx < 0 else (rx0 - bx1)
        nx, ny = (1.0 if dx < 0 else -1.0), 0.0
        px, py = (rx1 if dx < 0 else rx0), max(by0, ry0)
    if gap < -1e-9 or gap > distance + 1e-9:
        return _NO_HIT
    return True, max(gap, 0.0), nx, ny, px, py


@njit(cache=True)
def box_cast_tiles(blocks, cx, cy, hw, hh, dx, dy, distance):
    """Nearest tile hit of an axis-aligned box sweep: ``(found, distance, nx, ny, px, py)``."""
    reach_x = abs(dx) * distance
    reach_y = abs(dy) * distance
    x0 = cx - hw - (reach_x if dx < 0 else 0.0)
    x1 = cx + hw + (reach_x if dx > 0 else 0.0)
    y0 = cy - hh - (reach_y if dy < 0 else 0.0)
    y1 = cy + hh + (reach_y if dy > 0 else 0.0)
    eps = 1e-9
    cx0, cx1, cy0, cy1 = _cell_range(blocks, x0 - eps - BOX_CAST_SKIN, x1 + eps + BOX_CAST_SKIN,
                                     y0 - eps - BOX_CAST_SKIN, y1 + eps + BOX_CAST_SKIN)
    best = _NO_HIT
    for i in range(cx0, cx1 + 1):
        for j in range(cy0, cy1 + 1):
            if blocks[i, j] == 0:
                continue
            rx = i - HALF_WORLD
            ry = j - HALF_WORLD
            hit = _sweep_rect(cx, cy, hw, hh, dx, dy, distance,
                              rx - BOX_CAST_SKIN, ry - BOX_CAST_SKIN,
                              rx + 1.0 + BOX_CAST_SKIN, ry + 1.0 + BOX_CAST_SKIN)
            if hit[0] and (not best[0] or hit[1] < best[1]):
                best = hit
    return best


@njit(cache=True)
def _ray_rect(ox, oy, dx, dy, distance, x0, y0, x1, y1):
    if x0 < ox < x1 and y0 < oy < y1:
        return True, 0.0, -dx, -dy, ox, oy
    t_near, t_far = -math.inf, math.inf
    nx, ny = 0.0, 0.0
    for axis in range(2):
        if axis == 0:
            o, d, lo, hi = ox, dx, x0, x1
        else:
            o, d, lo, hi = oy, dy, y0, y1
        if abs(d) < 1e-12:
            if o < lo or o > hi:
                return _NO_HIT
            continue
        t1, t2 = (lo - o) / d, (hi - o) / d
        if t1 < t2:
            enter, leave = t1, t2
        else:
            enter, leave = t2, t1
        if enter > t_near:
            t_near = enter
            sign = -1.0 if t1 < t2 else 1.0
            if axis == 0:
                nx, ny = sign, 0.0
            else:
                nx, ny = 0.0, sign
        t_far = min(t_far, leave)
        if t_near > t_far:
            return _NO_HIT
    if t_near < 0:
        if t_far >= 0:
            return True, 0.0, -dx, -dy, ox, oy
        return _NO_HIT
    if t_near > distance:
        return _NO_HIT
    return True, t_near, nx, ny, ox + dx * t_near, oy + dy * t_near


@njit(cache=True)
def raycast_tiles(blocks, ox, oy, dx, dy, distance):
    """Nearest tile hit of a ray with a *normalised* direction."""
    ex, ey = ox + dx * distance, oy + dy * distance
    cx0, cx1, cy0, cy1 = _cell_range(blocks, min(ox, ex) - 1e-9, max(ox, ex) + 1e-9,
                                     min(oy, ey) - 1e-9, max(oy, ey) + 1e-9)
    best = _NO_HIT
    for i in range(cx0, cx1 + 1):
        for j in range(cy0, cy1 + 1):
            if blocks[i, j] == 0:
                continue
            rx = i - HALF_WORLD
            ry = j - HALF_WORLD
            hit = _ray_rect(ox, oy, dx, dy, distance, rx, ry, rx + 1.0, ry + 1.0)
            if hit[0] and (not best[0] or hit[1] < best[1]):
                best = hit
    return best


@njit(cache=True)
def overlap_box_tiles(blocks, cx, cy, hw, hh):
    cx0, cx1, cy0, cy1 = _cell_range(blocks, cx - hw, cx + hw, cy - hh, cy + hh)
    for i in range(cx0, cx1 + 1):
        for j in range(cy0, cy1 + 1):
            if blocks[i, j] == 0:
                continue
            rx = i - HALF_WORLD
            ry = j - HALF_WORLD
            if min(cx + hw, rx + 1.0) - max(cx - hw, rx) > 1e-9 and \
                    min(cy + hh, ry + 1.0) - max(cy - hh, ry) > 1e-9:
                return True
    return False


@njit(cache=True)
def touching_chunks(blocks, cx, cy, hw, hh, reach, out):
    """Chunk ids (``cx * 65536 + cy``) of tiles within ``reach`` of the core box.

    Fills ``out`` (unique ids, scan order) and returns the count, or ``-1`` when a tile lies so
    close to the ``reach`` boundary that ``math.hypot`` rounding could decide differently
    (the caller then uses the reference implementation).
    """
    cx0, cx1, cy0, cy1 = _cell_range(blocks, cx - hw - reach, cx + hw + reach,
                                     cy - hh - reach, cy + hh + reach)
    count = 0
    for i in range(cx0, cx1 + 1):
        for j in range(cy0, cy1 + 1):
            if blocks[i, j] == 0:
                continue
            rx = i - HALF_WORLD
            ry = j - HALF_WORLD
            dx = max(rx - (cx + hw), (cx - hw) - (rx + 1.0), 0.0)
            dy = max(ry - (cy + hh), (cy - hh) - (ry + 1.0), 0.0)
            h = math.hypot(dx, dy)
            if abs(h - reach) < 1e-9:
                return -1
            if h <= reach:
                chunk = (i // 64) * 65536 + (j // 64)
                known = False
                for k in range(count):
                    if out[k] == chunk:
                        known = True
                        break
                if not known:
                    if count >= out.shape[0]:
                        return -1
                    out[count] = chunk
                    count += 1
    return count


@njit(cache=True)
def polygons_near(aabb, x0, y0, x1, y1, out):
    """Ascending indices of polygons whose bounding box meets the query box; returns the count."""
    count = 0
    for k in range(aabb.shape[0]):
        if aabb[k, 0] <= x1 and aabb[k, 2] >= x0 and aabb[k, 1] <= y1 and aabb[k, 3] >= y0:
            if count >= out.shape[0]:
                return -1
            out[count] = k
            count += 1
    return count
