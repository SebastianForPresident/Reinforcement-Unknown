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
The fluid kernel advances a numpy-compatible PCG64 state array, consuming exactly one ``integers(0, 2)`` draw per
two-sided move, like the oracle (see the PCG64 section; a Generator object is never passed to numba).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

KERNEL_VERSION = 3
HALF_WORLD = 512.0
BOX_CAST_SKIN = 0.0055
_NO_HIT = (False, 0.0, 0.0, 0.0, 0.0, 0.0)


# ------------------------------------------------------------------ PCG64 (numpy-compatible)
# Passing a ``numpy.random.Generator`` into a numba function leaks ~600 bytes per call (numba 0.59), which showed up as
# steady memory growth in long training runs. The fluid kernels therefore take the generator *state* as a uint64 array
# ``[state_lo, state_hi, inc_lo, inc_hi, has_uint32, uinteger]`` and advance it with this port of numpy's PCG64
# XSL-RR 128/64 and of ``Generator.integers(0, 2)`` (one buffered 32-bit output per draw, result = its top bit).
# ``SeededDecisionStream.export_state/import_state`` move the state between numpy and this array; the test suite
# checks the port against numpy over millions of draws.
_M_HI = np.uint64(0x2360ED051FC65DA4)
_M_LO = np.uint64(0x4385DF649FCCF645)
_U32 = np.uint64(0xFFFFFFFF)
_S32 = np.uint64(32)


@njit(cache=True)
def _mul64(a, b):
    a_lo, a_hi = a & _U32, a >> _S32
    b_lo, b_hi = b & _U32, b >> _S32
    ll, lh, hl, hh = a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi
    mid = (ll >> _S32) + (lh & _U32) + (hl & _U32)
    return hh + (lh >> _S32) + (hl >> _S32) + (mid >> _S32), (ll & _U32) | (mid << _S32)


@njit(cache=True)
def _next_uint64(P):
    s_lo, s_hi = P[0], P[1]
    hi, lo = _mul64(s_lo, _M_LO)
    hi = hi + s_hi * _M_LO + s_lo * _M_HI
    new_lo = lo + P[2]
    carry = np.uint64(1) if new_lo < lo else np.uint64(0)
    new_hi = hi + P[3] + carry
    P[0], P[1] = new_lo, new_hi
    xored = new_hi ^ new_lo
    rot = new_hi >> np.uint64(58)
    return (xored >> rot) | (xored << ((np.uint64(64) - rot) & np.uint64(63)))


@njit(cache=True)
def rng_next_bit(P):
    """``Generator.integers(0, 2)``: advances the state; returns 0 or 1."""
    if P[4] != np.uint64(0):
        P[4] = np.uint64(0)
        word = P[5]
    else:
        value = _next_uint64(P)
        P[4] = np.uint64(1)
        P[5] = value >> _S32
        word = value & _U32
    return (word >> np.uint64(31)) & np.uint64(1)


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
def fluid_sweep(fluids, blocks, xmin, xmax, ymin, ymax, width, height, counters, push_enabled, rstate):
    """One sweep over ``[xmin,xmax) x [ymin,ymax)``.

    ``counters`` = int64[tile_cooldown, water_move_count, push_events]; ``rstate`` = the PCG64 state array
    (advanced in place). Returns the number of decisions drawn.
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
                direction = 1 if rng_next_bit(rstate) else -1
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
    half_x = blocks.shape[0] * 0.5            # 512.0 for the canonical 1024 x 1024 world, 64.0 for the XY1 mini-world
    half_y = blocks.shape[1] * 0.5
    cx0 = max(int(math.floor(x0 + half_x)), 0)
    cx1 = min(int(math.floor(x1 + half_x)), blocks.shape[0] - 1)
    cy0 = max(int(math.floor(y0 + half_y)), 0)
    cy1 = min(int(math.floor(y1 + half_y)), blocks.shape[1] - 1)
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
            rx = i - blocks.shape[0] * 0.5
            ry = j - blocks.shape[1] * 0.5
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
            rx = i - blocks.shape[0] * 0.5
            ry = j - blocks.shape[1] * 0.5
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
            rx = i - blocks.shape[0] * 0.5
            ry = j - blocks.shape[1] * 0.5
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
            rx = i - blocks.shape[0] * 0.5
            ry = j - blocks.shape[1] * 0.5
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


# ------------------------------------------------------------------ convex static-object polygons
# Polygons are passed as ``verts`` (float64 [total, 2], counter-clockwise) plus ``offs`` (int64
# [P + 1] start index of each polygon); ``aabb`` (float64 [P, 4]) holds their bounding boxes.
# These are literal ports of ``ground_query._sweep_polygon`` / ``_ray_polygon_distance`` /
# ``_convex_overlap`` including the reference's ``round(x, 12)`` hull coordinates.
@njit(cache=True)
def _two_product(a, b):
    """Error-free product (Dekker/Veltkamp): ``a * b == p + e`` exactly, without FMA."""
    p = a * b
    t = 134217729.0 * a
    ah = t - (t - a)
    al = a - ah
    t = 134217729.0 * b
    bh = t - (t - b)
    bl = b - bh
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


@njit(cache=True)
def round12(x):
    """Python's ``round(x, 12)`` (correctly rounded decimal, half-even) for |x| < 9007."""
    if abs(x) >= 9007.0:
        return x
    hi, lo = _two_product(x, 1e12)
    k = np.rint(hi)
    r = (hi - k) + lo
    if r > 0.5:
        k += 1.0
    elif r < -0.5:
        k -= 1.0
    return k / 1e12


@njit(cache=True)
def _hull(px, py, n, out):
    """Monotone-chain hull of ``n`` already rounded, unique, lexicographically sorted points.

    Returns the vertex count (hull vertices are written to ``out``), or 0 when fewer than 3 points.
    """
    if n < 3:
        return 0
    lower = np.empty((n, 2))
    upper = np.empty((n, 2))
    nl = 0
    for i in range(n):
        while nl >= 2 and ((lower[nl - 1, 0] - lower[nl - 2, 0]) * (py[i] - lower[nl - 2, 1])
                           - (lower[nl - 1, 1] - lower[nl - 2, 1]) * (px[i] - lower[nl - 2, 0])) <= 0:
            nl -= 1
        lower[nl, 0] = px[i]
        lower[nl, 1] = py[i]
        nl += 1
    nu = 0
    for i in range(n - 1, -1, -1):
        while nu >= 2 and ((upper[nu - 1, 0] - upper[nu - 2, 0]) * (py[i] - upper[nu - 2, 1])
                           - (upper[nu - 1, 1] - upper[nu - 2, 1]) * (px[i] - upper[nu - 2, 0])) <= 0:
            nu -= 1
        upper[nu, 0] = px[i]
        upper[nu, 1] = py[i]
        nu += 1
    m = 0
    for i in range(nl - 1):
        out[m, 0] = lower[i, 0]
        out[m, 1] = lower[i, 1]
        m += 1
    for i in range(nu - 1):
        out[m, 0] = upper[i, 0]
        out[m, 1] = upper[i, 1]
        m += 1
    return m


@njit(cache=True)
def _ray_poly(vx, vy, n, ox, oy, dx, dy):
    """``_ray_polygon_distance``: (found, t)."""
    best = 0.0
    found = False
    for i in range(n):
        ax, ay = vx[i], vy[i]
        j = i + 1 if i + 1 < n else 0
        bx, by = vx[j], vy[j]
        ex, ey = bx - ax, by - ay
        denom = dx * ey - dy * ex
        if abs(denom) < 1e-12:
            continue
        t = ((ax - ox) * ey - (ay - oy) * ex) / denom
        u = ((ax - ox) * dy - (ay - oy) * dx) / denom
        if t >= -1e-12 and -1e-12 <= u <= 1 + 1e-12 and ((not found) or t < best):
            best = max(t, 0.0)
            found = True
    return found, best


@njit(cache=True)
def sweep_polygon(verts, offs, k, cx, cy, hw, hh, dx, dy, distance):
    """``GroundWorld._sweep_polygon``: (found, distance, nx, ny, px, py) for polygon ``k``."""
    start, stop = offs[k], offs[k + 1]
    n = stop - start
    total = n * 4
    xs = np.empty(total)
    ys = np.empty(total)
    cxs = np.array([-hw, hw, hw, -hw])
    cys = np.array([-hh, -hh, hh, hh])
    idx = 0
    for i in range(n):
        for j in range(4):
            xs[idx] = round12((verts[start + i, 0] - cxs[j]) - cx)
            ys[idx] = round12((verts[start + i, 1] - cys[j]) - cy)
            idx += 1
    order = np.argsort(ys, kind="mergesort")
    order = order[np.argsort(xs[order], kind="mergesort")]
    ux = np.empty(total)
    uy = np.empty(total)
    m = 0
    for q in range(total):
        i = order[q]
        if m > 0 and ux[m - 1] == xs[i] and uy[m - 1] == ys[i]:
            continue
        ux[m] = xs[i]
        uy[m] = ys[i]
        m += 1
    hull = np.empty((2 * m + 2, 2))
    hn = _hull(ux, uy, m, hull)
    if hn == 0:
        return _NO_HIT
    inside = True
    for i in range(hn):
        j = i + 1 if i + 1 < hn else 0
        if (hull[j, 0] - hull[i, 0]) * (0.0 - hull[i, 1]) - (hull[j, 1] - hull[i, 1]) * (0.0 - hull[i, 0]) <= 1e-12:
            inside = False
            break
    if inside:
        return True, 0.0, -dx, -dy, cx, cy
    found, t = _ray_poly(hull[:hn, 0], hull[:hn, 1], hn, 0.0, 0.0, dx, dy)
    if (not found) or t > distance + 1e-9:
        return _NO_HIT
    return True, t, -dx, -dy, cx + dx * t, cy + dy * t


@njit(cache=True)
def ray_polygon(verts, offs, k, ox, oy, dx, dy):
    start, stop = offs[k], offs[k + 1]
    return _ray_poly(verts[start:stop, 0], verts[start:stop, 1], stop - start, ox, oy, dx, dy)


@njit(cache=True)
def convex_overlap(verts, offs, k, cx, cy, hw, hh):
    """``_convex_overlap(polygon, query box corners)`` (separating-axis test)."""
    start, stop = offs[k], offs[k + 1]
    n = stop - start
    bx = np.array([cx - hw, cx + hw, cx + hw, cx - hw])
    by = np.array([cy - hh, cy - hh, cy + hh, cy + hh])
    for which in range(2):
        m = n if which == 0 else 4
        for i in range(m):
            j = i + 1 if i + 1 < m else 0
            if which == 0:
                ex = verts[start + j, 0] - verts[start + i, 0]
                ey = verts[start + j, 1] - verts[start + i, 1]
            else:
                ex, ey = bx[j] - bx[i], by[j] - by[i]
            ax, ay = -ey, ex
            pa_max, pa_min, pb_max, pb_min = -np.inf, np.inf, -np.inf, np.inf
            for q in range(n):
                v = verts[start + q, 0] * ax + verts[start + q, 1] * ay
                pa_max = max(pa_max, v)
                pa_min = min(pa_min, v)
            for q in range(4):
                v = bx[q] * ax + by[q] * ay
                pb_max = max(pb_max, v)
                pb_min = min(pb_min, v)
            if pa_max <= pb_min + 1e-12 or pb_max <= pa_min + 1e-12:
                return False
    return True
