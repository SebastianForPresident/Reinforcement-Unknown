"""Analytic Physics2D-style queries against the canonical Ground layer.

CasU's standing controller issues ``Physics2D.BoxCast``/``Raycast``/``OverlapBox``
queries against layer 6 (tiles and static object colliders). Tiles are full unit
cells and the player queries are axis-aligned, so exact sweeps are cheap and do
not depend on PhysX. Convex static object colliders use a Minkowski-difference
ray test.

Box2D reports a cast ``distance`` shorter than the geometric gap by the contact
skin (measured: ~0.0055: polygon radius target separation plus tile extrusion),
and cross-axis near contact within the skin still counts as a hit. Both effects
are modelled by inflating every tile rectangle by ``BOX_CAST_SKIN``; the value is
validated against recorded source samples.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from . import backend as _backend

HALF_WORLD = 512.0
BOX_CAST_SKIN = 0.0055


@dataclass(frozen=True)
class Hit:
    distance: float
    normal: tuple[float, float]
    point: tuple[float, float]


class GroundWorld:
    """Ground-layer geometry: occupancy tiles plus optional convex polygons."""

    def __init__(self, blocks: np.ndarray, polygons=(), backend: str | None = None):
        self.blocks = blocks
        # compiled tile kernels (None -> readable reference implementation below)
        self._k = _backend.kernels_for(backend)
        self._near_buf = np.empty(64, dtype=np.int64)
        self._chunk_buf = np.empty(16, dtype=np.int64)
        self.width, self.height = blocks.shape
        # polygons: iterable of (name, np.ndarray[N,2] counter-clockwise vertices)
        self.polygons = list(polygons)
        self._aabb = np.zeros((len(self.polygons), 4))
        for index, (_, poly) in enumerate(self.polygons):
            self._aabb[index] = (poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max())

    def set_polygon(self, index: int, name: str, poly: np.ndarray) -> None:
        """Replace one convex polygon (scripted movers) and refresh its culling box."""
        self.polygons[index] = (name, poly)
        self._aabb[index] = (poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max())

    def _polygons_near(self, x0, y0, x1, y1):
        """Indices of polygons whose bounding box intersects the query box."""
        if not len(self.polygons):
            return ()
        if self._k is not None:
            count = self._k.polygons_near(self._aabb, float(x0), float(y0), float(x1), float(y1), self._near_buf)
            if count >= 0:
                return self._near_buf[:count].tolist()
        a = self._aabb
        return np.flatnonzero((a[:, 0] <= x1) & (a[:, 2] >= x0) & (a[:, 1] <= y1) & (a[:, 3] >= y0))

    # ----------------------------------------------------------------- tiles
    def solid(self, cx: int, cy: int) -> bool:
        if cx < 0 or cy < 0 or cx >= self.width or cy >= self.height:
            return False
        return bool(self.blocks[cx, cy])

    def block_at(self, wx: float, wy: float) -> int:
        cx, cy = int(math.floor(wx + HALF_WORLD)), int(math.floor(wy + HALF_WORLD))
        if cx < 0 or cy < 0 or cx >= self.width or cy >= self.height:
            return 0
        return int(self.blocks[cx, cy])

    def _tile_rects(self, x0, x1, y0, y1):
        """Solid unit cells overlapping the closed world interval ``[x0,x1]x[y0,y1]``."""
        cx0 = max(int(math.floor(x0 + HALF_WORLD)), 0)
        cx1 = min(int(math.floor(x1 + HALF_WORLD)), self.width - 1)
        cy0 = max(int(math.floor(y0 + HALF_WORLD)), 0)
        cy1 = min(int(math.floor(y1 + HALF_WORLD)), self.height - 1)
        if cx1 < cx0 or cy1 < cy0:
            return []
        sub = self.blocks[cx0:cx1 + 1, cy0:cy1 + 1]
        xs, ys = np.nonzero(sub)
        return [(cx0 + int(i) - HALF_WORLD, cy0 + int(j) - HALF_WORLD) for i, j in zip(xs, ys)]

    # ------------------------------------------------------------- box casts
    def box_cast(self, center, size, direction, distance) -> Hit | None:
        """Sweep an axis-aligned box along a cardinal ``direction`` (dx, dy).

        Mirrors ``Physics2D.BoxCast(angle=0)`` with ``queriesStartInColliders``:
        an initially overlapping box reports distance 0 and the reverse normal.
        """
        dx, dy = direction
        cx, cy = center
        hw, hh = size[0] / 2.0, size[1] / 2.0
        best: Hit | None = None
        reach_x = abs(dx) * distance
        reach_y = abs(dy) * distance
        x0, x1 = cx - hw - (reach_x if dx < 0 else 0.0), cx + hw + (reach_x if dx > 0 else 0.0)
        y0, y1 = cy - hh - (reach_y if dy < 0 else 0.0), cy + hh + (reach_y if dy > 0 else 0.0)
        eps = 1e-9
        if self._k is not None:
            found, dist, nx, ny, px, py = self._k.box_cast_tiles(
                self.blocks, float(cx), float(cy), hw, hh, float(dx), float(dy), float(distance))
            if found:
                best = Hit(dist, (nx, ny), (px, py))
        else:
            for rx, ry in self._tile_rects(x0 - eps - BOX_CAST_SKIN, x1 + eps + BOX_CAST_SKIN,
                                           y0 - eps - BOX_CAST_SKIN, y1 + eps + BOX_CAST_SKIN):
                hit = self._sweep_rect(cx, cy, hw, hh, dx, dy, distance,
                                       rx - BOX_CAST_SKIN, ry - BOX_CAST_SKIN,
                                       rx + 1.0 + BOX_CAST_SKIN, ry + 1.0 + BOX_CAST_SKIN)
                if hit is not None and (best is None or hit.distance < best.distance):
                    best = hit
        for index in self._polygons_near(x0 - BOX_CAST_SKIN, y0 - BOX_CAST_SKIN,
                                         x1 + BOX_CAST_SKIN, y1 + BOX_CAST_SKIN):
            hit = self._sweep_polygon(cx, cy, hw, hh, dx, dy, distance, self.polygons[index][1])
            if hit is not None and (best is None or hit.distance < best.distance):
                best = hit
        return best

    @staticmethod
    def _sweep_rect(cx, cy, hw, hh, dx, dy, distance, rx0, ry0, rx1, ry1):
        bx0, bx1, by0, by1 = cx - hw, cx + hw, cy - hh, cy + hh
        ox = min(bx1, rx1) - max(bx0, rx0)
        oy = min(by1, ry1) - max(by0, ry0)
        if ox > 1e-9 and oy > 1e-9:  # starts overlapping a collider
            return Hit(0.0, (-float(dx), -float(dy)), (cx, cy))
        if dy != 0:
            if ox <= 1e-9:
                return None
            gap = (by0 - ry1) if dy < 0 else (ry0 - by1)
            normal = (0.0, 1.0 if dy < 0 else -1.0)
            point_y = ry1 if dy < 0 else ry0
            point = (max(bx0, rx0), point_y)
        else:
            if oy <= 1e-9:
                return None
            gap = (bx0 - rx1) if dx < 0 else (rx0 - bx1)
            normal = (1.0 if dx < 0 else -1.0, 0.0)
            point = (rx1 if dx < 0 else rx0, max(by0, ry0))
        if gap < -1e-9 or gap > distance + 1e-9:
            return None
        return Hit(max(gap, 0.0), normal, point)

    @staticmethod
    def _sweep_polygon(cx, cy, hw, hh, dx, dy, distance, poly):
        # Minkowski difference P - B (convex hull of vertex differences); cast the
        # box centre as a ray from the origin of that space.
        corners = np.array([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])
        diffs = (poly[:, None, :] - corners[None, :, :]).reshape(-1, 2) - np.array([cx, cy])
        hull = _convex_hull(diffs)
        if hull is None:
            return None
        inside = _point_in_convex(hull, (0.0, 0.0))
        if inside:
            return Hit(0.0, (-float(dx), -float(dy)), (cx, cy))
        t = _ray_convex(hull, (float(dx), float(dy)))
        if t is None or t > distance + 1e-9:
            return None
        return Hit(t, (-float(dx), -float(dy)), (cx + dx * t, cy + dy * t))

    # ------------------------------------------------------------------ rays
    def raycast(self, origin, direction, distance) -> Hit | None:
        """First Ground hit of a ray; a ray starting inside a collider hits at 0."""
        ox, oy = origin
        dx, dy = direction
        norm = math.hypot(dx, dy)
        if norm < 1e-12 or distance <= 0:
            return None
        dx, dy = dx / norm, dy / norm
        best: Hit | None = None
        ex, ey = ox + dx * distance, oy + dy * distance
        if self._k is not None:
            found, dist, nx, ny, px, py = self._k.raycast_tiles(
                self.blocks, float(ox), float(oy), float(dx), float(dy), float(distance))
            if found:
                best = Hit(dist, (nx, ny), (px, py))
        else:
            for rx, ry in self._tile_rects(min(ox, ex) - 1e-9, max(ox, ex) + 1e-9,
                                           min(oy, ey) - 1e-9, max(oy, ey) + 1e-9):
                hit = _ray_rect(ox, oy, dx, dy, distance, rx, ry, rx + 1.0, ry + 1.0)
                if hit is not None and (best is None or hit.distance < best.distance):
                    best = hit
        for index in self._polygons_near(min(ox, ex) - 1e-9, min(oy, ey) - 1e-9,
                                         max(ox, ex) + 1e-9, max(oy, ey) + 1e-9):
            t = _ray_polygon_distance(self.polygons[index][1], (ox, oy), (dx, dy))
            if t is not None and t <= distance and (best is None or t < best.distance):
                best = Hit(t, (0.0, 0.0), (ox + dx * t, oy + dy * t))
        return best

    def touching_colliders(self, center, size, reach: float) -> set:
        """Collider identities within ``reach`` of an axis-aligned core box.

        Tiles belong to their 64x64 chunk's TilemapCollider2D (one collider per
        chunk); static objects are individual colliders. ``reach`` is the player's
        rounded-edge radius plus the contact tolerance.
        """
        cx, cy = center
        hw, hh = size[0] / 2.0, size[1] / 2.0
        found = set()
        count = -1
        if self._k is not None:
            count = self._k.touching_chunks(self.blocks, float(cx), float(cy), hw, hh, float(reach), self._chunk_buf)
            for chunk in self._chunk_buf[:count].tolist() if count >= 0 else ():
                found.add(("chunk", chunk >> 16, chunk & 0xFFFF))
        if count < 0:   # reference path (also the kernel's near-boundary / overflow fallback)
            for rx, ry in self._tile_rects(cx - hw - reach, cx + hw + reach,
                                           cy - hh - reach, cy + hh + reach):
                dx = max(rx - (cx + hw), (cx - hw) - (rx + 1.0), 0.0)
                dy = max(ry - (cy + hh), (cy - hh) - (ry + 1.0), 0.0)
                if math.hypot(dx, dy) <= reach:
                    found.add(("chunk", int((rx + HALF_WORLD) // 64), int((ry + HALF_WORLD) // 64)))
        for index in self._polygons_near(cx - hw - reach, cy - hh - reach, cx + hw + reach, cy + hh + reach):
            found.add(("object", int(index)))
        return found

    def overlap_box(self, center, size) -> bool:
        cx, cy = center
        hw, hh = size[0] / 2.0, size[1] / 2.0
        if self._k is not None:
            if self._k.overlap_box_tiles(self.blocks, float(cx), float(cy), hw, hh):
                return True
        else:
            for rx, ry in self._tile_rects(cx - hw, cx + hw, cy - hh, cy + hh):
                if min(cx + hw, rx + 1.0) - max(cx - hw, rx) > 1e-9 and \
                        min(cy + hh, ry + 1.0) - max(cy - hh, ry) > 1e-9:
                    return True
        corners = None
        for index in self._polygons_near(cx - hw, cy - hh, cx + hw, cy + hh):
            if corners is None:
                corners = np.array([(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)])
            if _convex_overlap(self.polygons[index][1], corners):
                return True
        return False


def _ray_rect(ox, oy, dx, dy, distance, x0, y0, x1, y1):
    if x0 < ox < x1 and y0 < oy < y1:  # inside (strict)
        return Hit(0.0, (-dx, -dy), (ox, oy))
    t_near, t_far = -math.inf, math.inf
    normal = (0.0, 0.0)
    for o, d, lo, hi, axis in ((ox, dx, x0, x1, 0), (oy, dy, y0, y1, 1)):
        if abs(d) < 1e-12:
            if o < lo or o > hi:
                return None
            continue
        t1, t2 = (lo - o) / d, (hi - o) / d
        enter, leave = (t1, t2) if t1 < t2 else (t2, t1)
        if enter > t_near:
            t_near = enter
            sign = -1.0 if t1 < t2 else 1.0
            normal = (sign, 0.0) if axis == 0 else (0.0, sign)
        t_far = min(t_far, leave)
        if t_near > t_far:
            return None
    if t_near < 0:
        return Hit(0.0, (-dx, -dy), (ox, oy)) if t_far >= 0 else None
    if t_near > distance:
        return None
    return Hit(t_near, normal, (ox + dx * t_near, oy + dy * t_near))


def _convex_hull(points: np.ndarray):
    pts = sorted({(round(float(x), 12), round(float(y), 12)) for x, y in points})
    if len(pts) < 3:
        return None

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _point_in_convex(hull: np.ndarray, point) -> bool:
    n = len(hull)
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        if (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]) <= 1e-12:
            return False
    return True


def _ray_convex(hull: np.ndarray, direction):
    return _ray_polygon_distance(hull, (0.0, 0.0), direction)


def _ray_polygon_distance(poly: np.ndarray, origin, direction):
    best = None
    n = len(poly)
    ox, oy = origin
    dx, dy = direction
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        denom = dx * ey - dy * ex
        if abs(denom) < 1e-12:
            continue
        t = ((ax - ox) * ey - (ay - oy) * ex) / denom
        u = ((ax - ox) * dy - (ay - oy) * dx) / denom
        if t >= -1e-12 and -1e-12 <= u <= 1 + 1e-12 and (best is None or t < best):
            best = max(t, 0.0)
    return best


def _convex_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            ex, ey = poly[(i + 1) % n][0] - poly[i][0], poly[(i + 1) % n][1] - poly[i][1]
            axis = np.array([-ey, ex])
            pa, pb = a @ axis, b @ axis
            if pa.max() <= pb.min() + 1e-12 or pb.max() <= pa.min() + 1e-12:
                return False
    return True
