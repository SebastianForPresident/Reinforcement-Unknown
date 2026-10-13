"""Static non-tile colliders of the canonical Layer 1 (shuttle doors, crates, pods...).

The tile arrays alone are not the full collision world: Unity also holds ~600
static object colliders on layers that collide with the player (see
``fixtures/*/physics_audit``). They are measured source data, loaded here with
checksum verification; dynamic or non-player-colliding objects are listed but
never simulated as static geometry.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .reference import FIXTURE_DIR

AUDIT_DIRNAME = "physics_audit"
WORLD_HALF_EXTENT = 512.0


@dataclass(frozen=True)
class ObjectCollider:
    name: str
    kind: str                       # "box" or "circle"
    center: tuple[float, float]     # Unity/Isaac world x, planar y
    size: tuple[float, float] | None = None
    rotation_deg: float = 0.0       # CCW in the planar x-y plane
    radius: float | None = None
    layer: int = 6
    body_type: str = "NoRigidbody(static)"

    def aabb(self) -> tuple[float, float, float, float]:
        cx, cy = self.center
        if self.kind == "circle":
            return cx - self.radius, cy - self.radius, cx + self.radius, cy + self.radius
        w, h = self.size
        a = math.radians(self.rotation_deg)
        ex = abs(w * math.cos(a)) / 2 + abs(h * math.sin(a)) / 2
        ey = abs(w * math.sin(a)) / 2 + abs(h * math.cos(a)) / 2
        return cx - ex, cy - ey, cx + ex, cy + ey

    def contains(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        dx, dy = px - self.center[0], py - self.center[1]
        if self.kind == "circle":
            return dx * dx + dy * dy <= self.radius * self.radius
        a = math.radians(self.rotation_deg)
        local_x = dx * math.cos(a) + dy * math.sin(a)
        local_y = -dx * math.sin(a) + dy * math.cos(a)
        return (np.abs(local_x) <= self.size[0] / 2 + 1e-9) & (np.abs(local_y) <= self.size[1] / 2 + 1e-9)


@dataclass(frozen=True)
class WorldObjects:
    colliders: tuple[ObjectCollider, ...]
    excluded_dynamic: tuple[dict, ...]
    manifest: dict

    def in_region(self, min_x: float, min_y: float, max_x: float, max_y: float):
        out = []
        for collider in self.colliders:
            x0, y0, x1, y1 = collider.aabb()
            if x1 >= min_x and x0 <= max_x and y1 >= min_y and y0 <= max_y:
                out.append(collider)
        return out


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_physics_audit(fixture_dir: str | Path = FIXTURE_DIR):
    """Return ``(WorldObjects, occupancy[x,y] uint8)``, verifying SHA-256 sums."""
    root = Path(fixture_dir) / AUDIT_DIRNAME
    manifest = json.loads((root / "physics_audit_manifest.json").read_text())
    if manifest.get("schema") != "casu.physics-audit.v1":
        raise ValueError("unsupported physics audit schema")
    spec = manifest["object_colliders"]
    objects_path = root / spec["file"]
    if _sha(objects_path) != spec["sha256"]:
        raise ValueError("object collider file SHA-256 mismatch")
    occ_spec = manifest["occupancy"]
    occ_path = root / occ_spec["file"]
    if _sha(occ_path) != occ_spec["sha256"]:
        raise ValueError("physics occupancy file SHA-256 mismatch")
    data = json.loads(objects_path.read_text())
    if data.get("schema") != "casu.object-colliders.v1" or len(data["colliders"]) != spec["count"]:
        raise ValueError("object collider file does not match its manifest")
    colliders = []
    for entry in data["colliders"]:
        if entry["kind"] == "box":
            if entry.get("edge_radius", 0.0) != 0.0:
                raise ValueError("rounded static box colliders are unsupported")
            colliders.append(ObjectCollider(
                entry["name"], "box", tuple(entry["center"]), tuple(entry["size"]),
                entry["rotation_deg"], None, entry["layer"], entry["body_type"]))
        else:
            colliders.append(ObjectCollider(
                entry["name"], "circle", tuple(entry["center"]), None, 0.0,
                entry["radius"], entry["layer"], entry["body_type"]))
    occupancy = np.fromfile(occ_path, dtype=np.uint8).reshape(
        occ_spec["shape"][0], occ_spec["shape"][1])
    return (WorldObjects(tuple(colliders), tuple(data["excluded_dynamic_initial_footprints"]),
                         manifest), occupancy)


def rasterize_cell_centers(colliders, width: int = 1024, height: int = 1024) -> np.ndarray:
    """Boolean ``[x, y]`` map of cells whose centre lies inside any collider."""
    covered = np.zeros((width, height), dtype=bool)
    for collider in colliders:
        x0, y0, x1, y1 = collider.aabb()
        ix0 = max(int(math.floor(x0 + width / 2 - 1)), 0)
        ix1 = min(int(math.ceil(x1 + width / 2 + 1)), width - 1)
        iy0 = max(int(math.floor(y0 + height / 2 - 1)), 0)
        iy1 = min(int(math.ceil(y1 + height / 2 + 1)), height - 1)
        px = (np.arange(ix0, ix1 + 1) - width / 2 + 0.5)[:, None]
        py = (np.arange(iy0, iy1 + 1) - height / 2 + 0.5)[None, :]
        covered[ix0:ix1 + 1, iy0:iy1 + 1] |= collider.contains(px, py)
    return covered


def ground_polygons(colliders, *, circle_segments: int = 24):
    """Layer-6 (Ground) static object colliders as convex polygons for Physics2D-style queries.

    Only Ground-layer objects are visible to the player's BoxCast/Raycast/OverlapBox
    queries; Default-layer statics still collide physically but are not queried.
    """
    polygons = []
    for collider in colliders:
        if collider.layer != 6:
            continue
        cx, cy = collider.center
        if collider.kind == "box":
            a = math.radians(collider.rotation_deg)
            ca, sa = math.cos(a), math.sin(a)
            hw, hh = collider.size[0] / 2, collider.size[1] / 2
            pts = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
            poly = np.array([(cx + x * ca - y * sa, cy + x * sa + y * ca) for x, y in pts])
        else:
            angles = np.linspace(0, 2 * math.pi, circle_segments, endpoint=False)
            poly = np.stack((cx + collider.radius * np.cos(angles),
                             cy + collider.radius * np.sin(angles)), axis=1)
        polygons.append((collider.name, poly))
    return polygons
