"""PhysX stand-in for the CasU player's Box2D ``BoxCollider2D`` (rounded box).

Unity's collider is a rectangle ``size`` expanded by ``edgeRadius`` with rounded
corners. PhysX has no 2-D rounded rectangle, so each collider is a compound of
native convex primitives (no triangle meshes, no convex-hull fallback): two
cross-expanded cuboids plus four corner cylinders, extruded along Y by the planar
depth.

Editing a shape's transform or scale at runtime invalidates the PhysX simulation
view, so continuous crouch is represented by ``levels + 1`` pre-authored
compounds (crouch 0..1) of which exactly one is collision-enabled. The source
interpolates the collider size continuously; the quantization error is at most
``0.5 * (standing_height / 2) / levels`` in height. Import only inside Kit.
"""
from __future__ import annotations

SHAPE_NAMES_ALL = ("CrossX", "CrossZ", "CornerNE", "CornerNW", "CornerSW", "CornerSE")


def _author_shape(stage, path, prim_type, material_path, rest_offset, contact_offset, enabled):
    from pxr import Sdf

    layer = stage.GetEditTarget().GetLayer()
    parent_path, name = path.rsplit("/", 1)
    spec = Sdf.PrimSpec(layer.GetPrimAtPath(parent_path), name, Sdf.SpecifierDef, prim_type)
    schemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]
    if material_path:
        schemas.append("MaterialBindingAPI")
    spec.SetInfo("apiSchemas", Sdf.TokenListOp.Create(prependedItems=schemas))
    Sdf.AttributeSpec(spec, "physics:collisionEnabled", Sdf.ValueTypeNames.Bool).default = bool(enabled)
    Sdf.AttributeSpec(spec, "physxCollision:restOffset", Sdf.ValueTypeNames.Float).default = float(rest_offset)
    Sdf.AttributeSpec(spec, "physxCollision:contactOffset", Sdf.ValueTypeNames.Float).default = float(contact_offset)
    if material_path:
        relationship = Sdf.RelationshipSpec(spec, "material:binding:physics")
        relationship.targetPathList.explicitItems = [Sdf.Path(material_path)]
    return spec


def _author_hull(stage, path, material_path, rest_offset, contact_offset, *, enabled, width, height, offset,
                 radius, depth, segments):
    """One convex-hull mesh: the rounded rectangle (quarter-circle corners of ``segments`` steps), extruded along Y.

    EXPERIMENTAL representation (see ``player_collider='convex_hull'``): the same union of two cross boxes and four
    corner cylinders as the compound, but a single shape, so a crouch-level change toggles 2 USD attributes instead of 12.
    """
    import math
    from pxr import Gf, Sdf, Vt

    spec = _author_shape(stage, path, "Mesh", material_path, rest_offset, contact_offset, enabled)
    schemas = list(spec.GetInfo("apiSchemas").prependedItems) + ["PhysicsMeshCollisionAPI", "PhysxConvexHullCollisionAPI"]
    spec.SetInfo("apiSchemas", Sdf.TokenListOp.Create(prependedItems=schemas))
    Sdf.AttributeSpec(spec, "physics:approximation", Sdf.ValueTypeNames.Token).default = "convexHull"
    Sdf.AttributeSpec(spec, "physxConvexHullCollision:hullVertexLimit", Sdf.ValueTypeNames.Int).default = 255
    ring = []
    centers = ((width / 2, offset + height / 2, 0.0), (-width / 2, offset + height / 2, 90.0),
               (-width / 2, offset - height / 2, 180.0), (width / 2, offset - height / 2, 270.0))
    for cx, cz, start in centers:                      # counter-clockwise in the XZ plane
        for k in range(segments + 1):
            angle = math.radians(start + 90.0 * k / segments)
            ring.append((cx + radius * math.cos(angle), cz + radius * math.sin(angle)))
    n = len(ring)
    points = [Gf.Vec3f(x, -depth / 2, z) for x, z in ring] + [Gf.Vec3f(x, depth / 2, z) for x, z in ring]
    counts, indices = [], []
    counts.append(n); indices += list(range(n - 1, -1, -1))          # bottom cap (-Y)
    counts.append(n); indices += list(range(n, 2 * n))               # top cap (+Y)
    for i in range(n):                                               # side quads
        j = (i + 1) % n
        counts.append(4); indices += [i, j, n + j, n + i]
    Sdf.AttributeSpec(spec, "points", Sdf.ValueTypeNames.Point3fArray).default = Vt.Vec3fArray(points)
    Sdf.AttributeSpec(spec, "faceVertexCounts", Sdf.ValueTypeNames.IntArray).default = Vt.IntArray(counts)
    Sdf.AttributeSpec(spec, "faceVertexIndices", Sdf.ValueTypeNames.IntArray).default = Vt.IntArray(indices)
    Sdf.AttributeSpec(spec, "extent", Sdf.ValueTypeNames.Float3Array).default = Vt.Vec3fArray(
        [Gf.Vec3f(-width / 2 - radius, -depth / 2, offset - height / 2 - radius),
         Gf.Vec3f(width / 2 + radius, depth / 2, offset + height / 2 + radius)])
    return spec


class RoundedBoxCollider:
    """Pre-authored crouch levels of the rounded compound collider under ``root_path``."""

    def __init__(self, stage, root_path: str, *, width: float, standing_height: float,
                 crouch_height_ratio: float = 0.5, edge_radius: float, depth: float,
                 material_path: str | None, levels: int = 32,
                 rest_offset: float = 0.0155, contact_offset: float = 0.04,
                 corner_shape: str = "cylinder", representation: str = "compound",
                 hull_segments: int = 8):
        from pxr import Gf, Sdf, UsdGeom

        if representation not in ("compound", "convex_hull"):
            raise ValueError("representation must be 'compound' (validated) or 'convex_hull' (experimental)")
        self.representation = representation

        self.stage, self.root_path, self.levels = stage, root_path, int(levels)
        self.width, self.standing_height = float(width), float(standing_height)
        self.crouch_height_ratio, self.edge_radius, self.depth = (
            float(crouch_height_ratio), float(edge_radius), float(depth))
        self.active_level = 0
        self.shape_names = (("Hull",) if representation == "convex_hull" else
                            SHAPE_NAMES_ALL if corner_shape == "cylinder" else ("CrossX", "CrossZ"))
        layer = stage.GetEditTarget().GetLayer()
        for level in range(self.levels + 1):
            crouch = level / self.levels
            h, off = self.level_shape(level)
            level_path = f"{root_path}/Level_{level}"
            Sdf.PrimSpec(layer.GetPrimAtPath(root_path), f"Level_{level}", Sdf.SpecifierDef, "Xform")
            r, d, w = self.edge_radius, self.depth, self.width
            if representation == "convex_hull":
                _author_hull(stage, f"{level_path}/Hull", material_path, rest_offset, contact_offset,
                             enabled=(level == 0), width=w, height=h, offset=off, radius=r, depth=d,
                             segments=hull_segments)
                continue
            specs = {
                "CrossX": ("Cube", (0.0, off), (w + 2 * r, d, h)),
                "CrossZ": ("Cube", (0.0, off), (w, d, h + 2 * r)),
            }
            if corner_shape == "cylinder":
                specs.update({
                    "CornerNE": ("Cylinder", (w / 2, off + h / 2), None),
                    "CornerNW": ("Cylinder", (-w / 2, off + h / 2), None),
                    "CornerSW": ("Cylinder", (-w / 2, off - h / 2), None),
                    "CornerSE": ("Cylinder", (w / 2, off - h / 2), None),
                })
            elif corner_shape != "none":
                raise ValueError("corner_shape must be 'cylinder' or 'none'")
            for name, (prim_type, (cx, cz), scale) in specs.items():
                spec = _author_shape(stage, f"{level_path}/{name}", prim_type, material_path,
                                     rest_offset, contact_offset, enabled=(level == 0))
                if prim_type == "Cube":
                    Sdf.AttributeSpec(spec, "size", Sdf.ValueTypeNames.Double).default = 1.0
                    Sdf.AttributeSpec(spec, "xformOp:scale", Sdf.ValueTypeNames.Float3).default = Gf.Vec3f(*scale)
                    order = ["xformOp:translate", "xformOp:scale"]
                else:
                    Sdf.AttributeSpec(spec, "radius", Sdf.ValueTypeNames.Double).default = r
                    Sdf.AttributeSpec(spec, "height", Sdf.ValueTypeNames.Double).default = d
                    Sdf.AttributeSpec(spec, "axis", Sdf.ValueTypeNames.Token,
                                      variability=Sdf.VariabilityUniform).default = "Y"
                    order = ["xformOp:translate"]
                Sdf.AttributeSpec(spec, "xformOp:translate", Sdf.ValueTypeNames.Double3).default = Gf.Vec3d(cx, 0.0, cz)
                Sdf.AttributeSpec(spec, "xformOpOrder", Sdf.ValueTypeNames.TokenArray,
                                  variability=Sdf.VariabilityUniform).default = order
        self._enabled_attrs = {}

    def level_shape(self, level: int):
        crouch = level / self.levels
        h = self.standing_height * (1.0 - (1.0 - self.crouch_height_ratio) * crouch)
        off = -self.standing_height * (1.0 - self.crouch_height_ratio) * 0.5 * crouch
        return h, off

    def level_for_crouch(self, crouch_amount: float) -> int:
        t = 0.0 if crouch_amount < 0.0 else 1.0 if crouch_amount > 1.0 else crouch_amount
        return int(round(t * self.levels))

    def _attrs(self, level: int):
        from pxr import UsdPhysics

        if level not in self._enabled_attrs:
            self._enabled_attrs[level] = [
                UsdPhysics.CollisionAPI(self.stage.GetPrimAtPath(f"{self.root_path}/Level_{level}/{n}"))
                .GetCollisionEnabledAttr() for n in self.shape_names]
        return self._enabled_attrs[level]

    def set_level(self, level: int) -> None:
        if level == self.active_level:
            return
        for attr in self._attrs(self.active_level):
            attr.Set(False)
        for attr in self._attrs(level):
            attr.Set(True)
        self.active_level = level


def set_levels(colliders, levels) -> None:
    """Switch each collider to its target crouch level, skipping unchanged ones.

    Deliberately *not* wrapped in one ``Sdf.ChangeBlock``: batching the 12 attribute writes of all
    worlds into a single USD notice is ~2x cheaper but changes PhysX's shape update ordering and
    therefore the physics (0.1 m divergence within the first macrostep on the batched tape), so the
    sequential, baseline-identical order is kept. See ``perf/OPTIMIZATION_LOG.md``.
    """
    for collider, level in zip(colliders, levels):
        if level != collider.active_level:
            collider.set_level(level)
