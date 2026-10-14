"""Author static per-source-chunk terrain meshes into the active USD stage.

Import this module only from inside Isaac Sim/Kit so its USD/PhysX dependencies
resolve from the pinned runtime rather than host Python.
"""
from __future__ import annotations

from time import perf_counter

from .geometry import chunk_surface_mesh, rectangles_for_chunk


def author_layer1_collision(stage, blocks, *, chunk_size: int = 64,
                            half_thickness: float = 0.1,
                            root_path: str = "/World/CasUTerrain",
                            world_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
                            only_chunks=None) -> dict:
    """Create one static triangle-mesh collider per nonempty source chunk.

    The array is x-first. Each 64x64 chunk remains a separate collider and is
    closed on its own boundaries. The physics approximation is explicitly
    `none` (triangle mesh), not a convex hull that would fill empty regions.
    """
    import numpy as np
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics
    import isaaclab.sim as sim_utils

    if blocks.ndim != 2:
        raise ValueError("terrain must be a 2D array")
    root = UsdGeom.Xform.Define(stage, root_path)
    root.GetPrim().SetMetadata("documentation", "CasU seed 391728405 static chunk collision")
    begin = perf_counter()
    chunk_count = 0
    vertex_count = 0
    face_count = 0
    occupied = 0
    chunk_filter = None if only_chunks is None else {tuple(map(int, pair)) for pair in only_chunks}
    for chunk_x in range((blocks.shape[0] + chunk_size - 1) // chunk_size):
        for chunk_y in range((blocks.shape[1] + chunk_size - 1) // chunk_size):
            if chunk_filter is not None and (chunk_x, chunk_y) not in chunk_filter:
                continue
            x0, y0 = chunk_x * chunk_size, chunk_y * chunk_size
            cells = blocks[x0:min(x0 + chunk_size, blocks.shape[0]),
                           y0:min(y0 + chunk_size, blocks.shape[1])]
            cell_count = int(np.count_nonzero(cells))
            if cell_count == 0:
                continue
            points, quads = chunk_surface_mesh(blocks, chunk_x, chunk_y,
                                               chunk_size=chunk_size,
                                               half_thickness=half_thickness)
            if len(points) == 0 or len(quads) == 0:
                raise RuntimeError(f"nonempty chunk {(chunk_x, chunk_y)} produced no mesh")
            path = f"{root_path}/Chunk_{chunk_x}_{chunk_y}"
            mesh = UsdGeom.Mesh.Define(stage, path)
            mesh.CreatePointsAttr(points)
            if world_offset != (0.0, 0.0, 0.0):
                xform = UsdGeom.Xformable(mesh.GetPrim())
                xform.AddTranslateOp().Set(world_offset)
            triangles = np.stack((quads[:, (0, 1, 2)], quads[:, (0, 2, 3)]), axis=1)
            triangles = triangles.reshape(-1, 3)
            mesh.CreateFaceVertexCountsAttr(np.full(len(triangles), 3, dtype=np.int32))
            mesh.CreateFaceVertexIndicesAttr(triangles.reshape(-1))
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            sim_utils.define_collision_properties(
                path, sim_utils.CollisionPropertiesCfg(collision_enabled=True), stage=stage
            )
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
            mesh_collision.CreateApproximationAttr("none")
            mesh.GetPrim().CreateAttribute(
                "casu:chunkXY", Sdf.ValueTypeNames.Int2, custom=True
            ).Set(Gf.Vec2i(chunk_x, chunk_y))
            chunk_count += 1
            occupied += cell_count
            vertex_count += len(points)
            face_count += len(quads)
    return {
        "nonempty_chunk_collider_count": chunk_count,
        "occupied_cells": occupied,
        "mesh_vertex_count": vertex_count,
        "mesh_quad_count": face_count,
        "half_thickness": half_thickness,
        "build_seconds": perf_counter() - begin,
        "collision_approximation": "none",
        "representation": "one closed static triangle mesh per nonempty 64x64 CasU chunk",
    }


def _collision_offset_schemas(contact_offset):
    """API schemas + attribute writer for explicit PhysX contact/rest offsets (None = defaults)."""
    if contact_offset is None:
        return []
    return ["PhysxCollisionAPI"]


def _author_chunk_cuboids(stage, chunk_path, rectangles, *, half_thickness,
                          world_offset, material_path, contact_offset=None):
    """Write one static Cube collider per rectangle directly as Sdf prim specs.

    This is equivalent to the Isaac Lab ``CuboidCfg`` spawner (a unit Cube
    with translate/scale ops and ``PhysicsCollisionAPI``) but avoids its
    per-prim material prim and USD notification cost (about 5 ms each).
    A single shared physics material is bound by relationship.
    """
    from pxr import Gf, Sdf

    layer = stage.GetEditTarget().GetLayer()
    parent = layer.GetPrimAtPath(chunk_path)
    api_schemas = ["PhysicsCollisionAPI", *_collision_offset_schemas(contact_offset)]
    if material_path:
        api_schemas.append("MaterialBindingAPI")
    count = 0
    with Sdf.ChangeBlock():
        for rect in rectangles:
            center_x, center_z = rect.center_xy
            spec = Sdf.PrimSpec(
                parent, f"Rect_{rect.cell_x}_{rect.cell_y}_{rect.width}_{rect.height}",
                Sdf.SpecifierDef, "Cube")
            spec.SetInfo("apiSchemas", Sdf.TokenListOp.Create(prependedItems=api_schemas))
            Sdf.AttributeSpec(spec, "size", Sdf.ValueTypeNames.Double).default = 1.0
            Sdf.AttributeSpec(spec, "xformOp:translate", Sdf.ValueTypeNames.Double3).default = (
                Gf.Vec3d(center_x + world_offset[0], world_offset[1], center_z + world_offset[2]))
            Sdf.AttributeSpec(spec, "xformOp:scale", Sdf.ValueTypeNames.Float3).default = (
                Gf.Vec3f(float(rect.width), float(2 * half_thickness), float(rect.height)))
            Sdf.AttributeSpec(spec, "xformOpOrder", Sdf.ValueTypeNames.TokenArray,
                              variability=Sdf.VariabilityUniform).default = [
                "xformOp:translate", "xformOp:scale"]
            Sdf.AttributeSpec(spec, "physics:collisionEnabled", Sdf.ValueTypeNames.Bool).default = True
            if contact_offset is not None:
                Sdf.AttributeSpec(spec, "physxCollision:contactOffset", Sdf.ValueTypeNames.Float).default = float(contact_offset)
                Sdf.AttributeSpec(spec, "physxCollision:restOffset", Sdf.ValueTypeNames.Float).default = 0.0
            if material_path:
                relationship = Sdf.RelationshipSpec(spec, "material:binding:physics")
                relationship.targetPathList.explicitItems = [Sdf.Path(material_path)]
            count += 1
    return count


def author_layer1_box_collision(stage, blocks, *, chunk_size: int = 64,
                                half_thickness: float = 0.1,
                                physics_material=None,
                                physics_material_path: str | None = None,
                                only_chunks=None,
                                replace_chunks=False,
                                root_path: str = "/World/CasUTerrain",
                                world_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
                                contact_offset: float | None = None) -> dict:
    """Create chunk-preserving static cuboids from exact occupied rectangles.

    ``contact_offset`` sets the PhysX contact distance of the terrain shapes (rest offset 0).
    A pair's contact distance is the *sum* of both shapes' offsets; leaving terrain at the
    PhysX default makes a body sliding over a seam between coplanar boxes brake against
    the neighbouring box's edge (speculative contact), so the env passes a tiny value.

    This is the straightforward correctness representation used to validate
    contacts while the lower-prim-count triangle-mesh path is characterized.
    It creates one cube collider per merged raster rectangle, never per tile.
    """
    import numpy as np
    from pxr import UsdGeom

    if blocks.ndim != 2:
        raise ValueError("terrain must be a 2D array")
    if physics_material is not None and not physics_material_path:
        raise ValueError("physics_material requires an absolute shared physics_material_path")
    UsdGeom.Xform.Define(stage, root_path)
    begin = perf_counter()
    collider_count = 0
    occupied = 0
    nonempty_chunk_count = 0
    chunk_filter = None if only_chunks is None else {tuple(map(int, pair)) for pair in only_chunks}
    for chunk_x in range((blocks.shape[0] + chunk_size - 1) // chunk_size):
        for chunk_y in range((blocks.shape[1] + chunk_size - 1) // chunk_size):
            if chunk_filter is not None and (chunk_x, chunk_y) not in chunk_filter:
                continue
            x0, y0 = chunk_x * chunk_size, chunk_y * chunk_size
            cell_count = int(np.count_nonzero(blocks[x0:min(x0 + chunk_size, blocks.shape[0]),
                                                    y0:min(y0 + chunk_size, blocks.shape[1])]))
            if not cell_count:
                continue
            nonempty_chunk_count += 1
            chunk_path = f"{root_path}/Chunk_{chunk_x}_{chunk_y}"
            if replace_chunks:
                existing = stage.GetPrimAtPath(chunk_path)
                if existing and existing.IsValid():
                    stage.RemovePrim(chunk_path)
            UsdGeom.Xform.Define(stage, chunk_path)
            occupied += cell_count
            collider_count += _author_chunk_cuboids(
                stage, chunk_path, rectangles_for_chunk(
                    blocks, chunk_x, chunk_y, chunk_size=chunk_size,
                    world_shape=blocks.shape),
                half_thickness=half_thickness, world_offset=world_offset,
                material_path=(physics_material_path
                               if physics_material is not None else None),
                contact_offset=contact_offset)
    return {
        "nonempty_chunk_count": nonempty_chunk_count,
        "occupied_cells": occupied,
        "box_collider_count": collider_count,
        "build_seconds": perf_counter() - begin,
        "half_thickness": half_thickness,
        "representation": "one static cuboid per merged occupancy rectangle within each source chunk",
    }


def author_static_object_colliders(stage, colliders, *, half_thickness: float = 0.1,
                                   physics_material_path: str | None = None,
                                   root_path: str = "/World/CasUObjects",
                                   contact_offset: float | None = None) -> dict:
    """Author the measured non-tile static colliders (doors, crates, pods...).

    Boxes become scaled/rotated unit Cubes and circles become Y-axis Cylinders,
    using the same planar-to-3D extrusion as the tile cuboids. ``colliders`` are
    :class:`casu_grid.world_objects.ObjectCollider` records.
    """
    import math
    from pxr import Gf, Sdf, UsdGeom

    begin = perf_counter()
    UsdGeom.Xform.Define(stage, root_path)
    layer = stage.GetEditTarget().GetLayer()
    parent = layer.GetPrimAtPath(root_path)
    api_schemas = ["PhysicsCollisionAPI", *_collision_offset_schemas(contact_offset)]
    if physics_material_path:
        api_schemas.append("MaterialBindingAPI")
    count = 0
    with Sdf.ChangeBlock():
        for index, collider in enumerate(colliders):
            cx, cy = collider.center
            if collider.kind == "box":
                spec = Sdf.PrimSpec(parent, f"Box_{index}", Sdf.SpecifierDef, "Cube")
                Sdf.AttributeSpec(spec, "size", Sdf.ValueTypeNames.Double).default = 1.0
                order = ["xformOp:translate", "xformOp:rotateY", "xformOp:scale"]
                Sdf.AttributeSpec(spec, "xformOp:rotateY", Sdf.ValueTypeNames.Float).default = (
                    float(-collider.rotation_deg))
                Sdf.AttributeSpec(spec, "xformOp:scale", Sdf.ValueTypeNames.Float3).default = Gf.Vec3f(
                    float(collider.size[0]), float(2 * half_thickness), float(collider.size[1]))
            elif collider.kind == "circle":
                spec = Sdf.PrimSpec(parent, f"Circle_{index}", Sdf.SpecifierDef, "Cylinder")
                Sdf.AttributeSpec(spec, "radius", Sdf.ValueTypeNames.Double).default = float(collider.radius)
                Sdf.AttributeSpec(spec, "height", Sdf.ValueTypeNames.Double).default = float(2 * half_thickness)
                Sdf.AttributeSpec(spec, "axis", Sdf.ValueTypeNames.Token,
                                  variability=Sdf.VariabilityUniform).default = "Y"
                order = ["xformOp:translate"]
            else:
                raise ValueError(f"unsupported object collider kind {collider.kind!r}")
            spec.SetInfo("apiSchemas", Sdf.TokenListOp.Create(prependedItems=api_schemas))
            Sdf.AttributeSpec(spec, "xformOp:translate", Sdf.ValueTypeNames.Double3).default = (
                Gf.Vec3d(cx, 0.0, cy))
            Sdf.AttributeSpec(spec, "xformOpOrder", Sdf.ValueTypeNames.TokenArray,
                              variability=Sdf.VariabilityUniform).default = order
            Sdf.AttributeSpec(spec, "physics:collisionEnabled", Sdf.ValueTypeNames.Bool).default = True
            if contact_offset is not None:
                Sdf.AttributeSpec(spec, "physxCollision:contactOffset", Sdf.ValueTypeNames.Float).default = float(contact_offset)
                Sdf.AttributeSpec(spec, "physxCollision:restOffset", Sdf.ValueTypeNames.Float).default = 0.0
            spec.SetInfo("documentation", collider.name)
            if physics_material_path:
                relationship = Sdf.RelationshipSpec(spec, "material:binding:physics")
                relationship.targetPathList.explicitItems = [Sdf.Path(physics_material_path)]
            count += 1
    return {"object_collider_count": count, "build_seconds": perf_counter() - begin,
            "representation": "static scaled/rotated Cube and Y-axis Cylinder prims from measured Unity colliders"}
