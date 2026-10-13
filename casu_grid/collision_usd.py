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

    if blocks.ndim != 2 or blocks.shape[0] % chunk_size or blocks.shape[1] % chunk_size:
        raise ValueError("terrain dimensions must be 2D multiples of chunk_size")
    root = UsdGeom.Xform.Define(stage, root_path)
    root.GetPrim().SetMetadata("documentation", "CasU seed 391728405 static chunk collision")
    begin = perf_counter()
    chunk_count = 0
    vertex_count = 0
    face_count = 0
    occupied = 0
    chunk_filter = None if only_chunks is None else {tuple(map(int, pair)) for pair in only_chunks}
    for chunk_x in range(blocks.shape[0] // chunk_size):
        for chunk_y in range(blocks.shape[1] // chunk_size):
            if chunk_filter is not None and (chunk_x, chunk_y) not in chunk_filter:
                continue
            x0, y0 = chunk_x * chunk_size, chunk_y * chunk_size
            cells = blocks[x0:x0 + chunk_size, y0:y0 + chunk_size]
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


def author_layer1_box_collision(stage, blocks, *, chunk_size: int = 64,
                                half_thickness: float = 0.1,
                                physics_material=None,
                                physics_material_path: str | None = None,
                                only_chunks=None,
                                replace_chunks=False,
                                root_path: str = "/World/CasUTerrain",
                                world_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> dict:
    """Create chunk-preserving static cuboids from exact occupied rectangles.

    This is the straightforward correctness representation used to validate
    contacts while the lower-prim-count triangle-mesh path is characterized.
    It creates one cube collider per merged raster rectangle, never per tile.
    """
    import numpy as np
    from pxr import UsdGeom
    import isaaclab.sim as sim_utils

    if blocks.ndim != 2 or blocks.shape[0] % chunk_size or blocks.shape[1] % chunk_size:
        raise ValueError("terrain dimensions must be 2D multiples of chunk_size")
    UsdGeom.Xform.Define(stage, root_path)
    begin = perf_counter()
    collider_count = 0
    occupied = 0
    nonempty_chunk_count = 0
    chunk_filter = None if only_chunks is None else {tuple(map(int, pair)) for pair in only_chunks}
    for chunk_x in range(blocks.shape[0] // chunk_size):
        for chunk_y in range(blocks.shape[1] // chunk_size):
            if chunk_filter is not None and (chunk_x, chunk_y) not in chunk_filter:
                continue
            x0, y0 = chunk_x * chunk_size, chunk_y * chunk_size
            cell_count = int(np.count_nonzero(blocks[x0:x0 + chunk_size,
                                                    y0:y0 + chunk_size]))
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
            for rect in rectangles_for_chunk(blocks, chunk_x, chunk_y,
                                             chunk_size=chunk_size):
                center_x, center_z = rect.center_xy
                path = (f"{chunk_path}/Rect_{rect.cell_x}_{rect.cell_y}_"
                        f"{rect.width}_{rect.height}")
                cfg = sim_utils.CuboidCfg(
                    size=(float(rect.width), float(2 * half_thickness),
                          float(rect.height)),
                    physics_material=physics_material,
                    physics_material_path=physics_material_path or "material",
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True))
                cfg.func(path, cfg, translation=(center_x + world_offset[0],
                                                  world_offset[1],
                                                  center_z + world_offset[2]))
                collider_count += 1
    return {
        "nonempty_chunk_count": nonempty_chunk_count,
        "occupied_cells": occupied,
        "box_collider_count": collider_count,
        "build_seconds": perf_counter() - begin,
        "half_thickness": half_thickness,
        "representation": "one static cuboid per merged occupancy rectangle within each source chunk",
    }
