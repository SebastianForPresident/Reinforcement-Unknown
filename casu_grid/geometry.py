"""Chunk-local exact raster-to-rectangle terrain collision geometry.

The source tiles use Unity Grid colliders (unit square occupancy). This module
represents the union of those occupied cells as non-overlapping axis-aligned
rectangles, separately per source 64x64 chunk. It does not merge across chunk
boundaries. Rectangle partitioning preserves the raster occupancy exactly;
engine contact solver parity remains a separate measurement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .reference import Layer1Reference


@dataclass(frozen=True)
class TerrainRectangle:
    chunk_x: int
    chunk_y: int
    cell_x: int
    cell_y: int
    width: int
    height: int
    world_width: int = 1024
    world_height: int = 1024

    @property
    def center_xy(self) -> tuple[float, float]:
        return (self.cell_x + self.width / 2 - self.world_width / 2,
                self.cell_y + self.height / 2 - self.world_height / 2)


def _row_runs(row: np.ndarray) -> Iterator[tuple[int, int]]:
    """Yield half-open occupied x intervals from a 1D boolean array."""
    edges = np.diff(np.pad(row.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return iter(zip(starts.tolist(), ends.tolist()))


def _chunk_row_runs(occupancy: np.ndarray) -> list[list[tuple[int, int]]]:
    """:func:`_row_runs` of every row of ``occupancy`` (x-first) in one vectorised pass; same runs, same order within a row."""
    width, height = occupancy.shape
    padded = np.zeros((width + 2, height), dtype=np.int8)
    padded[1:-1] = occupancy
    edges = padded[1:] - padded[:-1]                           # (width + 1, height)
    start_x, start_y = np.nonzero(edges == 1)                  # column-major scan: ascending y, then x -- use a stable sort by row
    end_x, end_y = np.nonzero(edges == -1)
    order_s, order_e = np.argsort(start_y, kind="stable"), np.argsort(end_y, kind="stable")
    start_x, start_y, end_x, end_y = start_x[order_s], start_y[order_s], end_x[order_e], end_y[order_e]
    rows = [[] for _ in range(height)]
    for y, a, b in zip(start_y.tolist(), start_x.tolist(), end_x.tolist()):      # runs of a row alternate start/end, so zip pairs them
        rows[y].append((a, b))
    return rows


def rectangles_for_chunk(blocks: np.ndarray, chunk_x: int, chunk_y: int,
                         *, chunk_size: int = 64,
                         world_shape: tuple[int, int] = (1024, 1024)) -> list[TerrainRectangle]:
    """Decompose nonzero cells into exact non-overlapping rectangles.

    Identical x-runs on consecutive y-rows are extended vertically. All
    coordinates are grid-cell coordinates; output remains local to one chunk.
    """
    if blocks.ndim != 2:
        raise ValueError("terrain must be a 2D x-first array")
    x0, y0 = chunk_x * chunk_size, chunk_y * chunk_size
    x1, y1 = min(x0 + chunk_size, blocks.shape[0]), min(y0 + chunk_size, blocks.shape[1])
    if x0 < 0 or y0 < 0 or x0 >= blocks.shape[0] or y0 >= blocks.shape[1]:
        raise ValueError("chunk lies outside terrain bounds")
    chunk_height = y1 - y0
    occupancy = blocks[x0:x1, y0:y1] != 0
    active: dict[tuple[int, int], int] = {}
    output: list[TerrainRectangle] = []
    row_runs = _chunk_row_runs(occupancy)
    for local_y in range(chunk_height):
        runs = set(row_runs[local_y])
        for run, start_y in list(active.items()):
            if run not in runs:
                output.append(TerrainRectangle(chunk_x, chunk_y,
                                               x0 + run[0], y0 + start_y,
                                               run[1] - run[0], local_y - start_y,
                                               world_shape[0], world_shape[1]))
                del active[run]
        for run in runs:
            active.setdefault(run, local_y)
    for run, start_y in active.items():
        output.append(TerrainRectangle(chunk_x, chunk_y, x0 + run[0],
                                       y0 + start_y, run[1] - run[0],
                                       chunk_height - start_y, world_shape[0], world_shape[1]))
    return output


def rectangles_for_world(reference: Layer1Reference,
                         *, chunk_size: int = 64) -> list[TerrainRectangle]:
    """Build all occupied terrain rectangles, preserving source chunks."""
    blocks = reference.terrain
    result = []
    for chunk_x in range((blocks.shape[0] + chunk_size - 1) // chunk_size):
        for chunk_y in range((blocks.shape[1] + chunk_size - 1) // chunk_size):
            result.extend(rectangles_for_chunk(blocks, chunk_x, chunk_y,
                                               chunk_size=chunk_size,
                                               world_shape=blocks.shape))
    return result


def rect_to_isaac_spec(rect: TerrainRectangle) -> dict:
    """Map a raster rectangle to an Isaac cuboid in the X/Z terrain plane."""
    center_x, center_y = rect.center_xy
    return {
        "chunk_xy": [rect.chunk_x, rect.chunk_y],
        "grid_origin_xy": [rect.cell_x, rect.cell_y],
        "grid_size_xy": [rect.width, rect.height],
        "isaac_xyz": [center_x, 0.0, center_y],
        "size_xyz": [float(rect.width), 0.2, float(rect.height)],
        "prim_path": (f"/World/CasUTerrain/Chunk_{rect.chunk_x}_{rect.chunk_y}/"
                      f"Rect_{rect.cell_x}_{rect.cell_y}_{rect.width}_{rect.height}"),
    }


def chunk_surface_mesh(blocks: np.ndarray, chunk_x: int, chunk_y: int,
                       *, chunk_size: int = 64,
                       half_thickness: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Create a closed, chunk-local surface mesh for occupied grid cells.

    The source plane maps to Isaac X/Z, so Y is the small extrusion axis and
    source gravity remains Isaac -Z. Faces shared by occupied cells inside a
    chunk are removed. Chunk edges stay closed independently, preserving the
    source's segmentation and making seam behavior observable.

    Returns float32 vertices `(N,3)` and int32 quad indices `(M,4)`.
    """
    if blocks.ndim != 2:
        raise ValueError("terrain must be a 2D x-first array")
    x0, y0 = chunk_x * chunk_size, chunk_y * chunk_size
    x1, y1 = min(x0 + chunk_size, blocks.shape[0]), min(y0 + chunk_size, blocks.shape[1])
    if x0 < 0 or y0 < 0 or x0 >= blocks.shape[0] or y0 >= blocks.shape[1]:
        raise ValueError("chunk lies outside terrain bounds")
    occ = blocks[x0:x1, y0:y1] != 0
    vertex_ids: dict[tuple[int, int, int], int] = {}
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    low_y, high_y = -1, 1

    def add_face(corners: tuple[tuple[int, int, int], ...]) -> None:
        ids = []
        for point in corners:
            index = vertex_ids.get(point)
            if index is None:
                index = len(vertices)
                vertex_ids[point] = index
                vertices.append((float(point[0]), float(point[1]), float(point[2])))
            ids.append(index)
        faces.append(tuple(ids))

    for local_x, local_z in np.argwhere(occ):
        x, z = int(local_x), int(local_z)
        # End caps: winding points outward along +/- Isaac Y.
        add_face(((x, low_y, z), (x + 1, low_y, z),
                  (x + 1, low_y, z + 1), (x, low_y, z + 1)))
        add_face(((x, high_y, z), (x, high_y, z + 1),
                  (x + 1, high_y, z + 1), (x + 1, high_y, z)))
        if x == 0 or not occ[x - 1, z]:
            add_face(((x, low_y, z), (x, low_y, z + 1),
                      (x, high_y, z + 1), (x, high_y, z)))
        if x == occ.shape[0] - 1 or not occ[x + 1, z]:
            add_face(((x + 1, low_y, z), (x + 1, high_y, z),
                      (x + 1, high_y, z + 1), (x + 1, low_y, z + 1)))
        if z == 0 or not occ[x, z - 1]:
            add_face(((x, low_y, z), (x, high_y, z),
                      (x + 1, high_y, z), (x + 1, low_y, z)))
        if z == occ.shape[1] - 1 or not occ[x, z + 1]:
            add_face(((x, low_y, z + 1), (x + 1, low_y, z + 1),
                      (x + 1, high_y, z + 1), (x, high_y, z + 1)))

    if not faces:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 4), dtype=np.int32)
    points = np.asarray(vertices, dtype=np.float32)
    points[:, 1] *= float(half_thickness)
    points[:, 0] += x0 - blocks.shape[0] / 2
    points[:, 2] += y0 - blocks.shape[1] / 2
    return points, np.asarray(faces, dtype=np.int32)
