"""Inspectable small terrain-collider fixtures for CasU-to-Isaac studies.

Mission 11 measurements found every non-air tile in the captured Layer 1
using ``Tile.ColliderType.Grid``. These fixtures model each occupied tile as
one unit cuboid in an X/Z plane; Y is the shallow extrusion axis. They are
small test geometry only and never expand to the canonical world.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CollisionCell:
    x: int
    y: int
    block_id: int


@dataclass(frozen=True)
class CollisionFixture:
    name: str
    cells: tuple[CollisionCell, ...]


def fixture_set() -> tuple[CollisionFixture, ...]:
    """Controlled layouts covering isolated, group, seam, wall, glass, edit."""
    def fixture(name: str, rows: Iterable[tuple[int, int, int]]) -> CollisionFixture:
        return CollisionFixture(name, tuple(CollisionCell(*row) for row in rows))

    return (
        fixture("isolated_rock", ((100, 100, 1),)),
        fixture("adjacent_soil", ((120, 100, 16), (121, 100, 16))),
        fixture("group_2x2_concrete", ((140, 100, 5), (141, 100, 5),
                                        (140, 101, 5), (141, 101, 5))),
        fixture("short_platform", tuple((x, 120, 1) for x in range(160, 168))),
        fixture("platform_crossing_chunk_seam", tuple((x, 200, 16) for x in range(62, 66))),
        fixture("vertical_wall", tuple((300, y, 6) for y in range(100, 108))),
        fixture("glass_tile", ((320, 100, 7),)),
        fixture("tile_destruction_before", ((340, 100, 1),)),
        fixture("tile_destruction_after", ()),
    )


def cell_center_world_xy(x: int, y: int, width: int = 1024,
                         height: int = 1024) -> tuple[float, float]:
    """Match WorldGeneration.BlockToWorldPos cell centers."""
    return x - width / 2 + 0.5, y - height / 2 + 0.5


def chunk_xy(x: int, y: int, chunk_size: int = 64) -> tuple[int, int]:
    return x // chunk_size, y // chunk_size


def isaac_cell_spec(cell: CollisionCell, *, width: int = 1024,
                    height: int = 1024) -> dict:
    """Return the explicit static cuboid transform and chunk owner."""
    world_x, world_y = cell_center_world_xy(cell.x, cell.y, width, height)
    cx, cy = chunk_xy(cell.x, cell.y)
    return {
        "block_id": cell.block_id,
        "grid_xy": [cell.x, cell.y],
        "world_xy": [world_x, world_y],
        "isaac_xyz": [world_x, 0.0, world_y],
        "size_xyz": [1.0, 0.2, 1.0],
        "chunk_xy": [cx, cy],
        "prim_path": f"/World/CasUFixtures/Chunk_{cx}_{cy}/Cell_{cell.x}_{cell.y}",
    }


def fixture_specs(fixture: CollisionFixture) -> list[dict]:
    return [isaac_cell_spec(cell) for cell in fixture.cells]
