"""Randomized 128 x 128 XY1 mini-world reference, independent of canonical XY1."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from . import CasUGridState

MINI_WIDTH = 128
MINI_HEIGHT = 128
MINI_FIXTURE_ID = "xy1-mini-128x128-randomized-v1"

# Frequencies measured from the canonical CasU Layer 1 raster. Infinirock is
# reserved for the deliberately indestructible noisy side barriers.
_MATERIALS = np.asarray([1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 15, 16], dtype=np.uint16)
_WEIGHTS = np.asarray([160065, 60910, 14807, 8392, 2428, 463, 642, 394, 129, 12046,
                       2914, 33935, 8214], dtype=np.float64)
_WEIGHTS /= _WEIGHTS.sum()


def generate_mini_terrain(seed: int) -> np.ndarray:
    """Generate varied, CasU-material terrain while leaving a clear center spawn."""
    rng = np.random.default_rng(int(seed))
    width, height = MINI_WIDTH, MINI_HEIGHT
    blocks = np.zeros((width, height), dtype=np.uint16)

    # Continuous CasU-like bottom floor. Material varies, but every floor tile
    # is solid and standable; the upper face is level as in the source layer.
    blocks[:, :2] = rng.choice(np.asarray([1, 2, 15, 16], dtype=np.uint16),
                               size=(width, 2), p=[.46, .24, .17, .13])

    # Each boundary remains solid and Infinirock, with a noisy inward profile.
    left_depth = np.clip(3 + rng.integers(-1, 3, size=height), 2, 6)
    right_depth = np.clip(3 + rng.integers(-1, 3, size=height), 2, 6)
    for y in range(height):
        blocks[:int(left_depth[y]), y] = 14
        blocks[width-int(right_depth[y]):, y] = 14

    # Random coarse occupancy creates deposits and cavities instead of a uniform
    # wall. Occupied cells use measured CasU material proportions.
    coarse = rng.random((width, height))
    for _ in range(3):
        padded = np.pad(coarse, 1, mode="edge")
        coarse = (padded[:-2, 1:-1] + padded[2:, 1:-1] +
                  padded[1:-1, :-2] + padded[1:-1, 2:] + coarse) / 5.0
    # Match the canonical Layer 1 solid-cell fraction (~30.6%); the smooth
    # spatial field supplies connected deposits and cavities around that mean.
    threshold = 0.55 + 0.035 * np.sin(np.arange(height)[None, :] / 7.0)
    deposits = coarse > threshold
    deposits[:, :7] = False
    deposits[:, -1] = False  # open ceiling: never author a cap row
    for x in range(7, width-7):
        for y in range(7, height-1):
            if deposits[x, y] and blocks[x, y] == 0:
                blocks[x, y] = rng.choice(_MATERIALS, p=_WEIGHTS)

    # Clear only cells intersecting the standing body at the exact center pose.
    # This is collision clearance, not a surrounding chamber or spawn structure.
    center_x, center_y = width // 2, height // 2
    blocks[center_x-1:center_x+2, center_y-2:center_y+4] = 0
    # The bottom remains continuous and the top edge stays open except for side walls.
    return blocks


@dataclass(frozen=True)
class MiniWorldReference:
    seed: int
    terrain: np.ndarray

    @property
    def manifest(self) -> dict:
        encoded = self.terrain.tobytes(order="C")
        digest = hashlib.sha256(encoded).hexdigest()
        manifest = {
            "schema": "casu.xy1-mini-world.v1",
            "fixture_id": MINI_FIXTURE_ID,
            "content_id": "sha256:" + digest,
            "seed": int(self.seed),
            "world": {"width": MINI_WIDTH, "height": MINI_HEIGHT,
                      "chunk_size": 64, "array_order": "x-first; flat C-order; y fastest"},
            "runtime": {"player_spawn_grid_xy": [MINI_WIDTH // 2, MINI_HEIGHT // 2],
                        "player_world_position_xy": [0.5, 0.5]},
            "world_variant": "randomized_casu_terrain_no_pod_open_top_noisy_infinirock_sides",
        }
        return manifest

    @property
    def spawn_grid_xy(self):
        return MINI_WIDTH // 2, MINI_HEIGHT // 2

    @property
    def spawn_world_xy(self):
        return 0.5, 0.5

    def new_state(self, decisions=()):
        state = CasUGridState(MINI_WIDTH, MINI_HEIGHT,
                              camera_x=MINI_WIDTH // 2, camera_y=MINI_HEIGHT // 2,
                              decisions=decisions)
        state.blocks[...] = self.terrain
        return state


def load_mini_world_reference(seed: int = 391728405) -> MiniWorldReference:
    return MiniWorldReference(int(seed), generate_mini_terrain(int(seed)))
