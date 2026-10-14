"""The vectorised row-run extraction and rectangle decomposition equal the per-row reference, element for element and in order."""
import unittest

import numpy as np

from casu_grid import geometry
from casu_grid.geometry import TerrainRectangle, _row_runs, rectangles_for_chunk
from casu_grid.xy1_mini_world import generate_mini_terrain


def reference_rectangles(blocks, chunk_x, chunk_y, chunk_size=64, world_shape=(1024, 1024)):
    """The pre-Optimization-IV implementation (one numpy pad/diff per row)."""
    x0, y0 = chunk_x * chunk_size, chunk_y * chunk_size
    x1, y1 = min(x0 + chunk_size, blocks.shape[0]), min(y0 + chunk_size, blocks.shape[1])
    chunk_height = y1 - y0
    occupancy = blocks[x0:x1, y0:y1] != 0
    active, output = {}, []
    for local_y in range(chunk_height):
        runs = set(_row_runs(occupancy[:, local_y]))
        for run, start_y in list(active.items()):
            if run not in runs:
                output.append(TerrainRectangle(chunk_x, chunk_y, x0 + run[0], y0 + start_y, run[1] - run[0], local_y - start_y, world_shape[0], world_shape[1]))
                del active[run]
        for run in runs:
            active.setdefault(run, local_y)
    for run, start_y in active.items():
        output.append(TerrainRectangle(chunk_x, chunk_y, x0 + run[0], y0 + start_y, run[1] - run[0], chunk_height - start_y, world_shape[0], world_shape[1]))
    return output


class RowRuns(unittest.TestCase):
    def test_random_occupancies(self):
        rng = np.random.default_rng(3)
        for density in (0.0, 0.05, 0.3, 0.5, 0.9, 1.0):
            for shape in ((64, 64), (64, 20), (13, 64), (1, 5), (64, 1)):
                occ = rng.random(shape) < density
                got = geometry._chunk_row_runs(occ)
                want = [list(_row_runs(occ[:, y])) for y in range(shape[1])]
                self.assertEqual(got, want)

    def test_rectangles_equal_reference_on_mini_worlds(self):
        for seed in (391728405, 391728406, 17):
            blocks = generate_mini_terrain(seed)
            for cx in range(2):
                for cy in range(2):
                    got = rectangles_for_chunk(blocks, cx, cy, world_shape=blocks.shape)
                    want = reference_rectangles(blocks, cx, cy, world_shape=blocks.shape)
                    self.assertEqual(got, want)             # same rectangles in the same order

    def test_rectangles_equal_reference_after_edits(self):
        rng = np.random.default_rng(9)
        blocks = generate_mini_terrain(5)
        for _ in range(40):
            x, y = int(rng.integers(0, 128)), int(rng.integers(0, 128))
            blocks[x, y] = 0 if blocks[x, y] else 3
            cx, cy = x // 64, y // 64
            self.assertEqual(rectangles_for_chunk(blocks, cx, cy, world_shape=blocks.shape),
                             reference_rectangles(blocks, cx, cy, world_shape=blocks.shape))


if __name__ == "__main__":
    unittest.main()
