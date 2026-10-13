"""Pure layout and coordinate checks for bounded collision fixtures."""
import unittest

import numpy as np

from casu_grid.collision import (
    cell_center_world_xy,
    chunk_xy,
    fixture_set,
    fixture_specs,
)
from casu_grid.geometry import (
    chunk_surface_mesh,
    rectangles_for_chunk,
    rect_to_isaac_spec,
)


class CollisionFixtureTests(unittest.TestCase):
    def test_fixture_inventory_and_cell_layouts(self):
        fixtures = {fixture.name: fixture for fixture in fixture_set()}
        self.assertEqual(set(fixtures), {
            "isolated_rock", "adjacent_soil", "group_2x2_concrete",
            "short_platform", "platform_crossing_chunk_seam", "vertical_wall",
            "glass_tile", "tile_destruction_before", "tile_destruction_after",
        })
        self.assertEqual(len(fixtures["isolated_rock"].cells), 1)
        self.assertEqual(len(fixtures["adjacent_soil"].cells), 2)
        self.assertEqual(len(fixtures["group_2x2_concrete"].cells), 4)
        self.assertEqual(len(fixtures["short_platform"].cells), 8)
        seam = fixture_specs(fixtures["platform_crossing_chunk_seam"])
        self.assertEqual([row["chunk_xy"][0] for row in seam], [0, 0, 1, 1])
        self.assertEqual(len(fixtures["vertical_wall"].cells), 8)
        self.assertEqual(fixtures["tile_destruction_after"].cells, ())

    def test_cell_centers_embed_xy_in_isaac_xz(self):
        self.assertEqual(cell_center_world_xy(0, 0), (-511.5, -511.5))
        self.assertEqual(cell_center_world_xy(1023, 1023), (511.5, 511.5))
        self.assertEqual(cell_center_world_xy(64, 65), (-447.5, -446.5))
        seam = fixture_specs({f.name: f for f in fixture_set()}[
            "platform_crossing_chunk_seam"])
        self.assertEqual(seam[0]["isaac_xyz"], [-449.5, 0.0, -311.5])
        self.assertEqual(seam[2]["isaac_xyz"], [-447.5, 0.0, -311.5])
        self.assertEqual(seam[1]["prim_path"].split("/")[3], "Chunk_0_3")
        self.assertEqual(seam[2]["prim_path"].split("/")[3], "Chunk_1_3")

    def test_chunk_indexing_matches_64_cell_partition(self):
        for index, expected in ((0, 0), (63, 0), (64, 1), (1023, 15)):
            self.assertEqual(chunk_xy(index, index), (expected, expected))

    def test_rectangle_decomposition_preserves_exact_chunk_occupancy(self):
        blocks = np.zeros((128, 64), dtype=np.uint16)
        blocks[0, 0] = 1
        blocks[1:4, 0:2] = 5
        blocks[63:65, 10] = 16  # Must remain split at the original chunk seam.
        blocks[4:12, 20] = 7
        blocks[7:10, 21:24] = 7
        rects = rectangles_for_chunk(blocks, 0, 0)
        rects.extend(rectangles_for_chunk(blocks, 1, 0))
        raster = np.zeros_like(blocks, dtype=np.uint8)
        for rect in rects:
            self.assertGreater(rect.width, 0)
            self.assertGreater(rect.height, 0)
            self.assertTrue(np.all(raster[rect.cell_x:rect.cell_x + rect.width,
                                         rect.cell_y:rect.cell_y + rect.height] == 0))
            raster[rect.cell_x:rect.cell_x + rect.width,
                   rect.cell_y:rect.cell_y + rect.height] = 1
        np.testing.assert_array_equal(raster, blocks != 0)
        seam_rects = [r for r in rects if r.cell_y == 10]
        self.assertEqual([(r.chunk_x, r.cell_x, r.width) for r in seam_rects],
                         [(0, 63, 1), (1, 64, 1)])

    def test_rectangle_transform_preserves_cell_footprint_and_chunk_path(self):
        blocks = np.zeros((64, 64), dtype=np.uint16)
        blocks[10:13, 20:22] = 1
        rect = rectangles_for_chunk(blocks, 0, 0)[0]
        spec = rect_to_isaac_spec(rect)
        self.assertEqual(spec["isaac_xyz"], [-500.5, 0.0, -491.0])
        self.assertEqual(spec["size_xyz"], [3.0, 0.2, 2.0])
        self.assertIn("Chunk_0_0", spec["prim_path"])

    def test_surface_mesh_is_closed_and_keeps_source_extrusion_axis(self):
        blocks = np.zeros((4, 4), dtype=np.uint16)
        blocks[1, 1] = 1
        vertices, faces = chunk_surface_mesh(blocks, 0, 0, chunk_size=4)
        self.assertEqual(vertices.shape, (8, 3))
        self.assertEqual(faces.shape, (6, 4))
        np.testing.assert_allclose(vertices.min(axis=0), [-1.0, -0.1, -1.0])
        np.testing.assert_allclose(vertices.max(axis=0), [0.0, 0.1, 0.0])
        edges = {}
        normals = []
        for face in faces:
            p = vertices[face]
            normal = np.cross(p[1] - p[0], p[2] - p[0])
            normals.append(tuple(np.sign(normal).astype(int)))
            for a, b in zip(face, np.roll(face, -1)):
                edge = tuple(sorted((int(a), int(b))))
                edges[edge] = edges.get(edge, 0) + 1
        self.assertTrue(all(count == 2 for count in edges.values()))
        self.assertEqual(set(normals), {
            (-1, 0, 0), (1, 0, 0), (0, -1, 0),
            (0, 1, 0), (0, 0, -1), (0, 0, 1),
        })

    def test_surface_mesh_removes_shared_cell_faces(self):
        blocks = np.zeros((4, 4), dtype=np.uint16)
        blocks[1:3, 1] = 1
        _, faces = chunk_surface_mesh(blocks, 0, 0, chunk_size=4)
        # Two extruded cells have 10 boundary quads; a shared internal side
        # would add two more faces.
        self.assertEqual(faces.shape, (10, 4))


if __name__ == "__main__":
    unittest.main()
