import unittest

import numpy as np

from casu_grid.reference import load_layer1_reference
from casu_grid.world_objects import ObjectCollider, load_physics_audit, rasterize_cell_centers


class WorldObjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = load_layer1_reference()
        cls.objects, cls.occupancy = load_physics_audit()

    def test_tile_cells_are_fully_covered_by_source_colliders(self):
        solid = self.reference.terrain != 0
        centre = (self.occupancy & 1) != 0
        corners = (self.occupancy & 30) == 30
        self.assertTrue(np.all(centre[solid]))
        self.assertTrue(np.all(corners[solid]))

    def test_static_objects_explain_every_non_tile_physics_cell(self):
        solid = self.reference.terrain != 0
        extra = ((self.occupancy & 1) != 0) & ~solid
        self.assertEqual(int(extra.sum()), 3900)
        static_cover = rasterize_cell_centers(self.objects.colliders)
        unexplained = extra & ~static_cover
        # What static boxes/circles do not cover must be the initial footprint
        # of an excluded dynamic collider, so nothing is silently missing.
        dynamic_cover = np.zeros_like(extra)
        for item in self.objects.excluded_dynamic:
            cx, cy = item["bounds_center"]
            w, h = item["bounds_size"]
            box = ObjectCollider("dyn", "box", (cx, cy), (w + 0.1, h + 0.1))
            dynamic_cover |= rasterize_cell_centers([box])
        self.assertEqual(int((unexplained & ~dynamic_cover).sum()), 0)

    def test_spawn_shuttle_doors_are_present(self):
        doors = [c for c in self.objects.colliders if "shuttledoor" in c.name]
        self.assertEqual(sorted(round(c.center[0], 1) for c in doors), [-5.5, 5.5])
        for door in doors:
            self.assertEqual(door.size, (1.0, 6.0))

    def test_checksum_tampering_is_rejected(self):
        import tempfile, shutil
        from pathlib import Path
        from casu_grid.reference import FIXTURE_DIR
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "fixture"
            shutil.copytree(FIXTURE_DIR / "physics_audit", copy / "physics_audit")
            path = copy / "physics_audit" / "object_colliders.json"
            path.write_text(path.read_text().replace('"jumppad', '"jumppadX', 1))
            with self.assertRaises(ValueError):
                load_physics_audit(copy)


if __name__ == "__main__":
    unittest.main()
