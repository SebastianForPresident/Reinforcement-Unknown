import unittest

from casu_grid.legacy_probe_traces import PROBES, load_probe


class LegacyProbeTests(unittest.TestCase):
    def test_all_probes_load_with_complete_tick_rows(self):
        for name in PROBES:
            with self.subTest(probe=name):
                trace = load_probe(name)
                horizon = len(trace.actions)
                self.assertEqual(len(trace.pre_fixed), 10 * horizon)
                self.assertEqual(len(trace.post_physics), 10 * horizon)
                self.assertEqual(trace.pre_fixed[0]["move_dir"], [0.0, 0.0])
                self.assertTrue(trace.pre_fixed[0]["grounded"])
                self.assertIn("rb_position", trace.post_physics[0]["extra"])

    def test_seam_crossing_probe_geometry(self):
        trace = load_probe("chunk_seam_cross")
        x = [row["extra"]["rb_position"][0] for row in trace.post_physics]
        self.assertLess(x[0], -448.0)            # chunk seam at world x = -448 (tile 64)
        self.assertGreater(x[-1], -448.0)


class HullConfigTests(unittest.TestCase):
    def test_hull_segments_validated(self):
        from casu_grid.isaac_navigation_env import IsaacLayer1EnvConfig
        IsaacLayer1EnvConfig(player_collider="convex_hull", player_hull_segments=8).validate()
        for bad in (1, 31):
            with self.assertRaises(ValueError):
                IsaacLayer1EnvConfig(player_hull_segments=bad).validate()


if __name__ == "__main__":
    unittest.main()
