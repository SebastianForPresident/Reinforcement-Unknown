import unittest

from casu_grid.studio import stats


class StatsTests(unittest.TestCase):
    def test_wilson_matches_known_values(self):
        lo, hi = stats.wilson(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)
        self.assertEqual(stats.wilson(0, 0), (0.0, 1.0))
        self.assertEqual(stats.wilson(0, 50)[0], 0.0)

    def test_difference_matches_reference_values(self):
        d = stats.difference(28, 96, 17, 96)
        self.assertAlmostEqual(d["difference"], 11 / 96)
        self.assertAlmostEqual(d["p_value"], 0.0606, places=3)       # pooled two-proportion z test, computed independently
        self.assertLess(d["low"], 0.0)                                # includes zero: not significant at 5 %
        self.assertGreater(d["high"], 0.2)

    def test_clear_difference_excludes_zero(self):
        d = stats.difference(60, 100, 10, 100)
        self.assertGreater(d["low"], 0.3)
        self.assertLess(d["p_value"], 1e-9)

    def test_identical_rates(self):
        d = stats.difference(10, 50, 10, 50)
        self.assertEqual(d["difference"], 0.0)
        self.assertAlmostEqual(d["p_value"], 1.0)
        self.assertTrue(d["low"] < 0 < d["high"])


if __name__ == "__main__":
    unittest.main()
