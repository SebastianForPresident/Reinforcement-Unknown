import unittest

import numpy as np

from casu_grid.workbench.ppo import navigation_metrics


def episodes(rows):
    keys = ("world", "reason", "length", "initial_distance", "best_distance", "final_distance", "blocks_destroyed", "return")
    return {k: [r[i] for r in rows] for i, k in enumerate(keys)}


def active(**kw):
    base = {"decisions": [64, 64, 64], "goal_distance": [10.0, 20.0, 1e6], "best_distance": [5.0, 10.0, 15.0], "initial_distance": [20.0, 20.0, 30.0],
            "blocks_destroyed": [0, 1, 2], "outside_world": [0.0, 0.0, 1.0]}
    base.update(kw)
    return {k: np.asarray(v, dtype=np.float32 if k not in ("decisions",) else np.int64) for k, v in base.items()}


class NavigationMetricsTests(unittest.TestCase):
    def test_no_finished_episode_gives_no_sample_not_zero(self):
        m = navigation_metrics(episodes([]), active(), 0.0, 4096)
        self.assertEqual(m["nav/completed_episodes"], 0)
        self.assertIsNone(m["nav/arrival_rate"])
        self.assertIsNone(m["nav/decisions_to_arrival_mean"])
        self.assertIsNone(m["nav/completed_progress_mean"])
        self.assertEqual(m["nav/arrivals"], 0)
        self.assertEqual(m["nav/arrivals_per_1k_decisions"], 0.0)       # zero arrivals in 4096 decisions IS a measurement

    def test_arrivals_timeouts_and_rates(self):
        eps = episodes([(0, "arrival", 100, 20.0, 1.0, 1.0, 3, 10.0), (1, "arrival", 300, 10.0, 0.5, 0.5, 0, 10.0), (2, "time_limit", 500, 10.0, 8.0, 40.0, 5, 1.0)])
        m = navigation_metrics(eps, None, None, 4000)
        self.assertEqual((m["nav/arrivals"], m["nav/timeouts"], m["nav/completed_episodes"]), (2, 1, 3))
        self.assertAlmostEqual(m["nav/arrival_rate"], 2 / 3)
        self.assertAlmostEqual(m["nav/arrivals_per_1k_decisions"], 0.5)
        self.assertEqual(m["nav/decisions_to_arrival_mean"], 200.0)
        self.assertAlmostEqual(m["nav/completed_progress_mean"], np.mean([1 - 1 / 20, 1 - 0.5 / 10, 1 - 8 / 10]))
        self.assertNotIn("nav/active_episodes", m)

    def test_layer_exits_are_counted_separately_from_arrivals_and_timeouts(self):
        eps = episodes([(0, "arrival", 100, 20.0, 1.0, 1.0, 3, 10.0), (1, "layer_exit", 40, 10.0, 6.0, 80.0, 5, 0.3), (2, "layer_exit", 60, 10.0, 9.0, 90.0, 5, 0.1),
                        (3, "time_limit", 500, 10.0, 8.0, 40.0, 5, 1.0)])
        m = navigation_metrics(eps, None, None, 4000)
        self.assertEqual((m["nav/arrivals"], m["nav/layer_exits"], m["nav/timeouts"], m["nav/completed_episodes"]), (1, 2, 1, 4))
        self.assertAlmostEqual(m["nav/layer_exits_per_1k_decisions"], 0.5)
        self.assertEqual(m["nav/decisions_to_layer_exit_mean"], 50.0)
        self.assertAlmostEqual(m["nav/arrival_rate"], 0.25)                  # layer exits are finished episodes that did not arrive
        none = navigation_metrics(episodes([]), None, None, 4000)
        self.assertEqual(none["nav/layer_exits"], 0)
        self.assertIsNone(none["nav/decisions_to_layer_exit_mean"])

    def test_agents_outside_the_world_do_not_corrupt_distance_statistics(self):
        m = navigation_metrics(episodes([]), active(), 10.0, 4096)
        self.assertAlmostEqual(m["nav/active_outside_world_fraction"], 1 / 3)
        self.assertEqual(m["nav/active_in_world_episodes"], 2)
        self.assertAlmostEqual(m["nav/active_distance_mean"], 15.0)         # the free-falling agent (1e6 tiles) is excluded
        self.assertAlmostEqual(m["nav/active_best_distance_mean"], 10.0)      # best-so-far distance is valid for every agent
        self.assertAlmostEqual(m["nav/blocks_destroyed_per_1k_decisions"], 1000 * 10.0 / 4096)

    def test_all_agents_outside_gives_no_current_distance(self):
        m = navigation_metrics(episodes([]), active(outside_world=[1, 1, 1]), 0.0, 4096)
        self.assertIsNone(m["nav/active_distance_mean"])
        self.assertEqual(m["nav/active_outside_world_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
