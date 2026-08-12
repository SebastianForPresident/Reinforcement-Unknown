import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import ReinforcePPO


class ObservedReturnTests(unittest.TestCase):
    def test_lookahead_is_discounted_and_has_no_bootstrap(self):
        rewards = np.asarray([1, 2, 3, 4, 100], dtype=np.float32)
        starts = np.zeros(5, dtype=np.float32)
        actual = ReinforcePPO.observed_returns(rewards, starts, .5, 2)
        np.testing.assert_allclose(actual, [2.75, 4.5, 30.0])

    def test_returns_stop_at_episode_boundary(self):
        rewards = np.asarray([1, 2, 30, 40], dtype=np.float32)
        starts = np.asarray([1, 0, 1, 0], dtype=np.float32)
        actual = ReinforcePPO.observed_returns(rewards, starts, .5, 2)
        np.testing.assert_allclose(actual, [2.0, 2.0])

    def test_prefix_gae_bootstraps_at_train_boundary(self):
        buffer = SimpleNamespace(
            rewards=np.asarray([[1], [2], [99]], dtype=np.float32),
            values=np.asarray([[10], [20], [30]], dtype=np.float32),
            episode_starts=np.zeros((3, 1), dtype=np.float32),
        )
        advantages, returns = ReinforcePPO.gae_prefix(buffer, 2, .5, 1.0)
        np.testing.assert_allclose(advantages, [-0.5, -3.0])
        np.testing.assert_allclose(returns, [9.5, 17.0])

    def test_gamma_cutoff_is_below_one_percent(self):
        self.assertLess(.99 ** 459, .01)
        self.assertGreater(.99 ** 458, .009)


if __name__ == "__main__":
    unittest.main()
