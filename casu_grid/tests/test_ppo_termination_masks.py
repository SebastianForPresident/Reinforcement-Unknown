import unittest

import torch

from casu_grid.workbench.ppo import _generalized_advantage_estimate as gae


def run(terminated, truncated, next_value):
    """Two decisions of one world; the episode ends after decision 0 (terminal or truncation), decision 1 belongs to the next episode."""
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.tensor([[0.5], [0.4]])
    terminated_t = torch.tensor([[float(terminated)], [0.0]])
    dones = torch.tensor([[float(terminated or truncated)], [0.0]])
    next_values = torch.tensor([[next_value], [0.3]])
    return gae(rewards, values, dones, terminated_t, next_values, gamma=0.99, gae_lambda=0.95)


class TerminationMaskTests(unittest.TestCase):
    def test_true_terminal_does_not_bootstrap(self):
        adv_low, ret_low = run(True, False, next_value=0.0)
        adv_high, ret_high = run(True, False, next_value=1000.0)
        self.assertTrue(torch.equal(adv_low, adv_high))                 # the value of the post-exit state is irrelevant
        self.assertAlmostEqual(float(adv_low[0]), 1.0 - 0.5, places=6)    # delta = reward - value, nothing leaks in from the next episode

    def test_truncation_bootstraps(self):
        adv_low, _ = run(False, True, next_value=0.0)
        adv_high, _ = run(False, True, next_value=10.0)
        self.assertGreater(float(adv_high[0]), float(adv_low[0]) + 9.0)

    def test_advantage_recursion_stops_at_either_kind_of_end(self):
        for terminated, truncated in ((True, False), (False, True)):
            a1, _ = run(terminated, truncated, 0.0)
            rewards = torch.tensor([[1.0], [200.0]])                      # the next episode's rewards must not leak into this episode's advantage
            a2, _ = gae(rewards, torch.tensor([[0.5], [0.4]]), torch.tensor([[1.0], [0.0]]), torch.tensor([[float(terminated)], [0.0]]),
                        torch.tensor([[0.0], [0.3]]), 0.99, 0.95)
            self.assertAlmostEqual(float(a1[0]), float(a2[0]), places=6)


if __name__ == "__main__":
    unittest.main()


class CheckpointTerminationContractTests(unittest.TestCase):
    """A checkpoint can only be resumed under the termination contract it was trained with."""

    def trainer(self, termination):
        from casu_grid.tests.test_ppo_execution import FakeVectorEnv, build
        t = build("optimized")
        t.environment_contract = None if termination is None else {"termination": termination}
        return t

    def test_resume_under_the_same_contract_works_and_a_different_one_is_refused(self):
        import tempfile
        from pathlib import Path
        from casu_grid.workbench.contracts import TERMINATION_V1, TERMINATION_V2_LAYER_EXIT
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.pt"
            self.trainer(TERMINATION_V2_LAYER_EXIT).save(path)
            sidecar = __import__("json").loads((Path(tmp) / "c.pt.json").read_text())
            self.assertEqual(sidecar["environment_contract"], {"termination": TERMINATION_V2_LAYER_EXIT})
            self.trainer(TERMINATION_V2_LAYER_EXIT).load(path)                        # same contract: fine
            with self.assertRaisesRegex(ValueError, "different experiments"):
                self.trainer(TERMINATION_V1).load(path)                               # reference XY1 must not resume a v2 checkpoint
            # a checkpoint without a contract id is the reference XY1
            self.trainer(None).save(path)
            self.trainer(TERMINATION_V1).load(path)
            with self.assertRaisesRegex(ValueError, "different experiments"):
                self.trainer(TERMINATION_V2_LAYER_EXIT).load(path)
            self.assertEqual(self.trainer(TERMINATION_V1).load_policy(path)["termination_contract"], TERMINATION_V1)

    def test_contract_ids(self):
        from casu_grid.workbench.contracts import termination_contract_id as tid
        self.assertEqual(tid({"navigation_mode": True}), "xy1-termination-v1")
        self.assertEqual(tid({"navigation_mode": True, "bottom_exit_terminates": True}), "xy1-termination-v2-layer-exit")
        self.assertEqual(tid({"navigation_mode": True, "terminate_out_of_world": True}), "xy1-termination-v1x-out-of-world-truncation")
        self.assertEqual(tid({}), "layer1-default")
