import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import CounterfactualReplay as replay  # noqa: E402
import Types  # noqa: E402
import PrefixReplay  # noqa: E402


class CounterfactualReplayTests(unittest.TestCase):
    def test_default_tape_is_valid_categorical_cb1(self):
        tape = replay.default_tape(100)
        self.assertEqual(len(tape), 100)
        limits = (3, 2, 3, 2, 9, 11, 2)
        for action in tape:
            self.assertEqual(len(action), 7)
            for value, limit in zip(action, limits):
                self.assertGreaterEqual(value, 0)
                self.assertLess(value, limit)

    def test_first_difference_reports_nested_path(self):
        self.assertEqual(
            replay.first_difference({"a": {"b": 1}}, {"a": {"b": 2}})["path"],
            "root.a.b",
        )

    def test_observation_difference_reports_wire_field(self):
        left = np.zeros(1, dtype=Types.OBSERVATION_DTYPE)
        right = left.copy()
        right["LayerProgress"] = 1.0
        difference = replay.observation_difference(
            left.tobytes() + (0).to_bytes(8, "little"),
            right.tobytes() + (0).to_bytes(8, "little"),
        )
        self.assertEqual(difference["path"], "observation.LayerProgress")

    def test_snapshot_match_checks_independent_wire_baseline(self):
        left = {
            "state": {}, "arrays": {}, "observation_wire_sha256": "a",
            "observation_sha256": "a", "state_sha256": "s",
            "observation_base64": "AQ==",
        }
        right = dict(left)
        right["observation_wire_sha256"] = "b"
        self.assertEqual(
            replay.snapshot_match(left, right)["path"],
            "observation_wire_sha256",
        )

    def test_topology_presentation_disappearance_is_symmetric(self):
        baseline = [
            {"identity": "go:dust", "classification": "derived-rebuildable"},
            {"identity": "component:body", "classification": "causal"},
        ]
        live = [{"identity": "component:body", "classification": "causal"}]
        self.assertEqual(
            replay.causal_topology_keys(baseline),
            replay.causal_topology_keys(live),
        )

    def test_topology_causal_disappearance_still_fails(self):
        baseline = [
            {"identity": "component:body", "classification": "causal"},
        ]
        live = []
        self.assertNotEqual(
            replay.causal_topology_keys(baseline),
            replay.causal_topology_keys(live),
        )

    def test_presentation_ui_enabled_change_is_canonicalized(self):
        baseline = [{"identity": "component:globaldark-description-tmp", "enabled": True,
                     "classification": "presentation-ui-lifecycle"}]
        live = [{"identity": "component:globaldark-description-tmp", "enabled": False,
                 "classification": "presentation-ui-lifecycle"}]
        self.assertEqual(
            replay.causal_lifecycle_keys(baseline),
            replay.causal_lifecycle_keys(live),
        )
        self.assertEqual(replay.lifecycle_values(baseline), [])
        self.assertEqual(replay.lifecycle_values(live), [])

    def test_presentation_ui_topology_churn_still_blocks(self):
        baseline = [{"identity": "component:globaldark-description-tmp",
                     "classification": "presentation-ui-lifecycle"}]
        self.assertNotEqual(replay.causal_topology_keys(baseline), [])

    def test_causal_enabled_change_still_blocks(self):
        baseline = [{"identity": "component:player-body", "enabled": True,
                     "classification": "causal"}]
        live = [{"identity": "component:player-body", "enabled": False,
                 "classification": "causal"}]
        self.assertEqual(replay.causal_lifecycle_keys(baseline), ["component:player-body"])
        self.assertEqual(replay.causal_lifecycle_keys(live), ["component:player-body"])
        self.assertNotEqual(
            replay.lifecycle_values(baseline), replay.lifecycle_values(live),
        )

    def test_client_surfaces_owned_launch_failure_probe(self):
        client = replay.Client.__new__(replay.Client)
        client.failure_probe = lambda: "owned launch exited"
        self.assertEqual(client._launch_failure(), "owned launch exited")

    def test_prefix_detail_token_obeys_unity_session_contract(self):
        value = PrefixReplay.token(
            "generation-rng-isolation-aa-1789362100", "detail-left",
        )
        self.assertLessEqual(len(value), 64)
        self.assertRegex(value, r"^[a-z0-9-]+$")
        self.assertGreaterEqual(len(value), 8)


if __name__ == "__main__":
    unittest.main()
