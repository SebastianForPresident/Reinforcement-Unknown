import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import CriticRollouts
import IndependentCritic
import ObservationNormalization as ObsNorm
import Types
import Train


def valid_spatial(count, rng):
    shape = (count, ObsNorm.SPATIAL_CHANNELS, CriticRollouts.WIDTH,
             CriticRollouts.HEIGHT)
    spatial = np.zeros(shape, dtype=np.float32)
    spatial[:, 0] = rng.random(shape[-2:], dtype=np.float32)
    spatial[:, 1] = rng.integers(0, 2, size=(count, *shape[-2:]))
    spatial[:, 2] = rng.random(shape[-2:], dtype=np.float32)
    sleep = rng.integers(0, ObsNorm.SLEEP_QUALITY_COUNT,
                         size=(count, *shape[-2:]))
    fluid = rng.integers(0, ObsNorm.FLUID_TYPE_COUNT,
                         size=(count, *shape[-2:]))
    for index in range(ObsNorm.SLEEP_QUALITY_COUNT):
        spatial[:, 3 + index] = sleep == index
    for index in range(ObsNorm.FLUID_TYPE_COUNT):
        spatial[:, 7 + index] = fluid == index
    spatial[:, 14, Types.SIGHT_RANGE_X, Types.SIGHT_RANGE_Y] = 1.0
    for sample in range(count):
        x = int(rng.integers(0, CriticRollouts.WIDTH))
        y = int(rng.integers(0, CriticRollouts.HEIGHT))
        spatial[sample, 15, x, y] = 1.0
    return spatial


class CompactSpatialTests(unittest.TestCase):
    def test_compact_spatial_round_trip_is_exact(self):
        rng = np.random.default_rng(12)
        original = valid_spatial(3, rng)
        restored = CriticRollouts._unpack_spatial(
            CriticRollouts._pack_spatial(original)
        )
        np.testing.assert_array_equal(restored, original)

    def test_rollout_file_round_trip(self):
        rng = np.random.default_rng(3)
        steps = 4
        spatial = valid_spatial(steps, rng)
        final_spatial = valid_spatial(1, rng)[0]
        general = rng.standard_normal((steps, 11), dtype=np.float32)
        final_general = rng.standard_normal(11, dtype=np.float32)
        buffer = SimpleNamespace(
            observations={
                "general": general[:, None],
                "spatial": spatial[:, None],
            },
            rewards=np.arange(steps, dtype=np.float32)[:, None],
            episode_starts=np.asarray([[1], [0], [0], [0]], dtype=np.float32),
            actions=np.zeros((steps, 1, 7), dtype=np.float32),
            values=np.linspace(0, 1, steps, dtype=np.float32)[:, None],
            log_probs=np.linspace(-1, 0, steps, dtype=np.float32)[:, None],
            returns=np.linspace(1, 2, steps, dtype=np.float32)[:, None],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollout.npz"
            CriticRollouts.save_rollout(
                path,
                1234,
                buffer,
                {"general": final_general[None],
                 "spatial": final_spatial[None]},
                np.asarray([False]),
            )
            restored = CriticRollouts.load_rollout(path)

        self.assertEqual(restored.timestep, 1234)
        np.testing.assert_array_equal(restored.observations["general"], general)
        np.testing.assert_array_equal(restored.observations["spatial"], spatial)
        np.testing.assert_array_equal(restored.final_observation["general"],
                                      final_general)
        np.testing.assert_array_equal(restored.final_observation["spatial"],
                                      final_spatial)
        np.testing.assert_array_equal(restored.returns, buffer.returns[:, 0])


class IndependentCriticTests(unittest.TestCase):
    def test_initialization_and_training_do_not_change_policy(self):
        class Features(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 4)

            def forward(self, observations):
                return self.linear(observations["general"])

        policy = SimpleNamespace(
            features_extractor=Features(),
            mlp_extractor=SimpleNamespace(value_net=nn.Linear(4, 4)),
            value_net=nn.Linear(4, 1),
        )
        before = {
            name: value.detach().clone()
            for name, value in policy.features_extractor.state_dict().items()
        }
        critic = IndependentCritic.IndependentCritic(policy)
        optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
        rollout = SimpleNamespace(
            observations={"general": np.ones((8, 3), dtype=np.float32)},
            returns=np.ones(8, dtype=np.float32),
        )
        IndependentCritic.train_rollout(
            critic, optimizer, rollout, torch.device("cpu"),
            batch_size=4, epochs=2,
        )
        for name, value in policy.features_extractor.state_dict().items():
            torch.testing.assert_close(value, before[name])


class RolloutCheckpointTests(unittest.TestCase):
    def test_training_progress_callback_starts_and_updates(self):
        callback = Train.TrainingProgressCallback()
        callback.model = SimpleNamespace(
            num_timesteps=100,
            _total_timesteps=200,
            _num_timesteps_at_start=100,
        )
        with mock.patch("sys.stderr"):
            callback._on_training_start()
            self.assertTrue(callback._on_step())
            callback._on_training_end()

    def test_checkpoint_is_saved_only_after_five_complete_updates(self):
        class Model:
            def __init__(self):
                self.num_timesteps = 0
                self.saved = []

            def save(self, path):
                self.saved.append(Path(path).name)

        with tempfile.TemporaryDirectory() as directory:
            callback = Train.RolloutCheckpointCallback(
                directory, "cb1", updates_per_save=5
            )
            callback.model = Model()
            callback._on_rollout_start()
            for rollout in range(1, 6):
                callback.model.num_timesteps = rollout * 2048
                callback._on_rollout_end()
                # The update consumes the completed rollout before the next
                # rollout-start callback is issued.
                callback._on_rollout_start()

            self.assertEqual(callback.model.saved, ["cb1_10240_steps"])
            callback._on_training_end()
            self.assertEqual(callback.model.saved, ["cb1_10240_steps"])


if __name__ == "__main__":
    unittest.main()
