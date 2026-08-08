import csv
import sys
import tempfile
import unittest
import json
import threading
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import CasualtiesEnv
import Encoders
import EpisodeTrace
import ObservationEncoding
import ObservationNormalization as Normalization
import ProtocolValidation
import Reward
import Train
import Types


def live_observation():
    obs = np.zeros((), dtype=Types.OBSERVATION_DTYPE)
    obs["ProtocolVersion"] = Types.PROTOCOL_VERSION
    obs["WorldDimensions"]["X"] = 512
    obs["WorldDimensions"]["Y"] = 512
    obs["PlayerTilePosition"]["X"] = 256
    obs["PlayerTilePosition"]["Y"] = 400
    obs["LayerTimeRemaining"] = 1_000_000_000
    for name in (
        "BrainHealth", "Consciousness", "Stamina", "Energy",
        "BloodOxygen", "ClawHealth",
    ):
        obs[name] = 100.0
    return obs


class NormalizationTests(unittest.TestCase):
    def test_inputs_are_finite_bounded_and_categorical(self):
        obs = live_observation()
        obs["RelativeBlockMap"]["Health"][0, 0] = 420_133_760.0
        obs["RelativeBlockMap"]["Health"][1, 0] = 100.0
        obs["RelativeBlockMap"]["Health"][2, 0] = 99.7
        obs["RelativeBlockMap"]["Health"][3, 0] = 80.0
        obs["RelativeFluidMap"]["Type"][3, 4] = 6
        obs["Velocity"]["X"] = np.inf
        obs["Temperature"] = np.nan
        obs["RelativeLookPos"]["X"] = 4
        obs["RelativeLookPos"]["Y"] = -5

        processed = CasualtiesEnv.PreprocessObservation(obs)
        self.assertEqual(processed["spatial"].shape, (16, 85, 49))
        for value in processed.values():
            self.assertTrue(np.isfinite(value).all())
            self.assertGreaterEqual(float(value.min()), -1.0)
            self.assertLessEqual(float(value.max()), 1.0)
        self.assertTrue(np.isfinite(processed["spatial"]).all())
        self.assertGreaterEqual(float(processed["spatial"].min()), 0.0)
        self.assertLessEqual(float(processed["spatial"].max()), 1.0)

        # Infinirock is bounded, while current block damage remains distinct.
        self.assertAlmostEqual(float(processed["spatial"][0, 0, 0]), 1.0, places=6)
        self.assertEqual(float(processed["spatial"][1, 0, 0]), 1.0)
        self.assertEqual(float(processed["spatial"][1, 1, 0]), 0.0)
        self.assertGreater(
            float(processed["spatial"][0, 1, 0]),
            float(processed["spatial"][0, 3, 0]),
        )
        self.assertGreater(
            float(processed["spatial"][0, 1, 0])
            - float(processed["spatial"][0, 2, 0]),
            2e-4,
        )
        # Fluid 6 occupies exactly its categorical channel, not magnitude 6.
        fluid_slice = processed["spatial"][7:14, 3, 4]
        self.assertEqual(float(fluid_slice.sum()), 1.0)
        self.assertEqual(float(fluid_slice[6]), 1.0)


class ProtocolTests(unittest.TestCase):
    def test_absolute_position_uses_zero_based_world_bounds(self):
        obs = live_observation()
        obs["PlayerTilePosition"]["X"] = 511
        obs["PlayerTilePosition"]["Y"] = 511
        processed = CasualtiesEnv.PreprocessObservation(obs)
        ProtocolValidation.validate_reset_observation(obs, processed)

        obs["PlayerTilePosition"]["X"] = 512
        with self.assertRaisesRegex(RuntimeError, "outside world 512x512"):
            ProtocolValidation.validate_reset_observation(obs, processed)

    def test_observation_wait_rejects_reset_id_overrun(self):
        env = CasualtiesEnv.Env.__new__(CasualtiesEnv.Env)
        env.latest_obs = live_observation()
        env.latest_observation_id = 583
        env.obs_ready = threading.Event()

        previous_lock = CasualtiesEnv._observation_lock
        CasualtiesEnv._observation_lock = threading.Lock()
        try:
            with self.assertRaisesRegex(
                RuntimeError, r"expected 582, observed 583"
            ):
                env._wait_for_observation_id(582)
        finally:
            CasualtiesEnv._observation_lock = previous_lock

    def test_observation_wait_rejects_step_id_gap(self):
        env = CasualtiesEnv.Env.__new__(CasualtiesEnv.Env)
        env.latest_obs = live_observation()
        env.latest_observation_id = 584
        env.obs_ready = threading.Event()

        previous_lock = CasualtiesEnv._observation_lock
        CasualtiesEnv._observation_lock = threading.Lock()
        try:
            with self.assertRaisesRegex(
                RuntimeError, r"expected 583, observed 584"
            ):
                env._wait_for_observation_id(583)
        finally:
            CasualtiesEnv._observation_lock = previous_lock

    def test_ten_tick_action_contract(self):
        obs = live_observation()
        action = np.asarray([2, 1, 0, 1, 8, 0, 1], dtype=np.int64)
        expected = ProtocolValidation.wire_action(action)
        for name, value in zip(Types.POLICY_ACTION_DTYPE.names, expected):
            obs["PreviousAction"][name] = value
        obs["MacrostepPhysicsTicks"] = Types.POLICY_PHYSICS_TICKS
        obs["SimulationDeltaTime"] = 0.2
        processed = CasualtiesEnv.PreprocessObservation(obs)

        info = ProtocolValidation.validate_step_observation(
            obs, processed, action
        )
        self.assertEqual(info["macrostep_physics_ticks"], 10)
        self.assertAlmostEqual(info["simulation_delta_time"], 0.2, places=6)

        obs["PreviousAction"]["Attack"] = 0
        with self.assertRaisesRegex(RuntimeError, "Previous-action feedback mismatch"):
            ProtocolValidation.validate_step_observation(obs, processed, action)

    def test_reset_is_zero_tick_observation(self):
        obs = live_observation()
        processed = CasualtiesEnv.PreprocessObservation(obs)
        ProtocolValidation.validate_reset_observation(obs, processed)

    def test_player_dead_is_an_invariant_failure(self):
        obs = live_observation()
        obs["PlayerDead"] = True
        processed = CasualtiesEnv.PreprocessObservation(obs)
        with self.assertRaisesRegex(RuntimeError, "player is dead"):
            ProtocolValidation.validate_step_observation(
                obs, processed, np.zeros(7, dtype=np.int64)
            )


class ActionDispatchTests(unittest.TestCase):
    def test_reset_has_no_action_and_first_decode_dispatches_once(self):
        class Pipe:
            def __init__(self):
                self.messages = []

            def sendall(self, message):
                self.messages.append(message)

        class FakeServer:
            def __init__(self):
                self.action_pipe = Pipe()
                self.action_write_lock = threading.Lock()
                self.reset_requested = threading.Event()
                self.action_sequence = 0
                self.action_messages = []

            def SendDecodedAction(self):
                if self.reset_requested.is_set():
                    raise RuntimeError("action sent during reset")
                self.action_messages.append((
                    self.action_sequence,
                    self.move,
                    self.jump,
                    self.vertMove,
                    self.crouch,
                    self.lookdX,
                    self.lookdY,
                    self.attack,
                ))
                self.action_pipe.sendall(
                    f"ACTION {self.action_sequence}\n".encode("ascii")
                )

        fake_server = FakeServer()
        previous_server = CasualtiesEnv._server
        action = np.asarray([2, 1, 0, 1, 8, 0, 1], dtype=np.int64)
        try:
            CasualtiesEnv._server = fake_server
            CasualtiesEnv.SendReset(7)
            self.assertEqual(fake_server.action_pipe.messages, [b"RESET 7\n"])
            self.assertTrue(fake_server.reset_requested.is_set())

            # Env.reset clears this only after consuming RESET_READY.
            fake_server.reset_requested.clear()
            CasualtiesEnv.Decode(action)

            self.assertEqual(len(fake_server.action_messages), 1)
            self.assertEqual(
                fake_server.action_messages[0],
                (1, 1, 1, -1, 1, 4, -5, 1),
            )
            self.assertEqual(
                fake_server.action_pipe.messages,
                [b"RESET 7\n", b"ACTION 1\n"],
            )

            CasualtiesEnv.Decode(
                np.asarray([1, 0, 2, 0, 4, 5, 0], dtype=np.int64)
            )
            self.assertEqual(
                [message[0] for message in fake_server.action_messages],
                [1, 2],
            )
            self.assertEqual(
                fake_server.action_pipe.messages,
                [b"RESET 7\n", b"ACTION 1\n", b"ACTION 2\n"],
            )
        finally:
            CasualtiesEnv._server = previous_server


class TraceTests(unittest.TestCase):
    def test_watchdog_truncation_is_not_episode_completion(self):
        action = np.asarray([2, 1, 0, 1, 8, 0, 1], dtype=np.int64)
        obs = live_observation()
        obs["PreviousAction"]["MoveDirection"] = 1
        obs["PreviousAction"]["LookDX"] = 4

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            writer = EpisodeTrace.EpisodeTraceWriter(path)
            writer.begin_episode(1)
            writer.record(
                1, action, obs, 42, 0.0, {}, terminated=False, truncated=True
            )
            writer.finish_episode(complete=True)
            writer.close()

            with path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))

            self.assertEqual(row["episode_complete"], "False")
            self.assertEqual(row["action_move"], "2")
            self.assertEqual(row["action_look_dx"], "8")
            self.assertEqual(row["action_interact"], "0")
            self.assertEqual(row["obs_previousaction_movedirection"], "1")
            self.assertEqual(row["obs_previousaction_lookdx"], "4")


class RewardTests(unittest.TestCase):
    def test_only_new_deepest_progress_and_completion_pay(self):
        class State:
            pass

        state = State()
        obs = {"LayerProgress": 0.1}
        Reward.Reset(state, obs)
        obs["LayerProgress"] = 0.2
        self.assertAlmostEqual(Reward.Reward(obs, None, state), 1.0)
        obs["LayerProgress"] = 0.15
        self.assertEqual(Reward.Reward(obs, None, state), 0.0)
        obs["LayerProgress"] = 1.0
        self.assertAlmostEqual(Reward.Reward(obs, None, state), 18.0)
        self.assertEqual(Reward.Reward(obs, None, state), 0.0)


class EncoderTests(unittest.TestCase):
    def test_full_resolution_forward_backward(self):
        CasualtiesEnv.Init_General()
        observation_space = gym.spaces.Dict({
            "general": gym.spaces.Box(
                -1.0, 1.0, (CasualtiesEnv._general_input_dim,), np.float32
            ),
            "spatial": gym.spaces.Box(
                0.0, 1.0, (Normalization.SPATIAL_CHANNELS, 85, 49), np.float32
            ),
        })
        encoder = ObservationEncoding.CasualtiesFeatureExtractor(observation_space)
        spatial = torch.rand(2, Normalization.SPATIAL_CHANNELS, 85, 49)
        general = torch.rand(2, CasualtiesEnv._general_input_dim) * 2.0 - 1.0
        output = encoder({"spatial": spatial, "general": general})
        output.square().mean().backward()

        self.assertEqual(tuple(output.shape), (2, 256))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in encoder.parameters()
        ))
        self.assertFalse(any(
            isinstance(module, (torch.nn.MaxPool2d, torch.nn.AvgPool2d))
            for module in encoder.modules()
        ))
        for module in encoder.modules():
            if isinstance(module, torch.nn.Conv2d):
                self.assertEqual(module.stride, (1, 1))


class TrainingProtocolTests(unittest.TestCase):
    def test_resume_requires_matching_cb1_manifest(self):
        class FakeEnv:
            action_space = gym.spaces.MultiDiscrete([3, 2, 3, 2, 9, 11, 2])
            observation_space = gym.spaces.Dict({
                "general": gym.spaces.Box(-1.0, 1.0, (139,), np.float32),
                "spatial": gym.spaces.Box(
                    0.0, 1.0,
                    (Normalization.SPATIAL_CHANNELS, 85, 49), np.float32
                ),
            })

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "Pre-CB1 checkpoints"):
                Train.ValidateOrWriteProtocolManifest(
                    run_dir, FakeEnv(), resuming=True
                )
            Train.ValidateOrWriteProtocolManifest(
                run_dir, FakeEnv(), resuming=False
            )
            manifest = json.loads(
                (run_dir / Train.PROTOCOL_MANIFEST_NAME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["checkpoint"], "CB1")
            self.assertEqual(manifest["reward"], "VB1")
            self.assertEqual(manifest["protocol_version"], 1)
            self.assertEqual(manifest["gamma"], 0.99)
            self.assertEqual(manifest["gae_lambda"], 0.95)
            self.assertEqual(manifest["n_steps"], 2048)
            self.assertEqual(manifest["model_schema_version"], 1)
            Train.ValidateOrWriteProtocolManifest(
                run_dir, FakeEnv(), resuming=True
            )

            expected = manifest.copy()
            for key, incompatible_value in (
                ("gae_lambda", 0.90),
                ("n_steps", 1024),
                ("model_schema_version", 2),
            ):
                incompatible = expected.copy()
                incompatible[key] = incompatible_value
                (run_dir / Train.PROTOCOL_MANIFEST_NAME).write_text(
                    json.dumps(incompatible), encoding="utf-8"
                )
                with self.assertRaisesRegex(RuntimeError, "protocol manifest"):
                    Train.ValidateOrWriteProtocolManifest(
                        run_dir, FakeEnv(), resuming=True
                    )
                (run_dir / Train.PROTOCOL_MANIFEST_NAME).write_text(
                    json.dumps(expected), encoding="utf-8"
                )


if __name__ == "__main__":
    unittest.main()
