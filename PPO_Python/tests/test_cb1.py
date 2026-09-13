import csv
import sys
import tempfile
import unittest
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
import ResourceBudget
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
    def test_v2_reduced_wire_payload_size(self):
        self.assertEqual(Types.PROTOCOL_VERSION, 2)
        self.assertEqual(Types.OBSERVATION_DTYPE.itemsize, 42_004)

    def test_training_truncation_limits_are_part_of_environment_contract(self):
        self.assertIsNone(CasualtiesEnv.TruncationReason(99_999, 25_000))
        self.assertEqual(
            CasualtiesEnv.TruncationReason(100_000, 99_999),
            "max_episode_steps",
        )
        self.assertEqual(
            CasualtiesEnv.TruncationReason(80_000, 5_000),
            "stagnant_best_depth",
        )
        self.assertIsNone(CasualtiesEnv.TruncationReason(79_999, 5_000))
        self.assertEqual(
            CasualtiesEnv.TruncationReason(1_000, 999, inactive_steps=75),
            "capable_inactivity",
        )
        self.assertIsNone(CasualtiesEnv.TruncationReason(
            9_999,
            0,
            mobile_stagnant_steps=9_999,
        ))
        self.assertEqual(
            CasualtiesEnv.TruncationReason(
                10_000,
                0,
                mobile_stagnant_steps=10_000,
            ),
            "mobile_stagnation",
        )
        self.assertEqual(
            CasualtiesEnv.AdvanceMobileStagnation(9_999, 0.0),
            10_000,
        )
        self.assertEqual(CasualtiesEnv.AdvanceMobileStagnation(9_999, 0.001), 0)

    def test_capable_inactivity_exempts_incapacity(self):
        obs = live_observation()
        previous_position = (256, 400)
        previous_health = obs["RelativeBlockMap"]["Health"].copy()
        self.assertEqual(
            CasualtiesEnv.InactivityStatus(
                obs, previous_position, previous_health
            ),
            {
                "capable": True,
                "tile_unchanged": True,
                "velocity_inactive": True,
                "block_health_unchanged": True,
            },
        )
        status = CasualtiesEnv.InactivityStatus(
            obs, previous_position, previous_health
        )
        self.assertTrue(CasualtiesEnv.PhysicallyInactive(status))

        obs["Velocity"]["X"] = 0.1
        status = CasualtiesEnv.InactivityStatus(
            obs, previous_position, previous_health
        )
        self.assertFalse(CasualtiesEnv.PhysicallyInactive(status))
        obs["Velocity"]["X"] = 0.0
        obs["RelativeBlockMap"]["Health"][0, 0] = 1.0
        status = CasualtiesEnv.InactivityStatus(
            obs, previous_position, previous_health
        )
        self.assertFalse(CasualtiesEnv.PhysicallyInactive(status))
        obs["RelativeBlockMap"]["Health"][0, 0] = 0.0
        for field, value in (
            ("TimeRagdolled", 1.0),
            ("Consciousness", 0.0),
            ("PlayerDead", True),
        ):
            incapacitated = live_observation()
            incapacitated[field] = value
            status = CasualtiesEnv.InactivityStatus(
                incapacitated, previous_position, previous_health
            )
            self.assertFalse(status["capable"])
            self.assertTrue(CasualtiesEnv.PhysicallyInactive(status))
            self.assertEqual(
                CasualtiesEnv.AdvanceInactivity(74, status, False),
                0,
            )
            self.assertEqual(
                CasualtiesEnv.AdvanceInactivity(
                    74, status, False, count_incapacity=True
                ),
                75,
            )

        self.assertEqual(
            CasualtiesEnv.AdvanceInactivity(74, status, True),
            0,
        )

        jittering = live_observation()
        jittering["TimeRagdolled"] = 1.0
        jittering["Velocity"]["X"] = 1.0
        jittering["PlayerTilePosition"]["X"] += 1
        jittering["RelativeBlockMap"]["Health"][0, 0] = 1.0
        jittering_status = CasualtiesEnv.InactivityStatus(
            jittering, previous_position, previous_health
        )
        self.assertFalse(CasualtiesEnv.PhysicallyInactive(jittering_status))
        self.assertEqual(
            CasualtiesEnv.AdvanceInactivity(74, jittering_status, True),
            0,
        )

        capable = CasualtiesEnv.InactivityStatus(
            live_observation(), previous_position, previous_health
        )
        self.assertEqual(CasualtiesEnv.AdvanceInactivity(74, capable, True), 0)

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

    def test_functional_godmode_requires_restored_stamina(self):
        obs = live_observation()
        obs["Stamina"] = 0.0
        processed = CasualtiesEnv.PreprocessObservation(obs)
        with self.assertRaisesRegex(RuntimeError, "Stamina=0.000"):
            ProtocolValidation.validate_reset_observation(obs, processed)

        ProtocolValidation.validate_reset_observation(
            obs, processed, natural_stamina=True
        )

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
            CasualtiesEnv.SendReset(7, 123456)
            self.assertEqual(
                fake_server.action_pipe.messages,
                [b"RESET 7 123456\n"],
            )
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
                [b"RESET 7 123456\n", b"ACTION 1\n"],
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
                [
                    b"RESET 7 123456\n",
                    b"ACTION 1\n",
                    b"ACTION 2\n",
                ],
            )
        finally:
            CasualtiesEnv._server = previous_server

    def test_world_seed_validation(self):
        self.assertEqual(CasualtiesEnv.NormalizeWorldSeed(0), 0)
        self.assertEqual(
            CasualtiesEnv.NormalizeWorldSeed(np.int64(2_147_483_647)),
            2_147_483_647,
        )
        for invalid in (-1, 2_147_483_648, True, 1.5, "17"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    CasualtiesEnv.NormalizeWorldSeed(invalid)


class TraceTests(unittest.TestCase):
    def test_watchdog_truncation_is_not_episode_completion(self):
        action = np.asarray([2, 1, 0, 1, 8, 0, 1], dtype=np.int64)
        obs = live_observation()
        obs["PreviousAction"]["MoveDirection"] = 1
        obs["PreviousAction"]["LookDX"] = 4

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            writer = EpisodeTrace.EpisodeTraceWriter(path)
            writer.begin_episode(1, world_seed=123456)
            writer.record(
                1, action, obs, 42, 0.0, {}, terminated=False, truncated=True
            )
            writer.finish_episode(complete=True)
            writer.close()

            with path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))

            self.assertEqual(row["episode_complete"], "False")
            self.assertEqual(row["world_seed"], "123456")
            self.assertEqual(row["action_move"], "2")
            self.assertEqual(row["action_look_dx"], "8")
            self.assertEqual(row["action_interact"], "0")
            self.assertEqual(row["obs_previousaction_movedirection"], "1")
            self.assertEqual(row["obs_previousaction_lookdx"], "4")

    def test_watchdog_penalty_and_gates_have_stable_trace_columns(self):
        action = np.asarray([1, 0, 1, 0, 4, 5, 0], dtype=np.int64)
        obs = live_observation()
        first_info = {
            "incomplete_episode": 0.0,
            "inactive_steps": 74,
            "mobile_stagnant_steps": 9_999,
            "max_mobile_stagnant_steps": 10_000,
            "continuous_progress_improved": False,
            "mobile_stagnation_timer_advanced": True,
            "inactivity_capable": True,
            "inactivity_incapacitated": False,
            "inactivity_physical_gates_met": True,
            "inactivity_tile_unchanged": True,
            "inactivity_velocity_inactive": True,
            "inactivity_block_health_unchanged": True,
            "inactivity_depth_unchanged": True,
            "inactivity_timer_advanced": True,
            "inactivity_timer_mode": "physical",
            "inactivity_timer_advanced_while_incapacitated": False,
            "watchdog_failed": False,
            "truncation_reason": None,
        }
        final_info = dict(first_info)
        final_info.update({
            "incomplete_episode": -10.0,
            "mobile_stagnant_steps": 10_000,
            "watchdog_failed": True,
            "truncation_reason": "mobile_stagnation",
        })

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            writer = EpisodeTrace.EpisodeTraceWriter(path)
            writer.begin_episode(1, world_seed=7)
            writer.record(74, action, obs, 74, 0.0, first_info, False, False)
            writer.record(75, action, obs, 75, -10.0, final_info, True, False)
            writer.finish_episode(complete=False)
            writer.close()

            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(rows[-1]["reward_incomplete_episode"], "-10.0")
            self.assertEqual(rows[-1]["reward_inactive_steps"], "74")
            self.assertEqual(
                rows[-1]["reward_mobile_stagnant_steps"], "10000"
            )
            self.assertEqual(
                rows[-1]["reward_max_mobile_stagnant_steps"], "10000"
            )
            self.assertEqual(
                rows[-1]["reward_continuous_progress_improved"], "False"
            )
            self.assertEqual(rows[-1]["reward_watchdog_failed"], "True")
            self.assertEqual(
                rows[-1]["reward_truncation_reason"], "mobile_stagnation"
            )
            self.assertEqual(rows[-1]["episode_complete"], "False")


class SuccessEpisodeLengthTests(unittest.TestCase):
    def test_success_mean_excludes_failures_from_terminal_window(self):
        callback = Train.SuccessEpisodeLengthCallback(window_size=4)
        callback.model = type("Model", (), {"num_timesteps": 123})()
        callback.locals = {
            "dones": [True, True, True],
            "infos": [
                {"episode_steps": 4_000, "watchdog_failed": False},
                {"episode_steps": 500, "watchdog_failed": True},
                {"episode_steps": 6_000, "watchdog_failed": False},
            ],
        }

        self.assertTrue(callback._on_step())
        self.assertEqual(callback.success_mean(), 5_000.0)

    def test_window_is_last_terminal_episodes_not_last_successes(self):
        callback = Train.SuccessEpisodeLengthCallback(window_size=2)
        callback.model = type("Model", (), {"num_timesteps": 123})()
        callback.locals = {
            "dones": [True, True, True],
            "infos": [
                {"episode_steps": 9_000, "watchdog_failed": False},
                {"episode_steps": 1_000, "watchdog_failed": True},
                {"episode_steps": 5_000, "watchdog_failed": False},
            ],
        }

        callback._on_step()
        self.assertEqual(
            [(length, succeeded) for length, succeeded, _ in callback.episodes],
            [(1_000, False), (5_000, True)],
        )
        self.assertEqual(callback.success_mean(), 5_000.0)

    def test_state_restores_only_episodes_at_or_before_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            state_path.write_text(json.dumps({
                "version": 1,
                "window_size": 100,
                "episodes": [
                    {
                        "length": 4_000,
                        "succeeded": True,
                        "terminal_global_step": 100,
                    },
                    {
                        "length": 500,
                        "succeeded": False,
                        "terminal_global_step": 200,
                    },
                ],
            }))
            callback = Train.SuccessEpisodeLengthCallback(state_path=state_path)

            class Model:
                num_timesteps = 150

            callback.model = Model()
            callback._on_training_start()

            self.assertEqual(callback.success_mean(), 4_000.0)
            self.assertEqual(len(callback.episodes), 1)


class RewardTests(unittest.TestCase):
    def test_only_new_deepest_progress_and_completion_pay(self):
        class State:
            pass

        state = State()
        obs = np.frombuffer(live_observation().tobytes(), dtype=Types.OBSERVATION_DTYPE)[0].copy()
        obs["LayerProgress"] = 0.125
        Reward.Reset(state, obs)
        obs["LayerProgress"] = 0.25
        self.assertAlmostEqual(Reward.Reward(obs, None, state), 1.25)
        self.assertEqual(state.last_reward_terms["incomplete_episode"], 0.0)
        obs["LayerProgress"] = 0.15
        self.assertEqual(Reward.Reward(obs, None, state), 0.0)
        obs["LayerProgress"] = 1.0
        self.assertAlmostEqual(Reward.Reward(obs, None, state), 17.5)
        self.assertEqual(Reward.Reward(obs, None, state), 0.0)

    def test_ineffective_swing_penalty_is_shadow_only(self):
        class State:
            pass

        state = State()
        obs = np.frombuffer(live_observation().tobytes(), dtype=Types.OBSERVATION_DTYPE)[0].copy()
        self.assertIsInstance(obs, np.void)
        obs["LayerProgress"] = 0.25
        obs["AttackSwingsExecuted"] = 1
        obs["AttackContactClawHealthLoss"] = 0.0
        Reward.Reset(state, obs)

        self.assertEqual(Reward.Reward(obs, None, state), 0.0)
        self.assertEqual(state.last_reward_terms["ineffective_swings"], 1)
        self.assertEqual(
            state.last_reward_terms["ineffective_swing_penalty_shadow"],
            -0.0001,
        )
        self.assertEqual(
            state.last_reward_terms["ineffective_swing_penalty"], 0.0
        )

    def test_claw_contact_exempts_executed_swing_from_shadow_miss(self):
        class State:
            pass

        state = State()
        obs = np.frombuffer(live_observation().tobytes(), dtype=Types.OBSERVATION_DTYPE)[0].copy()
        obs["LayerProgress"] = 0.25
        obs["AttackSwingsExecuted"] = 1
        obs["AttackContactClawHealthLoss"] = 0.01
        Reward.Reset(state, obs)

        self.assertEqual(Reward.Reward(obs, None, state), 0.0)
        self.assertEqual(state.last_reward_terms["ineffective_swings"], 0)
        self.assertEqual(
            state.last_reward_terms["ineffective_swing_penalty_shadow"], 0.0
        )

    def test_missed_attack_mode_penalizes_only_executed_noncontacts(self):
        class State:
            missed_attack_penalty_mode = True

        state = State()
        obs = np.frombuffer(
            live_observation().tobytes(), dtype=Types.OBSERVATION_DTYPE
        )[0].copy()
        obs["LayerProgress"] = 0.25
        Reward.Reset(state, obs)

        obs["AttackSwingsExecuted"] = 1
        obs["AttackContactClawHealthLoss"] = 0.0
        self.assertEqual(Reward.Reward(obs, None, state), -0.005)
        self.assertEqual(state.last_reward_terms["ineffective_swings"], 1)

        obs["AttackContactClawHealthLoss"] = 0.3
        self.assertEqual(Reward.Reward(obs, None, state), 0.0)
        self.assertEqual(state.last_reward_terms["ineffective_swings"], 0)

        obs["AttackSwingsExecuted"] = 0
        obs["AttackContactClawHealthLoss"] = 0.0
        self.assertEqual(Reward.Reward(obs, None, state), 0.0)

    def test_attack_budget_charges_every_executed_swing_immediately(self):
        class State:
            attack_budget_mode = True
            attack_budget = 2_000

        state = State()
        obs = np.frombuffer(
            live_observation().tobytes(), dtype=Types.OBSERVATION_DTYPE
        )[0].copy()
        obs["LayerProgress"] = 0.25
        Reward.Reset(state, obs)
        obs["AttackSwingsExecuted"] = 1
        obs["AttackContactClawHealthLoss"] = 1.0
        self.assertEqual(Reward.Reward(obs, None, state), -0.005)
        self.assertEqual(state.last_reward_terms["attack_budget_penalty"], -0.005)
        obs["AttackSwingsExecuted"] = 0
        self.assertEqual(Reward.Reward(obs, None, state), 0.0)

    def test_incomplete_episode_penalty_updates_reward_terms(self):
        class State:
            pass

        state = State()
        state.last_reward_terms = {"progress": 0.25, "reward": 0.25}
        self.assertEqual(Reward.ApplyIncompleteEpisodePenalty(state), -9.75)
        self.assertEqual(state.last_reward_terms["incomplete_episode"], -10.0)
        self.assertEqual(state.last_reward_terms["reward"], -9.75)


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
    def test_attack_budget_contract_and_normalization(self):
        rng = np.random.default_rng(90210)
        samples = [ResourceBudget.sample_budget(rng) for _ in range(100)]
        self.assertTrue(all(1_500 <= value <= 3_000 for value in samples))
        self.assertEqual(ResourceBudget.normalize_remaining(3_000), 1.0)
        self.assertEqual(ResourceBudget.normalize_remaining(1_500), 0.5)
        self.assertEqual(ResourceBudget.normalize_remaining(-1), 0.0)
        self.assertFalse(ResourceBudget.exceeded(1_500, 1_500))
        self.assertTrue(ResourceBudget.exceeded(1_501, 1_500))
        self.assertEqual(ResourceBudget.immediate_cost(1, 2_000), -0.005)
        self.assertEqual(ResourceBudget.immediate_cost(2_000, 2_000), -10.0)

    def test_attack_prior_pressure_and_annealing(self):
        self.assertEqual(
            Train.AttackPriorCoefficient(Train.CB1_INTERVENTION_SOURCE_STEP),
            0.1,
        )
        self.assertAlmostEqual(
            Train.AttackPriorCoefficient(
                Train.CB1_INTERVENTION_SOURCE_STEP + 500_000
            ),
            0.05,
        )
        self.assertEqual(
            Train.AttackPriorCoefficient(
                Train.CB1_INTERVENTION_SOURCE_STEP + 1_000_000
            ),
            0.0,
        )

        logits = torch.tensor([[0.0, 7.0]], requires_grad=True)
        distribution = SimpleNamespace(distribution=[
            torch.distributions.Categorical(logits=logits)
        ])
        loss, probability = Train.AttackPriorLoss(distribution)
        loss.backward()
        self.assertGreater(float(probability.detach()), 0.99)
        # Gradient descent subtracts this positive attack-logit gradient.
        self.assertGreater(float(logits.grad[0, 1]), 0.0)

    def test_resume_prefers_greatest_explicit_step_over_legacy_name(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            for name in (
                "casu_ppo_cb1_vb1_30001444_steps.zip",
                "casu_ppo_cb1_vb1_29995300_steps.zip",
                "casu_ppo_cb1_vb1_500k.zip",
            ):
                with Train.zipfile.ZipFile(run_dir / name, "w") as archive:
                    archive.writestr("valid", name)

            selected = Train.FindResumeModel(run_dir)

            self.assertEqual(
                selected.name,
                "casu_ppo_cb1_vb1_30001444_steps.zip",
            )

    def test_tensorboard_continuation_is_pinned_to_ppo_6(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            for run_id in range(6):
                (run_dir / "tensorboard" / f"PPO_{run_id}").mkdir(
                    parents=True,
                )

            Train.PrepareTensorboardContinuation(run_dir)

            self.assertTrue((run_dir / "tensorboard" / "PPO_6").is_dir())

    def test_tensorboard_continuation_rejects_newer_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "tensorboard" / "PPO_7").mkdir(parents=True)

            with self.assertRaisesRegex(RuntimeError, "newer than PPO_6"):
                Train.PrepareTensorboardContinuation(run_dir)

    def test_ppo6_intervention_manifest_pins_source_and_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            resume_model = (
                run_dir / "casu_ppo_cb1_vb1_51843364_steps.zip"
            )
            Train.ValidateOrWriteInterventionManifest(
                run_dir, resume_model
            )
            manifest = json.loads(
                (run_dir / Train.INTERVENTION_MANIFEST_NAME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["phase"], "PPO_6")
            self.assertEqual(manifest["source_step"], 51_843_364)
            self.assertEqual(
                manifest["experiment"],
                "actual_swing_claw_contact_shadow",
            )
            self.assertTrue(manifest["functional_stamina_reset"])
            self.assertTrue(manifest["capable_only_inactivity"])
            self.assertFalse(manifest["ineffective_swing_penalty_enabled"])
            self.assertEqual(
                manifest["ineffective_swing_penalty_scale"], 0.0001
            )

            Train.ValidateOrWriteInterventionManifest(
                run_dir, resume_model
            )

    def test_ppo6_missed_attack_manifest_pins_fixed_penalty(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            resume_model = (
                run_dir / "casu_ppo_cb1_vb1_51843364_steps.zip"
            )
            with patch.dict(
                os.environ, {"PPO_MISSED_ATTACK_PENALTY": "1"}, clear=False
            ):
                Train.ValidateOrWriteInterventionManifest(
                    run_dir, resume_model
                )
            manifest = json.loads(
                (run_dir / Train.INTERVENTION_MANIFEST_NAME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                manifest["experiment"], "missed_executed_attack_penalty"
            )
            self.assertEqual(manifest["missed_attack_penalty"], -0.005)
            self.assertEqual(manifest["contact_penalty"], 0.0)
            self.assertEqual(manifest["cooldown_hold_penalty"], 0.0)
            self.assertFalse(manifest["permission_or_phase_logic"])

    def test_ppo6_attack_prior_manifest_records_optimizer_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            resume_model = (
                run_dir / "casu_ppo_cb1_vb1_51843364_steps.zip"
            )
            with patch.dict(
                os.environ,
                {
                    "PPO_MISSED_ATTACK_PENALTY": "0",
                    "PPO_ATTACK_LOGIT_PRIOR": "1",
                },
                clear=False,
            ):
                Train.ValidateOrWriteInterventionManifest(
                    run_dir, resume_model
                )
            manifest = json.loads(
                (run_dir / Train.INTERVENTION_MANIFEST_NAME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                manifest["experiment"],
                "isolated_annealed_attack_logit_prior",
            )
            self.assertEqual(manifest["target_attack_probability"], 0.25)
            self.assertEqual(manifest["initial_prior_coefficient"], 0.1)
            self.assertEqual(manifest["prior_anneal_steps"], 1_000_000)
            self.assertFalse(manifest["sampling_or_action_override"])
            self.assertEqual(manifest["missed_attack_penalty"], 0.0)

    def test_ppo6_natural_exhaustion_prior_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            resume_model = (
                run_dir / "casu_ppo_cb1_vb1_51843364_steps.zip"
            )
            with patch.dict(
                os.environ,
                {
                    "PPO_ATTACK_BUDGET": "0",
                    "PPO_ATTACK_LOGIT_PRIOR": "1",
                    "PPO_NATURAL_STAMINA": "1",
                    "PPO_MISSED_ATTACK_PENALTY": "0",
                },
                clear=False,
            ):
                env = CasualtiesEnv.Env()
                manifest = Train.ProtocolManifest(env)
                Train.ValidateOrWriteInterventionManifest(
                    run_dir, resume_model
                )
                env.close()
            intervention = json.loads(
                (run_dir / Train.INTERVENTION_MANIFEST_NAME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                intervention["experiment"],
                "natural_exhaustion_with_annealed_attack_prior",
            )
            self.assertEqual(
                intervention["target_attack_probability"], 0.25
            )
            self.assertFalse(intervention["stamina_reset_by_functional_godmode"])
            self.assertEqual(
                intervention["incapacitated_inactivity_penalty"], -10.0
            )
            self.assertFalse(intervention["attack_budget"])
            self.assertIn("natural_stamina", manifest)

    def test_ppo6_prior_and_missed_attack_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            resume_model = (
                run_dir / "casu_ppo_cb1_vb1_51843364_steps.zip"
            )
            with patch.dict(
                os.environ,
                {
                    "PPO_NATURAL_STAMINA": "0",
                    "PPO_ATTACK_LOGIT_PRIOR": "1",
                    "PPO_MISSED_ATTACK_PENALTY": "1",
                    "PPO_ATTACK_BUDGET": "0",
                },
                clear=False,
            ):
                Train.ValidateOrWriteInterventionManifest(
                    run_dir, resume_model
                )
            manifest = json.loads(
                (run_dir / Train.INTERVENTION_MANIFEST_NAME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                manifest["experiment"],
                "annealed_attack_prior_with_missed_swing_penalty",
            )
            self.assertEqual(manifest["target_attack_probability"], 0.25)
            self.assertEqual(manifest["missed_attack_penalty"], -0.005)
            self.assertEqual(manifest["contact_penalty"], 0.0)
            self.assertEqual(manifest["cooldown_hold_penalty"], 0.0)
            self.assertTrue(manifest["functional_stamina_reset"])
            self.assertFalse(manifest["natural_stamina"])

    def test_ppo6_resource_budget_manifest_records_hard_constraint(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            resume_model = (
                run_dir / "casu_ppo_cb1_vb1_51843364_steps.zip"
            )
            with patch.dict(
                os.environ,
                {
                    "PPO_ATTACK_BUDGET": "1",
                    "PPO_ATTACK_LOGIT_PRIOR": "1",
                    "PPO_MISSED_ATTACK_PENALTY": "0",
                },
                clear=False,
            ):
                Train.ValidateOrWriteInterventionManifest(
                    run_dir, resume_model
                )
            manifest = json.loads(
                (run_dir / Train.INTERVENTION_MANIFEST_NAME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                manifest["experiment"],
                "dense_budget_conditioned_resource_efficient_descent",
            )
            self.assertEqual(manifest["policy_general_shape"], 140)
            self.assertEqual(manifest["missed_attack_penalty"], 0.0)
            self.assertEqual(manifest["attack_budget"]["minimum_budget"], 1500)
            self.assertEqual(manifest["attack_budget"]["maximum_budget"], 3000)
            self.assertTrue(manifest["attack_budget"]["termination"])
            self.assertEqual(manifest["unused_budget_reward"], 0.0)
            self.assertEqual(
                manifest["immediate_swing_cost"],
                "-10 / assigned episode budget",
            )

    def test_resume_requires_matching_cb1_manifest(self):
        class FakeEnv:
            max_episode_steps = CasualtiesEnv.MAX_EPISODE_STEPS
            max_stagnant_steps = CasualtiesEnv.MAX_STAGNANT_STEPS
            max_inactive_steps = CasualtiesEnv.MAX_INACTIVE_STEPS
            max_mobile_stagnant_steps = (
                CasualtiesEnv.MAX_MOBILE_STAGNANT_STEPS
            )
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
            self.assertEqual(manifest["protocol_version"], 2)
            self.assertEqual(manifest["gamma"], 0.99)
            self.assertEqual(manifest["gae_lambda"], 0.98)
            self.assertEqual(manifest["n_steps"], 2048)
            self.assertEqual(manifest["model_schema_version"], 1)
            self.assertEqual(manifest["max_episode_steps"], 100_000)
            self.assertEqual(manifest["max_stagnant_steps"], 75_000)
            self.assertEqual(manifest["max_inactive_steps"], 75)
            self.assertEqual(manifest["max_mobile_stagnant_steps"], 10_000)
            self.assertEqual(manifest["incomplete_episode_penalty"], -10.0)
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
