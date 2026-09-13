"""Exercise the real environment step under both permission contracts."""
import os
import unittest
from unittest.mock import patch, Mock
import numpy as np
from PPO_Python.tests.test_cb1 import live_observation
import CasualtiesEnv
import Reward
import AttackPermission


class PermissionTests(unittest.TestCase):
    def test_curriculum_promotes_only_after_demonstrated_survival(self):
        state = AttackPermission.new_curriculum_state()
        for _ in range(99):
            state, event = AttackPermission.record_curriculum_outcome(
                state, True
            )
            self.assertIsNone(event)
        state, event = AttackPermission.record_curriculum_outcome(state, True)
        self.assertEqual(event, "temperature_promoted")
        self.assertEqual(AttackPermission.curriculum_values(state), (1, 0.25))
        for expected_temperature in (0.5, 1.0):
            for _ in range(100):
                state, event = AttackPermission.record_curriculum_outcome(
                    state, True
                )
            self.assertEqual(
                AttackPermission.curriculum_values(state),
                (1, expected_temperature),
            )
        for _ in range(100):
            state, event = AttackPermission.record_curriculum_outcome(
                state, True
            )
        self.assertEqual(event, "duration_promoted")
        self.assertEqual(AttackPermission.curriculum_values(state), (2, 0.1))

    def test_one_step_curriculum_rewards_abstinence_with_continuation(self):
        with patch.dict(
            os.environ,
            {
                "PPO_ATTACK_TERMINAL": "1",
                "PPO_ATTACK_EXPLORATION": "1",
                "PPO_ATTACK_CURRICULUM": "1",
            },
            clear=True,
        ), patch.object(CasualtiesEnv, "EpisodeTraceWriter"):
            env = CasualtiesEnv.Env()
        env.attack_forbidden = True
        env.attack_cost = 1.0
        env.last_consumed_observation_id = 0
        obs = live_observation()
        env.best_layer_depth = int(obs["BestLayerDepth"])
        env.last_best_depth_step = 0
        env.previous_tile_position = (255, 400)
        env.previous_block_health = obs["RelativeBlockMap"]["Health"].copy()
        Reward.Reset(env, obs)
        action = np.array([1, 0, 1, 0, 4, 5, 0])
        with patch.object(
            env, "_wait_for_observation_id", return_value=(obs, 1)
        ), patch.object(CasualtiesEnv, "Decode"), patch.object(
            CasualtiesEnv.ProtocolValidation,
            "validate_step_observation",
            return_value={},
        ):
            processed, reward, terminated, _, info = env.step(action)
        self.assertFalse(terminated)
        self.assertEqual(reward, 0.0)
        self.assertTrue(info["curriculum_survived"])
        self.assertEqual(info["permission_phase_limit"], 1)
        self.assertFalse(env.attack_forbidden)
        self.assertEqual(processed["general"][-1], 0.0)
        self.assertEqual(
            env.attack_curriculum_state["recent_outcomes"], [1]
        )

    def test_exploration_temperature_schedule(self):
        source = AttackPermission.EXPLORATION_SOURCE_STEP
        horizon = AttackPermission.EXPLORATION_ANNEAL_STEPS
        self.assertEqual(AttackPermission.exploration_temperature(source), 0.1)
        self.assertAlmostEqual(
            AttackPermission.exploration_temperature(source + horizon // 2),
            0.55,
        )
        self.assertEqual(
            AttackPermission.exploration_temperature(source + horizon), 1.0
        )
        self.assertEqual(
            AttackPermission.exploration_temperature(source + 2 * horizon),
            1.0,
        )

    def test_prohibited_executed_swing_terminates_with_failure_penalty(self):
        with patch.dict(
            os.environ,
            {"PPO_ATTACK_TERMINAL": "1"},
            clear=True,
        ), patch.object(CasualtiesEnv, "EpisodeTraceWriter"):
            env = CasualtiesEnv.Env()
        env.attack_forbidden = True
        env.attack_cost = 1.0
        env.last_consumed_observation_id = 0
        obs = live_observation()
        obs["AttackSwingsExecuted"] = 1
        env.best_layer_depth = int(obs["BestLayerDepth"])
        env.last_best_depth_step = 0
        env.previous_tile_position = (255, 400)
        env.previous_block_health = obs["RelativeBlockMap"]["Health"].copy()
        Reward.Reset(env, obs)
        action = np.array([1, 0, 1, 0, 4, 5, 1])
        with patch.object(
            env, "_wait_for_observation_id", return_value=(obs, 1)
        ), patch.object(CasualtiesEnv, "Decode") as decode, patch.object(
            CasualtiesEnv.ProtocolValidation,
            "validate_step_observation",
            return_value={},
        ):
            _, reward, terminated, truncated, info = env.step(action)
        self.assertEqual(decode.call_args.args[0][6], 1)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, -10.0)
        self.assertEqual(info["attack_execution_penalty"], 0.0)
        self.assertEqual(info["incomplete_episode"], -10.0)
        self.assertTrue(info["attack_violation"])
        self.assertEqual(info["truncation_reason"], "attack_prohibited_swing")

    def test_terminal_contract_keeps_attack_in_policy(self):
        with patch.dict(
            os.environ,
            {"PPO_ATTACK_TERMINAL": "1"},
            clear=True,
        ):
            contract = AttackPermission.contract()
        self.assertTrue(contract["attack_available_in_policy"])
        self.assertEqual(contract["violation_penalty"], -10.0)
        self.assertEqual(contract["per_swing_penalty"], 0.0)

    def test_penalty_mode_keeps_attack_and_charges_executed_swing(self):
        with patch.dict(
            os.environ,
            {"PPO_ATTACK_PENALTY": "1"},
            clear=True,
        ), patch.object(CasualtiesEnv, "EpisodeTraceWriter"):
            env = CasualtiesEnv.Env()
        env.attack_forbidden = True
        env.attack_cost = 1.0
        env.episode_steps = 0
        env.last_consumed_observation_id = 0
        obs = live_observation()
        obs["AttackSwingsExecuted"] = 1
        env.best_layer_depth = int(obs["BestLayerDepth"])
        env.last_best_depth_step = 0
        env.previous_tile_position = (255, 400)
        env.previous_block_health = obs["RelativeBlockMap"]["Health"].copy()
        Reward.Reset(env, obs)
        action = np.array([1, 0, 1, 0, 4, 5, 1])
        with patch.object(
            env, "_wait_for_observation_id", return_value=(obs, 1)
        ), patch.object(CasualtiesEnv, "Decode") as decode, patch.object(
            CasualtiesEnv.ProtocolValidation,
            "validate_step_observation",
            return_value={},
        ):
            processed, reward, terminated, _, info = env.step(action)
        self.assertEqual(decode.call_args.args[0][6], 1)
        self.assertEqual(reward, -0.001)
        self.assertFalse(terminated)
        self.assertEqual(info["attack_cost"], 1.0)
        self.assertEqual(info["attack_execution_penalty"], -0.001)
        self.assertEqual(processed["general"][-1], 1.0)

    def test_penalty_contract_has_no_mask(self):
        with patch.dict(
            os.environ,
            {"PPO_ATTACK_PENALTY": "1"},
            clear=True,
        ):
            contract = AttackPermission.contract()
        self.assertTrue(contract["attack_available_in_all_phases"])
        self.assertNotIn("mask", contract)
        self.assertEqual(contract["attack_execution_penalty_scale"], 0.001)

    def test_phase_boundary_terminates_before_unblocking(self):
        for forbidden in (False, True):
            with self.subTest(forbidden=forbidden), patch.dict(os.environ, {"PPO_ATTACK_PERMISSION": "1", "PPO_ATTACK_PHASES": "1"}):
                with patch.object(CasualtiesEnv, "EpisodeTraceWriter"):
                    env = CasualtiesEnv.Env()
                env.attack_forbidden = forbidden
                env.permission_phase_steps = 149
                env.permission_no_depth_steps = 149
                env.episode_steps = 149
                obs = live_observation()
                env.last_consumed_observation_id = 149
                env.best_layer_depth = int(obs["BestLayerDepth"])
                env.last_best_depth_step = 0
                env.previous_tile_position = (255, 400)
                env.previous_block_health = obs["RelativeBlockMap"]["Health"].copy()
                Reward.Reset(env, obs)
                action = np.array([1, 0, 1, 0, 4, 5, 1])
                with patch.object(env, "_wait_for_observation_id", return_value=(obs, 150)), patch.object(CasualtiesEnv, "Decode"), patch.object(CasualtiesEnv.ProtocolValidation, "validate_step_observation", return_value={}):
                    processed, reward, terminated, _, info = env.step(action)
                self.assertEqual(terminated, forbidden)
                self.assertEqual(reward, -10.0 if forbidden else 0.0)
                self.assertEqual(info["incomplete_episode"], -10.0 if forbidden else 0.0)
                self.assertTrue(env.attack_forbidden)
                self.assertEqual(processed["general"][-1], 1)
                self.assertEqual(info["attack_forbidden"], forbidden)
                if forbidden:
                    self.assertEqual(info["truncation_reason"], "attack_forbidden_no_depth")
                    self.assertEqual(env.permission_phase_steps, 150)
                else:
                    self.assertEqual(env.permission_phase_steps, 0)
                    self.assertEqual(env.permission_no_depth_steps, 0)

    def test_forbidden_timeout_boundary(self):
        self.assertFalse(AttackPermission.stalled_attempt(True, 149))
        self.assertTrue(AttackPermission.stalled_attempt(True, 150))
        self.assertFalse(AttackPermission.stalled_attempt(False, 10000))

    def test_watchdog_and_execution_in_both_modes(self):
        for forbidden in (False, True):
            with self.subTest(forbidden=forbidden), patch.dict(os.environ, {"PPO_ATTACK_PERMISSION": "1"}):
                with patch.object(CasualtiesEnv, "EpisodeTraceWriter"):
                    env = CasualtiesEnv.Env()
                env.attack_forbidden = forbidden
                obs = live_observation()
                env.last_consumed_observation_id = 0
                env.best_layer_depth = int(obs["BestLayerDepth"])
                env.last_best_depth_step = 0
                env.previous_tile_position = (256, 400)
                env.previous_block_health = obs["RelativeBlockMap"]["Health"].copy()
                env.max_episode_steps = 1
                Reward.Reset(env, obs)
                action = np.array([1, 0, 1, 0, 4, 5, 1])
                with patch.object(env, "_wait_for_observation_id", return_value=(obs, 1)), patch.object(CasualtiesEnv, "Decode") as decode, patch.object(CasualtiesEnv.ProtocolValidation, "validate_step_observation", return_value={}):
                    processed, reward, terminated, truncated, info = env.step(action)
                self.assertEqual(decode.call_args.args[0][6], 0 if forbidden else 1)
                self.assertEqual(action[6], 1)
                self.assertEqual(reward, 0.0 if forbidden else -10.0)
                self.assertTrue(terminated)
                self.assertFalse(truncated)
                self.assertEqual(processed["general"].shape, (140,))
                self.assertEqual(processed["general"][-1], forbidden)
                self.assertEqual(info["attack_forbidden"], forbidden)
                self.assertEqual(env.episode_trace.record.call_args.args[5]["attack_forbidden"], forbidden)


if __name__ == "__main__":
    unittest.main()
