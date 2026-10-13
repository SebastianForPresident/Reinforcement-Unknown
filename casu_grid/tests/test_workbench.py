from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from casu_grid.workbench.contracts import validate_contracts
from casu_grid.workbench.engineering_smoke import ACTION, MODEL, OBSERVATION, PPO, REWARD
from casu_grid.workbench.model import ConfigurableActorCritic, count_parameters
from casu_grid.workbench.ppo import (
    PPOTrainer, _generalized_advantage_estimate, _restore_final_observations,
)
from casu_grid.isaac_navigation_env import IsaacLayer1EnvConfig
from casu_grid.player_controller import StandingAction, StandingControllerState, StandingPlayerController


class TinyGoalEnv:
    """Vectorized Gym-style adapter used only to exercise the PPO implementation."""

    def __init__(self, count=2, horizon=5):
        self.count, self.horizon = count, horizon
        self.progress = np.zeros(count, dtype=np.float32)
        self.elapsed = np.zeros(count, dtype=np.int64)

    def _obs(self):
        observation = np.zeros((self.count, 8), dtype=np.float32)
        observation[:, 0] = self.progress
        observation[:, 7] = (self.horizon - self.elapsed) / self.horizon
        return {"player_state": observation}

    def reset(self, seed=None):
        self.progress.fill(0)
        self.elapsed.fill(0)
        return self._obs(), {"seed": seed}

    def step(self, action):
        advance = np.asarray(action["advance"]).reshape(self.count)
        self.progress += advance.astype(np.float32) * (1.0 / self.horizon)
        self.elapsed += 1
        reward = advance.astype(np.float32) / self.horizon
        reached = self.progress >= 1.0
        timed_out = self.elapsed >= self.horizon
        done = reached | timed_out
        if done.any():
            self.progress[done] = 0
            self.elapsed[done] = 0
        info = {"reward_terms": {"progress_delta": reward.copy(),
                                 "goal": reached.astype(np.float32)}}
        return self._obs(), reward, reached, timed_out & ~reached, info


class WorkbenchTests(unittest.TestCase):
    def test_runtime_callback_contract_errors_name_the_field(self):
        from casu_grid.workbench.runtime_validation import checked_callbacks
        from casu_grid.workbench.contracts import ObservationContract, ObservationField, RewardContract, RewardTerm
        observations = ObservationContract(scalar=(ObservationField("state", (2,), "test.state"),))
        rewards = RewardContract(terms=(RewardTerm("progress", "test.progress", 1.0),))
        _, observe, reward = checked_callbacks(
            (lambda _actions: object(), lambda _env: {"state": np.zeros(3)},
             lambda *_: (0.0, {"unexpected": 1.0})), observations, rewards)
        with self.assertRaisesRegex(ValueError, "field state has shape"):
            observe(object())
        _, observe, reward = checked_callbacks(
            (None, lambda _env: {"state": np.zeros(2)},
             lambda *_: (0.0, {"unexpected": 1.0})), observations, rewards)
        self.assertEqual(observe(object())["state"].shape, (2,))
        with self.assertRaisesRegex(ValueError, "reward term keys"):
            reward(None, {}, False, False)
        action, _, _ = checked_callbacks(
            (lambda _actions: object(), lambda _env: {"state": np.zeros(2)},
             lambda *_: (0.0, {"progress": 0.0})), observations, rewards)
        with self.assertRaisesRegex(ValueError, "move_x, move_y, jump, and crouch"):
            action({})
        from casu_grid.player_controller import StandingAction
        action, _, _ = checked_callbacks(
            (lambda _actions: StandingAction(move_x=1), lambda _env: {"state": np.zeros(2)},
             lambda *_: (0.0, {"progress": 0.0})), observations, rewards)
        self.assertEqual(action({}).move_x, 1)

    def test_gae_bootstraps_time_limit_but_not_true_terminal(self):
        rewards = torch.tensor([[1.0], [2.0]])
        values = torch.tensor([[0.5], [0.5]])
        # First transition is truncated and must use its final-state value of
        # 3; second transition is an actual terminal and must not bootstrap.
        dones = torch.tensor([[1.0], [1.0]])
        terminated = torch.tensor([[0.0], [1.0]])
        next_values = torch.tensor([[3.0], [7.0]])
        advantage, returns = _generalized_advantage_estimate(
            rewards, values, dones, terminated, next_values,
            gamma=0.9, gae_lambda=0.95)
        self.assertAlmostEqual(float(advantage[0, 0]), 1.0 + 0.9 * 3.0 - 0.5, places=6)
        self.assertAlmostEqual(float(advantage[1, 0]), 2.0 - 0.5, places=6)
        self.assertTrue(torch.equal(returns, advantage + values))

    def test_auto_reset_vector_final_observation_is_used_for_bootstrap(self):
        next_observation = {"state": torch.tensor([[0.0, 0.0], [10.0, 10.0]])}
        info = {
            "final_observation": [
                {"state": np.asarray([3.0, 4.0], dtype=np.float32)}, None,
            ],
            "_final_observation": np.asarray([True, False]),
        }
        restored = _restore_final_observations(next_observation, info, torch.device("cpu"))
        self.assertTrue(torch.equal(restored["state"],
                                    torch.tensor([[3.0, 4.0], [10.0, 10.0]])))

    def test_isaac_environment_rejects_unimplemented_vector_stage(self):
        IsaacLayer1EnvConfig().validate()
        from casu_grid.isaac_navigation_env import IsaacLayer1NavigationEnv
        with self.assertRaisesRegex(NotImplementedError, "single-env Isaac stage"):
            IsaacLayer1NavigationEnv(IsaacLayer1EnvConfig(num_envs=2))

    def test_immersion_allows_jump_and_liquid_impulse_uses_source_units(self):
        controller = StandingPlayerController()
        controller.set_action(StandingAction(jump=True))
        command = controller.pre_physics_tick(
            velocity_x=0.0, velocity_y=0.0, grounded=False,
            in_water=True, dt=0.02)
        self.assertEqual(command.vertical_velocity_override, 26.0)
        # SearchBase LiquidAffect: velocity *= drag; velocity.y += buoyancy
        vx, vy = 4.0 * 0.915, -2.0 * 0.915 + 0.6
        self.assertAlmostEqual(vx, 3.66)
        self.assertAlmostEqual(vy, -1.23)

    def test_contracts_and_tensor_shapes(self):
        contracts = validate_contracts(ACTION, OBSERVATION, REWARD, MODEL, PPO)
        self.assertEqual(contracts["ppo"]["version"], 1)
        model = ConfigurableActorCritic(ACTION, OBSERVATION, MODEL)
        observation = {"player_state": torch.zeros(3, 8)}
        action, log_prob, value = model.act(observation)
        self.assertEqual(tuple(action["advance"].shape), (3,))
        self.assertEqual(tuple(log_prob.shape), (3,))
        self.assertEqual(tuple(value.shape), (3,))
        self.assertGreater(count_parameters(model), 0)

    def test_declared_observation_normalization_is_applied(self):
        from casu_grid.workbench.contracts import ObservationField, ObservationContract
        field = ObservationField("state", (2,), "test.state", "unit",
                                 {"low": -1.0, "high": 3.0})
        observations = ObservationContract(scalar=(field,))
        model = ConfigurableActorCritic(ACTION, observations, MODEL)
        normalized = model.features({"state": torch.tensor([[-1.0, 3.0],
                                                               [1.0, 1.0]])})
        raw_model = ConfigurableActorCritic(
            ACTION, ObservationContract(scalar=(ObservationField(
                "state", (2,), "test.state"),)), MODEL)
        raw = raw_model.features({"state": torch.tensor([[-1.0, 3.0],
                                                           [1.0, 1.0]])})
        # Explicitly check the input transform independently of learned layers.
        from casu_grid.workbench.model import _normalize_observation
        transformed = _normalize_observation(field, torch.tensor([[-1.0, 3.0],
                                                                    [1.0, 1.0]]))
        self.assertTrue(torch.equal(transformed, torch.tensor([[0.0, 1.0],
                                                               [0.5, 0.5]])))
        self.assertEqual(tuple(normalized.shape), tuple(raw.shape))

    def test_spatial_mask_suppresses_values_and_contract_validates_shape(self):
        from casu_grid.workbench.contracts import ObservationField, ObservationContract
        spatial = ObservationField("terrain", (1, 3, 3), "test.terrain", mask="valid")
        mask = ObservationField("valid", (1, 3, 3), "test.valid")
        observations = ObservationContract(spatial=(spatial, mask), scalar=(
            ObservationField("state", (2,), "test.state"),))
        validate_contracts(ACTION, observations, REWARD, MODEL, PPO)
        model = ConfigurableActorCritic(ACTION, observations, MODEL)
        mask_batch = torch.tensor([[[[1.0, 0.0, 1.0],
                                     [1.0, 0.0, 1.0],
                                     [1.0, 0.0, 1.0]]]])
        terrain = torch.ones((1, 1, 3, 3))
        prepared = model._prepare_spatial_values({"terrain": terrain, "valid": mask_batch})[0]
        self.assertEqual(float(prepared[0, 0, :, 1].sum()), 0.0)
        self.assertEqual(float(prepared[0, 0, :, 0].sum()), 3.0)
        with self.assertRaisesRegex(ValueError, "dimensions do not match"):
            validate_contracts(ACTION, ObservationContract(
                spatial=(spatial, ObservationField("valid", (1, 2, 2), "test.valid")),
                scalar=(ObservationField("state", (2,), "test.state"),)),
                REWARD, MODEL, PPO)

    def test_continuous_action_distribution_shape(self):
        from casu_grid.workbench.contracts import ActionContract, ActionHead
        actions = ActionContract(heads=(ActionHead("move", kind="continuous", size=2),))
        model = ConfigurableActorCritic(actions, OBSERVATION, MODEL)
        observation = {"player_state": torch.zeros(4, 8)}
        selected, log_prob, _ = model.act(observation)
        self.assertEqual(tuple(selected["move"].shape), (4, 2))
        self.assertEqual(tuple(log_prob.shape), (4,))

    def test_optional_recurrent_model_runs_sequence_and_backpropagates(self):
        from dataclasses import replace
        from casu_grid.workbench.contracts import ActionContract, ActionHead, ModelContract
        from casu_grid.workbench.model import ConfigurableActorCritic
        action_contract = ActionContract(heads=(ActionHead("advance", size=2),))
        model_contract = replace(MODEL, recurrent=True, recurrent_size=12)
        validate_contracts(action_contract, OBSERVATION, REWARD, model_contract, PPO)
        network = ConfigurableActorCritic(action_contract, OBSERVATION, model_contract)
        obs = {"player_state": torch.zeros(6, 8)}
        actions, log_prob, values, latent, state = network.act(
            obs, return_latent=True, return_state=True, sequence_length=3,
            initial_state=torch.zeros(1, 2, 12))
        self.assertEqual(tuple(values.shape), (6,))
        self.assertEqual(tuple(state.shape), (1, 2, 12))
        logp, entropy, evaluated = network.evaluate_actions(
            obs, latent, sequence_length=3, initial_state=torch.zeros(1, 2, 12))
        (logp.mean() + entropy.mean() + evaluated.mean()).backward()
        self.assertTrue(torch.isfinite(logp).all())

    def test_recurrent_ppo_update_and_checkpoint_state(self):
        from dataclasses import replace
        from casu_grid.workbench.contracts import ModelContract
        recurrent_model = replace(MODEL, recurrent=True, recurrent_size=10)
        recurrent_ppo = replace(PPO, rollout_steps=5, minibatch_size=1, epochs=1,
                                target_kl=None)
        trainer = PPOTrainer(TinyGoalEnv(count=2, horizon=3), ACTION, OBSERVATION,
                             recurrent_model, recurrent_ppo, seed=8)
        metrics = trainer.train_update(rollout_steps=5)
        self.assertEqual(metrics["update"], 1)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["policy_loss"])))
        self.assertEqual(tuple(trainer._recurrent_state.shape), (1, 2, 10))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "recurrent.pt"
            trainer.save(path)
            restored = PPOTrainer(TinyGoalEnv(count=2, horizon=3), ACTION,
                                  OBSERVATION, recurrent_model, recurrent_ppo, seed=3)
            restored.load(path)
            self.assertTrue(torch.equal(restored._recurrent_state,
                                        trainer._recurrent_state))
            resumed = restored.train_update(rollout_steps=3)
            self.assertEqual(resumed["update"], 2)

    def test_ppo_optimizer_update_and_checkpoint_roundtrip(self):
        trainer = PPOTrainer(TinyGoalEnv(), ACTION, OBSERVATION, MODEL, PPO, seed=81)
        before = [value.detach().clone() for value in trainer.model.parameters()]
        metrics = trainer.train_update(rollout_steps=12)
        self.assertEqual(metrics["update"], 1)
        self.assertGreater(metrics["optimizer_batches"], 0)
        self.assertTrue(np.isfinite(metrics["policy_loss"]))
        self.assertIn("reward_term/progress_delta", metrics)
        self.assertTrue(any(not torch.equal(old, new) for old, new in
                            zip(before, trainer.model.parameters())))
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "smoke.pt"
            trainer.save(checkpoint, metadata={"purpose": "engineering smoke"})
            restored = PPOTrainer(TinyGoalEnv(), ACTION, OBSERVATION, MODEL, PPO, seed=12)
            metadata = restored.load(checkpoint)
            self.assertEqual(metadata["purpose"], "engineering smoke")
            self.assertEqual(restored.update_count, 1)
            self.assertEqual(restored.total_steps, trainer.total_steps)
            restored_metrics = restored.train_update(rollout_steps=4)
            self.assertEqual(restored_metrics["update"], 2)
            incompatible = PPOTrainer(TinyGoalEnv(), ACTION, OBSERVATION, MODEL,
                                     PPO.__class__(learning_rate=2e-3), seed=12)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                incompatible.load(checkpoint)

    def test_atomic_checkpoint_save_preserves_previous_file_on_write_failure(self):
        from unittest.mock import patch
        trainer = PPOTrainer(TinyGoalEnv(), ACTION, OBSERVATION, MODEL, PPO, seed=19)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "latest.pt"
            checkpoint.write_bytes(b"previous valid checkpoint")
            with patch("casu_grid.workbench.ppo.torch.save", side_effect=OSError("simulated disk full")):
                with self.assertRaisesRegex(OSError, "simulated disk full"):
                    trainer.save(checkpoint)
            self.assertEqual(checkpoint.read_bytes(), b"previous valid checkpoint")
            self.assertEqual(list(Path(temporary).glob("*.tmp")), [])

    def test_termination_reasons_count_only_finished_worlds(self):
        class ReasonEnv(TinyGoalEnv):
            def step(self, action):
                obs, reward, reached, truncated, info = super().step(action)
                info["termination_reason"] = [
                    "goal" if r else "time_limit" if t else None for r, t in zip(reached, truncated)]
                return obs, reward, reached, truncated, info
        trainer = PPOTrainer(ReasonEnv(count=3, horizon=4), ACTION, OBSERVATION, MODEL, PPO, seed=5)
        metrics = trainer.train_update(rollout_steps=12)
        self.assertNotIn("termination/None", metrics)
        counted = sum(v for k, v in metrics.items() if k.startswith("termination/"))
        self.assertEqual(counted, metrics["completed_episodes"])


if __name__ == "__main__":
    unittest.main()
