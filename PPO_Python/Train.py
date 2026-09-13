from stable_baselines3 import PPO
from collections import deque
import traceback
from datetime import datetime
from pathlib import Path
import json
import re
import zipfile
import sys
import time
import statistics
import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import explained_variance
import CasualtiesEnv
import ObservationEncoding
import Reward
import Types
import AttackPermission
import ResourceBudget
import os


TARGET_TOTAL_TIMESTEPS = 60_000_000
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.98
PPO_N_STEPS = 2048
PPO_LEARNING_RATE = 1e-4
PPO_BATCH_SIZE = 256
PPO_N_EPOCHS = 4
PPO_TARGET_KL = 0.05
# A full CB1 checkpoint is approximately 200 MB. Saving every five PPO
# updates produced one checkpoint per 10,240 steps and exhausted the training
# filesystem. Keep saves on complete update boundaries, but retain only a
# coarse recovery cadence; training completion still writes an explicit final
# checkpoint. PPO_6 keeps the 25-update cadence selected at its lineage
# boundary: 51,200 decisions per complete post-optimizer checkpoint.
CB1_CHECKPOINT_UPDATES_PER_SAVE = 25
CB1_MODEL_SCHEMA_VERSION = 1
CB1_TENSORBOARD_RUN_ID = 6
CB1_INTERVENTION_SOURCE_STEP = 51_843_364
ATTACK_PRIOR_TARGET = 0.25
ATTACK_PRIOR_INITIAL_COEFFICIENT = 0.1
ATTACK_PRIOR_ANNEAL_STEPS = 1_000_000
PROTOCOL_MANIFEST_NAME = f"protocol_{Types.CHECKPOINT_NAME.lower()}.json"
INTERVENTION_MANIFEST_NAME = "intervention_ppo6.json"


def AttackPriorCoefficient(global_step):
    elapsed = max(int(global_step) - CB1_INTERVENTION_SOURCE_STEP, 0)
    remaining = max(1.0 - elapsed / ATTACK_PRIOR_ANNEAL_STEPS, 0.0)
    return ATTACK_PRIOR_INITIAL_COEFFICIENT * remaining


def AttackPriorLoss(distribution):
    attack_probability = distribution.distribution[-1].probs[:, 1]
    probability = attack_probability.clamp(1e-6, 1.0 - 1e-6)
    target = th.full_like(probability, ATTACK_PRIOR_TARGET)
    return (
        F.binary_cross_entropy(probability, target),
        attack_probability.mean(),
    )


def ProtocolManifest(env):
    result = {
        "checkpoint": Types.CHECKPOINT_NAME,
        "reward": Types.REWARD_NAME,
        "protocol_version": Types.PROTOCOL_VERSION,
        "physics_hz": 50,
        "gameplay_update_hz": 60,
        "render_independent_gameplay": True,
        "policy_hz": 5,
        "physics_ticks_per_action": Types.POLICY_PHYSICS_TICKS,
        "max_episode_steps": env.max_episode_steps,
        "max_stagnant_steps": env.max_stagnant_steps,
        "max_inactive_steps": env.max_inactive_steps,
        "max_mobile_stagnant_steps": env.max_mobile_stagnant_steps,
        "incomplete_episode_penalty": Reward.INCOMPLETE_EPISODE_PENALTY,
        "model_schema_version": CB1_MODEL_SCHEMA_VERSION,
        "gamma": PPO_GAMMA,
        "gae_lambda": PPO_GAE_LAMBDA,
        "n_steps": PPO_N_STEPS,
        "action_space_nvec": env.action_space.nvec.tolist(),
        "observation_shapes": {
            name: list(space.shape)
            for name, space in env.observation_space.spaces.items()
        },
    }
    if getattr(env, "permission_mode", False):
        result["model_schema_version"] = 2
        result["attack_permission"] = AttackPermission.contract()
    if getattr(env, "missed_attack_penalty_mode", False):
        result["missed_attack_penalty"] = {
            "scale": Reward.MISSED_ATTACK_PENALTY_SCALE,
            "signal": (
                "executed swing with zero macrostep claw-health loss"
            ),
        }
    if getattr(env, "attack_prior_mode", False):
        result["attack_logit_prior"] = {
            "target_attack_probability": ATTACK_PRIOR_TARGET,
            "initial_coefficient": ATTACK_PRIOR_INITIAL_COEFFICIENT,
            "anneal_steps": ATTACK_PRIOR_ANNEAL_STEPS,
            "scope": "attack categorical branch only",
        }
    if getattr(env, "natural_stamina_mode", False):
        result["natural_stamina"] = {
            "stamina_reset_by_functional_godmode": False,
            "energy_reset_by_functional_godmode": True,
            "incapacitated_inactivity_steps": env.max_inactive_steps,
            "incapacitated_inactivity_simulated_seconds": (
                env.max_inactive_steps / 5.0
            ),
            "terminal_penalty": Reward.INCOMPLETE_EPISODE_PENALTY,
        }
    if getattr(env, "attack_budget_mode", False):
        result["model_schema_version"] = 2
        result["attack_budget"] = ResourceBudget.contract()
    return result


def ValidateOrWriteProtocolManifest(run_dir, env, resuming):
    path = run_dir / PROTOCOL_MANIFEST_NAME
    expected = ProtocolManifest(env)
    if resuming:
        if not path.is_file():
            raise RuntimeError(
                f"Refusing to resume {run_dir}: missing {PROTOCOL_MANIFEST_NAME}. "
                "Pre-CB1 checkpoints are intentionally incompatible."
            )
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise RuntimeError(
                f"Refusing to resume {run_dir}: protocol manifest does not "
                "match the current CB1 observation/action contract."
            )
    else:
        path.write_text(
            json.dumps(expected, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def CheckpointStep(path):
    match = re.search(r"_(\d+)_steps\.zip$", Path(path).name)
    if match is None:
        raise ValueError(f"Checkpoint has no explicit global step: {path}")
    return int(match.group(1))


def ValidateOrWriteInterventionManifest(run_dir, resume_model):
    """Record the PPO_6 actual-swing/contact shadow experiment."""
    path = run_dir / INTERVENTION_MANIFEST_NAME
    expected = {
        "phase": "PPO_6",
        "source_checkpoint": (
            f"casu_ppo_cb1_vb1_{CB1_INTERVENTION_SOURCE_STEP}_steps.zip"
        ),
        "source_step": CB1_INTERVENTION_SOURCE_STEP,
        "experiment": "actual_swing_claw_contact_shadow",
        "wire_protocol_version": Types.PROTOCOL_VERSION,
        "policy_observation_shape_unchanged": True,
        "functional_stamina_reset": True,
        "capable_only_inactivity": True,
        "actual_swing_signal": "attack_cooldown_increase",
        "contact_signal": "pre_godmode_claw_health_loss",
        "ineffective_swing_penalty_scale": (
            Reward.INEFFECTIVE_SWING_PENALTY_SCALE
        ),
        "ineffective_swing_penalty_enabled": (
            Reward.INEFFECTIVE_SWING_PENALTY_ENABLED
        ),
    }
    if os.environ.get("PPO_ATTACK_BUDGET", "0") == "1":
        expected = {
            "phase": "PPO_6",
            "source_checkpoint": (
                f"casu_ppo_cb1_vb1_{CB1_INTERVENTION_SOURCE_STEP}_steps.zip"
            ),
            "source_step": CB1_INTERVENTION_SOURCE_STEP,
            "experiment": "dense_budget_conditioned_resource_efficient_descent",
            "wire_protocol_version": Types.PROTOCOL_VERSION,
            "model_schema_version": 2,
            "source_general_shape": 139,
            "policy_general_shape": 140,
            "zero_column_model_surgery": True,
            "initial_policy_outputs_preserved": True,
            "attack_budget": ResourceBudget.contract(),
            "missed_attack_penalty": 0.0,
            "unused_budget_reward": 0.0,
            "immediate_swing_cost": "-10 / assigned episode budget",
            "attack_logit_prior": {
                "target_attack_probability": ATTACK_PRIOR_TARGET,
                "initial_coefficient": ATTACK_PRIOR_INITIAL_COEFFICIENT,
                "anneal_steps": ATTACK_PRIOR_ANNEAL_STEPS,
            },
        }
    elif (
        os.environ.get("PPO_ATTACK_LOGIT_PRIOR", "0") == "1"
        and os.environ.get("PPO_MISSED_ATTACK_PENALTY", "0") == "1"
        and os.environ.get("PPO_NATURAL_STAMINA", "0") != "1"
    ):
        expected = {
            "phase": "PPO_6",
            "source_checkpoint": (
                f"casu_ppo_cb1_vb1_{CB1_INTERVENTION_SOURCE_STEP}_steps.zip"
            ),
            "source_step": CB1_INTERVENTION_SOURCE_STEP,
            "experiment": "annealed_attack_prior_with_missed_swing_penalty",
            "wire_protocol_version": Types.PROTOCOL_VERSION,
            "model_schema_version": CB1_MODEL_SCHEMA_VERSION,
            "policy_observation_shape_unchanged": True,
            "functional_stamina_reset": True,
            "sampling_or_action_override": False,
            "target_attack_probability": ATTACK_PRIOR_TARGET,
            "initial_prior_coefficient": ATTACK_PRIOR_INITIAL_COEFFICIENT,
            "prior_anneal_steps": ATTACK_PRIOR_ANNEAL_STEPS,
            "miss_definition": (
                "game-confirmed executed swing with zero macrostep "
                "pre-godmode claw-health loss"
            ),
            "missed_attack_penalty": -Reward.MISSED_ATTACK_PENALTY_SCALE,
            "contact_penalty": 0.0,
            "cooldown_hold_penalty": 0.0,
            "attack_budget": False,
            "natural_stamina": False,
        }
    elif (
        os.environ.get("PPO_ATTACK_LOGIT_PRIOR", "0") == "1"
        and os.environ.get("PPO_NATURAL_STAMINA", "0") == "1"
    ):
        expected = {
            "phase": "PPO_6",
            "source_checkpoint": (
                f"casu_ppo_cb1_vb1_{CB1_INTERVENTION_SOURCE_STEP}_steps.zip"
            ),
            "source_step": CB1_INTERVENTION_SOURCE_STEP,
            "experiment": "natural_exhaustion_with_annealed_attack_prior",
            "wire_protocol_version": Types.PROTOCOL_VERSION,
            "model_schema_version": CB1_MODEL_SCHEMA_VERSION,
            "policy_observation_shape_unchanged": True,
            "stamina_already_observable": True,
            "stamina_reset_by_functional_godmode": False,
            "energy_reset_by_functional_godmode": True,
            "other_functional_godmode_repairs_retained": True,
            "exhaustion_reward": 0.0,
            "incapacitated_inactivity_steps": 75,
            "incapacitated_inactivity_simulated_seconds": 15.0,
            "incapacitated_inactivity_penalty": (
                Reward.INCOMPLETE_EPISODE_PENALTY
            ),
            "target_attack_probability": ATTACK_PRIOR_TARGET,
            "initial_prior_coefficient": ATTACK_PRIOR_INITIAL_COEFFICIENT,
            "prior_anneal_steps": ATTACK_PRIOR_ANNEAL_STEPS,
            "sampling_or_action_override": False,
            "synthetic_attack_cost": 0.0,
            "missed_attack_penalty": 0.0,
            "attack_budget": False,
        }
    elif os.environ.get("PPO_ATTACK_LOGIT_PRIOR", "0") == "1":
        expected = {
            "phase": "PPO_6",
            "source_checkpoint": (
                f"casu_ppo_cb1_vb1_{CB1_INTERVENTION_SOURCE_STEP}_steps.zip"
            ),
            "source_step": CB1_INTERVENTION_SOURCE_STEP,
            "experiment": "isolated_annealed_attack_logit_prior",
            "wire_protocol_version": Types.PROTOCOL_VERSION,
            "model_schema_version": CB1_MODEL_SCHEMA_VERSION,
            "policy_observation_shape_unchanged": True,
            "attack_action_unchanged": True,
            "sampling_or_action_override": False,
            "target_attack_probability": ATTACK_PRIOR_TARGET,
            "initial_prior_coefficient": ATTACK_PRIOR_INITIAL_COEFFICIENT,
            "prior_anneal_steps": ATTACK_PRIOR_ANNEAL_STEPS,
            "prior_loss": "binary cross entropy on attack branch probability",
            "missed_attack_penalty": 0.0,
            "watchdog_penalty": Reward.INCOMPLETE_EPISODE_PENALTY,
        }
    elif os.environ.get("PPO_MISSED_ATTACK_PENALTY", "0") == "1":
        expected = {
            "phase": "PPO_6",
            "source_checkpoint": (
                f"casu_ppo_cb1_vb1_{CB1_INTERVENTION_SOURCE_STEP}_steps.zip"
            ),
            "source_step": CB1_INTERVENTION_SOURCE_STEP,
            "experiment": "missed_executed_attack_penalty",
            "wire_protocol_version": Types.PROTOCOL_VERSION,
            "model_schema_version": CB1_MODEL_SCHEMA_VERSION,
            "policy_observation_shape_unchanged": True,
            "attack_action_unchanged": True,
            "miss_definition": (
                "executed swing with zero macrostep claw-health loss"
            ),
            "missed_attack_penalty": -Reward.MISSED_ATTACK_PENALTY_SCALE,
            "contact_penalty": 0.0,
            "cooldown_hold_penalty": 0.0,
            "watchdog_penalty": Reward.INCOMPLETE_EPISODE_PENALTY,
            "permission_or_phase_logic": False,
        }
    elif (
        os.environ.get("PPO_ATTACK_PERMISSION", "0") == "1"
        or os.environ.get("PPO_ATTACK_PENALTY", "0") == "1"
        or os.environ.get("PPO_ATTACK_TERMINAL", "0") == "1"
    ):
        expected = AttackPermission.contract()
    if path.is_file():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise RuntimeError(
                f"Refusing PPO_6 continuation: {INTERVENTION_MANIFEST_NAME} "
                "does not match the current intervention contract."
            )
        return

    source_step = CheckpointStep(resume_model)
    if source_step != CB1_INTERVENTION_SOURCE_STEP:
        raise RuntimeError(
            "Refusing to begin PPO_6 from an ambiguous checkpoint: expected "
            f"{CB1_INTERVENTION_SOURCE_STEP}, selected {source_step}."
        )
    path.write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def PrepareTensorboardContinuation(run_dir):
    """Pin this intervention to PPO_6 while preserving SB3 resume semantics."""
    tensorboard_dir = run_dir / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    target = tensorboard_dir / f"PPO_{CB1_TENSORBOARD_RUN_ID}"
    target.mkdir(exist_ok=True)

    newer = []
    for path in tensorboard_dir.glob("PPO_*"):
        try:
            run_id = int(path.name.rsplit("_", 1)[1])
        except ValueError:
            continue
        if run_id > CB1_TENSORBOARD_RUN_ID:
            newer.append(path.name)
    if newer:
        raise RuntimeError(
            "Refusing ambiguous TensorBoard continuation: found runs newer "
            f"than PPO_{CB1_TENSORBOARD_RUN_ID}: {sorted(newer)}"
        )


class PausingPPO(PPO):
    """Freeze Unity while PPO performs an optimizer update."""

    def __init__(self, *args, pause_simulation=None, **kwargs):
        self._pause_simulation = pause_simulation
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        """Exclude the live Unity control callback from SB3 serialization.

        The callback closes over Server's socket/threading state.  That state
        is deliberately process-local and includes threading.Event locks,
        which cloudpickle cannot serialize.  The callback is restored by the
        training entry point after loading a checkpoint.
        """
        excluded = super()._excluded_save_params()
        if "_pause_simulation" not in excluded:
            excluded.append("_pause_simulation")
        return excluded

    def train(self):
        if self._pause_simulation is not None:
            self._pause_simulation()
        result = super().train()
        temperature = None
        if isinstance(self.policy, AttackPermission.TerminalExplorationPolicy):
            try:
                curriculum_enabled = self.env.get_attr(
                    "attack_curriculum_mode"
                )[0]
                if curriculum_enabled:
                    temperature = self.env.get_attr(
                        "attack_exploration_temperature"
                    )[0]
            except (AttributeError, IndexError):
                pass
        AttackPermission.configure_exploration(self, temperature)
        if isinstance(self.policy, AttackPermission.TerminalExplorationPolicy):
            try:
                self.env.set_attr(
                    "attack_exploration_temperature_applied",
                    self.policy.attack_temperature,
                )
            except AttributeError:
                pass
            self.logger.record(
                "train/prohibited_attack_temperature",
                self.policy.attack_temperature,
            )
        return result


class AttackPriorPPO(PausingPPO):
    """PPO with an annealed, attack-branch-only probability prior."""

    def attack_prior_coefficient(self):
        return AttackPriorCoefficient(self.num_timesteps)

    def evaluate_actions_with_prior(self, observations, actions):
        """Evaluate PPO actions and prior from one shared feature pass."""
        features = self.policy.extract_features(observations)
        if self.policy.share_features_extractor:
            latent_pi, latent_vf = self.policy.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.policy.mlp_extractor.forward_critic(vf_features)
        distribution = self.policy._get_action_dist_from_latent(latent_pi)
        values = self.policy.value_net(latent_vf)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        prior_loss, attack_probability = AttackPriorLoss(distribution)
        return values, log_prob, entropy, prior_loss, attack_probability

    def train(self):
        """Standard SB3 PPO update plus the annealed attack prior loss."""
        if self._pause_simulation is not None:
            self._pause_simulation()
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(
                self._current_progress_remaining
            )

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        prior_losses, attack_probabilities = [], []
        prior_coefficient = self.attack_prior_coefficient()
        continue_training = True
        loss = None
        approx_kl_divs = []

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                (
                    values,
                    log_prob,
                    entropy,
                    prior_loss,
                    attack_probability,
                ) = self.evaluate_actions_with_prior(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (
                        advantages - advantages.mean()
                    ) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean(
                    (th.abs(ratio - 1) > clip_range).float()
                ).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                prior_losses.append(prior_loss.item())
                attack_probabilities.append(attack_probability.item())
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + prior_coefficient * prior_loss
                )

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(
                        (th.exp(log_ratio) - 1) - log_ratio
                    ).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                if (
                    self.target_kl is not None
                    and approx_kl_div > 1.5 * self.target_kl
                ):
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            "Early stopping due to reaching max KL: "
                            f"{approx_kl_div:.2f}"
                        )
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/attack_prior_loss", np.mean(prior_losses))
        self.logger.record(
            "train/attack_probability", np.mean(attack_probabilities)
        )
        self.logger.record(
            "train/attack_prior_coefficient", prior_coefficient
        )
        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std", th.exp(self.policy.log_std).mean().item()
            )
        self.logger.record(
            "train/n_updates", self._n_updates, exclude="tensorboard"
        )
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


class TrainingProgressCallback(BaseCallback):
    """Display total PPO progress without requiring tqdm or rich."""

    def __init__(self, update_interval=1.0):
        super().__init__(verbose=0)
        self.update_interval = update_interval
        self.started_at = None
        self.last_update = 0.0

    def _on_training_start(self):
        self.started_at = time.monotonic()
        self.last_update = 0.0
        self._print_progress(force=True)

    def _on_step(self):
        self._print_progress()
        return True

    def _on_training_end(self):
        self._print_progress(force=True)
        sys.stderr.write("\n")
        sys.stderr.flush()

    def _print_progress(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_update < self.update_interval:
            return

        self.last_update = now
        current = self.model.num_timesteps
        target = max(self.model._total_timesteps, 1)
        fraction = min(max(current / target, 0.0), 1.0)
        bar_width = 30
        filled = int(bar_width * fraction)
        bar = "█" * filled + "░" * (bar_width - filled)

        elapsed = max(now - self.started_at, 0.001)
        rate = max((current - self.model._num_timesteps_at_start) / elapsed, 0.0)
        remaining = max(target - current, 0)
        eta_seconds = remaining / rate if rate > 0.0 else 0.0

        def format_duration(seconds):
            seconds = int(seconds)
            hours, seconds = divmod(seconds, 3600)
            minutes, seconds = divmod(seconds, 60)
            if hours:
                return f"{hours}h {minutes:02d}m"
            return f"{minutes}m {seconds:02d}s"

        message = (
            f"\rCB1 [{bar}] {fraction:6.2%} | "
            f"{current:,}/{target:,} steps | "
            f"{rate:5.1f} FPS | ETA {format_duration(eta_seconds)}"
        )
        sys.stderr.write(message)
        sys.stderr.flush()


class SuccessEpisodeLengthCallback(BaseCallback):
    """Log completion speed without treating short failures as improvement."""

    TAG = "rollout/success_ep_len_mean"
    STATE_VERSION = 1

    def __init__(self, window_size=100, state_path=None):
        super().__init__(verbose=0)
        self.episodes = deque(maxlen=int(window_size))
        self.state_path = None if state_path is None else Path(state_path)

    def _on_training_start(self):
        if self.state_path is None or not self.state_path.is_file():
            return
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if state.get("version") != self.STATE_VERSION:
            raise RuntimeError(
                f"Unsupported success episode state version in {self.state_path}"
            )
        restored = []
        for item in state.get("episodes", []):
            terminal_step = int(item["terminal_global_step"])
            if terminal_step <= int(self.model.num_timesteps):
                restored.append(
                    (int(item["length"]), bool(item["succeeded"]), terminal_step)
                )
        self.episodes.extend(restored[-self.episodes.maxlen :])

    def _on_step(self):
        for done, info in zip(
            self.locals.get("dones", ()),
            self.locals.get("infos", ()),
        ):
            if not done or "episode_steps" not in info:
                continue
            succeeded = not bool(info.get("watchdog_failed", False))
            succeeded = succeeded and not bool(
                info.get("TimeLimit.truncated", False)
            )
            self.episodes.append(
                (int(info["episode_steps"]), succeeded, int(self.model.num_timesteps))
            )
            self._save_state()
        return True

    def success_mean(self):
        lengths = [
            length for length, succeeded, _terminal_step in self.episodes
            if succeeded
        ]
        if not lengths:
            return None
        return statistics.fmean(lengths)

    def _save_state(self):
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        payload = {
            "version": self.STATE_VERSION,
            "window_size": self.episodes.maxlen,
            "episodes": [
                {
                    "length": length,
                    "succeeded": succeeded,
                    "terminal_global_step": terminal_step,
                }
                for length, succeeded, terminal_step in self.episodes
            ],
        }
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.state_path)

    def _on_rollout_end(self):
        mean = self.success_mean()
        if mean is not None:
            self.logger.record(self.TAG, mean)


class AttackBudgetCallback(BaseCallback):
    """Log whether constrained episodes finish within their assigned budget."""

    def __init__(self, window_size=100):
        super().__init__(verbose=0)
        self.outcomes = deque(maxlen=int(window_size))

    def _on_step(self):
        for done, info in zip(
            self.locals.get("dones", ()), self.locals.get("infos", ())
        ):
            if not done or "attack_budget" not in info:
                continue
            budget = max(int(info["attack_budget"]), 1)
            used = int(info["attack_budget_used"])
            exceeded = bool(info["attack_budget_exceeded"])
            succeeded = not bool(info.get("watchdog_failed", False))
            self.outcomes.append((used / budget, exceeded, succeeded))
        return True

    def _on_rollout_end(self):
        if not self.outcomes:
            return
        self.logger.record(
            "rollout/attack_budget_utilization_mean",
            statistics.fmean(value for value, _, _ in self.outcomes),
        )
        self.logger.record(
            "rollout/attack_budget_exceeded_rate",
            statistics.fmean(float(value) for _, value, _ in self.outcomes),
        )
        successful = [
            value for value, _, succeeded in self.outcomes if succeeded
        ]
        if successful:
            self.logger.record(
                "rollout/success_attack_budget_utilization_mean",
                statistics.fmean(successful),
            )


class RolloutCheckpointCallback(BaseCallback):
    """Save only after complete PPO updates, never midway through a rollout."""

    def __init__(self, save_path, name_prefix, updates_per_save=5):
        super().__init__(verbose=0)
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.updates_per_save = int(updates_per_save)
        self.completed_rollouts = 0
        self.last_saved_timestep = None

    def _on_step(self):
        return True

    def _on_rollout_start(self):
        # on_rollout_start for rollout N+1 occurs after train() has consumed
        # rollout N, so the saved policy and timestep form a complete boundary.
        if (
            self.completed_rollouts > 0
            and self.completed_rollouts % self.updates_per_save == 0
        ):
            self._save()

    def _on_rollout_end(self):
        self.completed_rollouts += 1

    def _on_training_end(self):
        self._save()

    def _save(self):
        timestep = int(self.model.num_timesteps)
        if timestep == self.last_saved_timestep:
            return
        self.save_path.mkdir(parents=True, exist_ok=True)
        path = self.save_path / f"{self.name_prefix}_{timestep}_steps"
        self.model.save(str(path))
        self.last_saved_timestep = timestep

def FindResumeModel(run_dir):
    """Select the greatest explicit step checkpoint in a run directory."""
    candidates = []
    legacy_candidates = []
    for path in run_dir.glob("casu_ppo*.zip"):
        if not path.is_file() or not zipfile.is_zipfile(path):
            continue
        match = re.search(r"_(\d+)_steps\.zip$", path.name)
        if match is None:
            legacy_candidates.append(path)
            continue
        candidates.append((int(match.group(1)), path.stat().st_mtime, path))

    if candidates:
        return max(candidates, key=lambda item: (item[0], item[1]))[2]
    if legacy_candidates:
        return max(legacy_candidates, key=lambda path: path.stat().st_mtime)

    raise FileNotFoundError(
        f"No resumable PPO model found in {run_dir}. Expected a "
        "valid casu_ppo*.zip file."
    )

def Begin_Training(env, pause_simulation=None, resume_dir=None):
    if resume_dir is None:
        run_dir = Path("checkpoints") / (
            f"{Types.CHECKPOINT_NAME}_"
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        resume_model = None
    else:
        run_dir = Path(resume_dir).expanduser().resolve()
        if not run_dir.is_dir():
            raise NotADirectoryError(f"Resume directory does not exist: {run_dir}")
        resume_model = FindResumeModel(run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        ValidateOrWriteProtocolManifest(
            run_dir,
            env,
            resuming=resume_model is not None,
        )
    except Exception:
        env.close()
        raise

    checkpoint_callback = RolloutCheckpointCallback(
        save_path=str(run_dir),
        name_prefix="casu_ppo_cb1_vb1",
        updates_per_save=CB1_CHECKPOINT_UPDATES_PER_SAVE,
    )

    try:
        print(f"Training run directory: {run_dir}")
        if resume_model is None:
            model = PausingPPO(
                "MultiInputPolicy",
                env,
                policy_kwargs={
                    "features_extractor_class": ObservationEncoding.CasualtiesFeatureExtractor,
                },
                gamma=PPO_GAMMA,
                gae_lambda=PPO_GAE_LAMBDA,
                n_steps=PPO_N_STEPS,
                learning_rate=PPO_LEARNING_RATE,
                batch_size=PPO_BATCH_SIZE,
                n_epochs=PPO_N_EPOCHS,
                target_kl=PPO_TARGET_KL,
                device="auto",
                tensorboard_log=str(run_dir / "tensorboard"),
                pause_simulation=pause_simulation,
            )
            reset_num_timesteps = True
            total_timesteps = TARGET_TOTAL_TIMESTEPS
        else:
            print(f"Resuming model: {resume_model}")
            PrepareTensorboardContinuation(run_dir)
            ValidateOrWriteInterventionManifest(run_dir, resume_model)
            model_class = (
                AttackPriorPPO
                if getattr(env, "attack_prior_mode", False)
                else PausingPPO
            )
            model = model_class.load(
                str(resume_model),
                env=env,
                policy_kwargs={
                    "features_extractor_class": ObservationEncoding.CasualtiesFeatureExtractor,
                },
                device="auto",
                tensorboard_log=str(run_dir / "tensorboard"),
                custom_objects={
                    "learning_rate": PPO_LEARNING_RATE,
                    "gamma": PPO_GAMMA,
                    "gae_lambda": PPO_GAE_LAMBDA,
                    "batch_size": PPO_BATCH_SIZE,
                    "n_epochs": PPO_N_EPOCHS,
                    "target_kl": PPO_TARGET_KL,
                    # A checkpoint written by an experimental collector may
                    # have a longer physical collection window. Plain CB1
                    # always restores its original optimizer rollout size.
                    "n_steps": PPO_N_STEPS,
                },
            )
            model._pause_simulation = pause_simulation
            env.ConfigureAttackCurriculum(run_dir)
            AttackPermission.configure_exploration(
                model,
                env.attack_exploration_temperature
                if env.attack_curriculum_mode
                else None,
            )
            env.attack_exploration_temperature_applied = (
                model.policy.attack_temperature
                if isinstance(
                    model.policy,
                    AttackPermission.TerminalExplorationPolicy,
                )
                else 1.0
            )
            reset_num_timesteps = False
            total_timesteps = max(
                0,
                TARGET_TOTAL_TIMESTEPS - model.num_timesteps,
            )
            print(
                f"Resuming from {model.num_timesteps} steps; "
                f"training {total_timesteps} additional steps "
                f"toward {TARGET_TOTAL_TIMESTEPS}."
            )

        callback = CallbackList([
            checkpoint_callback,
            SuccessEpisodeLengthCallback(
                state_path=run_dir / "success_ep_len_state.json"
            ),
            AttackBudgetCallback(),
            TrainingProgressCallback(),
        ])

        model.learn(
            total_timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name="PPO",
        )
        final_model = run_dir / (
            f"casu_ppo_cb1_vb1_{int(model.num_timesteps)}_steps"
        )
        if not final_model.with_suffix(".zip").is_file():
            model.save(str(final_model))
        print(f"Final model saved to: {final_model}.zip")
        return model
    except Exception:
        print(f"Training hiccup in {run_dir}!")
        traceback.print_exc()
    finally:
        env.close()
