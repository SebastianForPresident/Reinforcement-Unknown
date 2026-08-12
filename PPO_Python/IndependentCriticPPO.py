"""CB1 PPO variant with a separately trained, recent-window value network."""

import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from stable_baselines3 import PPO

import CriticRollouts
import IndependentCritic
from Train import PausingPPO


CRITIC_MODES = ("shadow", "active")


class IndependentCriticPPO(PausingPPO):
    """Preserve PPO's actor while decoupling critic data reuse and updates."""

    def __init__(self, *args, **kwargs):
        self._independent_critic_config = None
        self.independent_critic = None
        self.critic_optimizer = None
        self._critic_final_observation = None
        self._critic_final_dones = None
        super().__init__(*args, **kwargs)

    def _setup_model(self):
        super()._setup_model()
        if self._independent_critic_config is not None:
            self._create_independent_critic()

    def _create_independent_critic(self):
        self.independent_critic = IndependentCritic.IndependentCritic(
            self.policy
        ).to("cpu")
        learning_rate = float(self._independent_critic_config["learning_rate"])
        self.critic_optimizer = torch.optim.Adam(
            self.independent_critic.parameters(), lr=learning_rate
        )

    def _move_independent_critic(self, device):
        """Move critic parameters and Adam state together between CPU/CUDA."""
        device = torch.device(device)
        self.independent_critic.to(device)
        for state in self.critic_optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    def enable_independent_critic(
        self,
        rollout_directory,
        mode="shadow",
        learning_rate=1e-4,
        batch_size=64,
        history_rollouts=32,
        train_rollouts_per_update=2,
        epochs_per_rollout=1,
        critic_state_path=None,
    ):
        if mode not in CRITIC_MODES:
            raise ValueError(f"Critic mode must be one of {CRITIC_MODES}")
        if self.independent_critic is not None:
            raise RuntimeError("Independent critic is already enabled")
        self._independent_critic_config = {
            "rollout_directory": str(Path(rollout_directory).resolve()),
            "mode": mode,
            "learning_rate": float(learning_rate),
            "batch_size": int(batch_size),
            "history_rollouts": int(history_rollouts),
            "train_rollouts_per_update": int(train_rollouts_per_update),
            "epochs_per_rollout": int(epochs_per_rollout),
            "critic_state_path": (
                None
                if critic_state_path is None
                else str(Path(critic_state_path).resolve())
            ),
        }
        self._create_independent_critic()

    @property
    def critic_mode(self):
        if self._independent_critic_config is None:
            return None
        return self._independent_critic_config["mode"]

    def set_critic_mode(self, mode):
        if mode not in CRITIC_MODES:
            raise ValueError(f"Critic mode must be one of {CRITIC_MODES}")
        if self._independent_critic_config is None:
            raise RuntimeError("Independent critic is not enabled")
        self._independent_critic_config["mode"] = mode

    def _excluded_save_params(self):
        # Keep frequent SB3 actor checkpoints at their existing size. The
        # independent critic is persisted atomically in one separate snapshot.
        excluded = super()._excluded_save_params()
        for name in ("independent_critic", "critic_optimizer"):
            if name not in excluded:
                excluded.append(name)
        return excluded

    def save_critic_state(self, path):
        if self.independent_critic is None:
            raise RuntimeError("Independent critic is not enabled")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        os.close(descriptor)
        try:
            torch.save(
                {
                    "version": 1,
                    "num_timesteps": int(self.num_timesteps),
                    "config": dict(self._independent_critic_config),
                    "critic": self.independent_critic.state_dict(),
                    "optimizer": self.critic_optimizer.state_dict(),
                },
                temporary_name,
            )
            with open(temporary_name, "rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    def load_critic_state(self, path):
        if self.independent_critic is None:
            raise RuntimeError("Independent critic is not enabled")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if int(state.get("version", -1)) != 1:
            raise ValueError(f"Unsupported independent critic state: {path}")
        self.independent_critic.load_state_dict(state["critic"], strict=True)
        self.critic_optimizer.load_state_dict(state["optimizer"])
        return int(state["num_timesteps"])

    def collect_rollouts(self, *args, **kwargs):
        complete = super().collect_rollouts(*args, **kwargs)
        if complete and self.independent_critic is not None:
            self._critic_final_observation = {
                name: np.asarray(value).copy()
                for name, value in self._last_obs.items()
            }
            self._critic_final_dones = np.asarray(
                self._last_episode_starts, dtype=bool
            ).copy()
        return complete

    def _window(self):
        config = self._independent_critic_config
        return CriticRollouts.RolloutWindow(
            config["rollout_directory"],
            max_rollouts=config["history_rollouts"],
        )

    def _current_rollout(self):
        buffer = self.rollout_buffer
        return CriticRollouts.CriticRollout(
            timestep=self.num_timesteps,
            observations={
                name: CriticRollouts._as_single_env(value)
                for name, value in buffer.observations.items()
            },
            final_observation={
                name: np.asarray(value)[0]
                for name, value in self._critic_final_observation.items()
            },
            rewards=CriticRollouts._as_single_env(buffer.rewards),
            episode_starts=CriticRollouts._as_single_env(buffer.episode_starts),
            dones=self._critic_final_dones,
            actions=CriticRollouts._as_single_env(buffer.actions),
            old_values=CriticRollouts._as_single_env(buffer.values),
            old_log_probs=CriticRollouts._as_single_env(buffer.log_probs),
            returns=CriticRollouts._as_single_env(buffer.returns),
        )

    def _train_independent_critic(self, window):
        config = self._independent_critic_config
        self._move_independent_critic(self.device)
        paths = window.paths()
        count = min(config["train_rollouts_per_update"], len(paths))
        if count == 0:
            return np.nan

        # Rotate deterministically through recent data instead of repeatedly
        # fitting only the newest file. The current rollout is not archived
        # until after this update, so it is always a held-out evaluation set.
        end = (self._n_updates * count) % len(paths)
        selected = [paths[(end + offset) % len(paths)] for offset in range(count)]
        losses = []
        for path in selected:
            rollout = CriticRollouts.load_rollout(path)
            losses.append(
                IndependentCritic.train_rollout(
                    self.independent_critic,
                    self.critic_optimizer,
                    rollout,
                    self.device,
                    batch_size=config["batch_size"],
                    epochs=config["epochs_per_rollout"],
                    max_grad_norm=self.max_grad_norm,
                )
            )
        return float(np.nanmean(losses))

    def _replace_rollout_advantages(self, rollout):
        values = IndependentCritic.predict(
            self.independent_critic,
            rollout.observations,
            self.device,
            batch_size=self._independent_critic_config["batch_size"],
        )
        final_observations = {
            name: np.expand_dims(value, axis=0)
            for name, value in rollout.final_observation.items()
        }
        final_values = IndependentCritic.predict(
            self.independent_critic,
            final_observations,
            self.device,
            batch_size=1,
        )
        self.rollout_buffer.values[:, 0] = values
        self.rollout_buffer.compute_returns_and_advantage(
            last_values=torch.as_tensor(final_values, device=self.device),
            dones=rollout.dones,
        )

    def train(self):
        if self.independent_critic is None:
            return super().train()
        if self._pause_simulation is not None:
            self._pause_simulation()

        # Avoid PausingPPO.train(), which would request the same pause twice.
        window = self._window()
        critic_train_loss = self._train_independent_critic(window)
        current = self._current_rollout()
        metrics = IndependentCritic.evaluate(
            self.independent_critic,
            current,
            self.device,
            batch_size=self._independent_critic_config["batch_size"],
        )

        if self.critic_mode == "active":
            self._replace_rollout_advantages(current)
            self._move_independent_critic("cpu")
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
            original_vf_coef = self.vf_coef
            self.vf_coef = 0.0
            try:
                # Call PPO directly, bypassing PausingPPO's duplicate pause.
                # The shared value
                # output is evaluated but receives no value-loss gradient.
                PPO.train(self)
            finally:
                self.vf_coef = original_vf_coef
        else:
            # Shadow mode leaves ordinary CB1 actor+shared-critic learning
            # byte-for-byte on its existing route.
            self._move_independent_critic("cpu")
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
            PPO.train(self)

        self.logger.record("critic_independent/train_loss", critic_train_loss)
        self.logger.record("critic_independent/heldout_loss", metrics["loss"])
        self.logger.record(
            "critic_independent/heldout_explained_variance",
            metrics["explained_variance"],
        )
        self.logger.record(
            "critic_independent/shared_heldout_loss", metrics["shared_loss"]
        )
        self.logger.record(
            "critic_independent/shared_heldout_explained_variance",
            metrics["shared_explained_variance"],
        )
        self.logger.record(
            "critic_independent/window_rollouts", len(window.paths())
        )

        window.add(
            self.num_timesteps,
            self.rollout_buffer,
            self._critic_final_observation,
            self._critic_final_dones,
        )
        critic_state_path = self._independent_critic_config.get(
            "critic_state_path"
        )
        if critic_state_path is not None:
            # train() began by obtaining Unity's acknowledged pause. Persist
            # here, before the next action resumes simulation.
            self.save_critic_state(critic_state_path)
