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
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import ObservationEncoding
import Reward
import Types


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
# checkpoint.
CB1_CHECKPOINT_UPDATES_PER_SAVE = 100
CB1_MODEL_SCHEMA_VERSION = 1
CB1_TENSORBOARD_RUN_ID = 5
PROTOCOL_MANIFEST_NAME = f"protocol_{Types.CHECKPOINT_NAME.lower()}.json"


def ProtocolManifest(env):
    return {
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


def PrepareTensorboardContinuation(run_dir):
    """Pin this intervention to PPO_5 while preserving SB3 resume semantics."""
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
        return super().train()


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
            model = PausingPPO.load(
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
