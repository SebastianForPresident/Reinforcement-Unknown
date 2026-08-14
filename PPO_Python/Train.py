from stable_baselines3 import PPO
import traceback
from datetime import datetime
from pathlib import Path
import json
import zipfile
import sys
import time
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import ObservationEncoding
import Types


TARGET_TOTAL_TIMESTEPS = 3_000_000
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_N_STEPS = 2048
PPO_LEARNING_RATE = 1e-4
PPO_BATCH_SIZE = 256
PPO_N_EPOCHS = 4
PPO_TARGET_KL = 0.05
CB1_MODEL_SCHEMA_VERSION = 1
PROTOCOL_MANIFEST_NAME = f"protocol_{Types.CHECKPOINT_NAME.lower()}.json"


def ProtocolManifest(env):
    return {
        "checkpoint": Types.CHECKPOINT_NAME,
        "reward": Types.REWARD_NAME,
        "protocol_version": Types.PROTOCOL_VERSION,
        "physics_hz": 50,
        "policy_hz": 5,
        "physics_ticks_per_action": Types.POLICY_PHYSICS_TICKS,
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
    """Select the newest completed PPO model in an existing run directory."""
    candidates = sorted(
        (
            path
            for path in run_dir.glob("casu_ppo*.zip")
            if path.is_file() and zipfile.is_zipfile(path)
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

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
        updates_per_save=5,
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
            final_model = run_dir / "casu_ppo_cb1_vb1_500k"
            reset_num_timesteps = True
            total_timesteps = TARGET_TOTAL_TIMESTEPS
        else:
            print(f"Resuming model: {resume_model}")
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
            final_model = run_dir / "casu_ppo_cb1_vb1_500k"
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
            TrainingProgressCallback(),
        ])

        model.learn(
            total_timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name="PPO",
        )
        model.save(str(final_model))
        print(f"Final model saved to: {final_model}.zip")
        return model
    except Exception:
        print(f"Training hiccup in {run_dir}!")
        traceback.print_exc()
    finally:
        env.close()
