import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
import traceback
from datetime import datetime
from pathlib import Path
import zipfile
import sys
import time
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import ObservationEncoding


TARGET_TOTAL_TIMESTEPS = 1_000_000


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
            f"\rC12 [{bar}] {fraction:6.2%} | "
            f"{current:,}/{target:,} steps | "
            f"{rate:5.1f} FPS | ETA {format_duration(eta_seconds)}"
        )
        sys.stderr.write(message)
        sys.stderr.flush()

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
        run_dir = Path("checkpoints") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        resume_model = None
    else:
        run_dir = Path(resume_dir).expanduser().resolve()
        if not run_dir.is_dir():
            raise NotADirectoryError(f"Resume directory does not exist: {run_dir}")
        resume_model = FindResumeModel(run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(run_dir),
        name_prefix="casu_ppo",
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
                device="auto",
                tensorboard_log=str(run_dir / "tensorboard"),
                pause_simulation=pause_simulation,
            )
            final_model = run_dir / "casu_ppo_c13_1m_final"
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
            )
            model._pause_simulation = pause_simulation
            final_model = run_dir / "casu_ppo_c13_1m_final"
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
