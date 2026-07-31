import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
import traceback
from datetime import datetime
from pathlib import Path
import zipfile
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
        save_freq=50_000,
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
            final_model = run_dir / "casu_ppo_final"
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
            final_model = run_dir / "casu_ppo_resumed_final"
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

        model.learn(
            total_timesteps,
            callback=checkpoint_callback,
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
