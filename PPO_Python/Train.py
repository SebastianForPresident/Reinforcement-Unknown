import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
import traceback
from datetime import datetime
from pathlib import Path


class PausingPPO(PPO):
    """Freeze Unity while PPO performs an optimizer update."""

    def __init__(self, *args, pause_simulation=None, **kwargs):
        self._pause_simulation = pause_simulation
        super().__init__(*args, **kwargs)

    def train(self):
        if self._pause_simulation is not None:
            self._pause_simulation()
        return super().train()


def FindResumeModel(run_dir):
    """Select the newest completed PPO model in an existing run directory."""
    candidates = sorted(
        run_dir.glob("casu_ppo*_final.zip"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    start_model = run_dir / "casu_ppo_start.zip"
    if start_model.is_file():
        return start_model

    raise FileNotFoundError(
        f"No resumable PPO model found in {run_dir}. Expected a "
        "casu_ppo*_final.zip or casu_ppo_start.zip file."
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
        save_freq=15_000,
        save_path=str(run_dir),
        name_prefix="casu_ppo",
    )

    try:
        print(f"Training run directory: {run_dir}")
        if resume_model is None:
            model = PausingPPO(
                "MlpPolicy",
                env,
                device="auto",
                tensorboard_log=str(run_dir / "tensorboard"),
                pause_simulation=pause_simulation,
            )
            final_model = run_dir / "casu_ppo_final"
            reset_num_timesteps = True
        else:
            print(f"Resuming model: {resume_model}")
            model = PausingPPO.load(
                str(resume_model),
                env=env,
                device="auto",
                tensorboard_log=str(run_dir / "tensorboard"),
            )
            model._pause_simulation = pause_simulation
            final_model = run_dir / "casu_ppo_resumed_final"
            reset_num_timesteps = False

        model.learn(
            250000,
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
