import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
import traceback
from datetime import datetime
from pathlib import Path

def Begin_Training(env):
    run_dir = Path("checkpoints") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=15_000,
        save_path=str(run_dir),
        name_prefix="casu_ppo",
    )

    try:
        print(f"Training run directory: {run_dir}")
        model = PPO(
            "MlpPolicy",
            env,
            device="auto",
            tensorboard_log=str(run_dir / "tensorboard"),
        )
        model.learn(250000, callback=checkpoint_callback)
        model.save(str(run_dir / "casu_ppo_final"))
        print(f"Final model saved to: {run_dir / 'casu_ppo_final.zip'}")
        return model
    except Exception:
        print(f"Training hiccup in {run_dir}!")
        traceback.print_exc()
    finally:
        env.close()
