"""Entry point for the CB1 observed-return PPO experiment."""

from pathlib import Path
import traceback

from stable_baselines3.common.callbacks import CallbackList

import ObservationEncoding
import ReinforcePPO
import Train


def Begin_Reinforce_Training(env, pause_simulation, resume_dir, mode="shadow"):
    run_dir = Path(resume_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Resume directory does not exist: {run_dir}")
    resume_model = Train.FindResumeModel(run_dir)
    Train.ValidateOrWriteProtocolManifest(run_dir, env, resuming=True)

    checkpoint_callback = Train.RolloutCheckpointCallback(
        save_path=str(run_dir),
        name_prefix="casu_ppo_cb1_vb1",
        updates_per_save=5,
    )
    try:
        print(f"Training run directory: {run_dir}")
        print(f"Resuming CB1 actor: {resume_model}")
        model = ReinforcePPO.ReinforceLookaheadPPO.load(
            str(resume_model),
            env=env,
            policy_kwargs={
                "features_extractor_class": (
                    ObservationEncoding.CasualtiesFeatureExtractor
                ),
            },
            device="auto",
            tensorboard_log=str(run_dir / "tensorboard"),
            custom_objects={
                "learning_rate": Train.PPO_LEARNING_RATE,
                "batch_size": Train.PPO_BATCH_SIZE,
                "n_epochs": Train.PPO_N_EPOCHS,
                "target_kl": Train.PPO_TARGET_KL,
                "n_steps": Train.PPO_N_STEPS + 458,
                "reinforce_mode": mode,
                "reinforce_train_steps": Train.PPO_N_STEPS,
                "reinforce_lookahead_steps": 458,
            },
        )
        model._pause_simulation = pause_simulation
        model.reinforce_mode = mode
        model.reinforce_train_steps = Train.PPO_N_STEPS
        model.reinforce_lookahead_steps = 458
        total_timesteps = max(0, Train.TARGET_TOTAL_TIMESTEPS - model.num_timesteps)
        print(
            f"Resuming from {model.num_timesteps} steps in REINFORCE {mode} "
            f"mode; each update trains 2,048 steps after collecting 458 "
            f"lookahead steps."
        )
        callbacks = CallbackList([
            checkpoint_callback,
            Train.TrainingProgressCallback(),
        ])
        model.learn(
            total_timesteps,
            callback=callbacks,
            reset_num_timesteps=False,
            tb_log_name="PPO",
        )
        final_model = run_dir / "casu_ppo_cb1_vb1_2m"
        model.save(str(final_model))
        print(f"Final model saved to: {final_model}.zip")
        return model
    except Exception:
        print(f"REINFORCE {mode} training hiccup in {run_dir}!")
        traceback.print_exc()
    finally:
        env.close()

