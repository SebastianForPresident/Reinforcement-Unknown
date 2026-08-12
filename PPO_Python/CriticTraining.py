"""Entry point for the continuous CB1 independent-critic training phase."""

from pathlib import Path
import traceback

from stable_baselines3.common.callbacks import CallbackList

import IndependentCriticPPO
import ObservationEncoding
import Train


CRITIC_STATE_NAME = "independent_critic_cb1_latest.pt"


def Begin_Critic_Training(
    env,
    pause_simulation,
    resume_dir,
    mode="shadow",
):
    """Resume the same CB1 actor lineage with an independent critic enabled."""
    run_dir = Path(resume_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Resume directory does not exist: {run_dir}")
    resume_model = Train.FindResumeModel(run_dir)
    Train.ValidateOrWriteProtocolManifest(run_dir, env, resuming=True)
    critic_state_path = run_dir / CRITIC_STATE_NAME
    rollout_directory = run_dir / "critic_rollouts"

    checkpoint_callback = Train.RolloutCheckpointCallback(
        save_path=str(run_dir),
        name_prefix="casu_ppo_cb1_vb1",
        updates_per_save=5,
    )

    try:
        print(f"Training run directory: {run_dir}")
        print(f"Resuming CB1 actor: {resume_model}")
        model = IndependentCriticPPO.IndependentCriticPPO.load(
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
            },
        )
        model._pause_simulation = pause_simulation
        if model.independent_critic is None:
            model.enable_independent_critic(
                rollout_directory,
                mode=mode,
                critic_state_path=critic_state_path,
            )
            print("Initialized independent critic from current CB1 value path.")
        else:
            model.set_critic_mode(mode)

        if critic_state_path.is_file():
            critic_timestep = model.load_critic_state(critic_state_path)
            print(
                "Restored independent critic snapshot from "
                f"{critic_timestep} actor steps."
            )

        total_timesteps = max(0, Train.TARGET_TOTAL_TIMESTEPS - model.num_timesteps)
        print(
            f"Resuming from {model.num_timesteps} steps in {mode} mode; "
            f"training {total_timesteps} additional steps toward "
            f"{Train.TARGET_TOTAL_TIMESTEPS}."
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
        model.save_critic_state(critic_state_path)
        print(f"Final model saved to: {final_model}.zip")
        return model
    except Exception:
        print(f"Independent-critic training hiccup in {run_dir}!")
        traceback.print_exc()
    finally:
        env.close()
