from stable_baselines3 import PPO
import traceback
from datetime import datetime
from pathlib import Path


LOG_EVERY = 100

def Infer(env, checkpoint_path, world_seed=None):
    log_file = Path("inference_debug.log")
    try:
        model = PPO.load(checkpoint_path, env=env)

        reset_options = (
            {"world_seed": world_seed} if world_seed is not None else None
        )
        obs, info = env.reset(options=reset_options)
        previous_action = None
        step = 0

        with log_file.open("a", encoding="utf-8") as log:
            log.write(
                f"\n[{datetime.now().isoformat(timespec='seconds')}] "
                f"checkpoint={checkpoint_path} "
                f"world_seed={info.get('world_seed')}\n"
            )

            while True:
                action, _ = model.predict(obs, deterministic=False)
                action_list = action.tolist()
                action_changed = previous_action is None or action_list != previous_action

                obs, reward, terminated, truncated, info = env.step(action)
                raw_obs = env.latest_obs
                step += 1

                if (step % LOG_EVERY == 0) or (action_changed and step <= 10):
                    log.write(
                        f"step={step} action_changed={action_changed} "
                        f"action={action_list} reward={reward:.5f} "
                        f"progress={info['progress']:.5f} "
                        f"completion={info['completion']:.1f} "
                        f"ticks={info.get('macrostep_physics_ticks', 0)} "
                        f"sim_dt={info.get('simulation_delta_time', 0.0):.3f} "
                        f"layer={float(raw_obs['LayerProgress']):.5f} "
                        f"dead={bool(raw_obs['PlayerDead'])} "
                        f"stamina={float(raw_obs['Stamina']):.2f} "
                        f"energy={float(raw_obs['Energy']):.2f} "
                        f"pain={float(raw_obs['AveragePain']):.2f} "
                        f"happiness={float(raw_obs['TotalHappiness']):.2f}\n"
                    )
                    log.flush()

                previous_action = action_list

                if terminated or truncated:
                    log.write(f"episode_end step={step} terminated={terminated} truncated={truncated}\n")
                    log.flush()
                    obs, info = env.reset(options=reset_options)
                    previous_action = None

    except KeyboardInterrupt:
        print("Inference stopped.")
    except Exception:
        traceback.print_exc()
    finally:
        env.close()
