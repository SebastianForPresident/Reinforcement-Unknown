"""VB1 completion curriculum: new deepest progress and completion only."""


PROGRESS_REWARD_SCALE = 10.0
COMPLETION_BONUS = 10.0


def Reset(env, obs):
    """Reset the monotonic progress frontier for a new procedural cave."""
    env.best_progress = float(obs["LayerProgress"])
    env.completion_bonus_applied = False
    env.last_reward_terms = {}


def Reward(obs, act, env):
    current_progress = float(obs["LayerProgress"])
    previous_best_progress = env.best_progress
    best_progress = max(current_progress, previous_best_progress)
    progress_delta = best_progress - previous_best_progress
    progress_reward = PROGRESS_REWARD_SCALE * progress_delta

    completion_bonus = 0.0
    if best_progress >= 1.0 and not env.completion_bonus_applied:
        completion_bonus = COMPLETION_BONUS
        env.completion_bonus_applied = True

    reward = progress_reward + completion_bonus
    env.best_progress = best_progress
    env.last_reward_terms = {
        "best_progress": best_progress,
        "progress_delta": progress_delta,
        "progress": progress_reward,
        "completion": completion_bonus,
        "reward": reward,
    }
    return float(reward)
