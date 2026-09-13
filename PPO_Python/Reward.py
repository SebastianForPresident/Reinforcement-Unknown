"""VB1 completion curriculum: new deepest progress and completion only."""

import ResourceBudget


PROGRESS_REWARD_SCALE = 10.0
COMPLETION_BONUS = 10.0
INCOMPLETE_EPISODE_PENALTY = -10.0
INEFFECTIVE_SWING_PENALTY_SCALE = 0.0001
INEFFECTIVE_SWING_PENALTY_ENABLED = False
MISSED_ATTACK_PENALTY_SCALE = 0.005
CLAW_CONTACT_EPSILON = 1e-6
ATTACK_EXECUTION_PENALTY_SCALE = 0.001


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

    attack_swings_executed = int(obs["AttackSwingsExecuted"])
    attack_contact_claw_health_loss = float(
        obs["AttackContactClawHealthLoss"]
    )
    ineffective_swings = (
        attack_swings_executed
        if (
            attack_swings_executed > 0
            and attack_contact_claw_health_loss <= CLAW_CONTACT_EPSILON
        )
        else 0
    )
    ineffective_swing_penalty_shadow = (
        -INEFFECTIVE_SWING_PENALTY_SCALE * ineffective_swings
    )
    missed_attack_penalty_mode = bool(
        getattr(env, "missed_attack_penalty_mode", False)
    )
    ineffective_swing_penalty = (
        -MISSED_ATTACK_PENALTY_SCALE * ineffective_swings
        if missed_attack_penalty_mode
        else (
            ineffective_swing_penalty_shadow
            if INEFFECTIVE_SWING_PENALTY_ENABLED
            else 0.0
        )
    )
    attack_cost = float(getattr(env, "attack_cost", 0.0))
    attack_execution_penalty = (
        -ATTACK_EXECUTION_PENALTY_SCALE
        * attack_cost
        * attack_swings_executed
        * int(not bool(getattr(env, "terminal_attack_mode", False)))
    )
    attack_budget_penalty = (
        ResourceBudget.immediate_cost(
            attack_swings_executed, env.attack_budget
        )
        if bool(getattr(env, "attack_budget_mode", False))
        else 0.0
    )

    reward = (
        progress_reward
        + completion_bonus
        + ineffective_swing_penalty
        + attack_execution_penalty
        + attack_budget_penalty
    )
    env.best_progress = best_progress
    env.last_reward_terms = {
        "best_progress": best_progress,
        "progress_delta": progress_delta,
        "progress": progress_reward,
        "completion": completion_bonus,
        "attack_swings_executed": attack_swings_executed,
        "attack_contact_claw_health_loss": attack_contact_claw_health_loss,
        "ineffective_swings": ineffective_swings,
        "ineffective_swing_penalty_shadow": ineffective_swing_penalty_shadow,
        "ineffective_swing_penalty": ineffective_swing_penalty,
        "missed_attack_penalty_mode": missed_attack_penalty_mode,
        "attack_cost": attack_cost,
        "attack_execution_penalty": attack_execution_penalty,
        "attack_budget_penalty": attack_budget_penalty,
        "incomplete_episode": 0.0,
        "reward": reward,
    }
    return float(reward)


def ApplyIncompleteEpisodePenalty(env):
    """Penalize a watchdog-ended descent attempt as an actual task failure."""
    reward = float(env.last_reward_terms.get("reward", 0.0))
    reward += INCOMPLETE_EPISODE_PENALTY
    env.last_reward_terms["incomplete_episode"] = INCOMPLETE_EPISODE_PENALTY
    env.last_reward_terms["reward"] = reward
    return reward
