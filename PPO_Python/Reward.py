import numpy as np


PROGRESS_REWARD_SCALE = 10.0
HEALTH_DELTA_REWARD_SCALE = 1.5
STEP_COST = 0.001
DANGER_COST_SCALE = 0.002
RADLINE_COST_SCALE = 0.001
RADLINE_COST_CAP = 0.05
DEATH_PENALTY = 12.0
LAYER_COMPLETE_BONUS = 12.0


def _clip01(value):
    return float(np.clip(float(value), 0.0, 1.0))


def _high_is_safe(value, healthy, danger):
    if healthy <= danger:
        return 1.0 if float(value) >= healthy else 0.0
    return _clip01((float(value) - danger) / (healthy - danger))


def _low_is_safe(value, danger):
    return 1.0 - _clip01(float(value) / danger) if danger > 0.0 else 1.0


def _target_is_safe(value, target, tolerance):
    return 1.0 - _clip01(abs(float(value) - target) / tolerance) if tolerance > 0.0 else 1.0


def _reserve_is_safe(value, low, target, high):
    value = float(value)
    if value <= low or value >= high:
        return 0.0
    if value <= target:
        return _clip01((value - low) / (target - low))
    return _clip01((high - value) / (high - target))


def _health_potential(obs):
    """Estimate how safely the body can continue the descent."""
    qualities = (
        0.20 * _high_is_safe(obs["BrainHealth"], 100.0, 0.0),
        0.10 * _high_is_safe(obs["Consciousness"], 100.0, 30.0),
        0.15 * _high_is_safe(obs["BloodOxygen"], 100.0, 65.0),
        0.15 * _high_is_safe(obs["BloodVolume"], 100.0, 40.0),
        0.10 * _low_is_safe(obs["TotalBleedSpeed"], 0.134),
        0.05 * _low_is_safe(obs["InternalBleeding"], 40.0),
        0.05 * _target_is_safe(obs["BloodPressure"], 120.0, 50.0),
        0.05 * _low_is_safe(obs["FibrillationProgress"], 100.0),
        0.05 * _target_is_safe(obs["Temperature"], 37.0, 8.0),
        0.03 * _low_is_safe(obs["SepticShock"], 75.0),
        0.02 * _low_is_safe(obs["RadiationSickness"], 60.0),
        0.025 * _reserve_is_safe(obs["Hunger"], 10.0, 100.0, 175.0),
        0.025 * _reserve_is_safe(obs["Thirst"], 10.0, 100.0, 175.0),
    )
    return _clip01(sum(qualities))


# Reward V2: Paying closer attention to health stats to avoid critical states
def Reward(obs, env):
    current_progress = float(obs["LayerProgress"])
    current_health = _health_potential(obs)

    if env.previous_progress is None:
        env.previous_progress = current_progress
        env.previous_health = current_health

    progress_delta = current_progress - env.previous_progress
    health_delta = current_health - env.previous_health

    reward = (
        PROGRESS_REWARD_SCALE * progress_delta
        + HEALTH_DELTA_REWARD_SCALE * health_delta
        - STEP_COST
        - DANGER_COST_SCALE * (1.0 - current_health)
    )

    if obs["LayerTimeRemaining"] <= 0 and obs["RadLineDisplacement"] < 0:
        reward -= min(
            -float(obs["RadLineDisplacement"]) * RADLINE_COST_SCALE,
            RADLINE_COST_CAP,
        )

    if bool(obs["PlayerDead"]):
        reward -= DEATH_PENALTY
    elif current_progress >= 1.0:
        reward += LAYER_COMPLETE_BONUS

    env.previous_progress = current_progress
    env.previous_health = current_health
    return float(reward)
