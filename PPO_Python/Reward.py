"""V7 reward: safety first, then pursue deeper progress.

Objective:
    Preserve the agent's ability to continue while pursuing the best
    LayerProgress available.  New best progress earns ``10 * delta``;
    physiological deterioration is continuously penalized.  LayerTimeRemaining
    and RadLineDisplacement remain observations rather than hand-shaped
    rewards.

Capability and systemic state:
    Stamina, Shock, AveragePain, PainShock, BrainHealth, RadiationSickness,
    SicknessAmount, and TotalHappiness use continuous quadratic penalties.

Cardiovascular, respiration, and blood:
    BloodPressure, HeartRate, FibrillationProgress, PulmonaryEmbolism,
    StrokeAmount, BloodOxygen, RespiratoryRate, BloodVolume, TotalBleedSpeed,
    InternalBleeding, and Hemothorax are scored against game-informed
    thresholds or critical values.

Terminal events:
    Death applies a one-time ``-10`` penalty.  Completion applies a one-time
    ``+10`` bonus, with death taking precedence if both are reported.

The named scales below are deliberately small relative to progress and are
also retained in ``env.last_reward_terms`` for trace diagnostics.
"""

import numpy as np


PROGRESS_REWARD_SCALE = 10.0
EXHAUSTION_REWARD_SCALE = 0.001
SHOCK_REWARD_SCALE = 0.000001
PAIN_REWARD_SCALE = 0.001
PAIN_SHOCK_REWARD_SCALE = 0.001
BRAIN_HEALTH_REWARD_SCALE = 0.001
BLOOD_PRESSURE_REWARD_SCALE = 0.0025
HEART_RATE_REWARD_SCALE = 0.0015
FIBRILLATION_REWARD_SCALE = 0.003
PULMONARY_EMBOLISM_REWARD_SCALE = 0.003
STROKE_REWARD_SCALE = 0.004
BLOOD_OXYGEN_REWARD_SCALE = 0.004
RESPIRATORY_RATE_REWARD_SCALE = 0.0025
BLOOD_VOLUME_REWARD_SCALE = 0.003
TOTAL_BLEED_SPEED_REWARD_SCALE = 0.003
INTERNAL_BLEEDING_REWARD_SCALE = 0.003
HEMOTHORAX_REWARD_SCALE = 0.0015
RADIATION_SICKNESS_REWARD_SCALE = 0.002
SICKNESS_REWARD_SCALE = 0.0015
TEMPERATURE_REWARD_SCALE = 0.003
TOTAL_HAPPINESS_REWARD_SCALE = 0.003
DEATH_PENALTY = 10.0
COMPLETION_BONUS = 10.0


def Reset(env, obs):
    """Reset reward state for a new episode."""
    env.best_progress = 0.0
    env.death_penalty_applied = False
    env.completion_bonus_applied = False
    env.last_reward_terms = {}


def Reward(obs, act, env):
    current_progress = float(obs["LayerProgress"])
    previous_best_progress = env.best_progress
    best_progress = max(current_progress, previous_best_progress)
    progress_delta = best_progress - previous_best_progress
    progress_reward = PROGRESS_REWARD_SCALE * progress_delta

    stamina = float(obs["Stamina"])
    exhaustion = np.clip((100.0 - stamina) / 100.0, 0.0, 1.0)
    exhaustion_penalty = -EXHAUSTION_REWARD_SCALE * exhaustion ** 4

    shock = float(obs["Shock"])
    shock_excess = max(0.0, shock - 10.0)
    shock_penalty = -SHOCK_REWARD_SCALE * 0.05 * shock_excess ** 2

    average_pain = float(obs["AveragePain"])
    pain_excess = max(0.0, (average_pain - 10.0) / 90.0)
    pain_penalty = -PAIN_REWARD_SCALE * pain_excess ** 2

    pain_shock = float(obs["PainShock"])
    pain_shock_excess = max(0.0, (pain_shock - 0.3) / 0.7)
    pain_shock_penalty = -PAIN_SHOCK_REWARD_SCALE * pain_shock_excess ** 2

    brain_health = np.clip(float(obs["BrainHealth"]), 0.0, 100.0)
    brain_damage = (100.0 - brain_health) / 100.0
    brain_health_penalty = -BRAIN_HEALTH_REWARD_SCALE * brain_damage ** 2

    blood_pressure = float(obs["BloodPressure"])
    low_pressure_risk = np.clip((110.0 - blood_pressure) / 50.0, 0.0, 1.0)
    high_pressure_risk = np.clip((blood_pressure - 130.0) / 50.0, 0.0, 1.0)
    blood_pressure_risk = max(low_pressure_risk, high_pressure_risk)
    blood_pressure_penalty = -BLOOD_PRESSURE_REWARD_SCALE * blood_pressure_risk ** 2

    heart_rate = float(obs["HeartRate"])
    low_heart_rate_risk = np.clip((60.0 - heart_rate) / 40.0, 0.0, 1.0)
    high_heart_rate_risk = np.clip((heart_rate - 110.0) / 90.0, 0.0, 1.0)
    heart_rate_risk = max(low_heart_rate_risk, high_heart_rate_risk)
    heart_rate_penalty = -HEART_RATE_REWARD_SCALE * heart_rate_risk ** 2

    fibrillation_progress = np.clip(float(obs["FibrillationProgress"]), 0.0, 100.0)
    fibrillation_excess = max(0.0, (fibrillation_progress - 15.0) / 85.0)
    fibrillation_penalty = -FIBRILLATION_REWARD_SCALE * fibrillation_excess ** 2

    pulmonary_embolism_penalty = (
        -PULMONARY_EMBOLISM_REWARD_SCALE
        if bool(obs["HasPulmonaryEmbolism"])
        else 0.0
    )

    stroke_amount = np.clip(float(obs["StrokeAmount"]), 0.0, 100.0)
    stroke_penalty = -STROKE_REWARD_SCALE * (stroke_amount / 100.0) ** 2

    blood_oxygen = np.clip(float(obs["BloodOxygen"]), 0.0, 100.0)
    oxygen_deficit = max(0.0, (90.0 - blood_oxygen) / 90.0)
    blood_oxygen_penalty = -BLOOD_OXYGEN_REWARD_SCALE * oxygen_deficit ** 2

    respiratory_rate = np.clip(float(obs["RespiratoryRate"]), 0.0, 100.0)
    respiratory_deficit = np.clip((90.0 - respiratory_rate) / 80.0, 0.0, 1.0)
    respiratory_penalty = -RESPIRATORY_RATE_REWARD_SCALE * respiratory_deficit ** 2

    blood_volume = float(obs["BloodVolume"])
    blood_loss = np.clip((100.0 - blood_volume) / 200.0, 0.0, 1.0)
    blood_volume_penalty = -BLOOD_VOLUME_REWARD_SCALE * blood_loss ** 2

    total_bleed_speed = max(0.0, float(obs["TotalBleedSpeed"]))
    bleed_risk = np.clip(total_bleed_speed / 0.30, 0.0, 1.0)
    bleed_penalty = -TOTAL_BLEED_SPEED_REWARD_SCALE * bleed_risk ** 2

    internal_bleeding = max(0.0, float(obs["InternalBleeding"]))
    internal_bleeding_risk = np.clip(internal_bleeding / 50.0, 0.0, 1.0)
    internal_bleeding_penalty = (
        -INTERNAL_BLEEDING_REWARD_SCALE * internal_bleeding_risk ** 2
    )

    hemothorax = max(0.0, float(obs["Hemothorax"]))
    hemothorax_risk = np.clip(hemothorax / 70.0, 0.0, 1.0)
    hemothorax_penalty = -HEMOTHORAX_REWARD_SCALE * hemothorax_risk ** 2

    radiation_sickness = max(0.0, float(obs["RadiationSickness"]))
    radiation_risk = np.clip((radiation_sickness - 10.0) / 70.0, 0.0, 1.0)
    radiation_penalty = -RADIATION_SICKNESS_REWARD_SCALE * radiation_risk ** 2

    sickness_amount = max(0.0, float(obs["SicknessAmount"]))
    sickness_risk = np.clip((sickness_amount - 10.0) / 85.0, 0.0, 1.0)
    sickness_penalty = -SICKNESS_REWARD_SCALE * sickness_risk ** 2

    temperature = float(obs["Temperature"])
    low_temperature_risk = np.clip((35.5 - temperature) / 7.5, 0.0, 1.0)
    high_temperature_risk = np.clip((temperature - 38.0) / 3.5, 0.0, 1.0)
    temperature_risk = max(low_temperature_risk, high_temperature_risk)
    temperature_penalty = -TEMPERATURE_REWARD_SCALE * temperature_risk ** 2

    total_happiness = float(obs["TotalHappiness"])
    mental_risk = np.clip((-total_happiness - 30.0) / 70.0, 0.0, 1.0)
    total_happiness_penalty = -TOTAL_HAPPINESS_REWARD_SCALE * mental_risk ** 2

    death_penalty = 0.0
    completion_bonus = 0.0
    if bool(obs["PlayerDead"]):
        if not env.death_penalty_applied:
            death_penalty = -DEATH_PENALTY
            env.death_penalty_applied = True
    elif best_progress >= 1.0 and not env.completion_bonus_applied:
        completion_bonus = COMPLETION_BONUS
        env.completion_bonus_applied = True

    reward = (
        progress_reward
        + exhaustion_penalty
        + shock_penalty
        + pain_penalty
        + pain_shock_penalty
        + brain_health_penalty
        + blood_pressure_penalty
        + heart_rate_penalty
        + fibrillation_penalty
        + pulmonary_embolism_penalty
        + stroke_penalty
        + blood_oxygen_penalty
        + respiratory_penalty
        + blood_volume_penalty
        + bleed_penalty
        + internal_bleeding_penalty
        + hemothorax_penalty
        + radiation_penalty
        + sickness_penalty
        + temperature_penalty
        + total_happiness_penalty
        + death_penalty
        + completion_bonus
    )

    env.best_progress = best_progress
    env.last_reward_terms = {
        "best_progress": best_progress,
        "progress_delta": progress_delta,
        "progress": progress_reward,
        "exhaustion": exhaustion_penalty,
        "shock": shock_penalty,
        "pain": pain_penalty,
        "pain_shock": pain_shock_penalty,
        "brain_health": brain_health_penalty,
        "blood_pressure": blood_pressure_penalty,
        "heart_rate": heart_rate_penalty,
        "fibrillation": fibrillation_penalty,
        "pulmonary_embolism": pulmonary_embolism_penalty,
        "stroke": stroke_penalty,
        "blood_oxygen": blood_oxygen_penalty,
        "respiratory_rate": respiratory_penalty,
        "blood_volume": blood_volume_penalty,
        "total_bleed_speed": bleed_penalty,
        "internal_bleeding": internal_bleeding_penalty,
        "hemothorax": hemothorax_penalty,
        "radiation_sickness": radiation_penalty,
        "sickness": sickness_penalty,
        "temperature": temperature_penalty,
        "total_happiness": total_happiness_penalty,
        "death": death_penalty,
        "completion": completion_bonus,
        "reward": reward,
    }
    return float(reward)
