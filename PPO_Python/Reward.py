"""V9 reward: safety first, then pursue deeper progress.

Objective:
    Preserve the agent's ability to continue while pursuing the best
    LayerProgress available.  New best progress earns ``10 * delta``;
    physiological deterioration is scored through first-class state deltas.
    LayerTimeRemaining and RadLineDisplacement remain observations rather than
    hand-shaped rewards.

Capability and systemic state:
    Stamina and AveragePain are scored by their first-class step-to-step
    deltas. BrainHealth, BloodOxygen, TotalBleedSpeed, InternalBleeding,
    RadiationSickness, SicknessAmount, and Temperature are also scored by
    their step-to-step deltas. TotalHappiness remains an observation only.
    Ragdolling retains the V5/V6 action-gated stall guard: it is penalized
    only after prolonged ragdoll time without another depth milestone.

Cardiovascular, respiration, and blood:
    BloodOxygen, TotalBleedSpeed, InternalBleeding, and Temperature are scored
    by first-class changes rather than occupancy.

Terminal events:
    Death applies a one-time ``-10`` penalty.  Completion applies a one-time
    ``+10`` bonus, with death taking precedence if both are reported.

The named scales below are deliberately small relative to progress and are
also retained in ``env.last_reward_terms`` for trace diagnostics.
"""

import numpy as np


PROGRESS_REWARD_SCALE = 10.0
STAMINA_DELTA_REWARD_SCALE = 0.001
AVERAGE_PAIN_DELTA_REWARD_SCALE = 0.001
BRAIN_HEALTH_DELTA_REWARD_SCALE = 0.001
BLOOD_OXYGEN_DELTA_REWARD_SCALE = 0.004
TOTAL_BLEED_SPEED_DELTA_REWARD_SCALE = 0.003
INTERNAL_BLEEDING_DELTA_REWARD_SCALE = 0.003
RADIATION_SICKNESS_DELTA_REWARD_SCALE = 0.002
SICKNESS_DELTA_REWARD_SCALE = 0.0015
TEMPERATURE_DELTA_REWARD_SCALE = 0.003
RAGDOLL_DEPTH_MILESTONE_METERS = 10.0
RAGDOLL_STALL_GRACE_SECONDS = 30.0
RAGDOLL_STALL_DOUBLING_SECONDS = 10.0
RAGDOLL_STALL_PENALTY_BASE = 0.001
RAGDOLL_STALL_PENALTY_CAP = 0.01
DEATH_PENALTY = 10.0
COMPLETION_BONUS = 10.0


def Reset(env, obs):
    """Reset reward state for a new episode."""
    env.best_progress = 0.0
    env.previous_stamina = float(obs["Stamina"])
    env.previous_average_pain = float(obs["AveragePain"])
    env.previous_brain_health = float(obs["BrainHealth"])
    env.previous_blood_oxygen = float(obs["BloodOxygen"])
    env.previous_total_bleed_speed = float(obs["TotalBleedSpeed"])
    env.previous_internal_bleeding = float(obs["InternalBleeding"])
    env.previous_radiation_sickness = float(obs["RadiationSickness"])
    env.previous_sickness_amount = float(obs["SicknessAmount"])
    env.previous_temperature = float(obs["Temperature"])
    env.ragdoll_depth_milestone = float(obs["BestLayerDepth"])
    env.ragdoll_seconds_since_depth_milestone = 0.0
    env.previous_time_ragdolled = float(obs["TimeRagdolled"])
    env.death_penalty_applied = False
    env.completion_bonus_applied = False
    env.last_reward_terms = {}


def Reward(obs, act, env):
    current_progress = float(obs["LayerProgress"])
    previous_best_progress = env.best_progress
    best_progress = max(current_progress, previous_best_progress)
    progress_delta = best_progress - previous_best_progress
    progress_reward = PROGRESS_REWARD_SCALE * progress_delta

    current_best_depth = float(obs["BestLayerDepth"])
    current_time_ragdolled = float(obs["TimeRagdolled"])
    ragdoll_seconds_delta = max(
        0.0, current_time_ragdolled - env.previous_time_ragdolled
    )
    if current_best_depth >= (
        env.ragdoll_depth_milestone + RAGDOLL_DEPTH_MILESTONE_METERS
    ):
        env.ragdoll_depth_milestone = current_best_depth
        env.ragdoll_seconds_since_depth_milestone = 0.0
    else:
        env.ragdoll_seconds_since_depth_milestone += ragdoll_seconds_delta

    overdue_ragdoll_seconds = max(
        0.0,
        env.ragdoll_seconds_since_depth_milestone
        - RAGDOLL_STALL_GRACE_SECONDS,
    )
    ragdoll_stall_penalty = 0.0
    if act[21] == 1:
        ragdoll_stall_penalty = min(
            RAGDOLL_STALL_PENALTY_CAP,
            RAGDOLL_STALL_PENALTY_BASE
            * (
                2.0
                ** (overdue_ragdoll_seconds / RAGDOLL_STALL_DOUBLING_SECONDS)
                - 1.0
            ),
        )

    stamina = float(obs["Stamina"])
    stamina_delta = stamina - env.previous_stamina
    stamina_delta_reward = STAMINA_DELTA_REWARD_SCALE * stamina_delta

    average_pain = float(obs["AveragePain"])
    average_pain_delta = average_pain - env.previous_average_pain
    average_pain_delta_reward = (
        -AVERAGE_PAIN_DELTA_REWARD_SCALE * average_pain_delta
    )

    brain_health = float(obs["BrainHealth"])
    brain_health_delta = brain_health - env.previous_brain_health
    brain_health_delta_reward = (
        BRAIN_HEALTH_DELTA_REWARD_SCALE * brain_health_delta
    )

    blood_oxygen = float(obs["BloodOxygen"])
    blood_oxygen_delta = blood_oxygen - env.previous_blood_oxygen
    blood_oxygen_delta_reward = (
        BLOOD_OXYGEN_DELTA_REWARD_SCALE * blood_oxygen_delta
    )

    total_bleed_speed = float(obs["TotalBleedSpeed"])
    total_bleed_speed_delta = total_bleed_speed - env.previous_total_bleed_speed
    total_bleed_speed_delta_reward = (
        -TOTAL_BLEED_SPEED_DELTA_REWARD_SCALE * total_bleed_speed_delta
    )

    internal_bleeding = float(obs["InternalBleeding"])
    internal_bleeding_delta = internal_bleeding - env.previous_internal_bleeding
    internal_bleeding_delta_reward = (
        -INTERNAL_BLEEDING_DELTA_REWARD_SCALE * internal_bleeding_delta
    )

    radiation_sickness = float(obs["RadiationSickness"])
    radiation_sickness_delta = (
        radiation_sickness - env.previous_radiation_sickness
    )
    radiation_sickness_delta_reward = (
        -RADIATION_SICKNESS_DELTA_REWARD_SCALE * radiation_sickness_delta
    )

    sickness_amount = float(obs["SicknessAmount"])
    sickness_amount_delta = sickness_amount - env.previous_sickness_amount
    sickness_amount_delta_reward = (
        -SICKNESS_DELTA_REWARD_SCALE * sickness_amount_delta
    )

    temperature = float(obs["Temperature"])
    previous_temperature_deviation = abs(env.previous_temperature - 37.0)
    temperature_deviation = abs(temperature - 37.0)
    temperature_deviation_delta = (
        temperature_deviation - previous_temperature_deviation
    )
    temperature_delta_reward = (
        -TEMPERATURE_DELTA_REWARD_SCALE * temperature_deviation_delta
    )

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
        + stamina_delta_reward
        + average_pain_delta_reward
        + brain_health_delta_reward
        + blood_oxygen_delta_reward
        + total_bleed_speed_delta_reward
        + internal_bleeding_delta_reward
        + radiation_sickness_delta_reward
        + sickness_amount_delta_reward
        + temperature_delta_reward
        - ragdoll_stall_penalty
        + death_penalty
        + completion_bonus
    )

    env.best_progress = best_progress
    env.previous_stamina = stamina
    env.previous_average_pain = average_pain
    env.previous_brain_health = brain_health
    env.previous_blood_oxygen = blood_oxygen
    env.previous_total_bleed_speed = total_bleed_speed
    env.previous_internal_bleeding = internal_bleeding
    env.previous_radiation_sickness = radiation_sickness
    env.previous_sickness_amount = sickness_amount
    env.previous_temperature = temperature
    env.previous_time_ragdolled = current_time_ragdolled
    env.last_reward_terms = {
        "best_progress": best_progress,
        "progress_delta": progress_delta,
        "progress": progress_reward,
        "stamina": stamina_delta_reward,
        "stamina_delta": stamina_delta,
        "average_pain": average_pain_delta_reward,
        "average_pain_delta": average_pain_delta,
        "brain_health": brain_health_delta_reward,
        "brain_health_delta": brain_health_delta,
        "blood_oxygen": blood_oxygen_delta_reward,
        "blood_oxygen_delta": blood_oxygen_delta,
        "total_bleed_speed": total_bleed_speed_delta_reward,
        "total_bleed_speed_delta": total_bleed_speed_delta,
        "internal_bleeding": internal_bleeding_delta_reward,
        "internal_bleeding_delta": internal_bleeding_delta,
        "radiation_sickness": radiation_sickness_delta_reward,
        "radiation_sickness_delta": radiation_sickness_delta,
        "sickness": sickness_amount_delta_reward,
        "sickness_amount_delta": sickness_amount_delta,
        "temperature": temperature_delta_reward,
        "temperature_deviation_delta": temperature_deviation_delta,
        "ragdoll_stall_penalty": ragdoll_stall_penalty,
        "ragdoll_seconds_since_depth_milestone": (
            env.ragdoll_seconds_since_depth_milestone
        ),
        "ragdoll_depth_milestone": env.ragdoll_depth_milestone,
        "death": death_penalty,
        "completion": completion_bonus,
        "reward": reward,
    }
    return float(reward)
