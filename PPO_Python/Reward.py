"""V15 reward: progress, health deltas, and exhaustion stall protection."""


PROGRESS_REWARD_SCALE = 10.0
DEPTH_MILESTONE_METERS = 10.0
AVERAGE_PAIN_DELTA_REWARD_SCALE = 0.001
BRAIN_HEALTH_DELTA_REWARD_SCALE = 0.001
BLOOD_OXYGEN_DELTA_REWARD_SCALE = 0.004
TOTAL_BLEED_SPEED_DELTA_REWARD_SCALE = 0.003
INTERNAL_BLEEDING_DELTA_REWARD_SCALE = 0.003
RADIATION_SICKNESS_DELTA_REWARD_SCALE = 0.002
SICKNESS_DELTA_REWARD_SCALE = 0.0015
TEMPERATURE_DELTA_REWARD_SCALE = 0.003
EXHAUSTION_STAMINA_THRESHOLD = 10.0
EXHAUSTION_STALL_GRACE_SECONDS = 30.0
EXHAUSTION_STALL_DOUBLING_SECONDS = 10.0
EXHAUSTION_STALL_PENALTY_BASE = 0.001
EXHAUSTION_STALL_PENALTY_CAP = 0.01
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
    env.depth_milestone = float(obs["BestLayerDepth"])
    env.exhaustion_seconds_since_depth_milestone = 0.0
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
    depth_milestone_reached = current_best_depth >= (
        env.depth_milestone + DEPTH_MILESTONE_METERS
    )
    if depth_milestone_reached:
        env.depth_milestone = current_best_depth
        env.exhaustion_seconds_since_depth_milestone = 0.0

    stamina = float(obs["Stamina"])
    if not depth_milestone_reached and stamina < EXHAUSTION_STAMINA_THRESHOLD:
        env.exhaustion_seconds_since_depth_milestone += max(
            0.0, float(obs["SimulationDeltaTime"])
        )

    overdue_exhaustion_seconds = max(
        0.0,
        env.exhaustion_seconds_since_depth_milestone
        - EXHAUSTION_STALL_GRACE_SECONDS,
    )
    exhaustion_move_stall_penalty = 0.0
    exhaustion_jump_stall_penalty = 0.0
    exhaustion_attack_stall_penalty = 0.0
    exhaustion_stall_penalty = 0.0
    if (
        stamina < EXHAUSTION_STAMINA_THRESHOLD
        and overdue_exhaustion_seconds > 0.0
    ):
        exhaustion_stall_penalty = min(
            EXHAUSTION_STALL_PENALTY_CAP,
            EXHAUSTION_STALL_PENALTY_BASE
            * (
                2.0
                ** (
                    overdue_exhaustion_seconds
                    / EXHAUSTION_STALL_DOUBLING_SECONDS
                )
                - 1.0
            ),
        )
        exhaustion_move_stall_penalty = exhaustion_stall_penalty * (
            float(act[0]) != 1
        )
        exhaustion_jump_stall_penalty = exhaustion_stall_penalty * (
            act[1] == 1
        )
        exhaustion_attack_stall_penalty = exhaustion_stall_penalty * (
            act[6] == 1
        )
    exhaustion_stall_action_selected = (
        float(act[0]) != 1 or act[1] == 1 or act[6] == 1
    )
    if exhaustion_stall_action_selected:
        exhaustion_stall_penalty = max(
            exhaustion_move_stall_penalty,
            exhaustion_jump_stall_penalty,
            exhaustion_attack_stall_penalty,
        )
    else:
        exhaustion_stall_penalty = 0.0

    stamina_delta = stamina - env.previous_stamina

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
        + average_pain_delta_reward
        + brain_health_delta_reward
        + blood_oxygen_delta_reward
        + total_bleed_speed_delta_reward
        + internal_bleeding_delta_reward
        + radiation_sickness_delta_reward
        + sickness_amount_delta_reward
        + temperature_delta_reward
        - exhaustion_stall_penalty
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
    env.last_reward_terms = {
        "best_progress": best_progress,
        "progress_delta": progress_delta,
        "progress": progress_reward,
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
        "depth_milestone": env.depth_milestone,
        "exhaustion_stall_penalty": exhaustion_stall_penalty,
        "exhaustion_move_stall": exhaustion_move_stall_penalty,
        "exhaustion_jump_stall": exhaustion_jump_stall_penalty,
        "exhaustion_attack_stall": exhaustion_attack_stall_penalty,
        "exhaustion_stall_action_selected": exhaustion_stall_action_selected,
        "exhaustion_seconds_since_depth_milestone": (
            env.exhaustion_seconds_since_depth_milestone
        ),
        "death": death_penalty,
        "completion": completion_bonus,
        "reward": reward,
    }
    return float(reward)
