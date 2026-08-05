"""V13 reward: safety first, then pursue deeper progress.

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
    Exhaustion uses the same paused, action-gated stall logic while the
    stamina remains critical, with movement, jumping, and attacking sharing
    one capped penalty rather than stacking.
    Successful unarmed contacts are rewarded through observed ClawHealth
    decreases. Missed attacks receive no hit reward; the existing stamina
    delta cost remains in effect. Claw-hit reward has a depth-progress budget:
    the first 15 full contacts remain fully rewarded, the next 15 fade to
    zero, and another 10-meter depth milestone resets the budget.

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
# A full unarmed contact removes up to 0.3 ClawHealth.  At 0.02 this yields
# about +0.006, comfortably offsetting the roughly 0.001 stamina cost of an
# attack while keeping missed attacks unrewarded.
CLAW_HIT_REWARD_SCALE = 0.02
CLAW_HIT_MAX_HEALTH_LOSS = 0.3
CLAW_HIT_FULL_REWARD_CONTACTS = 15.0
CLAW_HIT_FALLOFF_CONTACTS = 15.0
RAGDOLL_DEPTH_MILESTONE_METERS = 10.0
RAGDOLL_STALL_GRACE_SECONDS = 30.0
RAGDOLL_STALL_DOUBLING_SECONDS = 10.0
RAGDOLL_STALL_PENALTY_BASE = 0.001
RAGDOLL_STALL_PENALTY_CAP = 0.01
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
    env.previous_claw_health = float(obs["ClawHealth"])
    env.ragdoll_depth_milestone = float(obs["BestLayerDepth"])
    env.claw_hit_health_loss_since_depth_milestone = 0.0
    env.ragdoll_seconds_since_depth_milestone = 0.0
    env.previous_time_ragdolled = float(obs["TimeRagdolled"])
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
    current_time_ragdolled = float(obs["TimeRagdolled"])
    ragdoll_seconds_delta = max(
        0.0, current_time_ragdolled - env.previous_time_ragdolled
    )
    depth_milestone_reached = current_best_depth >= (
        env.ragdoll_depth_milestone + RAGDOLL_DEPTH_MILESTONE_METERS
    )
    if depth_milestone_reached:
        env.ragdoll_depth_milestone = current_best_depth
        env.ragdoll_seconds_since_depth_milestone = 0.0
        env.exhaustion_seconds_since_depth_milestone = 0.0
        env.claw_hit_health_loss_since_depth_milestone = 0.0
    else:
        env.ragdoll_seconds_since_depth_milestone += ragdoll_seconds_delta

    overdue_ragdoll_seconds = max(
        0.0,
        env.ragdoll_seconds_since_depth_milestone
        - RAGDOLL_STALL_GRACE_SECONDS,
    )
    ragdoll_stall_penalty = 0.0
    # C12's legacy ragdoll action remains supported for compatibility, but it
    # is intentionally absent from C13's seven-action policy.
    if len(act) > 21 and act[21] == 1:
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
    stamina_delta_reward = STAMINA_DELTA_REWARD_SCALE * stamina_delta

    claw_health = float(obs["ClawHealth"])
    claw_health_loss = min(
        CLAW_HIT_MAX_HEALTH_LOSS,
        max(0.0, env.previous_claw_health - claw_health),
    )
    full_reward_loss_budget = (
        CLAW_HIT_FULL_REWARD_CONTACTS * CLAW_HIT_MAX_HEALTH_LOSS
    )
    falloff_loss_budget = (
        CLAW_HIT_FALLOFF_CONTACTS * CLAW_HIT_MAX_HEALTH_LOSS
    )
    hit_loss_start = env.claw_hit_health_loss_since_depth_milestone
    hit_loss_end = hit_loss_start + claw_health_loss

    full_rewarded_loss = max(
        0.0,
        min(hit_loss_end, full_reward_loss_budget)
        - min(hit_loss_start, full_reward_loss_budget),
    )
    falloff_start = max(hit_loss_start, full_reward_loss_budget)
    falloff_end = min(
        hit_loss_end,
        full_reward_loss_budget + falloff_loss_budget,
    )
    falloff_loss = max(0.0, falloff_end - falloff_start)
    falloff_rewarded_loss = falloff_loss * max(
        0.0,
        1.0
        - (
            (falloff_start - full_reward_loss_budget)
            + (falloff_end - full_reward_loss_budget)
        )
        / (2.0 * falloff_loss_budget),
    )
    claw_hit_rewardable_loss = full_rewarded_loss + falloff_rewarded_loss
    claw_hit_reward = CLAW_HIT_REWARD_SCALE * claw_hit_rewardable_loss
    claw_hit_reward_multiplier = (
        claw_hit_rewardable_loss / claw_health_loss
        if claw_health_loss > 0.0
        else max(
            0.0,
            min(
                1.0,
                (full_reward_loss_budget + falloff_loss_budget - hit_loss_start)
                / falloff_loss_budget,
            ),
        )
    )
    env.claw_hit_health_loss_since_depth_milestone = hit_loss_end

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
        + claw_hit_reward
        - ragdoll_stall_penalty
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
    env.previous_claw_health = claw_health
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
        "claw_hit_reward": claw_hit_reward,
        "claw_health_loss": claw_health_loss,
        "claw_hit_rewardable_loss": claw_hit_rewardable_loss,
        "claw_hit_reward_multiplier": claw_hit_reward_multiplier,
        "claw_hit_health_loss_since_depth_milestone": (
            env.claw_hit_health_loss_since_depth_milestone
        ),
        "ragdoll_stall_penalty": ragdoll_stall_penalty,
        "ragdoll_seconds_since_depth_milestone": (
            env.ragdoll_seconds_since_depth_milestone
        ),
        "ragdoll_depth_milestone": env.ragdoll_depth_milestone,
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
