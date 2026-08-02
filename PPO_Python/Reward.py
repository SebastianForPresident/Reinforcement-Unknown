"""Reward V6: descend while preserving the capacity to keep descending.

This reward deliberately operates on continuous observation fields rather than
the UI moodle levels.  Moodle thresholds inform the caution/critical ranges
below, but scoring the underlying state avoids threshold flicker and exposes
the immediate cost of harmful actions.

Inventory and crafting are intentionally out of scope.  This version rewards
layer progress and penalizes the physiological costs of locomotion, combat,
ragdoll impacts, and environmental exposure.  It also discourages the
non-progressing ragdoll exploit: cumulative ragdoll time becomes costly only
after thirty seconds without reaching another 10 m depth milestone, and now
only if the policy explicitly chooses to ragdoll in that same step.

All identical to V5 except the explicit radline penalty's removed
"""

import numpy as np


# A whole layer contributes about +12 through progress, with a completion
# bonus large enough to make finishing preferable to simply staying safe.
PROGRESS_REWARD_SCALE = 12.0
SAFETY_DELTA_REWARD_SCALE = 1.25
RISK_OCCUPANCY_COST = 0.006
STEP_COST = 0.0005
DEATH_PENALTY = 20.0
LAYER_COMPLETE_BONUS = 20.0

# Ragdolling is a useful traversal state in tight structures.  It becomes an
# exploit only when the body spends a long cumulative time ragdolled without
# reaching meaningfully deeper terrain.  These are game-time seconds/meters.
RAGDOLL_DEPTH_MILESTONE_METERS = 10.0
RAGDOLL_STALL_GRACE_SECONDS = 30.0
RAGDOLL_STALL_DOUBLING_SECONDS = 10.0
RAGDOLL_STALL_PENALTY_BASE = 0.001
RAGDOLL_STALL_PENALTY_CAP = 0.01


def _clip01(value):
    return float(np.clip(float(value), 0.0, 1.0))


def _rising_risk(value, caution, critical):
    """Risk for a value that becomes dangerous as it rises."""
    return _clip01((float(value) - caution) / (critical - caution))


def _falling_risk(value, caution, critical):
    """Risk for a value that becomes dangerous as it falls."""
    return _clip01((caution - float(value)) / (caution - critical))


def _outside_risk(value, low_caution, high_caution, low_critical, high_critical):
    """Risk outside a healthy range, with separate low/high critical bounds."""
    return max(
        _falling_risk(value, low_caution, low_critical),
        _rising_risk(value, high_caution, high_critical),
    )


def _weighted_mean(*weighted_values):
    total_weight = sum(weight for weight, _ in weighted_values)
    if total_weight <= 0.0:
        return 0.0
    return _clip01(sum(weight * value for weight, value in weighted_values) / total_weight)


def _limb_risk(obs):
    """Capture impact damage before it has time to become a vital-sign failure."""
    limbs = obs["Limbs"]

    skin_damage = 1.0 - float(np.mean(np.clip(limbs["SkinHealth"], 0.0, 100.0))) / 100.0
    muscle_damage = 1.0 - float(np.mean(np.clip(limbs["MuscleHealth"], 0.0, 100.0))) / 100.0
    structural_damage = float(
        np.mean(limbs["Broken"] | limbs["Dislocated"] | limbs["Dismembered"])
    )

    vital_limbs = limbs[limbs["IsVital"]]
    if len(vital_limbs):
        vital_integrity = min(
            float(np.min(np.clip(vital_limbs["SkinHealth"], 0.0, 100.0))) / 100.0,
            float(np.min(np.clip(vital_limbs["MuscleHealth"], 0.0, 100.0))) / 100.0,
        )
        vital_damage = 1.0 - vital_integrity
    else:
        vital_damage = 0.0

    # Ragdoll impacts damage blocks and limbs together.  TimeRagdolled is not
    # a moral penalty; it is a small immediate proxy for that loss of control.
    ragdoll_risk = _rising_risk(obs["TimeRagdolled"], 0.10, 1.00)

    return _weighted_mean(
        (0.20, skin_damage),
        (0.25, muscle_damage),
        (0.20, structural_damage),
        (0.25, vital_damage),
        (0.10, ragdoll_risk),
    )


def _physical_risk(obs):
    """Pain, exhaustion, shock, and limb integrity: immediate action costs."""
    return _weighted_mean(
        (0.25, _rising_risk(obs["AveragePain"], 10.0, 80.0)),
        (0.15, _rising_risk(obs["PainShock"], 0.10, 0.66)),
        (0.15, _rising_risk(obs["Shock"], 10.0, 60.0)),
        # Stamina starts costing reward immediately when it is spent.  At 15,
        # the game's exertion moodle is critical; at <1 it forces shock.
        (0.20, _falling_risk(obs["Stamina"], 100.0, 15.0)),
        (0.10, _falling_risk(obs["Energy"], 35.0, 7.0)),
        (0.15, _limb_risk(obs)),
    )


def _acute_risk(obs):
    """Vital-system risk that can turn a productive run into a short episode."""
    oxygen_risk = max(
        _falling_risk(obs["BloodOxygen"], 90.0, 45.0),
        _falling_risk(obs["RespiratoryRate"], 90.0, 10.0),
        1.0 if not bool(obs["Breathing"]) else 0.0,
    )
    circulation_risk = max(
        _outside_risk(obs["BloodPressure"], 96.0, 145.0, 60.0, 180.0),
        _outside_risk(obs["HeartRate"], 60.0, 110.0, 40.0, 200.0),
        _rising_risk(obs["FibrillationProgress"], 15.0, 75.0),
        _rising_risk(obs["StrokeAmount"], 0.0, 50.0),
        1.0 if bool(obs["HasPulmonaryEmbolism"]) else 0.0,
    )
    blood_loss_risk = max(
        _falling_risk(obs["BloodVolume"], 80.0, 30.0),
        _rising_risk(obs["TotalBleedSpeed"], 0.02, 0.134),
        _rising_risk(obs["InternalBleeding"], 5.0, 50.0),
        _rising_risk(obs["Hemothorax"], 40.0, 70.0),
    )

    return _weighted_mean(
        (0.18, _falling_risk(obs["BrainHealth"], 95.0, 30.0)),
        (0.20, _falling_risk(obs["Consciousness"], 90.0, 20.0)),
        (0.21, oxygen_risk),
        (0.23, circulation_risk),
        (0.18, blood_loss_risk),
    )


def _systemic_risk(obs):
    """Slower threats which nevertheless determine whether the run can continue."""
    limb_infection = float(np.max(obs["Limbs"]["InfectionAmount"]))
    hydration_risk = _outside_risk(obs["Thirst"], 75.0, 125.0, 0.0, 175.0)
    nutrition_risk = _outside_risk(obs["Hunger"], 75.0, 100.0, 0.0, 125.0)

    return _weighted_mean(
        (0.15, _outside_risk(obs["Temperature"], 35.5, 38.0, 28.0, 41.5)),
        (0.12, nutrition_risk),
        (0.12, hydration_risk),
        (0.10, _rising_risk(obs["SicknessAmount"], 10.0, 75.0)),
        (0.12, _rising_risk(obs["RadiationSickness"], 10.0, 80.0)),
        (0.10, _rising_risk(obs["VenomCurrent"], 2.0, 90.0)),
        (0.13, max(
            _rising_risk(limb_infection, 25.0, 80.0),
            _rising_risk(obs["SepticShock"], 10.0, 80.0),
        )),
        (0.08, _rising_risk(obs["OverEncumberance"], 0.05, 0.85)),
        (0.08, _rising_risk(obs["TraumaAmount"], 25.0, 80.0)),
    )


def RiskBreakdown(obs):
    """Return named continuous risks for logging and reward diagnostics."""
    physical = _physical_risk(obs)
    acute = _acute_risk(obs)
    systemic = _systemic_risk(obs)
    total = _weighted_mean(
        (0.34, physical),
        (0.46, acute),
        (0.20, systemic),
    )
    return {
        "physical_risk": physical,
        "acute_risk": acute,
        "systemic_risk": systemic,
        "risk": total,
    }


def Reset(env, obs):
    """Synchronize potential-based reward state with a freshly reset world."""
    env.previous_progress = float(obs["LayerProgress"])
    env.previous_risk = RiskBreakdown(obs)["risk"]
    env.ragdoll_depth_milestone = float(obs["BestLayerDepth"])
    env.ragdoll_seconds_since_depth_milestone = 0.0
    env.previous_time_ragdolled = float(obs["TimeRagdolled"])
    env.last_reward_terms = {}


def Reward(obs, act, env):
    """Return the scalar reward and preserve a named breakdown on the env."""
    current_progress = float(obs["LayerProgress"])
    risks = RiskBreakdown(obs)
    current_risk = risks["risk"]

    if env.previous_progress is None or env.previous_risk is None:
        Reset(env, obs)

    progress_delta = current_progress - env.previous_progress
    risk_delta = env.previous_risk - current_risk  # positive when the body becomes safer

    # Count only actual time spent ragdolled.  Standing pauses the accumulator
    # instead of clearing it, so periodically standing up cannot erase a long
    # unproductive ragdoll history.  The game's forced stand at 60 seconds is
    # likewise not an escape hatch.
    current_best_depth = float(obs["BestLayerDepth"])
    current_time_ragdolled = float(obs["TimeRagdolled"])
    ragdoll_seconds_delta = max(0.0, current_time_ragdolled - env.previous_time_ragdolled)

    if current_best_depth >= (env.ragdoll_depth_milestone + RAGDOLL_DEPTH_MILESTONE_METERS):
        env.ragdoll_depth_milestone = current_best_depth
        env.ragdoll_seconds_since_depth_milestone = 0.0
    else:
        env.ragdoll_seconds_since_depth_milestone += ragdoll_seconds_delta

    overdue_ragdoll_seconds = max(0.0, env.ragdoll_seconds_since_depth_milestone - RAGDOLL_STALL_GRACE_SECONDS)
    ragdoll_stall_penalty = 0.0
    if act[21] == 1: # ragdoll action, only penalize when the policy chooses to ragdoll
        ragdoll_stall_penalty = min(RAGDOLL_STALL_PENALTY_CAP, RAGDOLL_STALL_PENALTY_BASE * (2.0 ** (overdue_ragdoll_seconds / RAGDOLL_STALL_DOUBLING_SECONDS) - 1.0))

    progress_reward = PROGRESS_REWARD_SCALE * progress_delta
    safety_delta_reward = SAFETY_DELTA_REWARD_SCALE * risk_delta
    occupancy_penalty = RISK_OCCUPANCY_COST * current_risk
    reward = (
        progress_reward
        + safety_delta_reward
        - occupancy_penalty
        - ragdoll_stall_penalty
        - STEP_COST
    )

    death_penalty = 0.0
    completion_bonus = 0.0
    if bool(obs["PlayerDead"]):
        death_penalty = DEATH_PENALTY
        reward -= death_penalty
    elif current_progress >= 1.0:
        completion_bonus = LAYER_COMPLETE_BONUS
        reward += completion_bonus

    env.previous_progress = current_progress
    env.previous_risk = current_risk
    env.previous_time_ragdolled = current_time_ragdolled
    env.last_reward_terms = {
        **risks,
        "progress": progress_reward,
        "progress_delta": progress_delta,
        "safety_delta": safety_delta_reward,
        "risk_delta": risk_delta,
        "occupancy_penalty": occupancy_penalty,
        "ragdoll_seconds_since_depth_milestone": env.ragdoll_seconds_since_depth_milestone,
        "ragdoll_depth_milestone": env.ragdoll_depth_milestone,
        "ragdoll_stall_penalty": ragdoll_stall_penalty,
        "death": death_penalty,
        "completion": completion_bonus,
        "reward": float(reward),
    }
    return float(reward)
