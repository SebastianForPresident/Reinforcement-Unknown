import gymnasium as gym
import numpy as np
import threading
import time
import win32file
from ObservationFlattener import build_plan, flatten

_server = None
_flatten_plan = None

def Start(server):
    global _server, _listener
    _server = server

def Preprocess(obs):
    global _flatten_plan

    t0 = time.perf_counter()
    if _flatten_plan is None:
        _flatten_plan = build_plan(obs.dtype)

    flat = flatten(obs, _flatten_plan)
    t1 = time.perf_counter()
    return flat

def Decode(action):
    _server.move = action[0] - 1
    _server.jump = action[1]
    _server.vertMove = action[2] - 1
    _server.crouch = action[3]

    _server.lookdX = action[4] - 4
    _server.lookdY = action[5] - 5

    _server.attack = action[6]
    _server.interact = action[7]

    _server.targetSlotIndex = action[8]
    _server.selectedSlotIndex = action[9] - 1

    _server.dropItem = action[10]
    _server.moveItem = action[11]

    _server.selectedBagIndex = action[12] - 1

    _server.useItem = action[13]
    _server.useItemWorld = action[14]

    _server.selectedLimb = action[15]

    _server.useItemMedical = action[16]

    _server.selectedRecipe = action[17] - 1

    _server.favoriteItem = action[18]
    _server.switchMainHand = action[19]
    _server.trySleep = action[20]
    _server.ragdoll = action[21]

    _server.exercise = action[22] - 1

    _server.bark = action[23]
    _server.throw = action[24]

    _server.liquidAmount = action[25] * 5

    _server.drainLiquid = action[26]
    _server.pullLiquidFromWorld = action[27]

HEALTH_DT = 0.05  # Unity observations arrive at 20 Hz.
HEALTH_INTEGRAL_HORIZON = 10.0


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _high_quality(value, optimal, failure):
    """Quality for a value where larger values are better."""
    if optimal <= failure:
        return 1.0 if value >= optimal else 0.0
    return _clip01((float(value) - failure) / (optimal - failure))


def _low_quality(value, failure):
    """Quality for a value where zero is best."""
    return 1.0 - _clip01(float(value) / failure) if failure > 0.0 else 1.0


def _target_quality(value, target, tolerance):
    """Quality for a value with a preferred point rather than a max/min."""
    if tolerance <= 0.0:
        return 1.0 if float(value) == target else 0.0
    return 1.0 - _clip01(abs(float(value) - target) / tolerance)


def _reserve_quality(value, low, target, high):
    """Quality for reserves such as hunger/thirst with both low and high risks."""
    value = float(value)
    if value <= low or value >= high:
        return 0.0
    if value <= target:
        return _clip01((value - low) / (target - low))
    return _clip01((high - value) / (high - target))


def _bool_quality(value, desired):
    return 1.0 if bool(value) == desired else 0.0


HEALTH_WEIGHTS = {
    "alive": 5.0,
    "brain_health": 4.0,
    "blood_oxygen": 3.0,
    "blood_volume": 3.0,
    "fibrillation": 3.0,
    "bleeding": 3.0,
    "internal_bleeding": 3.0,
    "temperature": 2.5,
    "blood_pressure": 2.0,
    "stamina": 2.0,
    "energy": 2.0,
    "thirst": 2.0,
    "hunger": 1.5,
    "consciousness": 2.5,
    "limb_integrity": 3.0,
    "limb_conditions": 2.5,
    "heart_rate": 1.0,
    "total_happiness": 0.75,
    "immunity": 0.75,
}


def _health_qualities(obs):
    """Convert health observations into normalized quality channels."""
    q = {}

    # Circulation and breathing.
    q["heart_rate"] = _target_quality(obs["HeartRate"], 70.0, 70.0)
    q["fibrillation"] = _low_quality(obs["FibrillationProgress"], 100.0)
    q["fibrillation_forced"] = _bool_quality(obs["FibrillationForced"], False)
    q["fibrillation_rising"] = _bool_quality(obs["FibrillationRising"], False)
    q["pulmonary_embolism"] = _bool_quality(obs["HasPulmonaryEmbolism"], False)
    q["blood_oxygen"] = _high_quality(obs["BloodOxygen"], 100.0, 65.0)
    q["blood_volume"] = _high_quality(obs["BloodVolume"], 100.0, 40.0)
    q["blood_pressure"] = _target_quality(obs["BloodPressure"], 120.0, 50.0)
    q["blood_vessel_size"] = _target_quality(obs["BloodVesselSize"], 1.0, 0.15)
    q["blood_viscosity"] = _target_quality(obs["BloodViscosity"], 0.0, 100.0)
    q["bleeding"] = _low_quality(obs["TotalBleedSpeed"], 0.134)
    q["internal_bleeding"] = _low_quality(obs["InternalBleeding"], 40.0)
    q["hemothorax"] = _low_quality(obs["Hemothorax"], 50.0)
    q["venom_total"] = _low_quality(obs["VenomTotal"], 100.0)
    q["venom_current"] = _low_quality(obs["VenomCurrent"], 100.0)
    q["respiratory_rate"] = _high_quality(obs["RespiratoryRate"], 100.0, 10.0)
    q["breathing"] = _bool_quality(obs["Breathing"], True)

    # Reserves and general condition.
    q["hunger"] = _reserve_quality(obs["Hunger"], 40.0, 100.0, 125.0)
    q["thirst"] = _reserve_quality(obs["Thirst"], 40.0, 100.0, 175.0)
    q["stamina"] = _high_quality(obs["Stamina"], 100.0, 0.0)
    q["energy"] = _high_quality(obs["Energy"], 100.0, 0.0)
    q["immunity"] = _high_quality(obs["Immunity"], 100.0, 0.0)
    q["total_happiness"] = _high_quality(obs["TotalHappiness"], 100.0, -100.0)
    q["weight_offset"] = _target_quality(obs["WeightOffset"], 0.0, 40.0)
    q["dirtyness"] = _low_quality(obs["Dirtyness"], 100.0)
    q["claw_health"] = _high_quality(obs["ClawHealth"], 100.0, 0.0)
    q["brain_grow_sickness"] = _low_quality(obs["BrainGrowSickness"], 100.0)
    q["overdose"] = _low_quality(obs["OverdoseIndex"], 255.0)
    q["mind_wiped"] = _bool_quality(obs["MindWiped"], False)

    # Brain, pain, temperature, and sickness.
    q["brain_health"] = _high_quality(obs["BrainHealth"], 100.0, 0.0)
    q["consciousness"] = _high_quality(obs["Consciousness"], 100.0, 30.0)
    q["shock"] = _low_quality(obs["Shock"], 100.0)
    q["reversed_controls"] = _bool_quality(obs["ReversedControls"], False)
    q["brain_dying"] = _bool_quality(obs["BrainDying"], False)
    q["alive"] = _bool_quality(obs["PlayerDead"], False)
    q["stroke"] = _low_quality(obs["StrokeAmount"], 100.0)
    q["temperature"] = _target_quality(obs["Temperature"], 37.0, 8.0)
    q["average_pain"] = _low_quality(obs["AveragePain"], 100.0)
    q["pain_shock"] = _low_quality(obs["PainShock"], 1.0)
    q["hearing_loss"] = _low_quality(obs["HearingLoss"], 100.0)
    q["both_hands_unusable"] = _bool_quality(obs["BothHandsUnusable"], False)
    q["sickness"] = _low_quality(obs["SicknessAmount"], 100.0)
    q["septic_shock"] = _low_quality(obs["SepticShock"], 100.0)
    q["radiation_sickness"] = _low_quality(obs["RadiationSickness"], 100.0)
    q["trauma"] = _low_quality(obs["TraumaAmount"], 100.0)
    q["horrified"] = _low_quality(obs["HorrifiedLevel"], 100.0)
    q["disfigured"] = _bool_quality(obs["Disfigured"], False)
    q["eye_gone"] = _bool_quality(obs["EyeGone"], False)
    q["both_eyes_gone"] = _bool_quality(obs["BothEyesGone"], False)
    q["over_encumbered"] = _low_quality(obs["OverEncumberance"], 1.0)

    # Sleep and emergency state. These are intentionally lower-impact.
    q["bad_sleep"] = _low_quality(obs["BadSleepAmount"], 150.0)
    q["sleep_quality"] = _high_quality(obs["CurSleep"], 3.0, 0.0)
    q["tried_last_stand"] = _bool_quality(obs["TriedRollingLastStand"], False)

    # Aggregate limb state. Descriptor fields such as IsHead and DistanceToHeart
    # are deliberately not scored; they explain the other limb values.
    limbs = obs["Limbs"]
    if len(limbs) > 0:
        skin = np.clip(np.asarray(limbs["SkinHealth"], dtype=np.float32) / 100.0, 0.0, 1.0)
        muscle = np.clip(np.asarray(limbs["MuscleHealth"], dtype=np.float32) / 100.0, 0.0, 1.0)
        vital = np.asarray(limbs["IsVital"], dtype=bool)
        critical_skin = float(np.min(skin[vital])) if np.any(vital) else float(np.min(skin))
        critical_muscle = float(np.min(muscle[vital])) if np.any(vital) else float(np.min(muscle))
        q["limb_integrity"] = 0.5 * (critical_skin + critical_muscle)

        condition_channels = [
            1.0 - np.clip(np.asarray(limbs["Pain"], dtype=np.float32) / 100.0, 0.0, 1.0),
            1.0 - np.clip(np.asarray(limbs["InfectionAmount"], dtype=np.float32) / 100.0, 0.0, 1.0),
            1.0 - np.clip(np.asarray(limbs["DisinfectionTime"], dtype=np.float32) / 150.0, 0.0, 1.0),
            1.0 - np.clip(np.asarray(limbs["DislocationTimer"], dtype=np.float32) / 100.0, 0.0, 1.0),
            1.0 - np.clip(np.asarray(limbs["BoneHealTimer"], dtype=np.float32) / 100.0, 0.0, 1.0),
            1.0 - np.clip(np.asarray(limbs["Shrapnel"], dtype=np.float32) / 100.0, 0.0, 1.0),
            1.0 - np.clip(np.asarray(limbs["TotalBleedAmount"], dtype=np.float32) / 100.0, 0.0, 1.0),
            1.0 - np.asarray(limbs["Dislocated"], dtype=np.float32),
            1.0 - np.asarray(limbs["Broken"], dtype=np.float32),
            1.0 - np.asarray(limbs["Infected"], dtype=np.float32),
            1.0 - np.asarray(limbs["Dismembered"], dtype=np.float32),
        ]
        q["limb_conditions"] = float(np.mean(np.concatenate([channel.reshape(-1) for channel in condition_channels])))

    return {name: _clip01(value) for name, value in q.items()}


def _weighted_mean(values):
    total = 0.0
    weight_total = 0.0
    for name, value in values.items():
        weight = HEALTH_WEIGHTS.get(name, 1.0)
        total += weight * float(value)
        weight_total += weight
    return total / weight_total if weight_total else 0.0


def _reset_health_reward(env, obs):
    env.prev_health_qualities = _health_qualities(obs)
    env.health_integrals = {name: 0.0 for name in env.prev_health_qualities}


def Reward(obs, env):
    """Bootstrap reward: progress plus a weighted PID-like health signal."""
    current_layer_progress = float(obs["LayerProgress"])
    progress_reward = (current_layer_progress - env.prev_layer_progress) * 10.0

    current_qualities = _health_qualities(obs)
    if env.prev_health_qualities is None:
        _reset_health_reward(env, obs)
        previous_qualities = current_qualities
    else:
        previous_qualities = env.prev_health_qualities

    proportional_reward = -_weighted_mean({name: 1.0 - value for name, value in current_qualities.items()})
    derivative_reward = _weighted_mean({
        name: current_qualities[name] - previous_qualities.get(name, current_qualities[name])
        for name in current_qualities
    })

    integral_values = {}
    for name, quality in current_qualities.items():
        previous_integral = env.health_integrals.get(name, 0.0)
        centered_quality = 2.0 * quality - 1.0
        env.health_integrals[name] = float(np.clip(
            previous_integral + centered_quality * HEALTH_DT / HEALTH_INTEGRAL_HORIZON,
            -1.0,
            1.0,
        ))
        integral_values[name] = env.health_integrals[name]
    integral_reward = _weighted_mean(integral_values)

    reward = (
        progress_reward
        + 0.02 * proportional_reward
        + 0.01 * integral_reward
        + 0.25 * derivative_reward
    )

    if bool(obs["PlayerDead"]):
        reward -= 1.0
    else:
        reward += 0.001

    env.prev_layer_progress = current_layer_progress
    env.prev_health_qualities = current_qualities
    env.last_reward_components = {
        "progress": float(progress_reward),
        "health_proportional": float(0.02 * proportional_reward),
        "health_integral": float(0.01 * integral_reward),
        "health_derivative": float(0.25 * derivative_reward),
        "death": -1.0 if bool(obs["PlayerDead"]) else 0.0,
        "survival": 0.0 if bool(obs["PlayerDead"]) else 0.001,
    }
    return float(reward)

def SendReset():
    if _server is None:
        raise RuntimeError("PPO server has not been started")

    _server.reset_requested.set()
    with _server.action_write_lock:
        win32file.WriteFile(_server.action_pipe, b"RESET\n")

class Env(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.MultiDiscrete([3, 2, 3, 2, 9, 11, 2, 2, 25, 26, 2, 2, 33, 2, 2, 15, 2, 133, 2, 2, 2, 2, 4, 2, 2, 201, 2, 2])
        self.observation_space = gym.spaces.Box(low=-np.inf,high=np.inf,shape=(452933,),dtype=np.float32)

        self.latest_obs = None
        self.obs_ready = threading.Event()
        self.prev_layer_progress = 0.0
        self.prev_health_qualities = None
        self.health_integrals = {}
        self.last_reward_components = {}

        self.max_episode_steps = 5000
        self.episode_steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.episode_steps = 0

        self.obs_ready.clear()
        SendReset()
        self.obs_ready.wait()

        _server.reset_requested.clear()

        obs = self.latest_obs
        self.prev_layer_progress = float(obs["LayerProgress"])
        _reset_health_reward(self, obs)
        return Preprocess(obs), {}

    def step(self, action):
        Decode(action)

        self.obs_ready.clear()
        self.obs_ready.wait()

        self.episode_steps += 1

        obs = self.latest_obs
        reward = Reward(obs, self)
        terminated = bool(obs["PlayerDead"]) or obs["LayerProgress"] >= 1.0
        truncated = self.episode_steps >= self.max_episode_steps
        return Preprocess(obs), reward, terminated, truncated, dict(self.last_reward_components)

    def close(self):
        if _server is not None:
            _server.Shutdown()
        
