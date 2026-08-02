import gymnasium as gym
import numpy as np
import threading
import time
import Reward
from EpisodeTrace import EpisodeTraceWriter
from ObservationFlattener import flatten, build_plan
import Types

_server = None
_general_flatten_plans = {}
_general_input_dim = None
_observation_lock = None

# These are world/player values rather than spatial grids or repeated entity
# collections. Keep the order explicit so GeneralEncoder.input_dim is stable.
GENERAL_FIELD_NAMES = (
    "Velocity",
    "IsRight",
    "MaxSpeed",
    "RelativeLookPos",
    "JumpCooldown",
    "Grounded",
    "TimeSinceGrounded",
    "StandingOn",
    "TimeRagdolled",
    "CrawlTime",
    "InWater",
    "LiquidSlipTime",
    "LiquidRagdollBar",
    "LiquidDrinkTime",
    "CanWalljumpLeft",
    "CanWalljumpRight",
    "AttackCooldown",
    "CrouchAmount",
    "Crouching",
    "IsClimbing",
    "ClimbableProgress",
    "ClimbVelocity",
    "HeartRate",
    "FibrillationProgress",
    "FibrillationForced",
    "FibrillationRising",
    "HasPulmonaryEmbolism",
    "BloodOxygen",
    "BloodVolume",
    "BloodPressure",
    "BloodVesselSize",
    "BloodViscosity",
    "TotalBleedSpeed",
    "InternalBleeding",
    "Hemothorax",
    "VenomTotal",
    "VenomCurrent",
    "RespiratoryRate",
    "Breathing",
    "Adrenaline",
    "CurAdrenaline",
    "StimulantMultiplier",
    "OnHardStimulants",
    "OpiateHappiness",
    "AntidepressantHappiness",
    "BrainGrowSickness",
    "UsedNeuralBooster",
    "MindWiped",
    "Caffeinated",
    "OverdoseIndex",
    "WeightOffset",
    "Hunger",
    "Thirst",
    "Stamina",
    "Energy",
    "Immunity",
    "TotalHappiness",
    "Dirtyness",
    "ClawHealth",
    "BrainHealth",
    "Consciousness",
    "Shock",
    "ReversedControls",
    "BrainDying",
    "PlayerDead",
    "StrokeAmount",
    "Temperature",
    "ClothingTemperature",
    "AveragePain",
    "PainShock",
    "HearingLoss",
    "BothHandsUnusable",
    "SicknessAmount",
    "SepticShock",
    "RadiationSickness",
    "CorpsesSeen",
    "TraumaAmount",
    "HorrifiedLevel",
    "FocusedLevel",
    "Disfigured",
    "EyeGone",
    "BothEyesGone",
    "TotalEncumberance",
    "OverEncumberance",
    "MaxEncumberance",
    "Sleeping",
    "CurSleep",
    "BadSleepAmount",
    "GoodSleepTime",
    "ForcedSleepQuality",
    "UsingSleepingBag",
    "CanTakeNap",
    "TriedRollingLastStand",
    "LastStandTime",
    "STR",
    "RES",
    "INT",
    "STRProgress",
    "RESProgress",
    "INTProgress",
    "LayerProgress",
    "CurrentLayer",
    "BestLayerDepth",
    "LayerTimeRemaining",
    "RadLineDisplacement",
)

def GetGeneralValues(obs):
    """Return selected non-spatial, non-entity values as one float32 vector."""
    values = []

    for field_name in GENERAL_FIELD_NAMES:
        values.append(flatten(obs[field_name], _general_flatten_plans[field_name]))

    return np.concatenate(values).astype(np.float32, copy=False)

def EncodeGrid(grid):
    channels = []

    for field in grid.dtype.names:
        channels.append(grid[field].astype(np.float32))

    return np.ascontiguousarray(np.stack(channels, axis=0))

def Init_General():
    global _general_input_dim
    general_input_dim = 0

    for field_name in GENERAL_FIELD_NAMES:
        field_dtype = Types.OBSERVATION_DTYPE[field_name]

        plan = build_plan(field_dtype)
        _general_flatten_plans[field_name] = plan

        dummy = np.zeros((), dtype=field_dtype)
        general_input_dim += flatten(dummy, plan).size

    _general_input_dim = general_input_dim

def PreprocessObservation(obs):
    return {
        "general": GetGeneralValues(obs),
        "blocks": EncodeGrid(obs["RelativeBlockMap"]),
        "fluids": EncodeGrid(obs["RelativeFluidMap"])
    }

def Start(server):
    global _server, _listener, _observation_lock
    _server = server
    _observation_lock = server.observation_lock

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

    # PPO optimizer updates leave Unity paused.  Resume only after this fresh
    # policy action has been decoded, so no physics tick can replay the action
    # that preceded the update.
    _server.ResumeSimulation()

def SendReset():
    if _server is None:
        raise RuntimeError("PPO server has not been started")

    _server.reset_requested.set()
    with _server.action_write_lock:
        _server.action_pipe.sendall(b"RESET\n")

class Env(gym.Env):
    def __init__(self):
        Init_General()
        self.action_space = gym.spaces.MultiDiscrete([3, 2, 3, 2, 9, 11, 2, 2, 25, 26, 2, 2, 33, 2, 2, 15, 2, 133, 2, 2, 2, 2, 4, 2, 2, 201, 2, 2])
        self.observation_space = gym.spaces.Dict({
            "general": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(_general_input_dim,),
                dtype=np.float32,
            ),
            "blocks": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    len(Types.BLOCK_DTYPE.names),
                    Types.SIGHT_RANGE_X * 2 + 1,
                    Types.SIGHT_RANGE_Y * 2 + 1,
                ),
                dtype=np.float32,
            ),
            "fluids": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    len(Types.FLUID_TILE_DTYPE.names),
                    Types.SIGHT_RANGE_X * 2 + 1,
                    Types.SIGHT_RANGE_Y * 2 + 1,
                ),
                dtype=np.float32,
            ),
        })

        self.latest_obs = None
        self.latest_observation_id = None
        self.latest_decision_interval_ticks = 0
        self.last_consumed_observation_id = None
        self.last_decision_interval_ticks = 10
        self.obs_ready = threading.Event()
        self.previous_progress = None
        self.previous_risk = None
        self.last_reward_terms = {}

        # Match C6's roughly 8m20s wall-clock episode cap at the C6.5
        # conscious collection rate of about 20 decisions per second.
        self.max_episode_steps = 10000
        self.episode_steps = 0
        self.episode_number = 0
        self.episode_trace = EpisodeTraceWriter()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.episode_number += 1
        self.episode_trace.begin_episode(self.episode_number)
        self.episode_steps = 0

        with _observation_lock:
            reset_after_id = self.latest_observation_id
            self.obs_ready.clear()
        SendReset()
        obs, observation_id, _, decision_interval_ticks = self._wait_for_new_observation(reset_after_id)

        _server.reset_requested.clear()

        self.last_consumed_observation_id = observation_id
        self.last_decision_interval_ticks = decision_interval_ticks
        Reward.Reset(self, obs)
        return PreprocessObservation(obs), {}

    def _wait_for_new_observation(self, previous_id):
        """Return the newest observation whose ID follows previous_id.

        The receiver and this check share observation_lock.  That makes the
        ID check and event clear atomic with respect to receiver publication,
        avoiding a lost wakeup when an observation arrives at the boundary.
        """
        wait_started = time.perf_counter_ns()
        waited = False
        while True:
            with _observation_lock:
                current_id = self.latest_observation_id
                if (
                    current_id is not None
                    and (previous_id is None or current_id > previous_id)
                ):
                    obs = self.latest_obs
                    decision_interval_ticks = self.latest_decision_interval_ticks
                    self.obs_ready.clear()
                    _server.ProfileEvent(
                        "env_wait_observation",
                        (time.perf_counter_ns() - wait_started) / 1_000_000,
                        observation_id=current_id,
                        interval_ticks=decision_interval_ticks,
                    )
                    return obs, current_id, not waited, decision_interval_ticks

                self.obs_ready.clear()

            waited = True
            self.obs_ready.wait()

    def step(self, action):
        step_started = time.perf_counter_ns()
        Decode(action)

        obs, observation_id, _, decision_interval_ticks = self._wait_for_new_observation(
            self.last_consumed_observation_id
        )

        self.episode_steps += 1

        self.last_consumed_observation_id = observation_id
        self.last_decision_interval_ticks = decision_interval_ticks
        reward_started = time.perf_counter_ns()
        reward = Reward.Reward(obs, action, self)
        _server.ProfileEvent(
            "reward",
            (time.perf_counter_ns() - reward_started) / 1_000_000,
            observation_id=observation_id,
            interval_ticks=decision_interval_ticks,
        )
        terminated = bool(obs["PlayerDead"]) or obs["LayerProgress"] >= 1.0
        truncated = self.episode_steps >= self.max_episode_steps
        info = self.last_reward_terms.copy()
        self.episode_trace.record(
            self.episode_steps,
            action,
            obs,
            observation_id,
            reward,
            info,
            terminated,
            truncated,
        )
        if terminated or truncated:
            self.episode_trace.finish_episode()

        preprocess_started = time.perf_counter_ns()
        processed = PreprocessObservation(obs)
        _server.ProfileEvent(
            "observation_preprocess",
            (time.perf_counter_ns() - preprocess_started) / 1_000_000,
            observation_id=observation_id,
            interval_ticks=decision_interval_ticks,
        )
        _server.ProfileEvent(
            "env_step",
            (time.perf_counter_ns() - step_started) / 1_000_000,
            observation_id=observation_id,
            interval_ticks=decision_interval_ticks,
        )
        return processed, reward, terminated, truncated, info

    def close(self):
        self.episode_trace.finish_episode(complete=False)
        if _server is not None:
            _server.Shutdown()
        
