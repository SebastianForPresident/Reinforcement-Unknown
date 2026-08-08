import gymnasium as gym
import numpy as np
import threading
import Reward
from EpisodeTrace import EpisodeTraceWriter
import ObservationNormalization as ObsNorm
import ProtocolValidation
import Types

_server = None
_general_input_dim = None
_observation_lock = None
MAX_RESET_ATTEMPTS = 3

# CB1 adds previous-action and absolute-position context and uses a
# semantic encoder instead of flattening raw wire values.
GENERAL_FIELD_NAMES = ObsNorm.GENERAL_FIELD_NAMES

def GetGeneralValues(obs):
    return ObsNorm.encode_general(obs)

def Init_General():
    global _general_input_dim
    dummy = np.zeros((), dtype=Types.OBSERVATION_DTYPE)
    dummy["WorldDimensions"]["X"] = 1
    dummy["WorldDimensions"]["Y"] = 1
    _general_input_dim = GetGeneralValues(dummy).size

def PreprocessObservation(obs):
    return {
        "general": GetGeneralValues(obs),
        "spatial": ObsNorm.encode_spatial(obs),
    }

def Start(server):
    global _server, _observation_lock
    _server = server
    _observation_lock = server.observation_lock

def Decode(action):
    """Decode the seven-action CB1 policy into the sequenced wire protocol.

    The Unity harness and Server retain the 28 control fields plus a CB1
    decision sequence. Only seven controls affect locomotion and direct item
    use; every other field is explicitly reset on each policy decision.
    """
    if len(action) != 7:
        raise ValueError(f"CB1 expects 7 actions, received {len(action)}")

    _server.move = action[0] - 1
    _server.jump = action[1]
    _server.vertMove = action[2] - 1
    _server.crouch = action[3]

    _server.lookdX = action[4] - 4
    _server.lookdY = action[5] - 5

    _server.attack = action[6]

    # Legacy control fields intentionally disabled for CB1.
    _server.interact = 0
    _server.targetSlotIndex = 0
    _server.selectedSlotIndex = -1
    _server.dropItem = 0
    _server.moveItem = 0
    _server.selectedBagIndex = -1
    _server.useItem = 0
    _server.useItemWorld = 0
    _server.selectedLimb = 0
    _server.useItemMedical = 0
    _server.selectedRecipe = -1
    _server.favoriteItem = 0
    _server.switchMainHand = 0
    _server.trySleep = 0
    _server.ragdoll = 0
    _server.exercise = -1
    _server.bark = 0
    _server.throw = 0
    _server.liquidAmount = 0
    _server.drainLiquid = 0
    _server.pullLiquidFromWorld = 0

    # The bridge latches only a strictly newer sequence at a policy boundary.
    _server.action_sequence += 1

    # PPO optimizer updates leave Unity paused.  Resume only after this fresh
    # policy action has been decoded, so no physics tick can replay the action
    # that preceded the update.
    _server.SendDecodedAction()

def SendReset(reset_token):
    if _server is None:
        raise RuntimeError("PPO server has not been started")

    _server.reset_requested.set()
    with _server.action_write_lock:
        _server.action_pipe.sendall(f"RESET {reset_token}\n".encode("ascii"))

class Env(gym.Env):
    def __init__(self):
        Init_General()
        # CB1 policy controls: move, jump, vertical move, crouch, look X,
        # look Y, and attack. Decode() supplies inactive defaults for the
        # retained legacy control fields and appends a decision sequence.
        self.action_space = gym.spaces.MultiDiscrete(
            [3, 2, 3, 2, 9, 11, 2]
        )
        self.observation_space = gym.spaces.Dict({
            "general": gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(_general_input_dim,),
                dtype=np.float32,
            ),
            "spatial": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(
                    ObsNorm.SPATIAL_CHANNELS,
                    Types.SIGHT_RANGE_X * 2 + 1,
                    Types.SIGHT_RANGE_Y * 2 + 1,
                ),
                dtype=np.float32,
            ),
        })

        self.latest_obs = None
        self.latest_observation_id = None
        self.last_consumed_observation_id = None
        self.obs_ready = threading.Event()
        self.previous_progress = None
        self.last_reward_terms = {}

        # Ten million 5 Hz decisions is a multi-week technical watchdog, not
        # a learning horizon. Normal episodes end only on layer completion.
        self.max_episode_steps = 10_000_000
        self.episode_steps = 0
        self.episode_number = 0
        self.reset_token = 0
        self.episode_trace = EpisodeTraceWriter()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.episode_number += 1
        self.episode_trace.begin_episode(self.episode_number)
        self.episode_steps = 0

        with _observation_lock:
            self.obs_ready.clear()

        for reset_attempt in range(MAX_RESET_ATTEMPTS):
            self.reset_token += 1
            reset_token = self.reset_token
            SendReset(reset_token)

            # The acknowledgement carries the ID of the first observation
            # emitted after Unity has completed this reset. An observation
            # with a newer ID than the previous episode is not sufficient:
            # the last old-world packet can arrive after RESET was sent.
            first_observation_id = _server.WaitForResetReady(reset_token)
            obs, observation_id = self._wait_for_observation_id(
                first_observation_id
            )
            if not bool(obs["PlayerDead"]):
                break

            # A reset acknowledgement is tied to one exact post-reset
            # observation. If that observation is unusable, issue a new
            # tokenized reset rather than accepting another ambiguous packet.
        else:
            raise RuntimeError(
                "Unity reset did not produce a live observation after "
                f"{MAX_RESET_ATTEMPTS} attempts"
            )

        _server.reset_requested.clear()

        self.last_consumed_observation_id = observation_id
        Reward.Reset(self, obs)
        processed = PreprocessObservation(obs)
        ProtocolValidation.validate_reset_observation(obs, processed)
        return processed, {}

    def _wait_for_observation_id(self, expected_id):
        """Return exactly the observation identified by expected_id."""
        while True:
            with _observation_lock:
                current_id = self.latest_observation_id
                if current_id == expected_id:
                    obs = self.latest_obs
                    self.obs_ready.clear()
                    return obs, current_id

                if current_id is not None and current_id > expected_id:
                    raise RuntimeError(
                        "CB1 observation ID mismatch: "
                        f"expected {expected_id}, observed {current_id}"
                    )

                self.obs_ready.clear()

            self.obs_ready.wait()

    def step(self, action):
        Decode(action)

        if self.last_consumed_observation_id is None:
            raise RuntimeError("CB1 Env.step called before Env.reset")

        obs, observation_id = self._wait_for_observation_id(
            self.last_consumed_observation_id + 1
        )

        self.episode_steps += 1

        self.last_consumed_observation_id = observation_id
        processed = PreprocessObservation(obs)
        validation_info = ProtocolValidation.validate_step_observation(
            obs,
            processed,
            action,
        )
        reward = Reward.Reward(obs, action, self)
        terminated = obs["LayerProgress"] >= 1.0
        truncated = self.episode_steps >= self.max_episode_steps
        info = self.last_reward_terms.copy()
        info.update(validation_info)
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
            self.episode_trace.finish_episode(
                complete=bool(terminated and not truncated)
            )

        return processed, reward, terminated, truncated, info

    def close(self):
        self.episode_trace.close()
        if _server is not None:
            _server.Shutdown()
        
