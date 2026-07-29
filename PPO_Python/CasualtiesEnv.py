import gymnasium as gym
import numpy as np
import threading
from ObservationFlattener import build_plan, flatten
import Reward

_server = None
_flatten_plan = None

def Start(server):
    global _server, _listener
    _server = server

def Preprocess(obs):
    global _flatten_plan

    if _flatten_plan is None:
        _flatten_plan = build_plan(obs.dtype)

    flat = flatten(obs, _flatten_plan)
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
        self.action_space = gym.spaces.MultiDiscrete([3, 2, 3, 2, 9, 11, 2, 2, 25, 26, 2, 2, 33, 2, 2, 15, 2, 133, 2, 2, 2, 2, 4, 2, 2, 201, 2, 2])
        self.observation_space = gym.spaces.Box(low=-np.inf,high=np.inf,shape=(452933,),dtype=np.float32)

        self.latest_obs = None
        self.obs_ready = threading.Event()
        self.previous_progress = None
        self.previous_risk = None
        self.last_reward_terms = {}

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
        Reward.Reset(self, obs)
        return Preprocess(obs), {}

    def step(self, action):
        Decode(action)

        self.obs_ready.clear()
        self.obs_ready.wait()

        self.episode_steps += 1

        obs = self.latest_obs
        reward = Reward.Reward(obs, self)
        terminated = bool(obs["PlayerDead"]) or obs["LayerProgress"] >= 1.0
        truncated = self.episode_steps >= self.max_episode_steps
        return Preprocess(obs), reward, terminated, truncated, self.last_reward_terms.copy()

    def close(self):
        if _server is not None:
            _server.Shutdown()
        
