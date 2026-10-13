"""Runtime callbacks for the XY1 research configuration."""
from __future__ import annotations
import numpy as np
from .player_controller import StandingAction

def action_decoder(actions):
    def get(name, default):
        raw=np.asarray(actions.get(name,default)).reshape(-1)
        return int(raw[0]) if raw.size else int(default)
    return StandingAction(move_x=get("move_x",1)-1,jump=bool(get("jump",0)),
        move_y=get("move_y",1)-1,crouch=bool(get("crouch",0)),
        look_dx=get("look_dx",4)-4,look_dy=get("look_dy",5)-5,attack=bool(get("attack",0)))

def observation_encoder(env):
    from .xy1 import encode_xy1_observation
    return encode_xy1_observation(env)

def reward_function(previous,current,terminated,truncated):
    # The adapter owns the goal and invokes BestDistanceReward directly. This callback
    # exists to satisfy the generic callback contract without inventing a second state.
    return 0.0,{"distance_improvement":0.0,"arrival_bonus":0.0}
