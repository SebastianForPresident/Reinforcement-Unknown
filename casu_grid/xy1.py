"""XY1 point-navigation geometry, observation and reward contract."""
from __future__ import annotations
import math
import numpy as np

SPATIAL_SHAPE = (16, 85, 49)
GOAL_ACCEPTANCE_TILES = 1


def encode_xy1_observation(env):
    """Return historical CB1 16-channel terrain/fluid map plus two goal offsets.

    Tensor axes are (channel, x, y) like the historical CasU arrays. World y is up;
    increasing second array index is world y. Channel values are bounded [0,1].
    """
    slot, grid = env.slot, env.grid
    s = env.model.s
    px, py = grid.world_to_cell(s.x, s.y, grid.width, grid.height)
    cfg = getattr(env, "cfg", None)
    window_w, window_h = tuple(getattr(cfg, "observation_window_tiles", (85, 49)))
    left_x, left_y = window_w // 2, window_h // 2
    xs = np.arange(px - left_x, px + (window_w - left_x))
    ys = np.arange(py - left_y, py + (window_h - left_y))
    block_ids = np.zeros((window_w, window_h), dtype=np.int64)
    fluid_ids = np.zeros_like(block_ids)
    valid = ((xs[:, None] >= 0) & (xs[:, None] < grid.width) &
             (ys[None, :] >= 0) & (ys[None, :] < grid.height))
    cx = np.clip(xs, 0, grid.width-1); cy = np.clip(ys, 0, grid.height-1)
    block_ids[:] = grid.blocks[np.ix_(cx, cy)]
    fluid_ids[:] = grid.fluids[np.ix_(cx, cy)]
    block_ids[~valid] = 0; fluid_ids[~valid] = 7
    # Historical block channels: normalized health, extreme-health indicator,
    # toxicity, four sleep-quality one-hot channels, seven fluid one-hot channels,
    # player-center marker and cursor marker.
    from .state import BLOCK_HEALTH, TOXICITY
    EXTREME_BLOCK_HEALTH = 420133760.0
    health = np.zeros_like(block_ids, dtype=np.float32)
    for bid, hp in BLOCK_HEALTH.items():
        health[block_ids == bid] = min(1.0, math.log1p(min(hp, 15000.0)) / math.log1p(15000.0))
    health[block_ids == 14] = min(1.0, math.log1p(EXTREME_BLOCK_HEALTH)/math.log1p(EXTREME_BLOCK_HEALTH))
    extreme = (block_ids == 14).astype(np.float32)
    toxicity = np.zeros_like(health)
    for bid, value in TOXICITY.items(): toxicity[block_ids == bid] = min(1.0, value/2.5)
    channels = [health, extreme, toxicity]
    # Historical composition is 3 block descriptors + 4 sleep classes + 7 fluid classes + 2 markers.
    # This surrogate has no sleep-quality state, so those four channels are zero.
    channels.extend(np.zeros_like(health) for _ in range(4))
    for fluid in range(7): channels.append((fluid_ids == fluid).astype(np.float32))
    center = np.zeros((window_w,window_h),np.float32); center[left_x,left_y]=1.0
    cursor = np.zeros_like(center)
    look_x = px + int(s.current_action.look_dx); look_y = py + int(s.current_action.look_dy)
    ix, iy = look_x-(px-left_x), look_y-(py-left_y)
    if 0 <= ix < window_w and 0 <= iy < window_h: cursor[ix,iy]=1.0
    channels.extend((center,cursor))
    spatial=np.ascontiguousarray(np.stack(channels).astype(np.float32))
    goal=slot.goal_cell
    scalars=np.asarray([goal[0]-px, goal[1]-py],dtype=np.float32)
    return {"spatial": spatial, "goal_relative": scalars}


def within_arrival_tiles(position_cell, goal_cell):
    return abs(int(position_cell[0])-int(goal_cell[0])) <= 1 and abs(int(position_cell[1])-int(goal_cell[1])) <= 1


class BestDistanceReward:
    """0.1 times new all-time absolute distance improvement plus arrival bonus."""
    def __init__(self, best_distance: float, progress_weight: float = 0.1, arrival_bonus: float = 10.0):
        self.best_distance=float(best_distance); self.progress_weight=float(progress_weight); self.arrival_bonus=float(arrival_bonus)

    def reset(self, position, goal):
        self.best_distance=math.dist(tuple(position),tuple(goal))

    def __call__(self, position, goal, arrived=False):
        distance=math.dist(tuple(position),tuple(goal))
        improvement=max(0.0,self.best_distance-distance)
        self.best_distance=min(self.best_distance,distance)
        terms={"distance_improvement":self.progress_weight*improvement,
               "arrival_bonus":self.arrival_bonus if arrived else 0.0}
        return sum(terms.values()),terms
