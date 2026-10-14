"""XY1 128 x 128 mini-world, retaining the canonical XY1 learning contracts."""
from casu_grid.workbench.experiments.xy1_point_navigation import (
    ACTION, REWARD, MODEL, PPO, callbacks,
)
from casu_grid.workbench.contracts import ObservationContract, ObservationField

EXPERIMENT = {
    "id": "xy1-point-navigation-mini-128x128-v1",
    "name": "XY1 Point Navigation · Mini World 128 × 128",
    "description": "XY1 contracts in an independently randomized 128 × 128 CasU-style mini-world.",
    "hypothesis": "The same XY1 navigation policy and reward contracts apply to a compact randomized world.",
    "notes": "Separate environment variant. Center spawn; no pod or chamber; open top; noisy Infinirock sides. 64 × 64 FOV is provisional.",
    "parent": "xy1-point-navigation-v1",
    "created_utc": "2026-10-13T00:00:00+00:00",
    "config_version": 1,
}

OBSERVATION = ObservationContract(
    spatial=(ObservationField("spatial",(16,64,64),"xy1.historical_cb1_terrain_fluid","none"),),
    scalar=(ObservationField("goal_relative",(2,),"xy1.goal_tile_offset","custom",
        normalization_params={"convention":"unscaled signed tile offsets; x right positive, y up positive"}),),
    coordinate_convention="world-x-right, world-y-up; tensor axes (x,y); mini FOV 64 × 64; goals are sampled once in that viewport and remain fixed in world coordinates")

ENVIRONMENT = {
    "navigation_mode": True,
    "world_variant": "xy1_mini_128x128",
    "episode_length_steps": 100000,
    "player_collider": "compound",
    "num_envs": 1,
    "terrain_mode": "independent",
    "observation_window_tiles": [64, 64],
    "physics_dt": 0.02,
    "action_decimation": 10,
    "goal_window_tiles": [85, 49],
    "spawn": "world center with only full-body clearance; zero initial velocity",
    "arrival": "reference tile or eight adjacent tiles",
}
