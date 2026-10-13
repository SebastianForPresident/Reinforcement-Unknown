# XY1 — Experiment 1: goal-conditioned point navigation
# Horizon 100,000 decisions is provisional; exhaustion is an unsuccessful timeout.
EXPERIMENT = {
    "id": "xy1-point-navigation-v1", "name": "XY1 Point Navigation",
    "description": "Goal-conditioned Timmy point navigation with historical CB1 spatial channels.",
    "hypothesis": "A modest PPO policy can learn goal-directed movement from local terrain and relative destination offsets.",
    "notes": "Navigation only. Compound collider. No world-top cap. 100,000-decision technical ceiling.",
    "parent": None, "created_utc": "2026-10-13T00:00:00+00:00", "config_version": 1,
}

from casu_grid.workbench.contracts import (ActionContract, ActionHead, ObservationContract,
    ObservationField, RewardContract, RewardTerm, ModelContract, PPOContract)

ACTION = ActionContract(heads=(
    ActionHead("move_x",size=3), ActionHead("jump",size=2), ActionHead("move_y",size=3),
    ActionHead("crouch",size=2), ActionHead("look_dx",size=9), ActionHead("look_dy",size=11),
    ActionHead("attack",size=2),
), mapping={"move_x":{"0":-1,"1":0,"2":1},"jump":{"0":False,"1":True},
"move_y":{"0":-1,"1":0,"2":1},"crouch":{"0":False,"1":True},
"look_dx":{"0":-4,"1":-3,"2":-2,"3":-1,"4":0,"5":1,"6":2,"7":3,"8":4},
"look_dy":{"0":-5,"1":-4,"2":-3,"3":-2,"4":-1,"5":0,"6":1,"7":2,"8":3,"9":4,"10":5},
"attack":{"0":False,"1":True}})
OBSERVATION = ObservationContract(
    spatial=(ObservationField("spatial",(16,85,49),"xy1.historical_cb1_terrain_fluid","none"),),
    scalar=(ObservationField("goal_relative",(2,),"xy1.goal_tile_offset","custom",
        normalization_params={"convention":"unscaled signed tile offsets; x right positive, y up positive"}),),
    coordinate_convention="world-x-right, world-y-up; tensor axes (x,y); goal offsets are destination tile minus player tile in tiles")
REWARD = RewardContract(terms=(RewardTerm("distance_improvement","xy1.best_absolute_distance_delta",0.1),
                               RewardTerm("arrival_bonus","xy1.nine_tile_arrival",10.0,True)))
MODEL = ModelContract(spatial_channels=(16,32),spatial_kernels=(5,3),scalar_layers=(16,),fusion_layers=(64,))
PPO = PPOContract(learning_rate=3e-4,rollout_steps=64,minibatch_size=32,epochs=2,
                  gamma=0.99,gae_lambda=0.98,target_kl=0.03)
ENVIRONMENT = {"navigation_mode":True,"episode_length_steps":100000,"player_collider":"compound",
              "num_envs":1,"terrain_mode":"independent",
              "physics_dt":0.02,"action_decimation":10,"goal_window_tiles":[85,49],
              "spawn":"uniform valid tile centers; full standing collider clearance; zero initial velocity",
              "arrival":"reference tile or eight adjacent tiles"}

def callbacks():
    from casu_grid.xy1_callbacks import observation_encoder, reward_function
    return None, observation_encoder, reward_function
