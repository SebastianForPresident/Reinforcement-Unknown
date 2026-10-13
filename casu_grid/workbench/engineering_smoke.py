"""Explicit temporary engineering-only learning contract.

This is configuration scaffolding, not the user's selected Timmy experiment.
The short smoke objective exists only to exercise PPO plumbing.
"""
from casu_grid.workbench.contracts import (
    ActionContract, ActionHead, ModelContract, ObservationContract,
    ObservationField, PPOContract, RewardContract, RewardTerm,
)

ACTION = ActionContract(heads=(ActionHead("advance", size=2),), mapping={
    "advance": {"0": "stay", "1": "advance"},
})
OBSERVATION = ObservationContract(scalar=(
    ObservationField("player_state", (8,), "engineering.player_state", "custom"),
))
REWARD = RewardContract(terms=(
    RewardTerm("progress_delta", "engineering.progress_delta", 1.0),
    RewardTerm("goal", "engineering.goal_reached", 2.0, terminal=True),
), terminal_penalty=-0.25)
MODEL = ModelContract(spatial_channels=(), spatial_kernels=(), scalar_layers=(16,),
                      fusion_layers=(16,))
PPO = PPOContract(learning_rate=1e-3, rollout_steps=32, minibatch_size=16,
                  epochs=2, target_kl=None)


def callbacks():
    from casu_grid.isaac_navigation_env import engineering_navigation_callbacks
    return engineering_navigation_callbacks()
