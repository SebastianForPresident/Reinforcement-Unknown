"""SUPERSEDED EXPLORATORY VARIANT (termination ``xy1-termination-v1x-out-of-world-truncation``; kept only to reproduce the Research Studio V2 diagnostics).
Use ``xy1_point_navigation_mini_layer_exit`` (termination contract v2) instead.

EXPLORATORY VARIANT of the XY1 mini-world: an episode ends when the agent has left the world.

Identical to ``xy1_point_navigation_mini`` (same actions, observations, network, PPO, reward, goals, arrival rule, 100,000-decision ceiling, world
distribution) except for ONE termination rule: a running episode whose agent is outside the 128 x 128 world is *truncated* (reason ``out_of_world``,
no reward) and a fresh episode starts. In the experiment as defined, the floor can be dug through, nothing ends the episode of an agent that falls out of
the world, and in practice the whole population has fallen out within about 100 updates. This variant exists to measure what the early learning signal
looks like when the sample stream is not drained that way. It is NOT the validated XY1 contract and must not be used for the learning campaign without
an explicit decision to change the experiment.
"""
from casu_grid.workbench.experiments import xy1_point_navigation_mini as _base
from casu_grid.workbench.experiments.xy1_point_navigation_mini import ACTION, REWARD, MODEL, PPO, OBSERVATION, callbacks

EXPERIMENT = {
    **_base.EXPERIMENT,
    "id": "xy1-point-navigation-mini-128x128-guarded-v1",
    "name": "XY1 Mini World · episode ends when the agent leaves the world (SUPERSEDED exploratory variant)",
    "description": "The XY1 mini-world with one extra termination rule: an agent that left the 128 × 128 world ends its episode (truncation) instead of falling forever.",
    "hypothesis": "Without agents lost to free fall, the early PPO learning signal on XY1 is measurable.",
    "notes": "EXPLORATORY: differs from the validated XY1 contract in its termination rule only. Parent: xy1-point-navigation-mini-128x128-v1.",
    "parent": "xy1-point-navigation-mini-128x128-v1",
}

ENVIRONMENT = {**_base.ENVIRONMENT, "terminate_out_of_world": True}
