"""XY1 mini-world, TERMINATION CONTRACT v2: leaving through the bottom of the layer ends the episode.

Decision of the researcher: when Timmy exits through the bottom of the 128 x 128 mini-world, the navigation episode terminates and the world, Timmy and the goal are
re-initialised. This models CasU's bottom-of-layer transition (WorldGeneration.cs: ``body.y < -halfHeight + 3.1``, i.e. y < -60.9 in a world of height 128); it is not
player death. Identical to ``xy1_point_navigation_mini`` (the reproducible reference, termination contract ``xy1-termination-v1``) in every respect except the termination rule:

* the bottom exit is a TRUE terminal transition (``terminated = True``, reason ``layer_exit``): PPO does not bootstrap through it, unlike a time-limit truncation;
* it adds no reward and no penalty (reward formula, arrival bonus and the 100,000-decision truncation are unchanged);
* event precedence in a decision: goal arrival, then layer exit, then the decision ceiling;
* the floor stays destructible.

Checkpoints trained under this contract carry ``xy1-termination-v2-layer-exit`` and cannot be resumed under any other termination contract.
"""
from casu_grid.workbench.experiments import xy1_point_navigation_mini as _base
from casu_grid.workbench.experiments.xy1_point_navigation_mini import ACTION, REWARD, MODEL, PPO, OBSERVATION, callbacks

EXPERIMENT = {
    **_base.EXPERIMENT,
    "id": "xy1-point-navigation-mini-128x128-layer-exit-v2",
    "name": "XY1 Point Navigation · Mini World 128 × 128 · bottom exit ends the episode (termination v2)",
    "description": "The XY1 mini-world where exiting through the bottom of the layer (CasU's layer transition) terminates the episode and starts a freshly initialised world, Timmy and goal.",
    "hypothesis": "With the bottom-of-layer transition modelled as a terminal event, the sample stream stays populated and the XY1 navigation policy can be trained.",
    "notes": "Termination contract xy1-termination-v2-layer-exit. Differs from the reference xy1-point-navigation-mini-128x128-v1 in the termination rule only. Parent: xy1-point-navigation-mini-128x128-v1.",
    "parent": "xy1-point-navigation-mini-128x128-v1",
    "config_version": 2,
}

ENVIRONMENT = {**_base.ENVIRONMENT, "bottom_exit_terminates": True}
