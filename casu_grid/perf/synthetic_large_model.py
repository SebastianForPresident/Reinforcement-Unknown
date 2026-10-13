"""SYNTHETIC inference-load configuration -- NOT an approved Timmy architecture.

Same engineering-only contracts as ``workbench/engineering_smoke.py`` but with a much larger
policy/value network, used only to show how training throughput changes when the neural network
(rather than the simulator) becomes the bottleneck. Never read results from it as research.
"""
from casu_grid.workbench.contracts import ModelContract
from casu_grid.workbench.engineering_smoke import ACTION, OBSERVATION, PPO, REWARD, callbacks   # noqa: F401

MODEL = ModelContract(spatial_channels=(), spatial_kernels=(), scalar_layers=(512, 512, 512),
                      fusion_layers=(512, 512))
