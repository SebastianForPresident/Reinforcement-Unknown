"""User-editable learning contracts, independent of simulator internals.

These defaults are deliberately labeled engineering placeholders. They exist
to validate configuration, tensor shapes, model construction, and PPO plumbing;
they are not an approved Timmy action, observation, reward, or experiment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ActionHead:
    name: str
    kind: str = "categorical"  # categorical or continuous
    size: int = 3
    low: float = -1.0
    high: float = 1.0
    hold_physics_ticks: int = 10

    def validate(self) -> None:
        if not self.name:
            raise ValueError("action head name must not be empty")
        if self.kind not in {"categorical", "continuous"}:
            raise ValueError(f"unsupported action head kind: {self.kind}")
        if self.kind == "categorical" and self.size < 2:
            raise ValueError(f"categorical action {self.name} must have at least 2 values")
        if self.kind == "continuous" and not self.low < self.high:
            raise ValueError(f"continuous action {self.name} needs low < high")
        if self.hold_physics_ticks < 1:
            raise ValueError(f"action {self.name} hold duration must be positive")


@dataclass(frozen=True)
class ActionContract:
    heads: tuple[ActionHead, ...] = (
        ActionHead("move_x", size=3),
        ActionHead("jump", size=2),
        ActionHead("crouch", size=2),
    )
    mapping: dict[str, Any] = field(default_factory=lambda: {
        "move_x": {"0": -1, "1": 0, "2": 1},
        "jump": {"0": False, "1": True},
        "crouch": {"0": False, "1": True},
    })


@dataclass(frozen=True)
class ObservationField:
    name: str
    shape: tuple[int, ...]
    source: str
    normalization: str = "none"
    normalization_params: dict[str, Any] = field(default_factory=dict)
    mask: str | None = None

    def validate(self) -> None:
        if not self.name or not self.source:
            raise ValueError("observation fields require name and source")
        if not self.shape or any(int(d) <= 0 for d in self.shape):
            raise ValueError(f"observation {self.name} has invalid shape {self.shape}")
        if self.normalization not in {"none", "unit", "standardize", "custom"}:
            raise ValueError(f"unsupported normalization: {self.normalization}")
        if self.normalization == "unit":
            low = self.normalization_params.get("low", 0.0)
            high = self.normalization_params.get("high", 1.0)
            if isinstance(low, (int, float)) and isinstance(high, (int, float)) and low >= high:
                raise ValueError(f"unit normalization for {self.name} needs low < high")
        if self.normalization == "standardize":
            if "mean" not in self.normalization_params or "std" not in self.normalization_params:
                raise ValueError(f"standardize normalization for {self.name} needs mean and std")
            std = self.normalization_params["std"]
            if isinstance(std, (int, float)) and std <= 0:
                raise ValueError(f"standardize normalization for {self.name} needs std > 0")


@dataclass(frozen=True)
class ObservationContract:
    spatial: tuple[ObservationField, ...] = ()
    scalar: tuple[ObservationField, ...] = (
        ObservationField("player_state", (8,), "engineering.player_state", "custom"),
    )
    coordinate_convention: str = "world-x-right, world-y-up; grids x-first"


@dataclass(frozen=True)
class RewardTerm:
    name: str
    source: str
    weight: float
    terminal: bool = False


@dataclass(frozen=True)
class RewardContract:
    terms: tuple[RewardTerm, ...] = (
        RewardTerm("engineering_progress", "engineering.goal_progress", 1.0),
        RewardTerm("engineering_terminal", "engineering.goal_reached", 1.0, True),
    )
    terminal_penalty: float = 0.0


@dataclass(frozen=True)
class ModelContract:
    spatial_channels: tuple[int, ...] = (16, 32)
    spatial_kernels: tuple[int, ...] = (5, 3)
    scalar_layers: tuple[int, ...] = (64,)
    fusion_layers: tuple[int, ...] = (128,)
    recurrent: bool = False
    recurrent_size: int = 128


@dataclass(frozen=True)
class PPOContract:
    version: int = 1
    learning_rate: float = 3e-4
    rollout_steps: int = 256
    minibatch_size: int = 64
    epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03


def validate_contracts(actions: ActionContract, observations: ObservationContract,
                       rewards: RewardContract, model: ModelContract,
                       ppo: PPOContract) -> dict[str, Any]:
    if not actions.heads:
        raise ValueError("at least one action head is required")
    if len({head.name for head in actions.heads}) != len(actions.heads):
        raise ValueError("action head names must be unique")
    for head in actions.heads:
        head.validate()
    fields = observations.spatial + observations.scalar
    if not fields:
        raise ValueError("at least one observation field is required")
    if len({item.name for item in fields}) != len(fields):
        raise ValueError("observation field names must be unique")
    for item in fields:
        item.validate()
    fields_by_name = {item.name: item for item in fields}
    for item in observations.spatial:
        if item.mask is None:
            continue
        mask = fields_by_name.get(item.mask)
        if mask is None:
            raise ValueError(f"spatial observation {item.name} refers to missing mask {item.mask}")
        if len(mask.shape) not in {2, 3} or tuple(mask.shape[-2:]) != tuple(item.shape[-2:]):
            raise ValueError(f"mask {item.mask} spatial dimensions do not match {item.name}")
        if len(mask.shape) == 3 and mask.shape[0] not in {1, item.shape[0]}:
            raise ValueError(f"mask {item.mask} channels must be 1 or match {item.name}")
    if len({term.name for term in rewards.terms}) != len(rewards.terms):
        raise ValueError("reward term names must be unique")
    if any(not term.name or not term.source for term in rewards.terms):
        raise ValueError("reward terms require names and source mappings")
    if len(model.spatial_channels) != len(model.spatial_kernels):
        raise ValueError("each spatial encoder layer needs a kernel size")
    if any(x <= 0 for x in model.spatial_channels + model.spatial_kernels +
           model.scalar_layers + model.fusion_layers):
        raise ValueError("model layer dimensions must be positive")
    if model.recurrent and model.recurrent_size <= 0:
        raise ValueError("recurrent size must be positive when recurrence is enabled")
    if ppo.version < 1 or ppo.learning_rate <= 0:
        raise ValueError("PPO config version and learning rate must be positive")
    if min(ppo.rollout_steps, ppo.minibatch_size, ppo.epochs) <= 0:
        raise ValueError("PPO rollout, minibatch, and epoch counts must be positive")
    if not 0 <= ppo.gamma <= 1 or not 0 <= ppo.gae_lambda <= 1:
        raise ValueError("PPO gamma and GAE lambda must be in [0, 1]")
    if not 0 < ppo.clip_range < 1:
        raise ValueError("PPO clip range must be in (0, 1)")
    if ppo.target_kl is not None and ppo.target_kl <= 0:
        raise ValueError("target KL must be positive or None")
    return {
        "actions": asdict(actions),
        "observations": asdict(observations),
        "rewards": asdict(rewards),
        "model": asdict(model),
        "ppo": asdict(ppo),
    }
