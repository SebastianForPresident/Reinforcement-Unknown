"""Configurable actor-critic network for categorical/continuous action heads."""
from __future__ import annotations

import math
import os

import torch
from torch import nn
from torch.distributions import Categorical, Normal

from .contracts import ActionContract, ModelContract, ObservationContract


def _validate_args() -> bool:
    """Distribution argument validation costs ~20 us per construction and only turns bad inputs into a different
    error (non-finite values are still caught by the sampler / loss checks); kept on for the reference path."""
    return os.environ.get("CASU_PPO_EXECUTION", "optimized") == "reference"


def _mlp(input_size: int, hidden: tuple[int, ...]) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    width = input_size
    for next_width in hidden:
        layers.extend((nn.Linear(width, next_width), nn.Tanh()))
        width = next_width
    return nn.Sequential(*layers), width


def _normalize_observation(field, value: torch.Tensor) -> torch.Tensor:
    """Apply the explicit scalar/range transforms declared by a field."""
    params = field.normalization_params
    if field.normalization in {"none", "custom"}:
        # Custom normalization is performed by the research config's encoder
        # callback before this model receives the tensor.
        return value
    if field.normalization == "unit":
        low = torch.as_tensor(params.get("low", 0.0), dtype=value.dtype, device=value.device)
        high = torch.as_tensor(params.get("high", 1.0), dtype=value.dtype, device=value.device)
        if torch.any(high <= low):
            raise ValueError(f"unit normalization for {field.name} needs elementwise low < high")
        return (value - low) / (high - low)
    if field.normalization == "standardize":
        mean = torch.as_tensor(params["mean"], dtype=value.dtype, device=value.device)
        std = torch.as_tensor(params["std"], dtype=value.dtype, device=value.device)
        if torch.any(std <= 0):
            raise ValueError(f"standardize normalization for {field.name} needs std > 0")
        return (value - mean) / std
    raise ValueError(f"unsupported normalization: {field.normalization}")


class ConfigurableActorCritic(nn.Module):
    """Separate spatial/scalar encoders with configurable fusion and PPO heads."""

    def __init__(self, actions: ActionContract, observations: ObservationContract,
                 model: ModelContract):
        super().__init__()
        self.action_contract = actions
        self.observation_contract = observations
        self.spatial_fields = observations.spatial
        self.scalar_fields = observations.scalar
        self.spatial_encoder = nn.Identity()
        spatial_size = 0
        if self.spatial_fields:
            shapes = [item.shape for item in self.spatial_fields]
            if any(len(shape) != 3 for shape in shapes):
                raise ValueError("spatial observation shapes must be (channels, height, width)")
            if len({shape[1:] for shape in shapes}) != 1:
                raise ValueError("spatial fields must share height and width for fusion")
            in_channels = sum(shape[0] for shape in shapes)
            layers: list[nn.Module] = []
            width = in_channels
            height, width_px = shapes[0][1:]
            for out_channels, kernel in zip(model.spatial_channels, model.spatial_kernels):
                padding = kernel // 2
                layers.extend((nn.Conv2d(width, out_channels, kernel, padding=padding),
                               nn.ReLU()))
                width = out_channels
            self.spatial_encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
            spatial_size = width
        scalar_size = sum(math.prod(item.shape) for item in self.scalar_fields)
        self.scalar_encoder, scalar_out = _mlp(scalar_size, model.scalar_layers)
        self.fusion, fusion_size = _mlp(spatial_size + scalar_out, model.fusion_layers)
        if fusion_size == 0:
            fusion_size = spatial_size + scalar_out
        self.recurrent = model.recurrent
        self.recurrent_size = model.recurrent_size
        self.fusion_size = fusion_size
        self.recurrent_layer = (nn.GRU(fusion_size, model.recurrent_size, batch_first=True)
                                if model.recurrent else None)
        if model.recurrent:
            fusion_size = model.recurrent_size
        self.actor_heads = nn.ModuleDict()
        self.log_stds = nn.ParameterDict()
        for head in actions.heads:
            output_size = head.size
            self.actor_heads[head.name] = nn.Linear(fusion_size, output_size)
            if head.kind == "continuous":
                self.log_stds[head.name] = nn.Parameter(torch.zeros(head.size))
        self.critic = nn.Linear(fusion_size, 1)

    def features(self, obs: dict[str, torch.Tensor], *, sequence_length: int | None = None,
                 initial_state: torch.Tensor | None = None,
                 return_state: bool = False):
        features = []
        if self.spatial_fields:
            values = self._prepare_spatial_values(obs)
            if any(x.ndim != 4 for x in values):
                raise ValueError("batched spatial observation must have shape (batch, C, H, W)")
            features.append(self.spatial_encoder(torch.cat(values, dim=1)))
        if self.scalar_fields:
            values = []
            for field in self.scalar_fields:
                value = _normalize_observation(field, obs[field.name].float())
                values.append(value.reshape(value.shape[0], -1))
            features.append(self.scalar_encoder(torch.cat(values, dim=1)))
        result = torch.cat(features, dim=1)
        encoded = self.fusion(result)
        if not self.recurrent:
            return (encoded, None) if return_state else encoded
        if sequence_length is None or sequence_length < 1 or encoded.shape[0] % sequence_length:
            raise ValueError("recurrent features need a valid sequence_length dividing the batch")
        batch = encoded.shape[0] // sequence_length
        sequence = encoded.reshape(batch, sequence_length, self.fusion_size)
        output, state = self.recurrent_layer(sequence, initial_state)
        output = output.reshape(-1, self.recurrent_size)
        return (output, state) if return_state else output

    def _prepare_spatial_values(self, obs: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        values = []
        for field in self.spatial_fields:
            value = _normalize_observation(field, obs[field.name].float())
            if field.mask is not None:
                mask = obs[field.mask].float()
                if mask.ndim == 3:
                    mask = mask.unsqueeze(1)
                try:
                    value = value * mask
                except RuntimeError as exc:
                    raise ValueError(
                        f"mask {field.mask} cannot broadcast to spatial observation {field.name}"
                    ) from exc
            values.append(value)
        if any(value.ndim != 4 for value in values):
            raise ValueError("batched spatial observation must have shape (batch, C, H, W)")
        return values

    def distributions(self, obs: dict[str, torch.Tensor], *, sequence_length: int | None = None,
                      initial_state: torch.Tensor | None = None, return_state: bool = False):
        feature_result = self.features(obs, sequence_length=sequence_length,
                                       initial_state=initial_state,
                                       return_state=return_state)
        if return_state:
            features, state = feature_result
        else:
            features, state = feature_result, None
        distributions = {}
        for head in self.action_contract.heads:
            if head.kind == "categorical":
                distributions[head.name] = Categorical(logits=self.actor_heads[head.name](features), validate_args=_validate_args())
            else:
                mean = self.actor_heads[head.name](features)
                std = self.log_stds[head.name].exp().expand_as(mean)
                distributions[head.name] = Normal(mean, std, validate_args=_validate_args())
        values = self.critic(features).squeeze(-1)
        return (distributions, values, state) if return_state else (distributions, values)

    def act(self, obs: dict[str, torch.Tensor], deterministic: bool = False,
            return_latent: bool = False, *, sequence_length: int | None = None,
            initial_state: torch.Tensor | None = None, return_state: bool = False):
        result = self.distributions(obs, sequence_length=sequence_length,
                                    initial_state=initial_state, return_state=return_state)
        if return_state:
            distributions, value, next_state = result
        else:
            distributions, value = result
            next_state = None
        actions, latent_actions, log_prob = {}, {}, None
        for name, distribution in distributions.items():
            head = next(h for h in self.action_contract.heads if h.name == name)
            latent = (distribution.probs.argmax(dim=-1) if deterministic and
                      isinstance(distribution, Categorical) else
                      distribution.mean if deterministic else distribution.sample())
            latent_actions[name] = latent
            if head.kind == "continuous":
                squashed = torch.tanh(latent)
                action = head.low + (squashed + 1.0) * (head.high - head.low) * 0.5
                lp = distribution.log_prob(latent) - torch.log1p(-squashed.square() + 1e-6)
            else:
                action = latent
                lp = distribution.log_prob(latent)
            actions[name] = action
            if lp.ndim > 1:
                lp = lp.sum(-1)
            log_prob = lp if log_prob is None else log_prob + lp
        if return_latent:
            base = (actions, log_prob, value, latent_actions)
            return base + (next_state,) if return_state else base
        if return_state:
            return actions, log_prob, value, next_state
        return actions, log_prob, value

    def evaluate_actions(self, obs: dict[str, torch.Tensor], actions: dict[str, torch.Tensor],
                         *, sequence_length: int | None = None,
                         initial_state: torch.Tensor | None = None):
        distributions, value = self.distributions(obs, sequence_length=sequence_length,
                                                  initial_state=initial_state)
        log_prob, entropy = None, None
        for name, distribution in distributions.items():
            lp = distribution.log_prob(actions[name])
            ent = distribution.entropy()
            head = next(h for h in self.action_contract.heads if h.name == name)
            if head.kind == "continuous":
                squashed = torch.tanh(actions[name])
                lp = lp - torch.log1p(-squashed.square() + 1e-6)
            if lp.ndim > 1:
                lp, ent = lp.sum(-1), ent.sum(-1)
            log_prob = lp if log_prob is None else log_prob + lp
            entropy = ent if entropy is None else entropy + ent
        return log_prob, entropy, value


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
