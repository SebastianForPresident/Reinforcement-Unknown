"""Independent CB1 critic and recent-rollout supervised training utilities."""

from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class IndependentCritic(nn.Module):
    """A value network initialized from CB1 without sharing actor parameters."""

    def __init__(self, policy):
        super().__init__()
        self.features_extractor = deepcopy(policy.features_extractor)
        self.value_latent = deepcopy(policy.mlp_extractor.value_net)
        self.value_head = deepcopy(policy.value_net)

    def forward(self, observations):
        features = self.features_extractor(observations)
        latent = self.value_latent(features)
        return self.value_head(latent).flatten()


def observation_batch(observations, indices, device):
    return {
        name: torch.as_tensor(value[indices], device=device)
        for name, value in observations.items()
    }


@torch.no_grad()
def predict(critic, observations, device, batch_size=128):
    critic.eval()
    count = len(observations["general"])
    predictions = []
    for start in range(0, count, batch_size):
        indices = slice(start, min(start + batch_size, count))
        predictions.append(
            critic(observation_batch(observations, indices, device))
            .detach().cpu().numpy()
        )
    return np.concatenate(predictions).astype(np.float32, copy=False)


def explained_variance(predictions, targets):
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    variance = np.var(targets)
    if variance <= 1e-12:
        return np.nan
    return float(1.0 - np.var(targets - predictions) / variance)


def evaluate(critic, rollout, device, batch_size=128):
    predictions = predict(critic, rollout.observations, device, batch_size)
    targets = rollout.returns
    return {
        "loss": float(np.mean(np.square(predictions - targets))),
        "explained_variance": explained_variance(predictions, targets),
        "shared_loss": float(np.mean(np.square(rollout.old_values - targets))),
        "shared_explained_variance": explained_variance(
            rollout.old_values, targets
        ),
    }


def train_rollout(
    critic,
    optimizer,
    rollout,
    device,
    batch_size=64,
    epochs=1,
    max_grad_norm=0.5,
    rng=None,
):
    """Fit fixed TD(lambda) targets retained from the generating rollout."""
    critic.train()
    rng = np.random.default_rng() if rng is None else rng
    sample_count = len(rollout.returns)
    losses = []
    for _ in range(int(epochs)):
        order = rng.permutation(sample_count)
        for start in range(0, sample_count, batch_size):
            indices = order[start:start + batch_size]
            observations = observation_batch(rollout.observations, indices, device)
            targets = torch.as_tensor(
                rollout.returns[indices], dtype=torch.float32, device=device
            )
            predicted = critic(observations)
            loss = F.mse_loss(predicted, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else np.nan

