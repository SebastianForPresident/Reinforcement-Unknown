"""PPO with a critic-free, observed-reward lookahead alternative."""

import numpy as np
from stable_baselines3 import PPO

from Train import PausingPPO


REINFORCE_MODES = ("shadow", "active")


def observed_returns(rewards, episode_starts, gamma, lookahead):
    """Return discounted rewards through `lookahead` future transitions.

    The current reward plus the following `lookahead` rewards are observed.
    Returns never cross an episode reset.  With gamma=.99 and lookahead=458,
    the first omitted reward has weight gamma**459 < .01.
    """
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    episode_starts = np.asarray(episode_starts, dtype=bool).reshape(-1)
    train_steps = len(rewards) - int(lookahead)
    if train_steps <= 0:
        raise ValueError("lookahead must be shorter than the collected rollout")

    returns = np.empty(train_steps, dtype=np.float32)
    horizon = int(lookahead) + 1
    for start in range(train_steps):
        value = 0.0
        discount = 1.0
        stop = min(start + horizon, len(rewards))
        for index in range(start, stop):
            if index > start and episode_starts[index]:
                break
            value += discount * float(rewards[index])
            discount *= gamma
        returns[start] = value
    return returns


def gae_prefix(buffer, train_steps, gamma, gae_lambda):
    """Recreate ordinary GAE at the prefix boundary of a longer rollout."""
    rewards = np.asarray(buffer.rewards[:, 0], dtype=np.float32)
    values = np.asarray(buffer.values[:, 0], dtype=np.float32)
    starts = np.asarray(buffer.episode_starts[:, 0], dtype=bool)
    advantages = np.zeros(train_steps, dtype=np.float32)
    last_gae = 0.0
    for step in reversed(range(train_steps)):
        next_non_terminal = 1.0 - float(starts[step + 1])
        next_value = float(values[step + 1])
        delta = (
            float(rewards[step])
            + gamma * next_value * next_non_terminal
            - float(values[step])
        )
        last_gae = (
            delta
            + gamma * gae_lambda * next_non_terminal * last_gae
        )
        advantages[step] = last_gae
    return advantages, advantages + values[:train_steps]


class ReinforceLookaheadPPO(PausingPPO):
    """Collect a frozen-policy lookahead and exclude it from PPO training."""

    def __init__(
        self,
        *args,
        reinforce_mode="shadow",
        train_steps=2048,
        lookahead_steps=458,
        **kwargs,
    ):
        if reinforce_mode not in REINFORCE_MODES:
            raise ValueError(f"REINFORCE mode must be one of {REINFORCE_MODES}")
        self.reinforce_mode = reinforce_mode
        self.reinforce_train_steps = int(train_steps)
        self.reinforce_lookahead_steps = int(lookahead_steps)
        expected_steps = self.reinforce_train_steps + self.reinforce_lookahead_steps
        kwargs["n_steps"] = expected_steps
        super().__init__(*args, **kwargs)

    def train(self):
        if self._pause_simulation is not None:
            self._pause_simulation()

        buffer = self.rollout_buffer
        train_steps = self.reinforce_train_steps
        if buffer.buffer_size != train_steps + self.reinforce_lookahead_steps:
            raise RuntimeError("REINFORCE rollout buffer has the wrong length")

        reinforce_returns = observed_returns(
            buffer.rewards[:, 0],
            buffer.episode_starts[:, 0],
            self.gamma,
            self.reinforce_lookahead_steps,
        )
        # A scalar, action-independent baseline keeps this diagnostic entirely
        # critic-free. PPO will normalize the active advantages per minibatch.
        reinforce_advantages = reinforce_returns - np.mean(reinforce_returns)
        gae_advantages, gae_returns = gae_prefix(
            buffer, train_steps, self.gamma, self.gae_lambda
        )

        if np.std(reinforce_advantages) > 0 and np.std(gae_advantages) > 0:
            correlation = float(np.corrcoef(
                reinforce_advantages, gae_advantages
            )[0, 1])
        else:
            correlation = np.nan
        centered_gae = gae_advantages - np.mean(gae_advantages)
        sign_agreement = float(np.mean(
            np.sign(reinforce_advantages) == np.sign(centered_gae)
        ))
        self.logger.record("reinforce/advantage_correlation", correlation)
        self.logger.record("reinforce/advantage_sign_agreement", sign_agreement)
        self.logger.record(
            "reinforce/return_mean", float(np.mean(reinforce_returns))
        )
        self.logger.record(
            "reinforce/return_std", float(np.std(reinforce_returns))
        )
        self.logger.record(
            "reinforce/reward_nonzero_fraction",
            float(np.mean(np.asarray(buffer.rewards[:train_steps, 0]) != 0)),
        )

        if self.reinforce_mode == "active":
            buffer.advantages[:train_steps, 0] = reinforce_returns
            buffer.returns[:train_steps, 0] = reinforce_returns
        else:
            buffer.advantages[:train_steps, 0] = gae_advantages
            buffer.returns[:train_steps, 0] = gae_returns

        # SB3 samples only this prefix. The lookahead remains observed context
        # and can never acquire a policy/value gradient in this update.
        original_size = buffer.buffer_size
        original_vf_coef = self.vf_coef
        buffer.buffer_size = train_steps
        buffer.full = True
        buffer.generator_ready = False
        if self.reinforce_mode == "active":
            self.vf_coef = 0.0
        try:
            # Bypass PausingPPO.train(): Unity was already paused above.
            PPO.train(self)
        finally:
            self.vf_coef = original_vf_coef
            buffer.buffer_size = original_size
