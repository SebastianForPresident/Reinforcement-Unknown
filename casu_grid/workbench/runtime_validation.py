"""Validate researcher callback outputs at the environment boundary."""
from __future__ import annotations

import math
from numbers import Real
from typing import Callable

import numpy as np


def checked_callbacks(callbacks, observations, rewards):
    """Wrap user observation/reward callbacks with contract-aware error messages."""
    if callbacks is None:
        callbacks = (None, None, None)
    if not isinstance(callbacks, tuple) or len(callbacks) != 3:
        raise ValueError("callbacks() must return (action_decoder, observation_encoder, reward_function)")
    action_decoder, observation_encoder, reward_function = callbacks
    fields = observations.spatial + observations.scalar

    if action_decoder is not None:
        decode_callback = action_decoder
        def decode(actions):
            try:
                value = decode_callback(actions)
            except Exception as exc:
                raise ValueError(f"callbacks.action_decoder failed: {exc}") from exc
            required = ("move_x", "move_y", "jump", "crouch")
            if any(not hasattr(value, name) for name in required):
                raise ValueError("callbacks.action_decoder must return an action with move_x, move_y, jump, and crouch fields")
            for name in ("move_x", "move_y"):
                if getattr(value, name) not in (-1, 0, 1):
                    raise ValueError(f"callbacks.action_decoder field {name} must be -1, 0, or 1")
            for name in ("jump", "crouch"):
                if getattr(value, name) not in (False, True, 0, 1):
                    raise ValueError(f"callbacks.action_decoder field {name} must be boolean or 0/1")
            return value
        action_decoder = decode

    if observation_encoder is not None:
        encoder_callback = observation_encoder
        def observe(env):
            try:
                value = encoder_callback(env)
            except Exception as exc:
                raise ValueError(f"callbacks.observation_encoder failed: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError("callbacks.observation_encoder must return a dict keyed by observation field")
            expected = {field.name: field.shape for field in fields}
            if set(value) != set(expected):
                raise ValueError(f"callbacks.observation_encoder keys {sorted(value)} do not match contract {sorted(expected)}")
            checked = {}
            for name, shape in expected.items():
                try:
                    array = np.asarray(value[name], dtype=np.float32)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"callbacks.observation_encoder field {name} is not numeric: {exc}") from exc
                if tuple(array.shape) != tuple(shape):
                    raise ValueError(f"callbacks.observation_encoder field {name} has shape {array.shape}; expected {shape}")
                if not np.isfinite(array).all():
                    raise ValueError(f"callbacks.observation_encoder field {name} contains non-finite values")
                checked[name] = array
            return checked
        observation_encoder = observe

    if reward_function is not None:
        reward_callback = reward_function
        expected_terms = {term.name for term in rewards.terms}
        def reward(previous, current, terminated, truncated):
            try:
                value = reward_callback(previous, current, terminated, truncated)
            except Exception as exc:
                raise ValueError(f"callbacks.reward_function failed: {exc}") from exc
            if not isinstance(value, tuple) or len(value) != 2:
                raise ValueError("callbacks.reward_function must return (scalar_reward, reward_terms_dict)")
            scalar, terms = value
            if not isinstance(scalar, Real) or not math.isfinite(float(scalar)):
                raise ValueError("callbacks.reward_function scalar reward must be a finite number")
            if not isinstance(terms, dict):
                raise ValueError("callbacks.reward_function reward terms must be a dict")
            if set(terms) != expected_terms:
                raise ValueError(f"callbacks.reward_function reward term keys {sorted(terms)} do not match contract {sorted(expected_terms)}")
            normalized = {}
            for name, term_value in terms.items():
                if not isinstance(term_value, Real) or not math.isfinite(float(term_value)):
                    raise ValueError(f"callbacks.reward_function term {name} must be a finite scalar number")
                normalized[name] = float(term_value)
            return float(scalar), normalized
        reward_function = reward
    return action_decoder, observation_encoder, reward_function
