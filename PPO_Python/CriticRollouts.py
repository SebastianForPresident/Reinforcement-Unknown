"""Lossless, compact rollout storage for CB1 independent-critic training."""

from dataclasses import dataclass
import os
from pathlib import Path
import re
import tempfile

import numpy as np

import ObservationNormalization as ObsNorm
import Types


ARCHIVE_VERSION = 1
ROLLOUT_PATTERN = re.compile(r"critic_rollout_(\d+)_v(\d+)\.npz$")
WIDTH = Types.SIGHT_RANGE_X * 2 + 1
HEIGHT = Types.SIGHT_RANGE_Y * 2 + 1
TILE_COUNT = WIDTH * HEIGHT


def _as_single_env(array):
    """Remove SB3's singleton environment axis while preserving steps."""
    value = np.asarray(array)
    if value.ndim >= 2 and value.shape[1] == 1:
        return value[:, 0]
    return value


def _pack_spatial(spatial):
    spatial = np.asarray(spatial, dtype=np.float32)
    if spatial.shape[-3:] != (ObsNorm.SPATIAL_CHANNELS, WIDTH, HEIGHT):
        raise ValueError(f"Unexpected spatial rollout shape: {spatial.shape}")

    leading = spatial.shape[:-3]
    flat_extreme = spatial[..., 1, :, :].reshape(*leading, TILE_COUNT) > 0.5
    cursor_flat = spatial[..., 15, :, :].reshape(*leading, TILE_COUNT)
    cursor_index = np.argmax(cursor_flat, axis=-1).astype(np.uint16)
    cursor_present = np.any(cursor_flat > 0.5, axis=-1).astype(np.uint8)

    return {
        "spatial_health": spatial[..., 0, :, :].astype(np.float32),
        "spatial_extreme_bits": np.packbits(flat_extreme, axis=-1),
        "spatial_toxicity": spatial[..., 2, :, :].astype(np.float32),
        "spatial_sleep": np.argmax(spatial[..., 3:7, :, :], axis=-3).astype(np.uint8),
        "spatial_fluid": np.argmax(spatial[..., 7:14, :, :], axis=-3).astype(np.uint8),
        "spatial_cursor_index": cursor_index,
        "spatial_cursor_present": cursor_present,
    }


def _unpack_spatial(payload, prefix="spatial"):
    health = np.asarray(payload[f"{prefix}_health"], dtype=np.float32)
    leading = health.shape[:-2]
    spatial = np.zeros(
        (*leading, ObsNorm.SPATIAL_CHANNELS, WIDTH, HEIGHT),
        dtype=np.float32,
    )
    spatial[..., 0, :, :] = health

    bits = np.asarray(payload[f"{prefix}_extreme_bits"], dtype=np.uint8)
    extreme = np.unpackbits(bits, axis=-1, count=TILE_COUNT)
    spatial[..., 1, :, :] = extreme.reshape(*leading, WIDTH, HEIGHT)
    spatial[..., 2, :, :] = np.asarray(
        payload[f"{prefix}_toxicity"], dtype=np.float32
    )

    sleep = np.asarray(payload[f"{prefix}_sleep"], dtype=np.int64)
    fluid = np.asarray(payload[f"{prefix}_fluid"], dtype=np.int64)
    sleep_one_hot = np.eye(ObsNorm.SLEEP_QUALITY_COUNT, dtype=np.float32)[sleep]
    fluid_one_hot = np.eye(ObsNorm.FLUID_TYPE_COUNT, dtype=np.float32)[fluid]
    spatial[..., 3:7, :, :] = np.moveaxis(sleep_one_hot, -1, -3)
    spatial[..., 7:14, :, :] = np.moveaxis(fluid_one_hot, -1, -3)

    spatial[..., 14, Types.SIGHT_RANGE_X, Types.SIGHT_RANGE_Y] = 1.0
    cursor_index = np.asarray(payload[f"{prefix}_cursor_index"], dtype=np.int64)
    cursor_present = np.asarray(payload[f"{prefix}_cursor_present"], dtype=bool)
    cursor = spatial[..., 15, :, :].reshape(*leading, TILE_COUNT)
    for index in np.ndindex(leading):
        if cursor_present[index]:
            cursor[index + (cursor_index[index],)] = 1.0
    return spatial


def _prefixed_spatial(spatial, prefix):
    packed = _pack_spatial(spatial)
    return {
        key.replace("spatial", prefix, 1): value
        for key, value in packed.items()
    }


@dataclass
class CriticRollout:
    timestep: int
    observations: dict
    final_observation: dict
    rewards: np.ndarray
    episode_starts: np.ndarray
    dones: np.ndarray
    actions: np.ndarray
    old_values: np.ndarray
    old_log_probs: np.ndarray
    returns: np.ndarray


def save_rollout(path, timestep, rollout_buffer, final_observation, dones):
    """Atomically save one single-environment SB3 rollout."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    observations = {
        name: _as_single_env(value)
        for name, value in rollout_buffer.observations.items()
    }
    final = {
        name: np.asarray(value)[0]
        for name, value in final_observation.items()
    }

    payload = {
        "archive_version": np.asarray(ARCHIVE_VERSION, dtype=np.uint16),
        "timestep": np.asarray(timestep, dtype=np.uint64),
        "general": np.asarray(observations["general"], dtype=np.float32),
        "final_general": np.asarray(final["general"], dtype=np.float32),
        "rewards": _as_single_env(rollout_buffer.rewards).astype(np.float32),
        "episode_starts": _as_single_env(
            rollout_buffer.episode_starts
        ).astype(np.uint8),
        "dones": np.asarray(dones, dtype=np.uint8),
        "actions": _as_single_env(rollout_buffer.actions),
        "old_values": _as_single_env(rollout_buffer.values).astype(np.float32),
        "old_log_probs": _as_single_env(
            rollout_buffer.log_probs
        ).astype(np.float32),
        "returns": _as_single_env(rollout_buffer.returns).astype(np.float32),
        **_prefixed_spatial(observations["spatial"], "spatial"),
        **_prefixed_spatial(final["spatial"], "final_spatial"),
    }

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            # The compact categorical representation is already about an
            # order of magnitude smaller than dense 16-channel float32 input.
            # Store without DEFLATE so Unity's paused update is not dominated
            # by CPU compression latency.
            np.savez(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_rollout(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        version = int(payload["archive_version"])
        if version != ARCHIVE_VERSION:
            raise ValueError(
                f"Unsupported critic rollout version {version}: {path}"
            )
        observations = {
            "general": np.asarray(payload["general"], dtype=np.float32),
            "spatial": _unpack_spatial(payload, "spatial"),
        }
        final_observation = {
            "general": np.asarray(payload["final_general"], dtype=np.float32),
            "spatial": _unpack_spatial(payload, "final_spatial"),
        }
        return CriticRollout(
            timestep=int(payload["timestep"]),
            observations=observations,
            final_observation=final_observation,
            rewards=np.asarray(payload["rewards"], dtype=np.float32),
            episode_starts=np.asarray(payload["episode_starts"], dtype=np.float32),
            dones=np.asarray(payload["dones"], dtype=bool),
            actions=np.asarray(payload["actions"]),
            old_values=np.asarray(payload["old_values"], dtype=np.float32),
            old_log_probs=np.asarray(payload["old_log_probs"], dtype=np.float32),
            returns=np.asarray(payload["returns"], dtype=np.float32),
        )


class RolloutWindow:
    """Bounded on-disk window; old rollouts are removed after aging out."""

    def __init__(self, directory, max_rollouts=32):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_rollouts = int(max_rollouts)

    def paths(self):
        candidates = []
        for path in self.directory.glob("critic_rollout_*_v*.npz"):
            match = ROLLOUT_PATTERN.match(path.name)
            if match and int(match.group(2)) == ARCHIVE_VERSION:
                candidates.append((int(match.group(1)), path))
        return [path for _, path in sorted(candidates)]

    def add(self, timestep, rollout_buffer, final_observation, dones):
        path = self.directory / (
            f"critic_rollout_{int(timestep):012d}_v{ARCHIVE_VERSION}.npz"
        )
        save_rollout(path, timestep, rollout_buffer, final_observation, dones)
        expired = self.paths()[:-self.max_rollouts]
        for old_path in expired:
            old_path.unlink()
        return path
