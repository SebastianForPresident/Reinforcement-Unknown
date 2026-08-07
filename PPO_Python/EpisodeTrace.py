"""Episode-level CSV tracing for reward and action diagnostics."""

import csv
import json
import os
import struct
from datetime import datetime
from pathlib import Path
import zlib

import numpy as np

import Types


ACTION_NAMES = (
    "move",
    "jump",
    "vert_move",
    "crouch",
    "look_dx",
    "look_dy",
    "attack",
    "interact",
    "target_slot",
    "selected_slot",
    "drop_item",
    "move_item",
    "selected_bag",
    "use_item",
    "use_item_world",
    "selected_limb",
    "use_item_medical",
    "selected_recipe",
    "favorite_item",
    "switch_main_hand",
    "try_sleep",
    "ragdoll",
    "exercise",
    "bark",
    "throw",
    "liquid_amount",
    "drain_liquid",
    "pull_liquid_from_world",
)


BASE_COLUMNS = (
    "episode",
    "step",
    "observation_id",
    "reward",
    "terminated",
    "truncated",
    "episode_complete",
    *[f"action_{name}" for name in ACTION_NAMES],
    "obs_layer_progress",
    "obs_best_layer_depth",
    "obs_layer_time_remaining",
    "obs_radline_displacement",
    "obs_time_ragdolled",
    "obs_crawl_time",
    "obs_grounded",
    "obs_in_water",
    "obs_velocity_x",
    "obs_velocity_y",
    "obs_stamina",
    "obs_energy",
    "obs_average_pain",
    "obs_shock",
    "obs_consciousness",
    "obs_player_dead",
    "extra_info_json",
)

# Keep the trace useful for postmortems without serializing the large spatial
# and entity observation groups. These scalar health/vital fields are part of
# the live observation and are cheap to record compared with the full packet.
HEALTH_TRACE_FIELDS = (
    "HeartRate",
    "FibrillationProgress",
    "FibrillationForced",
    "FibrillationRising",
    "HasPulmonaryEmbolism",
    "BloodOxygen",
    "BloodVolume",
    "BloodPressure",
    "BloodVesselSize",
    "BloodViscosity",
    "TotalBleedSpeed",
    "InternalBleeding",
    "Hemothorax",
    "VenomTotal",
    "VenomCurrent",
    "RespiratoryRate",
    "Breathing",
    "BrainHealth",
    "StrokeAmount",
    "Temperature",
    "SicknessAmount",
    "SepticShock",
    "RadiationSickness",
    "TraumaAmount",
)


# The grid sidecar is deliberately separate from the scalar CSV.  It stores
# one player-relative frame per environment step in a compressed binary stream.
# Block health is quantized to tenths of a hit point, which is enough to retain
# the current-health signal while keeping each frame compact.
GRID_HISTORY_MAGIC = b"PPGRID1\0"
GRID_HISTORY_VERSION = 1
GRID_HISTORY_HEALTH_SCALE = 10
GRID_HISTORY_FRAME_HEADER = struct.Struct("<QII7b4fhhB")
GRID_HISTORY_FILE_HEADER = struct.Struct("<8sHHHHI")
GRID_HISTORY_FLUSH_INTERVAL = 128


class GridHistoryWriter:
    """Stream compact player-relative geometry frames beside the CSV trace."""

    def __init__(self, output_path, width, height):
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = int(width)
        self.height = int(height)
        self.tile_count = self.width * self.height
        self.health_bytes = self.tile_count * np.dtype("<u2").itemsize
        self.fluid_bytes = self.tile_count * np.dtype("u1").itemsize
        self.frame_size = (
            GRID_HISTORY_FRAME_HEADER.size
            + self.health_bytes
            + self.fluid_bytes
        )
        self.stride = max(1, int(os.environ.get("PPO_GRID_HISTORY_STRIDE", "1")))
        self._handle = self.path.open("wb")
        self._compressor = zlib.compressobj(level=1)
        self._frame_count = 0
        self._closed = False

        self._handle.write(
            GRID_HISTORY_FILE_HEADER.pack(
                GRID_HISTORY_MAGIC,
                GRID_HISTORY_VERSION,
                self.width,
                self.height,
                GRID_HISTORY_HEALTH_SCALE,
                self.frame_size,
            )
        )

    def record(self, episode, step, action, obs, observation_id):
        if self._closed or step % self.stride != 0:
            return

        action_values = [int(value) for value in action[:7]]
        action_values.extend([0] * (7 - len(action_values)))

        health = np.asarray(obs["RelativeBlockMap"]["Health"], dtype=np.float32)
        if health.shape != (self.width, self.height):
            raise ValueError(
                "Unexpected block grid shape: "
                f"expected {(self.width, self.height)}, got {health.shape}"
            )
        health = np.nan_to_num(health, nan=0.0, posinf=6553.5, neginf=0.0)
        health = np.clip(
            np.rint(health * GRID_HISTORY_HEALTH_SCALE),
            0,
            np.iinfo(np.uint16).max,
        ).astype("<u2", copy=False)

        fluids = np.asarray(obs["RelativeFluidMap"]["Type"], dtype=np.uint8)
        if fluids.shape != (self.width, self.height):
            raise ValueError(
                "Unexpected fluid grid shape: "
                f"expected {(self.width, self.height)}, got {fluids.shape}"
            )
        fluids = np.ascontiguousarray(fluids.astype("u1", copy=False))

        frame_header = GRID_HISTORY_FRAME_HEADER.pack(
            int(observation_id),
            int(episode),
            int(step),
            *action_values,
            float(obs["Velocity"]["X"]),
            float(obs["Velocity"]["Y"]),
            float(obs["LayerProgress"]),
            float(obs["Stamina"]),
            int(obs["RadLineDisplacement"]),
            int(obs["BestLayerDepth"]),
            int(bool(obs["Grounded"])),
        )
        payload = frame_header + health.tobytes(order="C") + fluids.tobytes(order="C")
        compressed = self._compressor.compress(payload)
        if compressed:
            self._handle.write(compressed)

        self._frame_count += 1
        if self._frame_count % GRID_HISTORY_FLUSH_INTERVAL == 0:
            pending = self._compressor.flush(zlib.Z_SYNC_FLUSH)
            if pending:
                self._handle.write(pending)
            self._handle.flush()

    def close(self):
        if self._closed:
            return
        pending = self._compressor.flush(zlib.Z_FINISH)
        if pending:
            self._handle.write(pending)
        self._handle.flush()
        self._handle.close()
        self._closed = True


def _read_exact(handle, size):
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("Grid history ended before its header was complete")
    return data


def iter_grid_history(path):
    """Yield decoded grid frames from a .bin.zlib sidecar in file order."""
    path = Path(path)
    with path.open("rb") as handle:
        header = _read_exact(handle, GRID_HISTORY_FILE_HEADER.size)
        magic, version, width, height, health_scale, frame_size = (
            GRID_HISTORY_FILE_HEADER.unpack(header)
        )
        if magic != GRID_HISTORY_MAGIC:
            raise ValueError(f"Not a PPO grid-history file: {path}")
        if version != GRID_HISTORY_VERSION:
            raise ValueError(f"Unsupported PPO grid-history version: {version}")

        tile_count = width * height
        expected_frame_size = (
            GRID_HISTORY_FRAME_HEADER.size
            + tile_count * np.dtype("<u2").itemsize
            + tile_count * np.dtype("u1").itemsize
        )
        if frame_size != expected_frame_size:
            raise ValueError(
                f"Unexpected grid frame size: header={frame_size}, "
                f"expected={expected_frame_size}"
            )

        decompressor = zlib.decompressobj()
        pending = bytearray()
        while True:
            compressed = handle.read(1024 * 1024)
            if not compressed:
                pending.extend(decompressor.flush())
                break
            pending.extend(decompressor.decompress(compressed))

            while len(pending) >= frame_size:
                frame = bytes(pending[:frame_size])
                del pending[:frame_size]
                metadata = GRID_HISTORY_FRAME_HEADER.unpack_from(frame)
                health_offset = GRID_HISTORY_FRAME_HEADER.size
                fluid_offset = health_offset + tile_count * np.dtype("<u2").itemsize
                health = np.frombuffer(
                    frame,
                    dtype="<u2",
                    count=tile_count,
                    offset=health_offset,
                ).astype(np.float32).reshape((width, height), order="C")
                health /= float(health_scale)
                fluids = np.frombuffer(
                    frame,
                    dtype="u1",
                    count=tile_count,
                    offset=fluid_offset,
                ).copy().reshape((width, height), order="C")

                yield {
                    "observation_id": metadata[0],
                    "episode": metadata[1],
                    "step": metadata[2],
                    "action": metadata[3:10],
                    "velocity_x": metadata[10],
                    "velocity_y": metadata[11],
                    "layer_progress": metadata[12],
                    "stamina": metadata[13],
                    "radline_displacement": metadata[14],
                    "best_layer_depth": metadata[15],
                    "grounded": bool(metadata[16]),
                    "health": health,
                    "fluids": fluids,
                }

        if pending:
            raise ValueError(
                f"Grid history contains a partial frame ({len(pending)} bytes)"
            )


def read_grid_frame(path, observation_id):
    """Return one frame by observation ID, or None when it is absent."""
    for frame in iter_grid_history(path):
        if frame["observation_id"] == int(observation_id):
            return frame
    return None


def _scalar(value):
    """Convert numpy scalar values to CSV-friendly Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(-1)[0].item()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def _json_default(value):
    converted = _scalar(value)
    if converted is not value:
        return converted
    return str(value)


def _record_scalar_observations(row, obs, prefix=""):
    """Record scalar observation leaves while skipping large fixed arrays."""
    dtype = getattr(obs, "dtype", None)
    if dtype is not None and dtype.names:
        for field_name in dtype.names:
            field_dtype = dtype.fields[field_name][0]
            if field_dtype.subdtype is not None:
                continue
            field_prefix = f"{prefix}_{field_name}" if prefix else field_name
            _record_scalar_observations(row, obs[field_name], field_prefix)
        return

    value = np.asarray(obs)
    if value.ndim == 0:
        row[f"obs_{prefix.lower()}"] = _scalar(obs)


class EpisodeTraceWriter:
    """Buffer one episode, then append its complete rows to a CSV file."""

    def __init__(self, output_path=None):
        if output_path is None:
            output_dir = Path(__file__).resolve().parent / "episode_traces"
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_path = output_dir / f"episode_trace_{timestamp}.csv"

        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        grid_stem = self.path.stem.replace(
            "episode_trace_", "episode_grid_history_", 1
        )
        self.grid_history = GridHistoryWriter(
            self.path.with_name(f"{grid_stem}.bin.zlib"),
            Types.SIGHT_RANGE_X * 2 + 1,
            Types.SIGHT_RANGE_Y * 2 + 1,
        )
        self._columns = None
        self._rows = []
        self._episode = None

    @property
    def episode_open(self):
        return self._episode is not None

    def begin_episode(self, episode):
        if self.episode_open:
            self.finish_episode(complete=False)
        self._episode = int(episode)
        self._rows = []

    def record(
        self,
        step,
        action,
        obs,
        observation_id,
        reward,
        info,
        terminated,
        truncated,
    ):
        if not self.episode_open:
            raise RuntimeError("Cannot record a step without an active episode")

        # C13 exposes seven policy actions while retaining the legacy 28-field
        # trace schema for tooling and comparisons with C11/C12.  The omitted
        # fields are recorded as zero (their inactive wire-protocol encoding).
        action_values = list(action)
        if len(action_values) > len(ACTION_NAMES):
            raise ValueError(
                f"Trace received {len(action_values)} actions, but the legacy "
                f"schema only has {len(ACTION_NAMES)} fields"
            )
        action_values.extend([0] * (len(ACTION_NAMES) - len(action_values)))
        row = {
            "episode": self._episode,
            "step": int(step),
            "observation_id": _scalar(observation_id),
            "reward": _scalar(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "episode_complete": "",
            "extra_info_json": "",
        }

        for index, name in enumerate(ACTION_NAMES):
            row[f"action_{name}"] = _scalar(action_values[index])

        row.update(
            {
                "obs_layer_progress": _scalar(obs["LayerProgress"]),
                "obs_best_layer_depth": _scalar(obs["BestLayerDepth"]),
                "obs_layer_time_remaining": _scalar(obs["LayerTimeRemaining"]),
                "obs_radline_displacement": _scalar(obs["RadLineDisplacement"]),
                "obs_time_ragdolled": _scalar(obs["TimeRagdolled"]),
                "obs_crawl_time": _scalar(obs["CrawlTime"]),
                "obs_grounded": _scalar(obs["Grounded"]),
                "obs_in_water": _scalar(obs["InWater"]),
                "obs_velocity_x": _scalar(obs["Velocity"]["X"]),
                "obs_velocity_y": _scalar(obs["Velocity"]["Y"]),
                "obs_stamina": _scalar(obs["Stamina"]),
                "obs_energy": _scalar(obs["Energy"]),
                "obs_average_pain": _scalar(obs["AveragePain"]),
                "obs_shock": _scalar(obs["Shock"]),
                "obs_consciousness": _scalar(obs["Consciousness"]),
                "obs_player_dead": _scalar(obs["PlayerDead"]),
            }
        )

        for field_name in HEALTH_TRACE_FIELDS:
            row[f"obs_{field_name.lower()}"] = _scalar(obs[field_name])

        _record_scalar_observations(row, obs)

        extra_info = {}
        for key, value in info.items():
            column = f"reward_{key}"
            converted = _scalar(value)
            if isinstance(converted, (bool, int, float, str)) or converted is None:
                row[column] = converted
            else:
                extra_info[key] = converted

        if extra_info:
            row["extra_info_json"] = json.dumps(extra_info, default=_json_default)

        self.grid_history.record(
            self._episode,
            step,
            action,
            obs,
            observation_id,
        )
        self._rows.append(row)

    def finish_episode(self, complete=True):
        if not self.episode_open:
            return
        if not self._rows:
            self._episode = None
            return

        for row in self._rows:
            row["episode_complete"] = bool(complete)

        self._append_rows(self._rows)
        self._rows = []
        self._episode = None

    def _append_rows(self, rows):
        discovered = []
        for row in rows:
            for key in row:
                if key not in discovered:
                    discovered.append(key)

        if self._columns is None:
            self._columns = list(BASE_COLUMNS)
            self._columns.extend(
                key for key in discovered if key not in self._columns
            )
        else:
            unknown = [key for key in discovered if key not in self._columns]
            if unknown:
                for row in rows:
                    extras = json.loads(row.get("extra_info_json") or "{}")
                    for key in unknown:
                        if key in row:
                            extras[key] = row.pop(key)
                    row["extra_info_json"] = json.dumps(extras, default=_json_default)

        write_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._columns, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
            handle.flush()

    def close(self):
        self.finish_episode(complete=False)
        self.grid_history.close()
