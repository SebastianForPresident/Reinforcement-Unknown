"""Episode-level CSV tracing for reward and action diagnostics."""

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np


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
