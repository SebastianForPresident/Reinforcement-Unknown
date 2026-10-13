"""Load packed CasU source player traces and rebuild their exact terrain."""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .body_model import BodyState, PolicyAction
from .reference import FIXTURE_DIR, load_layer1_reference

TRACE_DIR = FIXTURE_DIR / "source_traces"


@dataclass(frozen=True)
class SourceTrace:
    name: str
    actions: tuple[tuple[int, ...], ...]
    header: dict
    pre_fixed: list
    post_fixed: list
    post_physics: list
    terrain_edits: tuple[tuple[int, int, int, int], ...]
    terrain_support: dict | None

    def policy_actions(self):
        return [PolicyAction.from_categories(row) for row in self.actions]


def _read_gz(path: Path):
    with gzip.open(path, "rt") as stream:
        header = json.loads(stream.readline())
        return header, [json.loads(line) for line in stream]


def load_source_trace(name: str, root: Path = TRACE_DIR) -> SourceTrace:
    folder = root / name
    manifest = json.loads((folder / "manifest.json").read_text())
    loaded = {}
    for label, spec in manifest["files"].items():
        path = folder / spec["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError(f"{name}/{label}: SHA-256 mismatch")
        header, rows = _read_gz(path)
        if len(rows) != spec["rows"]:
            raise ValueError(f"{name}/{label}: row count mismatch")
        loaded[label] = (header, rows)
    edits, support = (), None
    terrain = manifest.get("terrain_fixture")
    if terrain:
        path = folder / terrain["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != terrain["sha256"]:
            raise ValueError(f"{name}: terrain fixture SHA-256 mismatch")
        data = json.loads(path.read_text())
        edits = tuple(tuple(int(v) for v in e) for e in data["cell_edits_x_y_old_new"])
        support = {k: data[k] for k in ("mode", "support_row_y", "runway_grid_bounds_inclusive",
                                        "cleared_air_bounds_inclusive")}
    return SourceTrace(name, tuple(tuple(a) for a in manifest["actions"]), loaded["post_fixed"][0],
                       loaded["pre_fixed"][1], loaded["post_fixed"][1], loaded["post_physics"][1],
                       edits, support)


def terrain_for_trace(trace: SourceTrace) -> np.ndarray:
    """Canonical terrain with the recorded fixture cell edits applied."""
    blocks = load_layer1_reference().terrain.copy()
    for x, y, old, new in trace.terrain_edits:
        if int(blocks[x, y]) != old:
            raise ValueError(f"terrain edit expects {old} at {(x, y)}, found {int(blocks[x, y])}")
        blocks[x, y] = new
    return blocks


def state_from_pre_fixed(row: dict, current_action: PolicyAction | None = None) -> BodyState:
    e = row["extra"]
    s = BodyState(
        x=e["body_position"][0], y=e["body_position"][1],
        vx=row["velocity"][0], vy=row["velocity"][1],
        grounded=bool(row["grounded"]),
        move_x=row["move_dir"][0], move_y=row["move_dir"][1],
        crouching=bool(e["crouching"]), crouch_amount=row["crouch_amount"],
        col_size=tuple(row["collider_size"]), col_offset=tuple(row["collider_offset"]),
        ended_jump=bool(row["ended_jump"]), wants_jump_input=bool(e["wants_jump_input"]),
        jump_cooldown=row["jump_cooldown"], time_since_grounded=row["time_since_grounded"],
        time_since_slid_left=e["time_since_slid_left"], time_since_slid_right=e["time_since_slid_right"],
        time_slid_for=e["time_slid_for"], sliding_left=bool(e["sliding_left"]),
        sliding_right=bool(e["sliding_right"]), first_wall_jump=bool(e["first_wall_jump"]),
        last_jumped_on_right_wall=bool(e["last_jumped_on_right_wall"]),
        ground_normal=tuple(e["ground_normal_dir"]),
        last_time_step_vx=e["last_time_step_velocity"][0], last_time_step_vy=e["last_time_step_velocity"][1],
        slippery=bool(row["slippery"]), gravity_scale=row["gravity_scale"],
        look_x=e["target_look_pos"][0], look_y=e["target_look_pos"][1])
    if current_action is not None:
        s.current_action = current_action
        s.last_action = current_action
    return s
