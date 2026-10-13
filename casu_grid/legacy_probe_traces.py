"""Adapt the Mission-11/12 CasU collision probes (``results/casu_*``) to the ``SourceTrace`` shape used by the player A/B.

Those probes recorded the real ``Body`` controller and Box2D at specific terrain features (wall, glass wall, isolated tile,
adjacent tiles, chunk seam, ledges), but only post-FixedUpdate and post-Physics2D rows, not the pre-FixedUpdate
controller state. Every probe starts grounded at rest, so the initial state is the default standing state at the recorded pose;
only the free-run comparison is meaningful (no teacher forcing, no decision resync). Terrain is the canonical world.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .source_traces import SourceTrace, load_source_trace

RESULTS = Path(__file__).resolve().parent / "results"
PROBES = {
    # name: (ticks file, post-physics file, smoke file) under results/
    "wall": ("casu_collision_wall_ticks_20261012.jsonl", "casu_collision_wall_postphysics_20261012.jsonl", "casu_collision_wall_smoke_20261012.json"),
    "glass_wall": ("casu_collision_glass_wall_ticks_20261012.jsonl", "casu_collision_glass_wall_postphysics_20261012.jsonl", "casu_collision_glass_wall_smoke_20261012.json"),
    "isolated_tile": ("casu_collision_isolated_tile_ticks_20261012.jsonl", "casu_collision_isolated_tile_postphysics_20261012.jsonl", "casu_collision_isolated_tile_smoke_20261012.json"),
    "adjacent_tiles": ("casu_collision_adjacent_tiles_ticks_20261012.jsonl", "casu_collision_adjacent_tiles_postphysics_20261012.jsonl", "casu_collision_adjacent_tiles_smoke_20261012.json"),
    "chunk_seam": ("casu_collision_chunk_seam_ticks_20261012.jsonl", "casu_collision_chunk_seam_postphysics_20261012.jsonl", "casu_collision_chunk_seam_smoke_20261012.json"),
    "chunk_seam_cross": ("casu_collision_chunk_seam_cross_ticks_20261012.jsonl", "casu_collision_chunk_seam_cross_postphysics_20261012.jsonl", "casu_collision_chunk_seam_cross_smoke_20261012.json"),
    "natural_flat_ledge": ("casu_player_natural_flat_ledge_ticks_c_20261012.jsonl", "casu_player_natural_flat_ledge_postphysics_c_20261012.jsonl", "casu_player_natural_flat_ledge_smoke_c_20261012.json"),
}


def load_probe(name: str, template: str = "flat_master") -> SourceTrace:
    ticks_file, post_file, smoke_file = (RESULTS / f for f in PROBES[name])
    smoke = json.loads(smoke_file.read_text())
    actions = tuple(tuple(a) for a in smoke["actions"])
    ticks = [json.loads(l) for l in ticks_file.read_text().splitlines()[1:]]
    post = [json.loads(l) for l in post_file.read_text().splitlines()[1:]]
    horizon = len(actions)

    def select(rows):
        picked, seen = [], set()
        for row in rows:
            key = (row["action_sequence"], row["macrostep_physics_ticks"])
            if 1 <= row["action_sequence"] <= horizon and row["macrostep_physics_ticks"] < 10 and key not in seen:
                seen.add(key)
                picked.append(row)
        if len(picked) != 10 * horizon:
            raise ValueError(f"{name}: expected {10 * horizon} rows, found {len(picked)}")
        return picked

    ticks, post = select(ticks), select(post)
    base = load_source_trace(template)
    extra_template = copy.deepcopy(base.pre_fixed[0]["extra"])
    pre = []
    for row in ticks:
        row = copy.deepcopy(row)
        extra = copy.deepcopy(extra_template)
        extra["body_position"] = list(row["position"])
        extra["rb_position"] = list(row["position"])
        extra["target_look_pos"] = [row["position"][0], row["position"][1], 0.0]
        row["extra"] = extra
        pre.append(row)
    pre[0]["move_dir"] = [0.0, 0.0]            # before the first action latch, as in the packed traces
    for row in post:
        row["extra"] = {"rb_position": list(row["position"]), "last_time_step_velocity": [0.0, 0.0]}
    return SourceTrace(name=f"probe:{name}", actions=actions, header=base.header, pre_fixed=pre, post_fixed=ticks,
                       post_physics=post, terrain_edits=(), terrain_support=None)
