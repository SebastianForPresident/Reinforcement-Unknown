"""Pack one CB1 player-physics reference session into a compact, checksummed bundle.

Inputs are the opt-in per-tick traces written by ``source_harness/CollisionProbe.cs``
(pre-FixedUpdate, post-FixedUpdate, post-Physics2D), the smoke result (action tape),
and the optional terrain-fixture edit list. Output goes to
``fixtures/<id>/source_traces/<name>/``.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

TRACES = {
    "pre_fixed": ("player-pre-fixed.jsonl", "pre_fixed_tick"),
    "post_fixed": ("player-fixed-tick.jsonl", "player_tick"),
    "post_physics": ("player-post-physics.jsonl", "post_physics_tick"),
}


def _round(value, digits=7):
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [_round(v, digits) for v in value]
    if isinstance(value, dict):
        return {k: _round(v, digits) for k, v in value.items()}
    return value


def pack(session: Path, out_dir: Path, name: str, horizon_actions: int, terrain: Path | None,
         events: str | None = None, event_log: Path | None = None):
    smoke = json.loads((session / "smoke_result.json").read_text())
    if smoke["status"] != "passed":
        raise ValueError("source session did not pass")
    actions = smoke["actions"]
    out = out_dir / name
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "casu.source-trace.v1", "name": name, "seed": smoke["seed"],
                "actions": actions, "action_ticks": 10, "files": {}}
    for label, (fname, key) in TRACES.items():
        header, rows = None, []
        per_sequence = {}
        for line in (session / fname).read_text().splitlines():
            record = json.loads(line)
            if record.get("record_type") in (None, "header"):
                header = record
            elif (record["record_type"] == key and 1 <= record["action_sequence"] <= horizon_actions
                  and record["macrostep_physics_ticks"] < 10
                  and per_sequence.setdefault(record["action_sequence"], set()).isdisjoint(
                      {record["macrostep_physics_ticks"]})):
                per_sequence[record["action_sequence"]].add(record["macrostep_physics_ticks"])
                for drop in ("frame", "fixed_time"):
                    record.pop(drop, None)
                if "liquid" in record:
                    record["liquid"] = {k: v for k, v in record["liquid"].items() if k != "samples"} \
                        if isinstance(record["liquid"], dict) else record["liquid"]
                rows.append(_round(record))
        if len(rows) != horizon_actions * 10:
            raise ValueError(f"{label}: expected {horizon_actions * 10} rows, found {len(rows)}")
        target = out / f"{label}.jsonl.gz"
        with gzip.GzipFile(target, "wb", mtime=0) as stream:
            stream.write((json.dumps(header) + "\n").encode())
            for row in rows:
                stream.write((json.dumps(row, separators=(",", ":")) + "\n").encode())
        manifest["files"][label] = {"file": target.name, "rows": len(rows),
                                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    if terrain is not None:
        shutil.copyfile(terrain, out / "terrain_fixture.json")
        manifest["terrain_fixture"] = {"file": "terrain_fixture.json",
                                       "sha256": hashlib.sha256(terrain.read_bytes()).hexdigest()}
    if events:
        parsed = []
        for item in events.split(";"):
            seq, x, y, block = (int(v) for v in item.split(":"))
            parsed.append({"action_sequence": seq, "cell": [x, y], "new_block": block})
        manifest["terrain_events"] = parsed
        if event_log is not None and event_log.is_file():
            manifest["terrain_event_source_log"] = [json.loads(l) for l in event_log.read_text().splitlines()]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--actions", type=int, required=True)
    parser.add_argument("--terrain", type=Path)
    parser.add_argument("--events", help="seq:x:y:block;... (PPO_DIAG_TERRAIN_EVENTS)")
    parser.add_argument("--event-log", type=Path)
    args = parser.parse_args()
    m = pack(args.session, args.out_dir, args.name, args.actions, args.terrain, args.events, args.event_log)
    print(args.name, {k: v["rows"] for k, v in m["files"].items()})
