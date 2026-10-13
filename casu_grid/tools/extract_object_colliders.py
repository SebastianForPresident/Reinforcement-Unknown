"""Convert a Unity non-tile collider dump into the project's object-collider sidecar.

Input is the ``occupancy-full.bin.colliders.jsonl`` written by the opt-in CB1
``PPO_DIAG_OCC_FULL`` probe (see ``casu_grid/MISSION13_WORLD_OBJECTS.md``).
Only static, enabled, non-trigger colliders on layers that physically collide
with the player body (Ground=6, Default=0) are kept; everything else is counted
and documented, never silently dropped.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

PLAYER_COLLIDING_LAYERS = {0: "Default", 6: "Ground"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert(dump: Path) -> tuple[list[dict], dict]:
    rows = [json.loads(line) for line in dump.read_text().splitlines() if line.strip()]
    summary = rows.pop()
    if summary.get("record_type") != "summary":
        raise ValueError("collider dump is missing its summary record")
    kept, excluded = [], collections.Counter()
    dynamic = []
    for row in rows:
        reason = None
        if row["type"] == "CompositeCollider2D":
            reason = "chunk composite (no effective paths)"
        elif row["layer"] not in PLAYER_COLLIDING_LAYERS:
            reason = f"layer {row['layer_name']} does not collide with the player"
        elif row["is_trigger"]:
            reason = "trigger"
        elif not row["enabled"] or not row["active_in_hierarchy"]:
            reason = "disabled or inactive"
        elif row["is_any_body"]:
            reason = "belongs to a living body"
        elif row["body_type"] == "Dynamic":
            reason = "dynamic rigidbody (movable; not static geometry)"
        if reason and row["body_type"] == "Dynamic" and row["layer"] in PLAYER_COLLIDING_LAYERS \
                and not row["is_trigger"] and not row["is_any_body"]:
            dynamic.append({"name": row["hierarchy"], "body_type": row["body_type"],
                            "bounds_center": row["bounds_center"], "bounds_size": row["bounds_size"]})
        if reason:
            excluded[f"{reason}: {row['hierarchy'].split('/')[0].replace('(Clone)', '')}"] += 1
            continue
        geometry = row["geometry"]
        entry = {
            "name": row["hierarchy"], "layer": row["layer"], "body_type": row["body_type"],
            "friction": row["friction"], "bounciness": row["bounciness"],
        }
        if geometry["kind"] == "box":
            c = geometry["world_corners"]
            edge_a = (c[1][0] - c[0][0], c[1][1] - c[0][1])
            edge_b = (c[3][0] - c[0][0], c[3][1] - c[0][1])
            entry.update(
                kind="box",
                center=[(c[0][0] + c[2][0]) / 2, (c[0][1] + c[2][1]) / 2],
                size=[math.hypot(*edge_a), math.hypot(*edge_b)],
                rotation_deg=math.degrees(math.atan2(edge_a[1], edge_a[0])),
                edge_radius=geometry["edge_radius"])
        elif geometry["kind"] == "circle":
            entry.update(kind="circle", center=geometry["world_center"],
                         radius=geometry["radius"] * max(geometry["lossy_scale"]))
        else:
            raise ValueError(f"unsupported static collider geometry: {geometry['kind']}")
        kept.append(entry)
    kept.sort(key=lambda e: (e["kind"], round(e["center"][0], 4), round(e["center"][1], 4), e["name"]))
    return kept, {"summary": summary, "excluded": dict(sorted(excluded.items())),
                  "dynamic": dynamic}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--occupancy", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    kept, report = convert(args.dump)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    objects_path = args.out_dir / "object_colliders.json"
    objects_path.write_text(json.dumps(
        {"schema": "casu.object-colliders.v1", "colliders": kept,
         "excluded_dynamic_initial_footprints": sorted(
             report["dynamic"], key=lambda d: (d["name"], d["bounds_center"]))},
        indent=0, sort_keys=True) + "\n")
    occupancy_path = args.out_dir / "physics_occupancy_u8_x_y.bin"
    occupancy_path.write_bytes(args.occupancy.read_bytes())
    manifest = {
        "schema": "casu.physics-audit.v1",
        "source": "CB1 staged runtime Physics2D.OverlapPoint audit, layer mask Ground(6)",
        "source_dump_sha256": _sha(args.dump),
        "object_colliders": {"file": objects_path.name, "sha256": _sha(objects_path),
                             "count": len(kept)},
        "occupancy": {
            "file": occupancy_path.name, "sha256": _sha(occupancy_path), "shape": [1024, 1024],
            "axis_order": "x,y", "dtype": "u1",
            "bit_flags": "1=cell centre overlap; 2/4/8/16 = (-,-)/(+,-)/(-,+)/(+,+) corner probes inset 0.1",
            "sampled": "after the first policy macrostep latch (physics world synced)"},
        "excluded_collider_counts": report["excluded"],
        "unity_collider_components_total": report["summary"]["non_tile_colliders_written"],
    }
    (args.out_dir / "physics_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"kept": len(kept), "excluded": report["excluded"]}, indent=1))


if __name__ == "__main__":
    main()
