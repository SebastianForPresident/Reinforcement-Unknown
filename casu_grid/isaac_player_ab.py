"""CasU-versus-Isaac player-motion A/B against recorded source traces.

Two modes, reported separately because they answer different questions:

* ``teacher`` -- engine fidelity. Each recorded source tick seeds the Isaac body
  (state, forces, gravity scale from the ported controller); one PhysX step is compared
  with the recorded Box2D result. Errors do not compound.
* ``free``    -- end-to-end fidelity. The recorded action tape drives the full Isaac
  environment from the first tick; per-tick state is compared with the source and the
  growth of the divergence is reported.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--trace", required=True, help="packed source trace name (e.g. flat_master)")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--modes", default="teacher,free")
parser.add_argument("--rest-offset", type=float, default=0.0155)
parser.add_argument("--contact-offset", type=float, default=0.04)
parser.add_argument("--crouch-levels", type=int, default=32)
parser.add_argument("--solver", type=int, default=1)
parser.add_argument("--max-depen", type=float, default=100.0)
parser.add_argument("--tag", default="")
parser.add_argument("--step-clear", type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from casu_grid.body_model import PolicyAction
from casu_grid.isaac_navigation_env import IsaacLayer1EnvConfig, IsaacLayer1NavigationEnv
from casu_grid.source_traces import load_source_trace, state_from_pre_fixed


def classify(source_post_physics, source_pre):
    contacts = source_post_physics["contacts"]
    if not contacts:
        return "airborne"
    ups = [c for c in contacts if c["normal"][1] > 0.5]
    sides = [c for c in contacts if abs(c["normal"][0]) > 0.5]
    downs = [c for c in contacts if c["normal"][1] < -0.5]
    if downs:
        return "ceiling"
    if ups and abs(source_pre["extra"]["last_time_step_velocity"][1]) > 5.0:
        return "landing"
    if sides and not ups:
        return "wall_air"
    if sides:
        return "wall_ground"
    return "ground"


def stats(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    return {"n": int(arr.size), "mean_abs": float(np.mean(np.abs(arr))),
            "p95_abs": float(np.percentile(np.abs(arr), 95)), "max_abs": float(np.max(np.abs(arr)))}


def teacher_forced(env, trace, actions):
    per_class = {}
    rows = []
    for i in range(1, len(trace.pre_fixed) - 1):
        pre, pp = trace.pre_fixed[i], trace.post_physics[i]
        seq = pre["action_sequence"]
        state = state_from_pre_fixed(pre, actions[seq - 1])
        env.load_body_state(state, snap_to_rest_gap=True)
        model, adapter, cfg = env.model, env.adapter, env.controller_config
        out = model.fixed_update()
        s = model.s
        adapter.write_state(s.x, s.y, s.vx, s.vy)
        adapter.apply_force(out.force[0], out.force[1] + cfg.mass * cfg.gravity_y * (out.gravity_scale - 1.0))
        adapter.step()
        x, y, vx, vy = adapter.read_state()
        sx, sy = pp["extra"]["rb_position"]
        svx, svy = pp["velocity"]
        cls = classify(pp, pre)
        entry = per_class.setdefault(cls, {"dx": [], "dy": [], "dvx": [], "dvy": []})
        entry["dx"].append(x - sx); entry["dy"].append(y - sy)
        entry["dvx"].append(vx - svx); entry["dvy"].append(vy - svy)
        rows.append({"i": i, "class": cls, "dx": x - sx, "dy": y - sy, "dvx": vx - svx, "dvy": vy - svy})
    return {"classes": {k: {m: stats(v) for m, v in d.items()} for k, d in per_class.items()},
            "tick_count": len(rows),
            "worst": sorted(rows, key=lambda r: -max(abs(r["dx"]), abs(r["dy"]), abs(r["dvx"]) / 25, abs(r["dvy"]) / 25))[:12]}


def free_running(env, trace, actions):
    state0 = state_from_pre_fixed(trace.pre_fixed[0], PolicyAction())
    env.load_body_state(state0)
    series = []
    for index, row in enumerate(trace.actions):
        env.step({"move_x": [row[0]], "jump": [row[1]], "crouch": [row[3]], "move_y": [row[2]]})
        for record in env.last_tick_records:
            i = index * 10 + record.tick
            src = trace.post_physics[i]
            sx, sy = src["extra"]["rb_position"]
            series.append({"i": i, "action": index + 1, "tick": record.tick,
                           "dx": record.post_physics["x"] - sx, "dy": record.post_physics["y"] - sy,
                           "dvx": record.post_physics["vx"] - src["velocity"][0],
                           "dvy": record.post_physics["vy"] - src["velocity"][1],
                           "isaac_grounded": record.post_physics["grounded"],
                           "source_grounded": bool(trace.pre_fixed[min(i + 1, len(trace.pre_fixed) - 1)]["grounded"])})
    def first_exceeding(key, limit):
        for row in series:
            if abs(row[key]) > limit:
                return {"action": row["action"], "tick": row["tick"], "value": row[key]}
        return None
    return {
        "ticks": len(series),
        "position_x": stats([r["dx"] for r in series]), "position_y": stats([r["dy"] for r in series]),
        "velocity_x": stats([r["dvx"] for r in series]), "velocity_y": stats([r["dvy"] for r in series]),
        "first_x_error_over_0p05": first_exceeding("dx", 0.05), "first_x_error_over_0p5": first_exceeding("dx", 0.5),
        "first_y_error_over_0p05": first_exceeding("dy", 0.05), "first_y_error_over_0p5": first_exceeding("dy", 0.5),
        "per_action_max_abs_dx": [max(abs(r["dx"]) for r in series if r["action"] == a)
                                  for a in range(1, len(trace.actions) + 1)],
        "per_action_max_abs_dy": [max(abs(r["dy"]) for r in series if r["action"] == a)
                                  for a in range(1, len(trace.actions) + 1)],
        "grounded_flag_mismatch_ticks": sum(r["isaac_grounded"] != r["source_grounded"] for r in series),
    }


def segment_resync(env, trace, actions):
    """Decision-level error: resync to the recorded state at each action boundary, run one action.

    The source's next boundary row already includes the *next* action's latch (jump
    impulse, move direction), so the same latch is applied to Isaac's final state
    before comparing.
    """
    rows = []
    for index in range(len(trace.actions) - 1):
        pre = trace.pre_fixed[index * 10]
        nxt = trace.pre_fixed[(index + 1) * 10]
        previous = actions[index - 1] if index else PolicyAction()
        env.load_body_state(state_from_pre_fixed(pre, previous), snap_to_rest_gap=True)
        row = trace.actions[index]
        env.step({"move_x": [row[0]], "jump": [row[1]], "crouch": [row[3]], "move_y": [row[2]]})
        env.model.apply_actions(latch=actions[index + 1])
        s = env.model.s
        nx, ny = nxt["extra"]["body_position"]
        window = range(index * 10, (index + 1) * 10)
        classes = {classify(trace.post_physics[i], trace.pre_fixed[i]) for i in window}
        if "landing" in classes:
            kind = "landing"
        elif classes & {"wall_air", "wall_ground", "ceiling"}:
            kind = "wall_or_ceiling"
        elif classes == {"airborne"}:
            kind = "air_only"
        elif "airborne" in classes:
            kind = "air_to_ground"
        else:
            kind = "ground_only"
        rows.append({"action": index + 1, "kind": kind, "dx": s.x - nx, "dy": s.y - ny,
                     "dvx": s.vx - nxt["velocity"][0], "dvy": s.vy - nxt["velocity"][1],
                     "grounded_match": bool(s.grounded) == bool(nxt["grounded"]),
                     "dcrouch": s.crouch_amount - nxt["crouch_amount"]})

    def group(selector):
        sel = [r for r in rows if selector(r)]
        if not sel:
            return {"n": 0}
        return {"n": len(sel), **{m: stats([r[m] for r in sel]) for m in ("dx", "dy", "dvx", "dvy", "dcrouch")},
                "grounded_mismatches": sum(not r["grounded_match"] for r in sel)}
    kinds = sorted({r["kind"] for r in rows})
    return {"all": group(lambda r: True), **{k: group(lambda r, k=k: r["kind"] == k) for k in kinds},
            "worst": sorted(rows, key=lambda r: -max(abs(r["dx"]), abs(r["dy"]), abs(r["dvx"]) / 25,
                                                     abs(r["dvy"]) / 25))[:8]}


def main():
    trace = load_source_trace(args.trace)
    actions = trace.policy_actions()
    cfg = IsaacLayer1EnvConfig(record_ticks=True, crouch_levels=args.crouch_levels,
                               player_rest_offset=args.rest_offset,
                               player_contact_offset=args.contact_offset,
                               physx_solver_type=args.solver, max_depenetration_velocity=args.max_depen,
                               step_up_clearance=args.step_clear)
    env = IsaacLayer1NavigationEnv(cfg)
    try:
        env.reset(seed=391728405)
        if trace.terrain_edits:
            env.apply_block_edits(trace.terrain_edits)
        result = {"schema": "casu.isaac-player-ab.v1", "trace": args.trace,
                  "rest_offset": args.rest_offset, "contact_offset": args.contact_offset,
                  "crouch_levels": args.crouch_levels, "modes": {}}
        for mode in args.modes.split(","):
            result["modes"][mode] = {"teacher": teacher_forced, "free": free_running,
                                     "segment": segment_resync}[mode](env, trace, actions)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=1) + "\n")
        print("AB_RESULT", args.output, flush=True)
    finally:
        env.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    app.close()
