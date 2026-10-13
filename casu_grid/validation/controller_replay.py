"""Teacher-forced validation of ``body_model`` against recorded CasU player traces.

Two checks per tick, both seeded from the *recorded* state so a single wrong
prediction cannot compound:

* ``fixed``  -- recorded pre-FixedUpdate state -> ``Body.FixedUpdate`` port vs the
  recorded post-FixedUpdate state.
* ``update`` -- recorded post-Physics2D state -> ``Body.Update`` (x1/x2), collision
  enter and held-action application vs the next recorded pre-FixedUpdate state.

Engine physics (Box2D vs PhysX) is deliberately out of scope here; this isolates
the source controller logic from the physics engine.
"""
from __future__ import annotations

from collections import defaultdict

from ..body_model import BodyConfig, StandingBodyModel
from ..ground_query import GroundWorld
from ..source_traces import load_source_trace, state_from_pre_fixed, terrain_for_trace


def _chunk_pairs(post_physics_row: dict) -> set:
    out = set()
    for contact in post_physics_row["contacts"]:
        px, py = contact["point"]
        nx, ny = contact["normal"]
        px, py = px - nx * 0.02, py - ny * 0.02
        out.add((contact["other_collider"], contact["collider"],
                 int((px + 512) // 64), int((py + 512) // 64)))
    return out


def validate_fixed_update(trace_name: str, config: BodyConfig | None = None):
    trace = load_source_trace(trace_name)
    world = GroundWorld(terrain_for_trace(trace))
    actions = trace.policy_actions()
    err, worst = defaultdict(float), {}
    for i, (pre, post) in enumerate(zip(trace.pre_fixed, trace.post_fixed)):
        if i == 0:  # fixture teleport tick: rb pose is synchronised only after it
            continue
        state = state_from_pre_fixed(pre, actions[pre["action_sequence"] - 1])
        model = StandingBodyModel(world, config or BodyConfig(), state)
        out = model.fixed_update()
        pairs = {
            "x": (state.x, post["extra"]["body_position"][0]),
            "y": (state.y, post["extra"]["body_position"][1]),
            "vx": (state.vx, post["velocity"][0]), "vy": (state.vy, post["velocity"][1]),
            "gravity_scale": (out.gravity_scale, post["gravity_scale"]),
            "ended_jump": (float(state.ended_jump), float(post["ended_jump"])),
            "max_speed": (model.actual_max_speed, post["max_speed"]),
        }
        for key, (a, b) in pairs.items():
            d = abs(a - b)
            if d >= err[key]:
                err[key], worst[key] = d, i
    return {"rows": len(trace.pre_fixed) - 1, "max_abs_error": dict(err), "worst_row": worst}


def _geometry_pairs(world, post_fixed_row: dict, reach: float, post_physics_row: dict | None = None) -> set:
    """Collider identities predicted from geometry at the pre- or post-physics pose."""
    size, offset = post_fixed_row["collider_size"], post_fixed_row["collider_offset"]
    x, y = (post_physics_row or post_fixed_row)["extra"]["body_position"]
    return world.touching_colliders((x + offset[0], y + offset[1]), size, reach)


def validate_update(trace_name: str, config: BodyConfig | None = None, tolerance: float = 1e-4,
                    event_source: str = "contacts", contact_reach: float = 0.13):
    trace = load_source_trace(trace_name)
    world = GroundWorld(terrain_for_trace(trace))
    actions = trace.policy_actions()
    err, mismatches = defaultdict(float), defaultdict(int)
    rows = 0
    for i in range(1, len(trace.pre_fixed) - 1):
        pre, post_physics = trace.pre_fixed[i], trace.post_physics[i]
        nxt = trace.pre_fixed[i + 1]
        seq, tick = pre["action_sequence"], pre["macrostep_physics_ticks"]
        if tick == 9 and seq >= len(actions):
            continue
        state = state_from_pre_fixed(pre, actions[seq - 1])
        model = StandingBodyModel(world, config or BodyConfig(), state)
        model.fixed_update()
        state.x, state.y = post_physics["extra"]["body_position"]
        state.vx, state.vy = post_physics["velocity"]
        if event_source == "contacts":
            entered = len(_chunk_pairs(post_physics) - _chunk_pairs(trace.post_physics[i - 1]))
        else:
            entered = len(
                _geometry_pairs(world, trace.post_fixed[i], contact_reach, trace.post_physics[i])
                - _geometry_pairs(world, trace.post_fixed[i - 1], contact_reach, trace.post_physics[i - 1]))
        for _ in range(entered):
            model.on_collision_enter()
        for _ in range(2 if (tick + 1) % 5 == 0 else 1):
            model.update()
        if tick == 9:
            model.apply_actions(latch=actions[seq])
        else:
            for _ in range(2 if (tick + 1) in (4, 8) else 1):
                model.apply_actions()
        e = nxt["extra"]
        pairs = {
            "grounded": (float(state.grounded), float(nxt["grounded"])),
            "crouch_amount": (state.crouch_amount, nxt["crouch_amount"]),
            "collider_height": (state.col_size[1], nxt["collider_size"][1]),
            "jump_cooldown": (state.jump_cooldown, nxt["jump_cooldown"]),
            "time_since_grounded": (state.time_since_grounded, nxt["time_since_grounded"]),
            "time_since_slid_left": (state.time_since_slid_left, e["time_since_slid_left"]),
            "time_since_slid_right": (state.time_since_slid_right, e["time_since_slid_right"]),
            "time_slid_for": (state.time_slid_for, e["time_slid_for"]),
            "sliding_left": (float(state.sliding_left), float(e["sliding_left"])),
            "sliding_right": (float(state.sliding_right), float(e["sliding_right"])),
            "ended_jump": (float(state.ended_jump), float(nxt["ended_jump"])),
            "wants_jump_input": (float(state.wants_jump_input), float(e["wants_jump_input"])),
            "move_x": (state.move_x, nxt["move_dir"][0]),
            "vy_after_jump": (state.vy, nxt["velocity"][1]),
            "ground_normal_x": (state.ground_normal[0], e["ground_normal_dir"][0]),
            "first_wall_jump": (float(state.first_wall_jump), float(e["first_wall_jump"])),
        }
        rows += 1
        for key, (a, b) in pairs.items():
            d = abs(a - b)
            err[key] = max(err[key], d)
            if d > tolerance:
                mismatches[key] += 1
    return {"rows": rows, "max_abs_error": dict(err), "mismatching_rows": dict(mismatches)}
