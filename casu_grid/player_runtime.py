"""Source-ordered macrostep loop shared by every Isaac environment.

The loop mirrors the CB1 harness schedule exactly (see ``body_model``): a latched
policy action is applied, then for each of ten 20 ms physics ticks
``FixedUpdate -> physics -> Update(x1|x2) -> held-action reapplication`` runs.
The physics engine is an injected adapter so the schedule is unit-testable
without Isaac and reusable for single and batched environments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .body_model import PolicyAction, StandingBodyModel
from .ground_query import GroundWorld

TICKS_PER_ACTION = 10
CONTACT_REACH = 0.125   # edge radius 0.1 + Box2D speculative margin, validated against traces


class PhysicsAdapter(Protocol):
    def write_state(self, x: float, y: float, vx: float, vy: float) -> None: ...
    def apply_force(self, fx: float, fy: float) -> None: ...
    def step(self) -> None: ...
    def read_state(self) -> tuple[float, float, float, float]: ...
    def set_crouch(self, crouch_amount: float) -> None: ...


@dataclass
class TickRecord:
    """Per-tick samples at the source's trace boundaries (for A/B and replays)."""
    tick: int
    pre_fixed: dict
    post_fixed: dict
    post_physics: dict
    entered_colliders: int
    stepped_up: bool


@dataclass
class RuntimeState:
    """Cross-tick state that is not part of the body model itself."""
    touching: frozenset = field(default_factory=frozenset)


def _sample(model: StandingBodyModel) -> dict:
    s = model.s
    return {"x": s.x, "y": s.y, "vx": s.vx, "vy": s.vy, "grounded": s.grounded,
            "crouch_amount": s.crouch_amount, "col_size": s.col_size, "col_offset": s.col_offset,
            "move_x": s.move_x, "ended_jump": s.ended_jump, "jump_cooldown": s.jump_cooldown,
            "time_since_grounded": s.time_since_grounded, "gravity_scale": s.gravity_scale,
            "sliding_left": s.sliding_left, "sliding_right": s.sliding_right}


def run_macrostep(model: StandingBodyModel, adapter: PhysicsAdapter, world: GroundWorld,
                  action: PolicyAction | None, runtime: RuntimeState, *,
                  ticks: int = TICKS_PER_ACTION, record: bool = False,
                  pre_tick_hook=None, post_fixed_hook=None,
                  post_tick_hook=None) -> list[TickRecord]:
    """Advance one policy decision. ``action`` None re-applies the held action.

    ``pre_tick_hook(model)`` runs before ``FixedUpdate`` (e.g. liquid effects) and
    ``post_fixed_hook(model)`` after it; both may edit ``model.s`` velocities.
    ``post_tick_hook(model, touched)`` runs after the tick's Update/apply phase
    (scripted objects, hazards).
    """
    s, cfg = model.s, model.cfg
    records: list[TickRecord] = []
    if action is not None:
        model.apply_actions(latch=action)
    for tick in range(ticks):
        pre = _sample(model) if record else None
        if pre_tick_hook is not None:
            pre_tick_hook(model)
        out = model.fixed_update()
        if post_fixed_hook is not None:
            post_fixed_hook(model)
        post_fixed = _sample(model) if record else None
        adapter.write_state(s.x, s.y, s.vx, s.vy)
        adapter.apply_force(out.force[0],
                            out.force[1] + cfg.mass * cfg.gravity_y * (out.gravity_scale - 1.0))
        adapter.step()
        s.x, s.y, s.vx, s.vy = adapter.read_state()
        post_physics = _sample(model) if record else None
        center = (s.x + s.col_offset[0], s.y + s.col_offset[1])
        touched = frozenset(world.touching_colliders(center, s.col_size, CONTACT_REACH))
        entered = len(touched - runtime.touching)
        runtime.touching = touched
        for _ in range(entered):
            model.on_collision_enter()
        completed = tick + 1
        for _ in range(2 if completed % 5 == 0 else 1):
            model.update()
        adapter.set_crouch(s.crouch_amount)
        if completed < ticks:  # the held action is reapplied after ticks 1..9
            for _ in range(2 if completed in (4, 8) else 1):
                model.apply_actions()
        if post_tick_hook is not None:
            post_tick_hook(model, touched)
        if record:
            records.append(TickRecord(tick, pre, post_fixed, post_physics, entered, out.stepped_up))
    return records
