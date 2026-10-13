"""CasU standing-player controller contract for Isaac-backed environments.

This module separates source controller state from PhysX integration. It emits
the per-fixed-tick force/velocity/gravity instructions; Isaac remains responsible
for rigid-body integration and contact response. Parameters are source-derived
defaults and deliberately exposed for measurement-driven calibration.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StandingAction:
    move_x: int = 0
    jump: bool = False
    crouch: bool = False
    move_y: int = 0

    def __post_init__(self):
        if self.move_x not in (-1, 0, 1):
            raise ValueError("move_x must be -1, 0, or 1")
        if self.move_y not in (-1, 0, 1):
            raise ValueError("move_y must be -1, 0, or 1")


@dataclass(frozen=True)
class StandingPhysicsConfig:
    mass_kg: float = 49.28
    collider_width: float = 1.6
    collider_height: float = 5.0
    collider_edge_radius: float = 0.1
    move_force: float = 4000.0
    max_speed: float = 25.0
    jump_speed: float = 26.0
    slowdown_amount: float = 0.2
    gravity: float = -55.0
    jump_release_gravity_scale: float = 2.5
    jump_cooldown: float = 0.25
    coyote_time: float = 0.11
    slippery_force_multiplier: float = 0.2

    def __post_init__(self):
        positive = (self.mass_kg, self.collider_width, self.collider_height,
                    self.move_force, self.max_speed, self.jump_speed,
                    self.jump_cooldown, self.coyote_time)
        if any(value <= 0 for value in positive):
            raise ValueError("mass, collider dimensions, speeds, and timers must be positive")
        if not 0 <= self.slowdown_amount <= 1:
            raise ValueError("slowdown_amount must be in [0, 1]")


@dataclass
class StandingControllerState:
    action: StandingAction = StandingAction()
    previous_action: StandingAction = StandingAction()
    ended_jump: bool = False
    wants_jump_input: bool = False
    jump_cooldown_remaining: float = 0.0
    time_since_grounded: float = 0.0
    crouch_amount: float = 0.0


@dataclass(frozen=True)
class TickCommand:
    horizontal_force: float
    vertical_velocity_override: float | None
    gravity_scale: float
    horizontal_velocity_override: float | None
    jump_started: bool
    collider_height: float
    collider_vertical_offset: float


class StandingPlayerController:
    """Stateful translation of the relevant standing ``Body`` mechanics.

    Call :meth:`set_action` once at each policy decision boundary, then
    :meth:`pre_physics_tick` once per source physics tick. Advance internal
    timers once per physics tick using the source fixed timestep.
    """

    def __init__(self, config: StandingPhysicsConfig | None = None,
                 state: StandingControllerState | None = None):
        self.config = config or StandingPhysicsConfig()
        self.state = state or StandingControllerState()

    def set_action(self, action: StandingAction, *, auto_jump: bool = False) -> None:
        """Latch one macrostep action and apply CasU's jump-input edges."""
        previous = self.state.action
        self.state.previous_action = previous
        self.state.action = action
        if (not previous.jump and action.jump) or (auto_jump and action.jump):
            self.state.wants_jump_input = True
        if previous.jump and not action.jump:
            self.state.wants_jump_input = False
            self.state.ended_jump = True

    def pre_physics_tick(self, *, velocity_x: float, velocity_y: float,
                         grounded: bool, moving_allowed: bool = True,
                         force_walk: bool = False, slippery: bool = False,
                         leg_speed_multiplier: float = 1.0,
                         can_jump: bool = True, in_water: bool = False,
                         crouch_min_fraction: float = 0.0,
                         measured_max_speed: float | None = None,
                         measured_move_force: float | None = None,
                         dt: float = 1.0 / 60.0) -> TickCommand:
        """Return source-ordered movement commands for one fixed tick.

        This intentionally does not integrate velocity or resolve contacts.
        ``Rigidbody2D.AddForce`` uses Force mode, so Isaac adapters should apply
        ``horizontal_force`` as force for one ``dt`` tick.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        cfg, state = self.config, self.state
        vx, vy = float(velocity_x), float(velocity_y)
        crouch_target = 1.0 if state.action.crouch else 0.0
        lerp_fraction = min(1.0, dt * 6.0)
        state.crouch_amount += (crouch_target - state.crouch_amount) * lerp_fraction
        state.crouch_amount = max(state.crouch_amount, crouch_min_fraction)
        if grounded and vy < 1.0:
            state.ended_jump = False

        gravity_scale = 2.5 if (
            vy > 0 and state.ended_jump and state.action.move_y != 1 and vy <= cfg.jump_speed
        ) else 1.0
        max_speed = max(0.0, cfg.max_speed * leg_speed_multiplier *
                        (0.5 if force_walk else 1.0))
        if measured_max_speed is not None:
            max_speed = max(0.0, float(measured_max_speed))
        actual_force = cfg.move_force * leg_speed_multiplier
        actual_force *= 0.5 if force_walk else 1.0
        actual_force *= 1.0 if moving_allowed else 0.0
        actual_force *= cfg.slippery_force_multiplier if slippery else 1.0
        if measured_move_force is not None:
            actual_force = float(measured_move_force)

        horizontal_force = 0.0
        if vx < max_speed and state.action.move_x > 0:
            horizontal_force = actual_force
        elif vx > -max_speed and state.action.move_x < 0:
            horizontal_force = -actual_force

        horizontal_override = None
        if grounded and vx > max_speed:
            horizontal_override = max_speed
            vx = max_speed
        elif grounded and vx < -max_speed:
            horizontal_override = -max_speed
            vx = -max_speed

        reversing_or_neutral = (
            state.action.move_x == 0 or
            (state.action.move_x > 0 and vx < 0) or
            (state.action.move_x < 0 and vx > 0)
        )
        if not slippery and reversing_or_neutral:
            amount = cfg.slowdown_amount * leg_speed_multiplier
            if not grounded:
                amount *= 0.5
            amount = min(1.0, max(0.0, amount))
            vx *= 1.0 - amount
            horizontal_override = vx

        jump_started = False
        vertical_override = None
        jump_available = grounded or self.state.time_since_grounded <= cfg.coyote_time or in_water
        if (state.wants_jump_input and state.jump_cooldown_remaining <= 0 and
                not force_walk and moving_allowed and can_jump and jump_available):
            vertical_override = cfg.jump_speed * max(0.0, leg_speed_multiplier)
            state.jump_cooldown_remaining = cfg.jump_cooldown
            state.time_since_grounded = cfg.coyote_time + dt
            state.ended_jump = False
            jump_started = True

        state.jump_cooldown_remaining = max(0.0, state.jump_cooldown_remaining - dt)
        if grounded:
            state.time_since_grounded = 0.0
        else:
            state.time_since_grounded += dt
        return TickCommand(
            horizontal_force, vertical_override, gravity_scale,
            horizontal_override, jump_started,
            cfg.collider_height * (1.0 - 0.5 * state.crouch_amount),
            -cfg.collider_height * 0.25 * state.crouch_amount)
