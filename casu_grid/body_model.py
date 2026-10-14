"""Source-ordered port of the CasU standing ``Body`` controller (navigation subset).

Everything here mirrors ``SearchBase/Body.cs`` plus the CB1 harness schedule:

per physics tick
  1. ``fixed_update``      -- ``Body.FixedUpdate`` (forces, velocity edits, step-up)
  2. physics engine        -- Box2D in CasU, PhysX in Isaac (not modelled here)
  3. ``update`` x1 or x2   -- manual ``Body.Update`` (timers, grounded/slide casts,
                              step-up, crouch); 2 when the completed tick % 5 == 0
  4. ``apply_actions`` x1 or x2 -- held action re-applied (2 after ticks 4 and 8);
                              at a macrostep boundary the new action is latched.

The module is pure Python/NumPy so it can be validated tick by tick against the
recorded CasU traces without Isaac. Gameplay systems outside navigation (items,
stamina decay, ragdoll recovery, climbing) are deliberately not modelled; see
``MISSION13`` notes for the explicit scope.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

from .ground_query import GroundWorld, Hit


def lerp(a: float, b: float, t: float) -> float:
    """``Mathf.Lerp`` (t clamped to [0, 1])."""
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return a + (b - a) * t


@dataclass(frozen=True)
class BodyConfig:
    max_speed: float = 25.0
    move_force: float = 4000.0
    jump_speed: float = 26.0
    slowdown_amount: float = 0.2
    wall_slide_slowdown: float = 0.1
    mass: float = 49.28
    orig_col_size: tuple[float, float] = (1.6, 5.0)
    edge_radius: float = 0.1
    gravity_y: float = -55.0
    leg_speed_mult: float = 0.9999999          # measured in the staged runtime
    actual_stamina_strength: float = 1.0
    jump_cooldown: float = 0.25
    coyote_time: float = 0.11
    wall_jump_window: float = 0.21
    update_dt: float = 1.0 / 60.0              # CB1 manual Body.Update delta
    fixed_dt: float = 0.02
    auto_jump: bool = False                    # PlayerCamera.autoJump
    slippery_blocks: frozenset = frozenset({27, 28})
    ragdoll_impact_speed: float = 52.0         # max(2*jumpSpeed, 1.5*jumpSpeed)
    step_up_clearance: float = 0.0             # extra lift after TryStepUp (engine-compatibility knob)


@dataclass(frozen=True)
class PolicyAction:
    move: int = 0       # -1, 0, 1
    jump: int = 0
    vertical: int = 0
    crouch: int = 0
    look_dx: int = 0
    look_dy: int = 0

    @staticmethod
    def from_categories(row) -> "PolicyAction":
        return PolicyAction(int(row[0]) - 1, int(row[1]), int(row[2]) - 1, int(row[3]),
                            int(row[4]) - 4, int(row[5]) - 5)


@dataclass
class BodyState:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    grounded: bool = True
    move_x: float = 0.0
    move_y: float = 0.0
    crouching: bool = False
    crouch_amount: float = 0.0
    col_size: tuple[float, float] = (1.6, 5.0)
    col_offset: tuple[float, float] = (0.0, 0.0)
    ended_jump: bool = False
    wants_jump_input: bool = False
    jump_cooldown: float = 0.0
    time_since_grounded: float = 0.0
    time_since_slid_left: float = 0.0
    time_since_slid_right: float = 0.0
    time_slid_for: float = 0.0
    sliding_left: bool = False
    sliding_right: bool = False
    first_wall_jump: bool = False
    last_jumped_on_right_wall: bool = False
    ground_normal: tuple[float, float] = (0.0, 0.0)
    last_time_step_vx: float = 0.0
    last_time_step_vy: float = 0.0
    slippery: bool = False
    standing_on: int = 0
    gravity_scale: float = 1.0
    look_x: float = 0.0     # targetLookPos (world)
    look_y: float = 0.0
    ragdoll_event: str | None = None
    # last applied/current policy actions (Last/Current in the harness)
    last_action: PolicyAction = field(default_factory=PolicyAction)
    current_action: PolicyAction = field(default_factory=PolicyAction)


@dataclass(frozen=True)
class FixedOutput:
    force: tuple[float, float]
    gravity_scale: float
    stepped_up: bool


class StandingBodyModel:
    def __init__(self, world: GroundWorld, config: BodyConfig | None = None,
                 state: BodyState | None = None):
        self.world = world
        self.cfg = config or BodyConfig()
        self.s = state or BodyState()

    # ------------------------------------------------------------- derived
    @property
    def actual_max_speed(self) -> float:
        c, s = self.cfg, self.s
        return min(c.leg_speed_mult * c.max_speed,
                   lerp(c.max_speed, c.max_speed * 0.5, s.crouch_amount))

    @property
    def actual_move_force(self) -> float:
        c, s = self.cfg, self.s
        return c.move_force * c.leg_speed_mult * (0.2 if s.slippery else 1.0)

    @property
    def actual_jump_speed(self) -> float:
        # legSpeedMult * jumpSpeed * Clamp01(1.45 - liquidSlipTime)
        return self.cfg.leg_speed_mult * self.cfg.jump_speed

    @property
    def actual_wall_slide_slowdown(self) -> float:
        c, s = self.cfg, self.s
        return c.wall_slide_slowdown * (1.0 / (1.0 + s.time_slid_for * 0.35)) * c.actual_stamina_strength

    def _center(self):
        s = self.s
        return (s.x + s.col_offset[0], s.y + s.col_offset[1])

    # ------------------------------------------------------- action input
    def apply_actions(self, *, latch: PolicyAction | None = None) -> None:
        """``PPOBridge.ApplyPPOActions`` for one application.

        ``latch`` is the freshly latched action at a macrostep boundary. The
        first application sees Last=old Current, then the edges are consumed.
        """
        s, c = self.s, self.cfg
        if latch is not None:
            s.last_action = s.current_action
            s.current_action = latch
        cur, last = s.current_action, s.last_action
        s.move_x, s.move_y = 0.0, 0.0
        if (last.jump == 0 and cur.jump == 1) or (c.auto_jump and cur.jump == 1):
            s.wants_jump_input = True
        if last.jump == 1 and cur.jump == 0:
            s.wants_jump_input = False
            s.ended_jump = True
        if s.wants_jump_input:
            self.jump()
            s.ended_jump = False
        s.move_x = float(cur.move)
        s.move_y = float(cur.vertical)
        s.crouching = cur.crouch == 1
        # targetLookPos = centre of the tile (playerTile + look offset)
        tile_x = math.floor(s.x + self.world.width / 2.0) + cur.look_dx
        tile_y = math.floor(s.y + self.world.height / 2.0) + cur.look_dy
        s.look_x, s.look_y = tile_x - self.world.width / 2.0 + 0.5, tile_y - self.world.height / 2.0 + 0.5
        # ConsumeLatchedPolicyEdges
        s.last_action = s.current_action

    def jump(self) -> bool:
        s, c = self.s, self.cfg
        if not (s.jump_cooldown <= 0.0):
            return False
        if s.grounded or s.time_since_grounded <= c.coyote_time:
            s.jump_cooldown = c.jump_cooldown
            s.vy = self.actual_jump_speed
            s.grounded = False
            s.wants_jump_input = False
            return True
        can_left = s.time_since_slid_left < c.wall_jump_window
        can_right = s.time_since_slid_right < c.wall_jump_window
        if (not can_left and not can_right) or (
                not s.first_wall_jump and (not can_left or not s.last_jumped_on_right_wall)
                and (not can_right or s.last_jumped_on_right_wall)):
            return False
        direction = 1.0 if can_left else -1.0
        s.last_jumped_on_right_wall = bool(can_right)
        s.first_wall_jump = False
        s.jump_cooldown = c.jump_cooldown
        s.vx, s.vy = direction * self.actual_jump_speed, self.actual_jump_speed
        s.sliding_right = False
        s.sliding_left = False
        s.wants_jump_input = False
        return True

    # --------------------------------------------------------- FixedUpdate
    def fixed_update(self) -> FixedOutput:
        s, c = self.s, self.cfg
        if s.grounded and s.vy < 1.0:
            s.ended_jump = False
        s.gravity_scale = 2.5 if (s.vy > 0.0 and s.ended_jump and s.move_y != 1.0
                                  and s.vy <= c.jump_speed) else 1.0
        max_speed = self.actual_max_speed
        force_x, force_y = 0.0, 0.0
        if s.vx < max_speed and s.move_x > 0.0:
            force_x += self.actual_move_force
        if s.vx > -max_speed and s.move_x < 0.0:
            force_x -= self.actual_move_force
        if s.grounded and s.vx > max_speed:
            s.vx = max_speed
        if s.grounded and s.vx < -max_speed:
            s.vx = -max_speed
        if s.grounded and s.ground_normal[0] != 0.0:
            force_x += -s.ground_normal[0] * c.mass * 52.5
            force_y += abs(s.ground_normal[0]) * c.mass * 22.5
        if (not s.slippery) and (s.move_x == 0.0 or (s.move_x > 0.0 and s.vx < 0.0)
                                 or (s.move_x < 0.0 and s.vx > 0.0)):
            factor = c.slowdown_amount * (1.0 if s.grounded else 0.5) * c.leg_speed_mult
            s.vx = s.vx * (1.0 - factor)
        stepped = self.try_step_up()
        if (s.sliding_left or s.sliding_right) and s.vy < 0.0 and not s.slippery:
            s.vy = s.vy * (1.0 - self.actual_wall_slide_slowdown)
        s.last_time_step_vx, s.last_time_step_vy = s.vx, s.vy
        return FixedOutput((force_x, force_y), s.gravity_scale, stepped)

    def try_step_up(self) -> bool:
        s, c = self.s, self.cfg
        # (targetLookPos - position).normalized.y > -0.9  unless crouching
        dx, dy = s.look_x - s.x, s.look_y - s.y
        norm = math.hypot(dx, dy)
        look_y = dy / norm if norm > 1e-9 else 0.0
        if (not (look_y > -0.9)) and s.crouching:
            return False
        direction = (s.move_x, 0.0)
        if direction[0] == 0.0:
            return False
        origin = (s.x, s.y - s.col_size[1] * 0.5 + s.col_offset[1])
        ray = self.world.raycast(origin, direction, s.col_size[0] * 0.5 + c.edge_radius * 5.0)
        if ray is None:
            return False
        probe = (ray.point[0] + direction[0] * 0.5, ray.point[1] + 1.0)
        if self.world.raycast(probe, (0.0, 1.0), s.col_size[1]) is not None:
            return False
        down = self.world.raycast(probe, (0.0, -1.0), 1.0)
        if down is None:
            return False
        vec_y = 0.1 + down.point[1] - ray.point[1] + c.step_up_clearance
        vec_x = direction[0] * (ray.distance - s.col_size[0] * 0.5)
        s.x += vec_x
        s.y += vec_y
        s.vx = s.last_time_step_vx
        return True

    # -------------------------------------------------------------- Update
    def update(self, dt: float | None = None) -> None:
        dt = self.cfg.update_dt if dt is None else dt
        s = self.s
        # HandleVariableUpdates
        s.time_since_slid_left += dt
        s.time_since_slid_right += dt
        s.jump_cooldown -= dt
        self._handle_grounded_state()
        self._handle_physics(dt)

    def _cast(self, size, direction, distance) -> Hit | None:
        return self.world.box_cast(self._center(), size, direction, distance)

    def _handle_grounded_state(self) -> None:
        s, c = self.s, self.cfg
        edge = c.edge_radius
        down = self._cast(s.col_size, (0.0, -1.0), edge + 0.2)
        right = self._cast(s.col_size, (1.0, 0.0), edge + 0.1)
        left = self._cast(s.col_size, (-1.0, 0.0), edge + 0.1)
        s.sliding_right = s.move_x > 0.0 and right is not None
        s.sliding_left = s.move_x < 0.0 and left is not None
        if s.sliding_right:
            s.time_since_slid_right = 0.0
        if s.sliding_left:
            s.time_since_slid_left = 0.0
        flag = down is not None
        if down is not None:
            s.ground_normal = down.normal
            block = self.world.block_at(down.point[0], down.point[1] - 0.5)
            if block > 0:
                s.standing_on = block
        s.slippery = bool(s.grounded and s.standing_on in c.slippery_blocks)
        s.grounded = flag

    def _handle_physics(self, dt: float) -> None:
        s, c = self.s, self.cfg
        if s.grounded:
            s.time_slid_for = 0.0
            s.first_wall_jump = True
            s.time_since_grounded = 0.0
        else:
            s.time_since_grounded += dt
            if s.sliding_left or s.sliding_right:
                s.time_slid_for += dt
        self.try_step_up()
        orig_x, orig_y = c.orig_col_size
        if self.world.overlap_box((s.x, s.y + orig_y * 0.25), (orig_x, orig_y * 0.5)):
            s.crouching = True
        target = 1.0 if s.crouching else 0.0
        s.crouch_amount = lerp(s.crouch_amount, target, dt * 6.0)
        # crouchAmount = max(crouchAmount, 1 - energy) with energy == 100 is a no-op.
        t = s.crouch_amount
        s.col_size = (lerp(orig_x, orig_x, t), lerp(orig_y, orig_y * 0.5, t))
        s.col_offset = (0.0, lerp(0.0, -orig_y * 0.25, t))

    # ---------------------------------------------------- collision events
    def on_collision_enter(self) -> None:
        """``Body.OnCollisionEnter2D``: landing squash or fall ragdoll."""
        s, c = self.s, self.cfg
        if s.last_time_step_vy < -max(self.actual_jump_speed * 2.0, c.jump_speed * 1.5):
            s.ragdoll_event = "hit_ground_hard"
            return
        s.crouch_amount -= s.last_time_step_vy * 0.03
