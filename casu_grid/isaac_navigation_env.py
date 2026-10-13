"""Isaac/PhysX adapter for canonical CasU Layer 1.

Import and construct this class only after ``AppLauncher`` has initialized Kit.
The world arrays stay in ``CasUGridState`` as the discrete source oracle; static
PhysX boxes are an Isaac geometry representation generated from those arrays.

Player control follows the source ``Body`` logic through
:mod:`casu_grid.body_model` and the CB1 tick schedule in
:mod:`casu_grid.player_runtime`; PhysX only integrates rigid-body motion and
resolves contacts. Grid-fluid scheduling follows CasU's fixed-tick manager;
body-fluid impulses use the source LiquidAffect coefficients.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any
import copy
import math
import time

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from .body_model import BodyConfig, BodyState, PolicyAction, StandingBodyModel
from .ground_query import GroundWorld
from .player_controller import StandingAction
from .player_runtime import RuntimeState, run_macrostep
from .reference import load_layer1_reference
from .shuttle_doors import ShuttleDoors
from .world_objects import ground_polygons

PLAYER_PATH = "/World/CasuPlayer"
LIQUID_COEFFICIENTS = {1: (0.6, 0.915), 2: (2.0, 0.8), 3: (0.5, 0.75),
                       4: (1.0, 0.8), 5: (0.6, 0.915), 6: (3.0, 0.7)}  # id -> (buoyancy, drag)


@dataclass(frozen=True)
class IsaacLayer1EnvConfig:
    seed: int = 391728405
    num_envs: int = 1
    physics_dt: float = 0.02
    action_decimation: int = 10
    episode_length_steps: int = 500
    spawn_mode: str = "canonical"  # canonical or fixture
    fixture_grid_xy: tuple[int, int] = (483, 1010)
    geometry_scope: str = "full"  # full or spawn_chunk (diagnostic only)
    terrain_extrusion_depth: float = 2.0
    include_world_objects: bool = True  # measured static non-tile colliders (shuttle doors, crates...)
    player_depth: float = 0.2
    contact_friction: float = 0.0       # measured: player "Entity" material friction is 0
    crouch_levels: int = 32             # quantization of the continuous crouch collider
    player_rest_offset: float = 0.0155  # calibrates the rest gap to the source's 0.1155
    player_contact_offset: float = 0.04
    rounded_player_collider: bool = True   # retained for config compatibility; always modelled
    fluid_rng_seed: int = 391728405
    liquid_affect_phase: str = "before_controller"  # before_controller or after_controller
    solver_position_iterations: int = 8
    solver_velocity_iterations: int = 8
    record_ticks: bool = False          # keep per-tick samples of the last macrostep (A/B tooling)
    use_cuda: bool = False

    def validate(self):
        if self.seed != 391728405:
            raise ValueError("this environment currently supports canonical seed 391728405 only")
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if self.physics_dt != 0.02:
            raise ValueError("CasU reference fixed timestep is 0.02 seconds")
        if self.action_decimation != 10:
            raise ValueError("CasU reference action cadence is ten physics ticks")
        if self.episode_length_steps <= 0:
            raise ValueError("episode_length_steps must be positive")
        if self.spawn_mode not in {"canonical", "fixture"}:
            raise ValueError("spawn_mode must be 'canonical' or 'fixture'")
        if self.geometry_scope not in {"full", "spawn_chunk"}:
            raise ValueError("geometry_scope must be 'full' or 'spawn_chunk'")
        if self.terrain_extrusion_depth <= 0 or self.player_depth <= 0:
            raise ValueError("extrusion dimensions must be positive")
        if self.contact_friction < 0:
            raise ValueError("contact_friction cannot be negative")
        if self.crouch_levels < 2:
            raise ValueError("crouch_levels must be at least 2")
        if self.liquid_affect_phase not in {"before_controller", "after_controller"}:
            raise ValueError("liquid_affect_phase must be before_controller or after_controller")


class _PhysxAdapter:
    """Binds the engine-agnostic macrostep loop to one Isaac rigid body."""

    def __init__(self, env: "IsaacLayer1NavigationEnv"):
        self.env = env

    def write_state(self, x, y, vx, vy):
        env = self.env
        pos = env.player.data.root_pos_w[0]
        if abs(float(pos[0]) - x) > 1e-6 or abs(float(pos[2]) - y) > 1e-6:
            pose = torch.tensor([[x, 0.0, y, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32,
                                device=env.sim.device)
            env.player.write_root_pose_to_sim(pose)
        velocity = torch.tensor([[vx, 0.0, vy, 0.0, 0.0, 0.0]], dtype=torch.float32,
                                device=env.sim.device)
        env.player.write_root_velocity_to_sim(velocity)

    def apply_force(self, fx, fy):
        env = self.env
        force = torch.tensor([[[fx, 0.0, fy]]], dtype=torch.float32, device=env.sim.device)
        torque = torch.zeros((1, 1, 3), dtype=torch.float32, device=env.sim.device)
        env.player.set_external_force_and_torque(force, torque, is_global=True)

    def step(self):
        env = self.env
        env.player.write_data_to_sim()
        env.sim.step(render=False)
        env.player.update(env.cfg.physics_dt)

    def read_state(self):
        player = self.env.player
        pos, vel = player.data.root_pos_w[0], player.data.root_lin_vel_w[0]
        return float(pos[0]), float(pos[2]), float(vel[0]), float(vel[2])

    def set_crouch(self, crouch_amount):
        body = self.env.collider
        body.set_level(body.level_for_crouch(crouch_amount))


class IsaacLayer1NavigationEnv(gym.Env):
    """Gymnasium API around a full canonical static world and standing player.

    Callback contracts:
      * ``action_decoder(action_dict) -> StandingAction``
      * ``observation_encoder(env) -> mapping[str, array]`` (unbatched values)
      * ``reward_function(previous, current, terminated, truncated) -> (scalar, terms)``

    The callbacks are owned by the selected research configuration. The built-in
    default returns a neutral reward and a compact engineering observation.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: IsaacLayer1EnvConfig | None = None, *,
                 action_decoder: Callable[[dict[str, Any]], StandingAction] | None = None,
                 observation_encoder: Callable[["IsaacLayer1NavigationEnv"], dict[str, Any]] | None = None,
                 reward_function: Callable[..., tuple[float, dict[str, float]]] | None = None):
        self.cfg = config or IsaacLayer1EnvConfig()
        self.cfg.validate()
        if self.cfg.num_envs != 1:
            raise NotImplementedError("single-env Isaac stage construction is the only validated path")
        self.reference = load_layer1_reference()
        from .state import SeededDecisionStream
        self.grid = self.reference.new_state()
        self.grid.rng = SeededDecisionStream(self.cfg.fluid_rng_seed)
        self._grid_initial = self.grid.snapshot()
        self._grid_initial["dirty_chunks"] = []
        self.action_decoder = action_decoder or self._default_action_decoder
        self.observation_encoder = observation_encoder or self._default_observation
        self.reward_function = reward_function or (lambda *_: (0.0, {}))
        self.elapsed_steps = 0
        self.physics_ticks = 0
        self._was_water = False
        self.closed = False
        self._last_state: dict[str, float] | None = None
        self.geometry_build_seconds = 0.0
        self.last_tick_records: list = []
        self.runtime = RuntimeState()
        self._construct_simulation()

    @property
    def num_envs(self):
        return self.cfg.num_envs

    @property
    def episode_step_seconds(self):
        return self.cfg.physics_dt * self.cfg.action_decimation

    # ------------------------------------------------------------ construction
    def _construct_simulation(self):
        import isaaclab.sim as sim_utils
        import isaacsim.core.utils.stage as stage_utils
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from pxr import Gf, UsdGeom, UsdPhysics
        from .collision_usd import author_layer1_box_collision, author_static_object_colliders
        from .isaac_player_body import RoundedBoxCollider

        if sim_utils.SimulationContext.instance() is not None:
            raise RuntimeError("Isaac SimulationContext already exists; close the previous environment first")
        device = "cuda:0" if self.cfg.use_cuda and torch.cuda.is_available() else "cpu"
        self.sim_cfg = sim_utils.SimulationCfg(
            dt=self.cfg.physics_dt, device=device, gravity=(0.0, 0.0, -55.0),
            physx=sim_utils.PhysxCfg(enable_ccd=True))
        self.sim = sim_utils.SimulationContext(self.sim_cfg)
        self.stage = stage_utils.get_current_stage()
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)

        self.material = sim_utils.RigidBodyMaterialCfg(
            static_friction=self.cfg.contact_friction,
            dynamic_friction=self.cfg.contact_friction,
            restitution=0.0)
        self.material_path = "/World/CasuPlayerMaterial"
        self.material.func(self.material_path, self.material)
        started = time.perf_counter()
        chunk_filter = None
        if self.cfg.geometry_scope == "spawn_chunk":
            spawn_xy = self._spawn_world_xy()
            grid_x, grid_y = int(math.floor(spawn_xy[0] + 512.0)), int(math.floor(spawn_xy[1] + 512.0))
            chunk_filter = [(grid_x // 64, grid_y // 64)]
        self.geometry_info = author_layer1_box_collision(
            self.stage, self.reference.terrain,
            half_thickness=self.cfg.terrain_extrusion_depth / 2,
            physics_material=self.material,
            physics_material_path=self.material_path,
            only_chunks=chunk_filter)

        self.world_objects = None
        self.object_geometry_info = None
        self.shuttle_doors = None
        self._door_bodies = []
        self._door_polygon_slots = []
        object_polygons = []
        if self.cfg.include_world_objects:
            from .world_objects import load_physics_audit
            self.world_objects, _ = load_physics_audit()
            colliders = list(self.world_objects.colliders)
            if chunk_filter is not None:
                kept = []
                for cx, cy in chunk_filter:
                    kept += self.world_objects.in_region(cx * 64 - 512.0, cy * 64 - 512.0,
                                                         cx * 64 - 448.0, cy * 64 - 448.0)
                colliders = list(dict.fromkeys(kept))
            doors = [c for c in colliders if "shuttledoor" in c.name]
            static = [c for c in colliders if "shuttledoor" not in c.name]
            self.object_geometry_info = author_static_object_colliders(
                self.stage, static, half_thickness=self.cfg.terrain_extrusion_depth / 2,
                physics_material_path=self.material_path)
            object_polygons = ground_polygons(static)
            if doors:
                self._build_shuttle_doors(sim_utils, RigidObject, RigidObjectCfg, doors, object_polygons)
        self.geometry_build_seconds = time.perf_counter() - started

        self.controller_config = BodyConfig()
        self.ground_world = GroundWorld(self.grid.blocks, object_polygons)
        self.model = StandingBodyModel(self.ground_world, self.controller_config)
        self.adapter = _PhysxAdapter(self)

        initial = self._initial_body_state()
        self.initial_position_x, self.initial_position_z = initial.x, initial.y
        rigid = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False, disable_gravity=False,
            solver_position_iteration_count=self.cfg.solver_position_iterations,
            solver_velocity_iteration_count=self.cfg.solver_velocity_iterations)
        spawn = sim_utils.CuboidCfg(
            size=(0.1, self.cfg.player_depth, 0.1), rigid_props=rigid,
            mass_props=sim_utils.MassPropertiesCfg(mass=self.controller_config.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            physics_material=self.material, physics_material_path=self.material_path)
        self.player = RigidObject(cfg=RigidObjectCfg(
            prim_path=PLAYER_PATH, spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(initial.x, 0.0, initial.y))))
        self.collider = RoundedBoxCollider(
            self.stage, PLAYER_PATH, width=self.controller_config.orig_col_size[0],
            standing_height=self.controller_config.orig_col_size[1],
            edge_radius=self.controller_config.edge_radius, depth=self.cfg.player_depth,
            material_path=self.material_path, levels=self.cfg.crouch_levels,
            rest_offset=self.cfg.player_rest_offset, contact_offset=self.cfg.player_contact_offset)

        anchor = UsdGeom.Xform.Define(self.stage, "/World/CasuPlanarAnchor")
        UsdGeom.Xformable(anchor).AddTranslateOp().Set(Gf.Vec3d(initial.x, 0.0, initial.y))
        anchor_body = UsdPhysics.RigidBodyAPI.Apply(anchor.GetPrim())
        anchor_body.CreateKinematicEnabledAttr(True)
        joint = UsdPhysics.Joint.Define(self.stage, "/World/CasuPlanarConstraint")
        joint.CreateBody0Rel().SetTargets(["/World/CasuPlanarAnchor"])
        joint.CreateBody1Rel().SetTargets([PLAYER_PATH])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
        for axis in (UsdPhysics.Tokens.transY, UsdPhysics.Tokens.rotX,
                     UsdPhysics.Tokens.rotY, UsdPhysics.Tokens.rotZ):
            limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
            limit.CreateLowAttr(1.0)
            limit.CreateHighAttr(-1.0)

        self.action_space = spaces.Dict({
            "move_x": spaces.MultiDiscrete([3]), "jump": spaces.MultiDiscrete([2]),
            "crouch": spaces.MultiDiscrete([2]),
        })
        self.observation_space = spaces.Dict({
            "player_state": spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32),
        })
        stage_utils.update_stage()
        self.sim.reset()
        self.sim.play()
        self.player.update(self.cfg.physics_dt)

    def _build_shuttle_doors(self, sim_utils, RigidObject, RigidObjectCfg, doors, object_polygons):
        """The spawn-shuttle doors are scripted movers: kinematic bodies instead of static prims."""
        half_thickness = self.cfg.terrain_extrusion_depth / 2
        for index, door in enumerate(sorted(doors, key=lambda c: c.center[0])):
            spawn = sim_utils.CuboidCfg(
                size=(door.size[0], 2 * half_thickness, door.size[1]),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                physics_material=self.material, physics_material_path=self.material_path)
            body = RigidObject(cfg=RigidObjectCfg(
                prim_path=f"/World/CasuShuttleDoor{index}", spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(door.center[0], 0.0, door.center[1]))))
            self._door_bodies.append((body, door))
        self.shuttle_doors = ShuttleDoors(
            trigger_boxes=((3.5, 506.5, 4.5, 507.5), (-4.5, 506.5, -3.5, 507.5)),
            base_centers=tuple(door.center for _, door in self._door_bodies))
        for body, door in self._door_bodies:
            self._door_polygon_slots.append(len(object_polygons))
            hw, hh = door.size[0] / 2, door.size[1] / 2
            cx, cy = door.center
            object_polygons.append((door.name, np.array([(cx - hw, cy - hh), (cx + hw, cy - hh),
                                                         (cx + hw, cy + hh), (cx - hw, cy + hh)])))

    def _apply_door_state(self):
        if not self._door_bodies:
            return
        for (body, door), slot, center in zip(self._door_bodies, self._door_polygon_slots,
                                              self.shuttle_doors.centers()):
            pose = torch.tensor([[center[0], 0.0, center[1], 1.0, 0.0, 0.0, 0.0]],
                                dtype=torch.float32, device=self.sim.device)
            body.write_root_pose_to_sim(pose)
            body.write_data_to_sim()
            hw, hh = door.size[0] / 2, door.size[1] / 2
            self.ground_world.polygons[slot] = (door.name, np.array([
                (center[0] - hw, center[1] - hh), (center[0] + hw, center[1] - hh),
                (center[0] + hw, center[1] + hh), (center[0] - hw, center[1] + hh)]))

    # ------------------------------------------------------------------ state
    def _spawn_world_xy(self):
        if self.cfg.spawn_mode == "canonical":
            xy = self.reference.spawn_world_xy
            return float(xy[0]), float(xy[1])
        x, y = self.cfg.fixture_grid_xy
        if not self.grid.get_block(x, y):
            raise ValueError(f"fixture support cell {(x, y)} is empty")
        # Source fixture placement: tile-centre x, collider standing on the support top
        # (rest gap of the source's rounded collider = 0.1155).
        return x - 511.5, y - 511.0 + self.controller_config.orig_col_size[1] / 2 + 0.1155

    def _initial_body_state(self) -> BodyState:
        x, y = self._spawn_world_xy()
        if self.cfg.spawn_mode == "canonical":
            # Measured at the first policy boundary of the canonical spawn: the body
            # starts crouched (crouchAmount 1.0) and grounded; the collider is still
            # standing-sized until the first Update.
            return BodyState(x=x, y=y, grounded=True, crouch_amount=1.0,
                             col_size=self.controller_config.orig_col_size, col_offset=(0.0, 0.0))
        return BodyState(x=x, y=y, grounded=True, col_size=self.controller_config.orig_col_size,
                         col_offset=(0.0, 0.0), first_wall_jump=True)

    def _default_action_decoder(self, action_dict):
        def get_head(name, default=0):
            if name not in action_dict:
                return default
            raw = np.asarray(action_dict[name]).reshape(-1)
            return int(raw[0]) if raw.size else default
        return StandingAction(move_x=get_head("move_x") - 1,
                              jump=bool(get_head("jump")),
                              crouch=bool(get_head("crouch")),
                              move_y=get_head("move_y") - 1)

    def _body_state(self):
        s = self.model.s
        return {"x": float(s.x), "y": float(s.y), "vx": float(s.vx), "vy": float(s.vy)}

    def _support_info(self, body=None):
        s = self.model.s
        return bool(s.grounded), bool(s.slippery)

    def set_block(self, x: int, y: int, block_id: int):
        """Mutate canonical discrete terrain and synchronize its source chunk."""
        self.grid.set_block(x, y, block_id)
        self._refresh_mutated_colliders()

    def _default_observation(self, _env):
        body = self._body_state()
        s = self.model.s
        return {"player_state": np.asarray([
            body["x"], body["y"], body["vx"], body["vy"], float(s.grounded),
            s.crouch_amount, max(s.jump_cooldown, 0.0),
            float(self.elapsed_steps) / self.cfg.episode_length_steps,
        ], dtype=np.float32)}

    def _format_observation(self):
        obs = self.observation_encoder(self)
        return {name: np.asarray(value, dtype=np.float32)[None, ...]
                for name, value in obs.items()}

    # ------------------------------------------------------------- tick hooks
    def _liquid_state(self):
        s = self.model.s
        cell_x, cell_y = self.grid.world_to_cell(s.x, s.y, self.grid.width, self.grid.height)
        fluid_id = self.grid.get_liquid(cell_x, cell_y)
        buoyancy, drag = LIQUID_COEFFICIENTS.get(fluid_id, (0.0, 0.0))
        return fluid_id > 0, buoyancy, drag

    def _apply_liquid(self, model):
        in_water, buoyancy, drag = self._liquid_state()
        if in_water:
            model.s.vx = model.s.vx * drag
            model.s.vy = model.s.vy * drag + buoyancy
        return in_water

    def _pre_tick(self, model):
        s = model.s
        in_water, _, _ = self._liquid_state()
        if in_water != self._was_water:
            self._was_water = in_water
        if in_water:
            # Body.Jump accepts bodyAffect.wasWater as an alternate support condition.
            s.time_since_grounded = 0.0
            s.first_wall_jump = True
        # PlayerCamera follows the standing body and supplies its position to
        # FluidManager.SimulationRangeIndex; the Update lerp is collapsed to the
        # latest pre-physics pose at this boundary.
        self.grid.camera_x, self.grid.camera_y = self.grid.world_to_cell(
            s.x, s.y, self.grid.width, self.grid.height)
        self.grid.simulation_step()
        self.physics_ticks += 1
        if self.cfg.liquid_affect_phase == "before_controller":
            self._apply_liquid(model)

    def _post_fixed(self, model):
        if self.cfg.liquid_affect_phase == "after_controller":
            self._apply_liquid(model)

    def _post_tick(self, model, touched):
        s = model.s
        if self.shuttle_doors is not None:
            x0, y0 = s.x - s.col_size[0] / 2, s.y + s.col_offset[1] - s.col_size[1] / 2
            body_box = (x0, y0, x0 + s.col_size[0], y0 + s.col_size[1])
            if self.shuttle_doors.tick(self.cfg.physics_dt, body_box):
                self._apply_door_state()
        if s.ragdoll_event is None:
            for kind, index in (t for t in touched if t[0] == "object"):
                if "jumppad" in self.ground_world.polygons[index][0]:
                    s.ragdoll_event = "jumppad"

    # ------------------------------------------------------------ lifecycle
    def _write_body_to_sim(self):
        s = self.model.s
        pose = torch.tensor([[s.x, 0.0, s.y, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32,
                            device=self.sim.device)
        self.player.write_root_pose_to_sim(pose)
        self.player.write_root_velocity_to_sim(torch.tensor(
            [[s.vx, 0.0, s.vy, 0.0, 0.0, 0.0]], dtype=torch.float32, device=self.sim.device))
        self.player.set_external_force_and_torque(
            torch.zeros((1, 1, 3), device=self.sim.device), torch.zeros((1, 1, 3), device=self.sim.device))
        self.player.write_data_to_sim()
        self.adapter.set_crouch(0.0)   # standing-sized collider, as at the source's first tick

    def reset(self, *, seed=None, options=None):
        if self.closed:
            raise RuntimeError("environment is closed")
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        current_blocks = self.grid.blocks.copy()
        self.grid.restore(copy.deepcopy(self._grid_initial))
        if not np.array_equal(current_blocks, self.grid.blocks):
            self.grid.dirty_chunks = {
                (cx, cy)
                for cx in range(self.grid.width // 64)
                for cy in range(self.grid.height // 64)
                if not np.array_equal(current_blocks[cx*64:(cx+1)*64, cy*64:(cy+1)*64],
                                      self.grid.blocks[cx*64:(cx+1)*64, cy*64:(cy+1)*64])
            }
            self._refresh_mutated_colliders()
        from .state import SeededDecisionStream
        self.grid.rng = SeededDecisionStream(self.cfg.fluid_rng_seed)
        self.model.s = self._initial_body_state()
        self.runtime = RuntimeState()
        if self.shuttle_doors is not None:
            self.shuttle_doors.restore({"activated": False, "progress": 0.0, "destroyed": False})
            self._apply_door_state()
        self.elapsed_steps = 0
        self.physics_ticks = 0
        self._was_water = False
        self._write_body_to_sim()
        self.sim.step(render=False)   # let PhysX register the freshly written state
        self._write_body_to_sim()
        self.player.update(self.cfg.physics_dt)
        self._last_state = self._body_state()
        return self._format_observation(), {"seed": seed, "canonical_content_id":
                                             self.reference.manifest["content_id"],
                                             "spawn_mode": self.cfg.spawn_mode}

    @staticmethod
    def _policy_action(action: StandingAction) -> PolicyAction:
        return PolicyAction(move=int(action.move_x), jump=int(bool(action.jump)),
                            vertical=int(action.move_y), crouch=int(bool(action.crouch)))

    def step(self, action_dict):
        if self.closed:
            raise RuntimeError("environment is closed")
        action = self.action_decoder(action_dict)
        records = run_macrostep(
            self.model, self.adapter, self.ground_world, self._policy_action(action), self.runtime,
            ticks=self.cfg.action_decimation, record=self.cfg.record_ticks,
            pre_tick_hook=self._pre_tick, post_fixed_hook=self._post_fixed,
            post_tick_hook=self._post_tick)
        self.last_tick_records = records
        self.elapsed_steps += 1
        current = self._body_state()
        ragdoll = self.model.s.ragdoll_event
        out_of_bounds = bool(abs(current["x"]) > 512.0 or abs(current["y"]) > 512.0)
        terminated = bool(out_of_bounds or ragdoll)
        truncated = self.elapsed_steps >= self.cfg.episode_length_steps
        reward, reward_terms = self.reward_function(self._last_state, current, terminated, truncated)
        self._last_state = current
        reason = ("ragdoll:" + ragdoll) if ragdoll else "world_bounds" if out_of_bounds \
            else "time_limit" if truncated else None
        info = {
            "reward_terms": reward_terms,
            "termination_reason": reason,
            "physics_ticks": self.cfg.action_decimation,
            "player": current,
            "grounded": bool(self.model.s.grounded),
            "canonical_content_id": self.reference.manifest["content_id"],
        }
        return self._format_observation(), np.asarray([reward], dtype=np.float32), \
            np.asarray([terminated]), np.asarray([truncated]), info

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()

    # --------------------------------------------------- snapshot / restore
    def snapshot(self):
        """Capture grid, player (model + PhysX root state), scripted objects and step index."""
        state = self.player.data.root_link_state_w[0].detach().cpu().clone()
        return {
            "schema": "casu.isaac-navigation-state.v2",
            "canonical_content_id": self.reference.manifest["content_id"],
            "grid": self.grid.snapshot(), "player_root_state_wxyz": state,
            "body_model": copy.deepcopy(self.model.s),
            "runtime_touching": sorted(self.runtime.touching),
            "shuttle_doors": None if self.shuttle_doors is None else self.shuttle_doors.snapshot(),
            "elapsed_steps": self.elapsed_steps,
            "collider_revision": self.grid.block_revision,
            "physics_ticks": self.physics_ticks,
            "was_water": self._was_water,
            "grid_tick": self.grid.tick,
            "fluid_rng_seed": self.cfg.fluid_rng_seed,
            "liquid_affect_phase": self.cfg.liquid_affect_phase,
        }

    def restore(self, snapshot):
        if snapshot.get("schema") != "casu.isaac-navigation-state.v2":
            raise ValueError("unsupported environment snapshot")
        if snapshot.get("canonical_content_id") != self.reference.manifest["content_id"]:
            raise ValueError("snapshot belongs to another canonical world")
        previous_blocks = self.grid.blocks.copy()
        before_revision = self.grid.block_revision
        self.grid.restore(snapshot["grid"])
        if before_revision != self.grid.block_revision or not np.array_equal(previous_blocks, self.grid.blocks):
            self.grid.dirty_chunks = {
                (cx, cy)
                for cx in range(self.grid.width // 64)
                for cy in range(self.grid.height // 64)
                if not np.array_equal(previous_blocks[cx*64:(cx+1)*64, cy*64:(cy+1)*64],
                                      self.grid.blocks[cx*64:(cx+1)*64, cy*64:(cy+1)*64])
            }
            self._refresh_mutated_colliders()
        self.model.s = copy.deepcopy(snapshot["body_model"])
        self.runtime = RuntimeState(frozenset(tuple(t) for t in snapshot["runtime_touching"]))
        if self.shuttle_doors is not None and snapshot["shuttle_doors"] is not None:
            self.shuttle_doors.restore(snapshot["shuttle_doors"])
        self.elapsed_steps = int(snapshot["elapsed_steps"])
        self.physics_ticks = int(snapshot["physics_ticks"])
        self._was_water = bool(snapshot.get("was_water", False))
        # A rigid-body pose/velocity write does not restore PhysX's contact solver
        # caches; reset the scene so continuation does not inherit warm-start impulses
        # from the abandoned rollout branch.
        self.sim.reset(soft=False)
        self.player.reset()
        self._apply_door_state()
        restored_body = snapshot["player_root_state_wxyz"].to(self.sim.device).unsqueeze(0).clone()
        self.player.write_root_link_state_to_sim(restored_body)
        self.player.set_external_force_and_torque(
            torch.zeros((1, 1, 3), device=self.sim.device), torch.zeros((1, 1, 3), device=self.sim.device))
        self.player.write_data_to_sim()
        self.adapter.set_crouch(self.model.s.crouch_amount)
        self.player.update(self.cfg.physics_dt)
        self._last_state = self._body_state()

    def _refresh_mutated_colliders(self):
        """Rebuild changed source chunk(s) as static boxes after grid edits."""
        chunks = getattr(self.grid, "dirty_chunks", set())
        if not chunks:
            return
        player_state = self.player.data.root_link_state_w[0].detach().clone()
        was_playing = self.sim.is_playing()
        from .collision_usd import author_layer1_box_collision
        for cx, cy in sorted(chunks):
            path = f"/World/CasUTerrain/Chunk_{cx}_{cy}"
            existing = self.stage.GetPrimAtPath(path)
            if existing and existing.IsValid():
                self.stage.RemovePrim(path)
            chunk = self.grid.blocks[cx * 64:(cx + 1) * 64,
                                     cy * 64:(cy + 1) * 64]
            if not np.count_nonzero(chunk):
                continue
            author_layer1_box_collision(
                self.stage, self.grid.blocks,
                half_thickness=self.cfg.terrain_extrusion_depth / 2,
                physics_material=self.material,
                physics_material_path=self.material_path,
                only_chunks=[(cx, cy)], replace_chunks=True)
        self.grid.dirty_chunks.clear()
        self.sim.reset(soft=False)
        self.player.reset()
        self.player.write_root_link_state_to_sim(player_state.unsqueeze(0))
        self.player.write_data_to_sim()
        if was_playing:
            self.sim.play()
        self.player.update(self.cfg.physics_dt)
        self._last_state = self._body_state()


def engineering_navigation_callbacks(target_x: float | None = None):
    """Explicitly temporary progress-to-x smoke objective callbacks."""
    def decode(actions):
        if "advance" in actions:
            advance = int(np.asarray(actions["advance"]).reshape(-1)[0])
            move = 0 if advance == 0 else 1
        else:
            move = int(np.asarray(actions.get("move_x", [1])).reshape(-1)[0]) - 1
        jump = bool(int(np.asarray(actions.get("jump", [0])).reshape(-1)[0]))
        crouch = bool(int(np.asarray(actions.get("crouch", [0])).reshape(-1)[0]))
        return StandingAction(move_x=move, jump=jump, crouch=crouch)

    def observe(env):
        body = env._body_state()
        s = env.model.s
        return {"player_state": np.asarray([
            body["x"] / 512.0, body["y"] / 512.0, body["vx"] / 25.0,
            body["vy"] / 26.0, float(s.grounded), s.crouch_amount,
            max(s.jump_cooldown, 0.0),
            env.elapsed_steps / env.cfg.episode_length_steps,
        ], dtype=np.float32)}

    goal = target_x

    def reward(previous, current, terminated, truncated):
        nonlocal goal
        if goal is None:
            origin_x = current["x"] if previous is None else previous["x"]
            goal = float(origin_x + 1.0)
        old_x = current["x"] if previous is None else previous["x"]
        before = min(max((old_x - (goal - 1.0)), 0.0), 1.0)
        after = min(max((current["x"] - (goal - 1.0)), 0.0), 1.0)
        progress = after - before
        reached = float(current["x"] >= goal)
        terms = {"progress_delta": float(progress), "goal": reached}
        return progress + 2.0 * reached, terms

    return decode, observe, reward
