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

from dataclasses import dataclass, field
from typing import Callable, Any
import copy
import math
import os
import time

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from .body_model import BodyConfig, BodyState, PolicyAction
from .env_slot import EnvSlot
from .player_controller import StandingAction
from .player_runtime import RuntimeState, run_macrostep
from .reference import load_layer1_reference
from .world_objects import ground_polygons, load_physics_audit

PLAYER_PATH = "/World/CasuPlayer"


def _gpu_overrides() -> dict:
    """EXPERIMENT hook: ``CASU_GPU_PHYSX='{"gpu_max_rigid_patch_count": 400000}'`` overrides PhysxCfg GPU buffers."""
    import json
    return json.loads(os.environ.get("CASU_GPU_PHYSX", "{}"))


def _default_workers() -> int:
    """Worker processes for the batched environment.

    ``CASU_NUM_WORKERS``: an integer (default 1 = everything in-process) or ``auto`` (stored as 0:
    chosen from the CPU count and world count, see ``isaac_sharded_env.resolve_worker_count``).
    """
    value = os.environ.get("CASU_NUM_WORKERS", "1").strip().lower()
    return 0 if value == "auto" else int(value)


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
    terrain_collider_delay_ticks: int = 0   # CB1 harness measured 30-38 (median 31): next-frame refresh
    include_world_objects: bool = True  # measured static non-tile colliders (shuttle doors, crates...)
    player_depth: float = 0.2
    contact_friction: float = 0.0       # measured: player "Entity" material friction is 0
    crouch_levels: int = 32             # quantization of the continuous crouch collider
    player_rest_offset: float = 0.0155  # calibrates the rest gap to the source's 0.1155
    player_contact_offset: float = 0.04   # speculative range for ~8 m/s landings; terrain stays tiny (seam ghosts)
    terrain_contact_offset: float = 0.001   # PhysX contact distance of terrain shapes (see collision_usd)
    # "compound" = validated cuboids+cylinders (default); "convex_hull" = EXPERIMENTAL single convex hull of the same
    # rounded rectangle (cheap crouch switching). Environment override: CASU_PLAYER_COLLIDER.
    player_collider: str = field(default_factory=lambda: os.environ.get("CASU_PLAYER_COLLIDER", "compound"))
    rounded_player_collider: bool = True   # retained for config compatibility; always modelled
    fluid_rng_seed: int = 391728405
    liquid_affect_phase: str = "before_controller"  # before_controller or after_controller
    max_depenetration_velocity: float = 100.0
    physx_solver_type: int = 1          # 0 = PGS (Box2D-like), 1 = TGS
    physx_enhanced_determinism: bool = False   # slower, more reproducible PhysX
    step_up_clearance: float = 0.0      # extra lift after TryStepUp so PhysX sees no tangent overlap
    solver_position_iterations: int = 8
    solver_velocity_iterations: int = 8
    record_ticks: bool = False          # keep per-tick samples of the last macrostep (A/B tooling)
    # Independent-terrain edits: PhysX picks up re-authored static colliders without a scene reset (60x
    # faster, 3.0 s -> 0.05 s per edit with 2 worlds; equal trajectories on wall/pit scenarios). Set True
    # to restore the old full ``sim.reset`` after every chunk rebuild.
    terrain_edit_scene_reset: bool = False
    use_cuda: bool = False              # EXPERIMENTAL: PhysX GPU pipeline (opt-in research backend, see perf/OPTIMIZATION2_LOG.md)
    physx_ccd: bool = True              # protected default; False exists only for GPU-pipeline experiments
    # Execution layout only (never changes the simulated experiment): split the worlds over this
    # many independent Isaac processes. 1 keeps everything in-process. See isaac_sharded_env.py.
    num_workers: int = field(default_factory=_default_workers)
    worker_physx_threads: int = 2       # PhysX CPU threads per worker when num_workers > 1 (1 or fewer hangs PhysX)
    worker_timeout_seconds: float = 900.0   # a worker that does not answer a request this long is declared failed

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
        if self.player_collider not in ("compound", "convex_hull"):
            raise ValueError("player_collider must be 'compound' or 'convex_hull'")
        if self.num_workers < 0 or self.worker_physx_threads < 2 or self.worker_timeout_seconds <= 0:
            raise ValueError("num_workers must be >= 0 (0 = auto), worker_physx_threads >= 2, worker_timeout_seconds > 0")
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
        if env.slot.physical_dirty:
            env._refresh_mutated_colliders()
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
        self.action_decoder = action_decoder or self._default_action_decoder
        self.observation_encoder = observation_encoder or self._default_observation
        self.reward_function = reward_function or (lambda *_: (0.0, {}))
        self.closed = False
        self.geometry_build_seconds = 0.0
        self.last_tick_records: list = []
        self.controller_config = BodyConfig(step_up_clearance=self.cfg.step_up_clearance)
        self.world_objects = None
        object_polygons, door_defs = [], []
        if self.cfg.include_world_objects:
            self.world_objects, _ = load_physics_audit()
            door_defs = [c for c in self.world_objects.colliders if "shuttledoor" in c.name]
            object_polygons = ground_polygons(
                [c for c in self.world_objects.colliders if "shuttledoor" not in c.name])
        self.slot = EnvSlot(self.cfg, self.reference, object_polygons=object_polygons,
                            door_defs=door_defs, fluid_seed=self.cfg.fluid_rng_seed,
                            body_config=self.controller_config)
        self.slot.on_doors_changed = self._apply_door_state
        self._door_bodies = []
        self._construct_simulation()

    # ----- state owned by the slot (kept as attributes for callers and tests)
    grid = property(lambda self: self.slot.grid)
    model = property(lambda self: self.slot.model)
    ground_world = property(lambda self: self.slot.ground_world)
    shuttle_doors = property(lambda self: self.slot.shuttle_doors)
    runtime = property(lambda self: self.slot.runtime, lambda self, v: setattr(self.slot, "runtime", v))
    elapsed_steps = property(lambda self: self.slot.elapsed_steps,
                             lambda self, v: setattr(self.slot, "elapsed_steps", v))
    physics_ticks = property(lambda self: self.slot.physics_ticks,
                             lambda self, v: setattr(self.slot, "physics_ticks", v))
    _last_state = property(lambda self: self.slot.last_state, lambda self, v: setattr(self.slot, "last_state", v))

    @property
    def num_envs(self):
        return self.cfg.num_envs

    @property
    def episode_step_seconds(self):
        return self.cfg.physics_dt * self.cfg.action_decimation

    def performance_info(self) -> dict:
        """Execution backend/layout for reproducible training metadata."""
        from . import backend
        return {"backend": backend.describe(), "num_workers": 1, "worlds": 1,
                "compiled_controller": False, "terrain_mode": "single", "worlds_per_worker": [1]}

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
            physx=sim_utils.PhysxCfg(enable_ccd=self.cfg.physx_ccd, **_gpu_overrides(), solver_type=self.cfg.physx_solver_type,
                                     enable_enhanced_determinism=self.cfg.physx_enhanced_determinism))
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
            only_chunks=chunk_filter, contact_offset=self.cfg.terrain_contact_offset)

        self.object_geometry_info = None
        if self.world_objects is not None:
            colliders = list(self.world_objects.colliders)
            if chunk_filter is not None:
                kept = []
                for cx, cy in chunk_filter:
                    kept += self.world_objects.in_region(cx * 64 - 512.0, cy * 64 - 512.0,
                                                         cx * 64 - 448.0, cy * 64 - 448.0)
                colliders = list(dict.fromkeys(kept))
            static = [c for c in colliders if "shuttledoor" not in c.name]
            self.object_geometry_info = author_static_object_colliders(
                self.stage, static, half_thickness=self.cfg.terrain_extrusion_depth / 2,
                physics_material_path=self.material_path, contact_offset=self.cfg.terrain_contact_offset)
            if self.slot.door_defs:
                self._build_shuttle_doors(sim_utils, RigidObject, RigidObjectCfg)
        self.geometry_build_seconds = time.perf_counter() - started

        self.adapter = _PhysxAdapter(self)

        initial = self.slot.initial_body_state()
        self.initial_position_x, self.initial_position_z = initial.x, initial.y
        rigid = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False, disable_gravity=False,
            max_depenetration_velocity=self.cfg.max_depenetration_velocity,
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
        from pxr import PhysxSchema
        rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(self.stage.GetPrimAtPath(PLAYER_PATH))
        rigid_api.CreateEnableCCDAttr().Set(bool(self.cfg.physx_ccd))   # the scene flag alone does not enable per-body CCD
        self.collider = RoundedBoxCollider(
            self.stage, PLAYER_PATH, width=self.controller_config.orig_col_size[0],
            standing_height=self.controller_config.orig_col_size[1],
            edge_radius=self.controller_config.edge_radius, depth=self.cfg.player_depth,
            material_path=self.material_path, levels=self.cfg.crouch_levels,
            rest_offset=self.cfg.player_rest_offset, contact_offset=self.cfg.player_contact_offset,
            representation=self.cfg.player_collider)

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

    def _build_shuttle_doors(self, sim_utils, RigidObject, RigidObjectCfg):
        """The spawn-shuttle doors are scripted movers: kinematic bodies instead of static prims."""
        half_thickness = self.cfg.terrain_extrusion_depth / 2
        for index, door in enumerate(self.slot.door_defs):
            spawn = sim_utils.CuboidCfg(
                size=(door.size[0], 2 * half_thickness, door.size[1]),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                physics_material=self.material, physics_material_path=self.material_path)
            body = RigidObject(cfg=RigidObjectCfg(
                prim_path=f"/World/CasuShuttleDoor{index}", spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(door.center[0], 0.0, door.center[1]))))
            self._door_bodies.append(body)

    def _apply_door_state(self):
        if not self._door_bodies:
            return
        for body, center in zip(self._door_bodies, self.slot.shuttle_doors.centers()):
            pose = torch.tensor([[center[0], 0.0, center[1], 1.0, 0.0, 0.0, 0.0]],
                                dtype=torch.float32, device=self.sim.device)
            body.write_root_pose_to_sim(pose)
            body.write_data_to_sim()

    # ------------------------------------------------------------------ state
    def _spawn_world_xy(self):
        return self.slot.spawn_world_xy()

    def _initial_body_state(self) -> BodyState:
        return self.slot.initial_body_state()

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
        return self.slot.body_dict()

    def _support_info(self, body=None):
        return self.slot.support_info()

    def set_block(self, x: int, y: int, block_id: int):
        """Mutate canonical discrete terrain now; colliders follow after the refresh delay.

        ``terrain_collider_delay_ticks == 0`` (default) rebuilds the source chunk's colliders
        immediately. The CB1 harness measured a next-render-frame refresh of 30-38 ticks.
        """
        self.slot.edit_block(x, y, block_id)
        if self.slot.physical_dirty:
            self._refresh_mutated_colliders()

    def apply_block_edits(self, edits):
        """Batch ``(x, y, expected_old, new)`` terrain edits applied immediately, one collider rebuild."""
        for x, y, old, new in edits:
            if self.grid.get_block(x, y) != old:
                raise ValueError(f"edit expects block {old} at {(x, y)}, found {self.grid.get_block(x, y)}")
            self.slot.edit_block(x, y, new, immediate=True)
        self._refresh_mutated_colliders()

    def load_body_state(self, state: BodyState, *, snap_to_rest_gap: bool = False):
        """Place the player at an arbitrary controller state (tooling / fixtures).

        ``snap_to_rest_gap`` lifts a body that sits slightly closer to its support than
        PhysX's rest offset (e.g. a pose recorded right after the source's step-up
        teleport) so the contact solver does not eject it.
        """
        self.model.s = copy.deepcopy(state)
        s = self.model.s
        if snap_to_rest_gap:
            center = (s.x + s.col_offset[0], s.y + s.col_offset[1])
            hit = self.ground_world.box_cast(center, s.col_size, (0.0, -1.0), 0.3)
            if hit is not None:
                deficit = 0.1155 - (hit.distance + 0.0055)
                if 0.0 < deficit < 0.03:
                    s.y += deficit
        center = (s.x + s.col_offset[0], s.y + s.col_offset[1])
        self.slot.runtime = RuntimeState(frozenset(self.ground_world.touching_colliders(
            center, s.col_size, 0.125)))
        self._write_body_to_sim()
        self.adapter.set_crouch(s.crouch_amount)
        self.player.update(self.cfg.physics_dt)
        self._last_state = self._body_state()

    def _default_observation(self, _env):
        return {"player_state": self.slot.default_observation_vector()}

    def _format_observation(self):
        obs = self.observation_encoder(self)
        return {name: np.asarray(value, dtype=np.float32)[None, ...]
                for name, value in obs.items()}

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
        if self.slot.reset_logical():
            self._refresh_mutated_colliders()
        self._apply_door_state()
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
        slot = self.slot
        self.last_tick_records = run_macrostep(
            slot.model, self.adapter, slot.ground_world, self._policy_action(action), slot.runtime,
            ticks=self.cfg.action_decimation, record=self.cfg.record_ticks,
            pre_tick_hook=slot.pre_tick, post_fixed_hook=slot.post_fixed, post_tick_hook=slot.post_tick)
        reward, reward_terms, terminated, truncated, reason, current = slot.finish_step(self.reward_function)
        info = {
            "reward_terms": reward_terms,
            "termination_reason": reason,
            "physics_ticks": self.cfg.action_decimation,
            "player": current,
            "grounded": bool(slot.model.s.grounded),
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
        snap = self.slot.snapshot_logical()
        snap.update({
            "schema": "casu.isaac-navigation-state.v2",
            "canonical_content_id": self.reference.manifest["content_id"],
            "player_root_state_wxyz": self.player.data.root_link_state_w[0].detach().cpu().clone(),
            "liquid_affect_phase": self.cfg.liquid_affect_phase,
        })
        return snap

    def restore(self, snapshot):
        if snapshot.get("schema") != "casu.isaac-navigation-state.v2":
            raise ValueError("unsupported environment snapshot")
        if snapshot.get("canonical_content_id") != self.reference.manifest["content_id"]:
            raise ValueError("snapshot belongs to another canonical world")
        if self.slot.restore_logical(snapshot):
            self._refresh_mutated_colliders()
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
        """Rebuild changed source chunk(s) as static boxes after physical terrain edits."""
        chunks = self.slot.physical_dirty
        if not chunks:
            return
        blocks = self.slot.physical_blocks
        player_state = self.player.data.root_link_state_w[0].detach().clone()
        was_playing = self.sim.is_playing()
        from .collision_usd import author_layer1_box_collision
        for cx, cy in sorted(chunks):
            path = f"/World/CasUTerrain/Chunk_{cx}_{cy}"
            existing = self.stage.GetPrimAtPath(path)
            if existing and existing.IsValid():
                self.stage.RemovePrim(path)
            if not np.count_nonzero(blocks[cx * 64:(cx + 1) * 64, cy * 64:(cy + 1) * 64]):
                continue
            author_layer1_box_collision(
                self.stage, blocks,
                half_thickness=self.cfg.terrain_extrusion_depth / 2,
                physics_material=self.material,
                physics_material_path=self.material_path,
                only_chunks=[(cx, cy)], replace_chunks=True,
                contact_offset=self.cfg.terrain_contact_offset)
        chunks.clear()
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
