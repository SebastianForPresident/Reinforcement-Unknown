"""Isaac/PhysX adapter for canonical CasU Layer 1.

Import and construct this class only after ``AppLauncher`` has initialized Kit.
The world arrays stay in ``CasUGridState`` as the discrete source oracle; static
PhysX boxes are an Isaac geometry representation generated from those arrays.

This adapter owns one isolated SimulationContext per environment instance.
Grid-fluid scheduling follows CasU's fixed-tick manager;
body-fluid impulses use the source LiquidAffect coefficients.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Any
import copy
import math
import time

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from .player_controller import StandingAction, StandingPhysicsConfig, StandingPlayerController
from .reference import load_layer1_reference


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
    player_depth: float = 0.2
    contact_friction: float = 0.0
    rounded_player_collider: bool = False
    fluid_rng_seed: int = 391728405
    liquid_affect_phase: str = "before_controller"  # before_controller or after_controller
    solver_position_iterations: int = 8
    solver_velocity_iterations: int = 8
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
        if self.liquid_affect_phase not in {"before_controller", "after_controller"}:
            raise ValueError("liquid_affect_phase must be before_controller or after_controller")


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
        from .state import CasUGridState
        from .state import SeededDecisionStream
        self.grid = self.reference.new_state()
        self.grid.rng = SeededDecisionStream(self.cfg.fluid_rng_seed)
        self._grid_initial = self.grid.snapshot()
        self._grid_initial["dirty_chunks"] = []
        self.controller_config = StandingPhysicsConfig()
        self.controller = StandingPlayerController(self.controller_config)
        self._player_shape_standing = True
        self.action_decoder = action_decoder or self._default_action_decoder
        self.observation_encoder = observation_encoder or self._default_observation
        self.reward_function = reward_function or (lambda *_: (0.0, {}))
        self.elapsed_steps = 0
        self.physics_ticks = 0
        self._was_water = False
        self.closed = False
        self._last_state: dict[str, float] | None = None
        self.geometry_build_seconds = 0.0
        self._player_shape_standing = True
        self._construct_simulation()

    @property
    def num_envs(self):
        return self.cfg.num_envs

    @property
    def episode_step_seconds(self):
        return self.cfg.physics_dt * self.cfg.action_decimation

    def _construct_simulation(self):
        import isaaclab.sim as sim_utils
        import isaacsim.core.utils.stage as stage_utils
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from pxr import Gf, UsdGeom, UsdPhysics
        from .collision_usd import author_layer1_box_collision

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
        self.geometry_build_seconds = time.perf_counter() - started

        initial_xy = self._spawn_world_xy()
        self.initial_position_x, self.initial_position_z = initial_xy
        body_size = (self.controller_config.collider_width, self.cfg.player_depth,
                     self.controller_config.collider_height)
        rigid = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False, disable_gravity=False,
            solver_position_iteration_count=self.cfg.solver_position_iterations,
            solver_velocity_iteration_count=self.cfg.solver_velocity_iterations)
        spawn = sim_utils.CuboidCfg(
            size=body_size, rigid_props=rigid,
            mass_props=sim_utils.MassPropertiesCfg(mass=self.controller_config.mass_kg),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=self.material, physics_material_path=self.material_path)
        self.player = RigidObject(cfg=RigidObjectCfg(
            prim_path="/World/CasuPlayer", spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(initial_xy[0], 0.0, initial_xy[1]))))
        self._player_shape_paths = ["/World/CasuPlayer"]
        self._author_crouch_collision_shapes(sim_utils, body_size, rigid)

        anchor = UsdGeom.Xform.Define(self.stage, "/World/CasuPlanarAnchor")
        UsdGeom.Xformable(anchor).AddTranslateOp().Set(Gf.Vec3d(initial_xy[0], 0.0, initial_xy[1]))
        anchor_body = UsdPhysics.RigidBodyAPI.Apply(anchor.GetPrim())
        anchor_body.CreateKinematicEnabledAttr(True)
        joint = UsdPhysics.Joint.Define(self.stage, "/World/CasuPlanarConstraint")
        joint.CreateBody0Rel().SetTargets(["/World/CasuPlanarAnchor"])
        joint.CreateBody1Rel().SetTargets(["/World/CasuPlayer"])
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

    def _spawn_world_xy(self):
        if self.cfg.spawn_mode == "canonical":
            xy = self.reference.spawn_world_xy
            return float(xy[0]), float(xy[1])
        x, y = self.cfg.fixture_grid_xy
        if not self.grid.get_block(x, y):
            raise ValueError(f"fixture support cell {(x, y)} is empty")
        # Source fixture placement uses the tile center and the original body height.
        return x - 511.5, y - 511.0 + self.controller_config.collider_height / 2

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
        self.player.update(self.cfg.physics_dt)
        position = self.player.data.root_pos_w[0]
        velocity = self.player.data.root_lin_vel_w[0]
        return {
            "x": float(position[0].item()), "y": float(position[2].item()),
            "vx": float(velocity[0].item()), "vy": float(velocity[2].item()),
        }

    def _support_info(self, body):
        foot = body["y"] - self.controller_config.collider_height / 2
        left = body["x"] - self.controller_config.collider_width / 2
        right = body["x"] + self.controller_config.collider_width / 2
        x_min = max(0, int(math.floor(left + 512.0)))
        x_max = min(self.grid.width - 1, int(math.floor(right + 512.0 - 1e-7)))
        y = int(math.floor(foot + 511.0 + 1e-7))
        if y < 0 or y >= self.grid.height:
            return False, False
        candidates = [int(self.grid.blocks[x, y]) for x in range(x_min, x_max + 1)
                      if self.grid.blocks[x, y] != 0]
        grounded = bool(candidates) and abs(foot - (y - 511.0)) <= 0.18 and body["vy"] <= 0.25
        slippery = grounded and any(value in (27, 28) for value in candidates)
        return grounded, slippery

    def set_block(self, x: int, y: int, block_id: int):
        """Mutate canonical discrete terrain and synchronize its source chunk."""
        self.grid.set_block(x, y, block_id)
        self._refresh_mutated_colliders()

    def _default_observation(self, _env):
        body = self._body_state()
        grounded, _ = self._support_info(body)
        return {"player_state": np.asarray([
            body["x"], body["y"], body["vx"], body["vy"], float(grounded),
            self.controller.state.crouch_amount,
            self.controller.state.jump_cooldown_remaining,
            float(self.elapsed_steps) / self.cfg.episode_length_steps,
        ], dtype=np.float32)}

    def _format_observation(self):
        obs = self.observation_encoder(self)
        return {name: np.asarray(value, dtype=np.float32)[None, ...]
                for name, value in obs.items()}

    def _write_velocity(self, *, vx=None, vy=None):
        state = self.player.data.root_link_state_w.clone()
        if vx is not None:
            state[0, 7] = float(vx)
        if vy is not None:
            state[0, 9] = float(vy)
        self.player.write_root_link_state_to_sim(state)

    def _author_crouch_collision_shapes(self, sim_utils, standing_size, rigid):
        """Author an offset crouch collider as a non-rigid child of the player."""
        from pxr import Gf, UsdGeom, UsdPhysics
        body = self.stage.GetPrimAtPath("/World/CasuPlayer")
        UsdPhysics.CollisionAPI.Apply(body)
        self._crouch_shape_path = "/World/CasuPlayer/CrouchShape"
        crouch_height = self.controller_config.collider_height * 0.5
        edge = self.controller_config.collider_edge_radius if self.cfg.rounded_player_collider else 0.0
        size = (self.controller_config.collider_width + 2.0 * edge, self.cfg.player_depth,
                crouch_height + 2.0 * edge)
        cfg = sim_utils.CuboidCfg(size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=self.material, physics_material_path=self.material_path)
        cfg.func(self._crouch_shape_path, cfg,
                 translation=(0.0, 0.0, -self.controller_config.collider_height * 0.25))
        shape = self.stage.GetPrimAtPath(self._crouch_shape_path)
        UsdPhysics.CollisionAPI.Apply(shape)
        UsdPhysics.CollisionAPI(shape).GetCollisionEnabledAttr().Set(False)
        self._standing_collider_prim = body
        self._crouch_collider_prim = shape

    def _update_player_shape(self, command):
        """Resize and offset the collision body to match the source standing collider."""
        if not self._standing_collider_prim.IsValid() or not self._crouch_collider_prim.IsValid():
            raise RuntimeError("player collision shape prim is missing")
        crouching = command.collider_height < self.controller_config.collider_height - 1e-4
        if crouching != (not self._player_shape_standing):
            from pxr import UsdPhysics
            UsdPhysics.CollisionAPI(self._standing_collider_prim).GetCollisionEnabledAttr().Set(not crouching)
            UsdPhysics.CollisionAPI(self._crouch_collider_prim).GetCollisionEnabledAttr().Set(crouching)
        self._player_shape_standing = not crouching

    def _physics_tick(self, action: StandingAction):
        body = self._body_state()
        grounded, slippery = self._support_info(body)
        cell_x, cell_y = self.grid.world_to_cell(body["x"], body["y"],
                                                 self.grid.width, self.grid.height)
        fluid_id = self.grid.get_liquid(cell_x, cell_y)
        buoyancy, drag = ({1: (0.6, 0.915), 2: (2.0, 0.8), 3: (0.5, 0.75),
                           4: (1.0, 0.8), 5: (0.6, 0.915), 6: (3.0, 0.7)}
                          .get(fluid_id, (0.0, 0.0)))
        in_water = fluid_id > 0
        if in_water != self._was_water:
            self._was_water = in_water
        if in_water:
            # Body.CanJump accepts bodyAffect.wasWater as an alternate support
            # condition; mirror its coyote timer reset while immersed.
            self.controller.state.time_since_grounded = 0.0
        if in_water and self.cfg.liquid_affect_phase == "before_controller":
            self._write_velocity(vx=body["vx"] * drag,
                                 vy=body["vy"] * drag + buoyancy)
            body = self._body_state()
        command = self.controller.pre_physics_tick(
            velocity_x=body["vx"], velocity_y=body["vy"], grounded=grounded,
            slippery=slippery, in_water=in_water, dt=self.cfg.physics_dt)
        self._update_player_shape(command)
        if command.horizontal_velocity_override is not None or command.vertical_velocity_override is not None:
            self._write_velocity(vx=command.horizontal_velocity_override,
                                 vy=command.vertical_velocity_override)
        gravity_extra = self.controller_config.mass_kg * -55.0 * (command.gravity_scale - 1.0)
        force = torch.tensor([[[command.horizontal_force, 0.0, gravity_extra]]],
                             dtype=torch.float32, device=self.sim.device)
        torque = torch.zeros((1, 1, 3), dtype=torch.float32, device=self.sim.device)
        self.player.set_external_force_and_torque(force, torque, is_global=True)
        self.player.write_data_to_sim()
        if in_water and self.cfg.liquid_affect_phase == "after_controller":
            # Applied after the controller command is committed, immediately
            # before PhysX, matching LiquidAffect's velocity multiply/add.
            self._write_velocity(vx=body["vx"] * drag,
                                 vy=body["vy"] * drag + buoyancy)
        # PlayerCamera follows the standing body's horizontal position and
        # supplies its position to FluidManager.SimulationRangeIndex. Its
        # Unity Update lerp is collapsed to the latest pre-physics body pose
        # at this boundary; the initial camera coordinate remains exact.
        self.grid.camera_x, self.grid.camera_y = self.grid.world_to_cell(
            body["x"], body["y"], self.grid.width, self.grid.height)
        self.grid.simulation_step()
        self.physics_ticks += 1
        self.sim.step(render=False)
        self.player.update(self.cfg.physics_dt)
        return command

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
        self.controller = StandingPlayerController(self.controller_config)
        self.elapsed_steps = 0
        self.physics_ticks = 0
        self._was_water = False
        pose = torch.tensor([[self.initial_position_x, 0.0, self.initial_position_z,
                              1.0, 0.0, 0.0, 0.0]], dtype=torch.float32,
                            device=self.sim.device)
        velocity = torch.zeros((1, 6), dtype=torch.float32, device=self.sim.device)
        self.player.write_root_pose_to_sim(pose)
        from pxr import UsdPhysics
        UsdPhysics.CollisionAPI(self._standing_collider_prim).GetCollisionEnabledAttr().Set(True)
        UsdPhysics.CollisionAPI(self._crouch_collider_prim).GetCollisionEnabledAttr().Set(False)
        self.player.write_root_velocity_to_sim(velocity)
        self.player.set_external_force_and_torque(
            torch.zeros((0, 3), device=self.sim.device), torch.zeros((0, 3), device=self.sim.device))
        self.player.write_data_to_sim()
        self.sim.step(render=False)
        self.player.update(self.cfg.physics_dt)
        self._last_state = self._body_state()
        return self._format_observation(), {"seed": seed, "canonical_content_id":
                                             self.reference.manifest["content_id"],
                                             "spawn_mode": self.cfg.spawn_mode}

    def step(self, action_dict):
        if self.closed:
            raise RuntimeError("environment is closed")
        action = self.action_decoder(action_dict)
        self.controller.set_action(action)
        for _ in range(self.cfg.action_decimation):
            self._physics_tick(action)
        self.elapsed_steps += 1
        current = self._body_state()
        terminated = bool(abs(current["x"]) > 512.0 or abs(current["y"]) > 512.0)
        truncated = self.elapsed_steps >= self.cfg.episode_length_steps
        reward, reward_terms = self.reward_function(self._last_state, current, terminated, truncated)
        self._last_state = current
        info = {
            "reward_terms": reward_terms,
            "termination_reason": "world_bounds" if terminated else "time_limit" if truncated else None,
            "physics_ticks": self.cfg.action_decimation,
            "player": current,
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

    def snapshot(self):
        """Capture source grid state, player root state, controller state, and step index."""
        state = self.player.data.root_link_state_w[0].detach().cpu().clone()
        return {
            "schema": "casu.isaac-navigation-state.v1",
            "canonical_content_id": self.reference.manifest["content_id"],
            "grid": self.grid.snapshot(), "player_root_state_wxyz": state,
            "controller": copy.deepcopy(self.controller.state),
            "elapsed_steps": self.elapsed_steps,
            "collider_revision": self.grid.block_revision,
            "physics_ticks": self.physics_ticks,
            "was_water": self._was_water,
            "grid_tick": self.grid.tick,
            "fluid_rng_seed": self.cfg.fluid_rng_seed,
            "liquid_affect_phase": self.cfg.liquid_affect_phase,
        }

    def restore(self, snapshot):
        if snapshot.get("schema") != "casu.isaac-navigation-state.v1":
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
        self.controller = StandingPlayerController(
            self.controller_config, state=copy.deepcopy(snapshot["controller"]))
        self._player_shape_standing = self.controller.state.crouch_amount <= 1e-5
        from pxr import UsdPhysics
        UsdPhysics.CollisionAPI(self._standing_collider_prim).GetCollisionEnabledAttr().Set(self._player_shape_standing)
        UsdPhysics.CollisionAPI(self._crouch_collider_prim).GetCollisionEnabledAttr().Set(not self._player_shape_standing)
        self.elapsed_steps = int(snapshot["elapsed_steps"])
        self._was_water = bool(snapshot.get("was_water", False))
        # A rigid-body pose/velocity write does not restore PhysX's contact
        # solver caches. Reset the scene timeline so continuation does not
        # inherit warm-start impulses from the abandoned rollout branch.
        self.sim.reset(soft=False)
        self.player.reset()
        restored_body = snapshot["player_root_state_wxyz"].to(self.sim.device).unsqueeze(0).clone()
        self.player.write_root_link_state_to_sim(restored_body)
        self.player.set_external_force_and_torque(
            torch.zeros((0, 3), device=self.sim.device),
            torch.zeros((0, 3), device=self.sim.device))
        self.player.write_data_to_sim()
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
    from .player_controller import StandingAction

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
        grounded, _ = env._support_info(body)
        return {"player_state": np.asarray([
            body["x"] / 512.0, body["y"] / 512.0, body["vx"] / 25.0,
            body["vy"] / 26.0, float(grounded), env.controller.state.crouch_amount,
            env.controller.state.jump_cooldown_remaining,
            env.elapsed_steps / env.cfg.episode_length_steps,
        ], dtype=np.float32)}

    previous_progress = {"x": None}
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
