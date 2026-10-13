"""Batched Isaac Lab adapter with one independent canonical state per slot.

Each slot has its own immutable canonical reference, mutable grid/RNG,
controller, episode counters, terrain namespace and rigid player body. PhysX is
advanced once per source fixed tick for the whole batch.
"""
from __future__ import annotations

import copy
import math
import time
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from .isaac_navigation_env import IsaacLayer1EnvConfig
from .player_controller import StandingAction, StandingPhysicsConfig, StandingPlayerController
from .reference import load_layer1_reference


class BatchedIsaacLayer1Env(gym.Env):
    """Single SimulationContext, N independently controlled CasU worlds."""

    metadata = {"render_modes": []}

    def __init__(self, config: IsaacLayer1EnvConfig, *,
                 action_decoder: Callable[[dict[str, Any]], StandingAction] | None = None,
                 observation_encoder: Callable[[Any], dict[str, Any]] | None = None,
                 reward_function: Callable[..., tuple[float, dict[str, float]]] | None = None,
                 use_full_mesh: bool = True):
        if config.num_envs < 2:
            raise ValueError("BatchedIsaacLayer1Env requires num_envs >= 2")
        # Validate the common source timing and seed without single-env restriction.
        if config.seed != 391728405 or config.physics_dt != 0.02 or config.action_decimation != 10:
            raise ValueError("batched worlds require canonical seed, 20 ms ticks, and ten-tick actions")
        if config.spawn_mode not in {"canonical", "fixture"}:
            raise ValueError("unsupported spawn mode")
        self.cfg = config
        self.count = int(config.num_envs)
        self.reference = load_layer1_reference()
        self.controller_config = StandingPhysicsConfig()
        self.action_decoder = action_decoder or self._default_action_decoder
        self.observation_encoder = observation_encoder or self._default_observation
        self.reward_function = reward_function or (lambda *_: (0.0, {}))
        self.world_spacing = 1040.0
        self.offsets = np.zeros((self.count, 3), dtype=np.float32)
        self.offsets[:, 0] = np.arange(self.count, dtype=np.float32) * self.world_spacing
        self.grids = []
        self.controllers = []
        self.elapsed_steps = np.zeros(self.count, dtype=np.int64)
        self.physics_ticks = np.zeros(self.count, dtype=np.int64)
        self.was_water = np.zeros(self.count, dtype=bool)
        self.last_states: list[dict[str, float] | None] = [None] * self.count
        self.closed = False
        self.geometry_build_seconds = 0.0
        self.geometry_by_env = []
        self.use_full_mesh = bool(use_full_mesh)
        self.spawn_chunks = []
        self.player_paths = []
        self.player_shape_paths = []
        self.player_shape_standing = np.ones(self.count, dtype=bool)
        self.spawn_local = [self._spawn_local_xy(i) for i in range(self.count)]

        from .state import SeededDecisionStream
        for i in range(self.count):
            grid = self.reference.new_state()
            grid.rng = SeededDecisionStream(self.cfg.fluid_rng_seed + i)
            self.grids.append(grid)
            self.controllers.append(StandingPlayerController(self.controller_config))

        self._construct_simulation(use_full_mesh=use_full_mesh)
        self.action_space = spaces.Dict({
            "move_x": spaces.MultiDiscrete([3]), "jump": spaces.MultiDiscrete([2]),
            "crouch": spaces.MultiDiscrete([2]),
        })
        self.observation_space = spaces.Dict({
            "player_state": spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32),
        })

    def _spawn_local_xy(self, index):
        if self.cfg.spawn_mode == "canonical":
            x, y = self.reference.spawn_world_xy
            return float(x), float(y)
        x, y = self.cfg.fixture_grid_xy
        if not self.reference.terrain[x, y]:
            raise ValueError(f"fixture support cell {(x, y)} is empty")
        return x - 511.5, y - 511.0 + self.controller_config.collider_height / 2

    def _construct_simulation(self, *, use_full_mesh):
        import isaaclab.sim as sim_utils
        import isaacsim.core.utils.stage as stage_utils
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from pxr import Gf, UsdGeom, UsdPhysics

        if sim_utils.SimulationContext.instance() is not None:
            raise RuntimeError("Isaac SimulationContext already exists; close the previous environment first")
        device = "cuda:0" if self.cfg.use_cuda and torch.cuda.is_available() else "cpu"
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=self.cfg.physics_dt, device=device, gravity=(0.0, 0.0, -55.0),
            physx=sim_utils.PhysxCfg(enable_ccd=True)))
        self.stage = stage_utils.get_current_stage()
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        self.material = sim_utils.RigidBodyMaterialCfg(
            static_friction=self.cfg.contact_friction,
            dynamic_friction=self.cfg.contact_friction, restitution=0.0)
        self.material_path = "/World/CasuPlayerMaterial"
        self.material.func(self.material_path, self.material)
        build_start = time.perf_counter()
        spawn_chunk = None
        if self.cfg.geometry_scope == "spawn_chunk":
            x, y = self.spawn_local[0]
            spawn_chunk = [(int(math.floor(x + 512)) // 64,
                            int(math.floor(y + 512)) // 64)]
        self.spawn_chunks = [] if spawn_chunk is None else list(spawn_chunk)
        for i in range(self.count):
            offset = tuple(float(v) for v in self.offsets[i])
            terrain_path = f"/World/Env_{i:05d}/CasUTerrain"
            if use_full_mesh:
                from .collision_usd import author_layer1_collision
                geometry = author_layer1_collision(
                    self.stage, self.reference.terrain,
                    half_thickness=self.cfg.terrain_extrusion_depth / 2,
                    root_path=terrain_path, world_offset=offset)
            else:
                from .collision_usd import author_layer1_box_collision
                geometry = author_layer1_box_collision(
                    self.stage, self.reference.terrain,
                    half_thickness=self.cfg.terrain_extrusion_depth / 2,
                    physics_material=self.material, physics_material_path=self.material_path,
                    only_chunks=spawn_chunk,
                    root_path=terrain_path, world_offset=offset)
            self.geometry_by_env.append(geometry)
        self.geometry_build_seconds = time.perf_counter() - build_start

        body_size = (self.controller_config.collider_width, self.cfg.player_depth,
                     self.controller_config.collider_height)
        spawn = sim_utils.CuboidCfg(
            size=body_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False, disable_gravity=False,
                solver_position_iteration_count=self.cfg.solver_position_iterations,
                solver_velocity_iteration_count=self.cfg.solver_velocity_iterations),
            mass_props=sim_utils.MassPropertiesCfg(mass=self.controller_config.mass_kg),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=self.material, physics_material_path=self.material_path)
        # RigidObject's regex view discovers every path matching env_.*/CasuPlayer.
        for i, local in enumerate(self.spawn_local):
            path = f"/World/Env_{i:05d}/CasuPlayer"
            world = self.offsets[i]
            position = (local[0] + float(world[0]), float(world[1]),
                        local[1] + float(world[2]))
            spawn_cfg = copy.deepcopy(spawn)
            spawn_cfg.func(path, spawn_cfg, translation=position)
            self.player_paths.append(path)
            from pxr import UsdPhysics
            body_prim = self.stage.GetPrimAtPath(path)
            UsdPhysics.CollisionAPI.Apply(body_prim)
            crouch_path = path + "/CrouchShape"
            crouch_height = self.controller_config.collider_height * 0.5
            edge = self.controller_config.collider_edge_radius if self.cfg.rounded_player_collider else 0.0
            crouch_cfg = sim_utils.CuboidCfg(
                size=(self.controller_config.collider_width + 2 * edge, self.cfg.player_depth,
                      crouch_height + 2 * edge),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                physics_material=self.material, physics_material_path=self.material_path)
            crouch_cfg.func(crouch_path, crouch_cfg,
                            translation=(position[0], position[1], position[2] - self.controller_config.collider_height * .25))
            crouch_prim = self.stage.GetPrimAtPath(crouch_path)
            UsdPhysics.CollisionAPI.Apply(crouch_prim).CreateCollisionEnabledAttr(False)
            self.player_shape_paths.append((path, crouch_path))
            anchor_path = f"/World/Env_{i:05d}/CasuPlanarAnchor"
            anchor = UsdGeom.Xform.Define(self.stage, anchor_path)
            UsdGeom.Xformable(anchor).AddTranslateOp().Set(Gf.Vec3d(*position))
            body_api = UsdPhysics.RigidBodyAPI.Apply(anchor.GetPrim())
            body_api.CreateKinematicEnabledAttr(True)
            joint = UsdPhysics.Joint.Define(self.stage, f"/World/Env_{i:05d}/CasuPlanarConstraint")
            joint.CreateBody0Rel().SetTargets([anchor_path])
            joint.CreateBody1Rel().SetTargets([path])
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0))
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
            joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
            for axis in (UsdPhysics.Tokens.transY, UsdPhysics.Tokens.rotX,
                         UsdPhysics.Tokens.rotY, UsdPhysics.Tokens.rotZ):
                limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
                limit.CreateLowAttr(1.0)
                limit.CreateHighAttr(-1.0)
        self.player = RigidObject(cfg=RigidObjectCfg(
            prim_path="/World/Env_[0-9]{5}/CasuPlayer", spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg()))
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
        if self.player.num_instances != self.count:
            raise RuntimeError(f"expected {self.count} indexed player bodies, found {self.player.num_instances}")

    @property
    def episode_step_seconds(self):
        return self.cfg.physics_dt * self.cfg.action_decimation

    def _default_action_decoder(self, actions):
        return [StandingAction(move_x=int(np.asarray(actions.get("move_x", np.ones(self.count))).reshape(self.count)[i]) - 1,
                               jump=bool(np.asarray(actions.get("jump", np.zeros(self.count))).reshape(self.count)[i]),
                               crouch=bool(np.asarray(actions.get("crouch", np.zeros(self.count))).reshape(self.count)[i]))
                for i in range(self.count)]

    def _decode_actions(self, actions):
        if self.action_decoder == self._default_action_decoder:
            return self._default_action_decoder(actions)
        # User callbacks retain the single-world contract; the vector adapter
        # invokes each independently with a one-row slice.
        decoded = []
        for i in range(self.count):
            row = {key: np.asarray(value)[i:i + 1] for key, value in actions.items()}
            decoded.append(self.action_decoder(row))
        return decoded

    def _body_state(self, i):
        player = self.player
        pos = player.data.root_pos_w[i]
        vel = player.data.root_lin_vel_w[i]
        offset = self.offsets[i]
        return {"x": float(pos[0].item() - offset[0]), "y": float(pos[2].item() - offset[2]),
                "vx": float(vel[0].item()), "vy": float(vel[2].item())}

    def _update_player_shape(self, i, command):
        """Switch between pre-authored standing and crouch collision shapes."""
        from pxr import UsdPhysics
        standing_path, crouch_path = self.player_shape_paths[i]
        standing_api = UsdPhysics.CollisionAPI(self.stage.GetPrimAtPath(standing_path))
        crouch_api = UsdPhysics.CollisionAPI(self.stage.GetPrimAtPath(crouch_path))
        crouching = command.collider_height < self.controller_config.collider_height - 1e-4
        if crouching != (not self.player_shape_standing[i]):
            standing_api.GetCollisionEnabledAttr().Set(not crouching)
            crouch_api.GetCollisionEnabledAttr().Set(crouching)
        self.player_shape_standing[i] = not crouching

    def _support_info(self, i, body):
        grid = self.grids[i]
        foot = body["y"] - self.controller_config.collider_height / 2
        left, right = body["x"] - .8, body["x"] + .8
        x0, x1 = max(0, int(math.floor(left + 512))), min(1023, int(math.floor(right + 512 - 1e-7)))
        y = int(math.floor(foot + 511 + 1e-7))
        if not 0 <= y < 1024:
            return False, False
        cells = [int(grid.blocks[x, y]) for x in range(x0, x1 + 1) if grid.blocks[x, y] != 0]
        grounded = bool(cells) and abs(foot - (y - 511)) <= .18 and body["vy"] <= .25
        return grounded, grounded and any(v in (27, 28) for v in cells)

    def _observe_one(self, i):
        proxy = type("Slot", (), {})()
        proxy.cfg = self.cfg
        proxy.grid = self.grids[i]
        proxy.controller = self.controllers[i]
        proxy.controller_config = self.controller_config
        proxy.elapsed_steps = int(self.elapsed_steps[i])
        proxy.reference = self.reference
        proxy._body_state = lambda: self._body_state(i)
        proxy._support_info = lambda body: self._support_info(i, body)
        return self.observation_encoder(proxy)

    def _format_observation(self):
        rows = [self._observe_one(i) for i in range(self.count)]
        return {key: np.stack([np.asarray(row[key], dtype=np.float32) for row in rows])
                for key in rows[0]}

    def _write_velocities(self, xs, ys):
        state = self.player.data.root_link_state_w.clone()
        for i in range(self.count):
            if xs[i] is not None: state[i, 7] = float(xs[i])
            if ys[i] is not None: state[i, 9] = float(ys[i])
        self.player.write_root_link_state_to_sim(state)

    def _physics_tick(self, actions):
        forces = torch.zeros((self.count, 1, 3), dtype=torch.float32, device=self.sim.device)
        torques = torch.zeros_like(forces)
        for i, action in enumerate(actions):
            body = self._body_state(i)
            grounded, slippery = self._support_info(i, body)
            grid = self.grids[i]
            cx, cy = grid.world_to_cell(body["x"], body["y"], grid.width, grid.height)
            fluid = grid.get_liquid(cx, cy)
            buoyancy, drag = ({1: (.6, .915), 2: (2., .8), 3: (.5, .75),
                               4: (1., .8), 5: (.6, .915), 6: (3., .7)}.get(fluid, (0., 0.)))
            in_water = fluid > 0
            if in_water: self.controllers[i].state.time_since_grounded = 0.0
            if in_water and self.cfg.liquid_affect_phase == "before_controller":
                body["vx"] *= drag; body["vy"] = body["vy"] * drag + buoyancy
                self._write_velocities([body["vx"] if j == i else None for j in range(self.count)],
                                       [body["vy"] if j == i else None for j in range(self.count)])
            command = self.controllers[i].pre_physics_tick(
                velocity_x=body["vx"], velocity_y=body["vy"], grounded=grounded,
                slippery=slippery, in_water=in_water, dt=self.cfg.physics_dt)
            self._update_player_shape(i, command)
            if command.horizontal_velocity_override is not None: body["vx"] = command.horizontal_velocity_override
            if command.vertical_velocity_override is not None: body["vy"] = command.vertical_velocity_override
            if command.horizontal_velocity_override is not None or command.vertical_velocity_override is not None:
                self._write_velocities(
                    [body["vx"] if j == i else None for j in range(self.count)],
                    [body["vy"] if j == i else None for j in range(self.count)])
            if in_water and self.cfg.liquid_affect_phase == "after_controller":
                body["vx"] *= drag; body["vy"] = body["vy"] * drag + buoyancy
                self._write_velocities([body["vx"] if j == i else None for j in range(self.count)],
                                       [body["vy"] if j == i else None for j in range(self.count)])
            forces[i, 0, 0] = command.horizontal_force
            forces[i, 0, 2] = self.controller_config.mass_kg * -55.0 * (command.gravity_scale - 1.0)
            grid.camera_x, grid.camera_y = grid.world_to_cell(body["x"], body["y"], grid.width, grid.height)
            grid.simulation_step()
            self.physics_ticks[i] += 1
        self.player.set_external_force_and_torque(forces, torques, is_global=True)
        self.player.write_data_to_sim()
        self.sim.step(render=False)
        self.player.update(self.cfg.physics_dt)

    def _reset_indices(self, indices):
        from .state import SeededDecisionStream
        ids = list(indices)
        env_ids = torch.as_tensor(ids, dtype=torch.int32, device=self.sim.device)
        for i in ids:
            grid = self.grids[i]
            initial = self.reference.new_state()
            old_blocks = grid.blocks.copy()
            grid.restore(initial.snapshot())
            self._sync_chunks(i, old_blocks, grid.blocks, reset_sim=False)
            grid.rng = SeededDecisionStream(self.cfg.fluid_rng_seed + i)
            self.controllers[i] = StandingPlayerController(self.controller_config)
            self.player_shape_standing[i] = True
            self.elapsed_steps[i] = self.physics_ticks[i] = 0
            self.was_water[i] = False
        root = self.player
        pos = torch.tensor([[self.spawn_local[i][0] + self.offsets[i, 0], self.offsets[i, 1],
                             self.spawn_local[i][1] + self.offsets[i, 2]] for i in ids],
                           dtype=torch.float32, device=self.sim.device)
        pose = torch.cat([pos, torch.tensor([[1., 0., 0., 0.]] * len(ids), device=self.sim.device)], dim=1)
        velocity = torch.zeros((len(ids), 6), dtype=torch.float32, device=self.sim.device)
        from pxr import UsdPhysics
        for i in ids:
            standing_path, crouch_path = self.player_shape_paths[i]
            UsdPhysics.CollisionAPI(self.stage.GetPrimAtPath(standing_path)).GetCollisionEnabledAttr().Set(True)
            UsdPhysics.CollisionAPI(self.stage.GetPrimAtPath(crouch_path)).GetCollisionEnabledAttr().Set(False)
        root.write_root_pose_to_sim(pose, env_ids=env_ids)
        root.write_root_velocity_to_sim(velocity, env_ids=env_ids)
        root.reset(env_ids=env_ids)
        root.set_external_force_and_torque(
            torch.zeros((len(ids), 1, 3), device=self.sim.device),
            torch.zeros((len(ids), 1, 3), device=self.sim.device), env_ids=env_ids)
        root.write_data_to_sim()
        self.sim.step(render=False)
        self.player.update(self.cfg.physics_dt)
        for i in ids: self.last_states[i] = self._body_state(i)

    def reset(self, *, seed=None, options=None):
        self._reset_indices(range(self.count))
        return self._format_observation(), {"seed": seed, "canonical_content_id": self.reference.manifest["content_id"]}

    def reset_done(self, mask):
        mask = np.asarray(mask, dtype=bool).reshape(self.count)
        self._reset_indices(np.flatnonzero(mask))
        return self._format_observation()

    def set_block(self, env_index: int, x: int, y: int, block_id: int):
        """Edit one world's terrain and rebuild its changed collision chunk."""
        if not 0 <= env_index < self.count:
            raise IndexError("environment index outside vector batch")
        before = self.grids[env_index].blocks.copy()
        self.grids[env_index].set_block(x, y, block_id)
        self._sync_chunks(env_index, before, self.grids[env_index].blocks)

    def _sync_chunks(self, env_index, before, after, *, reset_sim=True):
        changed = []
        for cx in range(after.shape[0] // 64):
            for cy in range(after.shape[1] // 64):
                sl = np.s_[cx*64:(cx+1)*64, cy*64:(cy+1)*64]
                if not np.array_equal(before[sl], after[sl]): changed.append((cx, cy))
        if self.cfg.geometry_scope == "spawn_chunk":
            changed = [pair for pair in changed if pair in self.spawn_chunks]
        if not changed: return
        from .collision_usd import author_layer1_box_collision, author_layer1_collision
        root = f"/World/Env_{env_index:05d}/CasUTerrain"
        offset = tuple(float(v) for v in self.offsets[env_index])
        for cx, cy in changed:
            path = f"{root}/Chunk_{cx}_{cy}"
            prim = self.stage.GetPrimAtPath(path)
            if prim and prim.IsValid(): self.stage.RemovePrim(path)
            if not np.count_nonzero(after[cx*64:(cx+1)*64, cy*64:(cy+1)*64]): continue
            if self.use_full_mesh:
                author_layer1_collision(self.stage, after,
                    half_thickness=self.cfg.terrain_extrusion_depth / 2,
                    root_path=root, world_offset=offset, only_chunks=[(cx, cy)])
            else:
                author_layer1_box_collision(self.stage, after,
                    half_thickness=self.cfg.terrain_extrusion_depth / 2,
                    physics_material=self.material, physics_material_path=self.material_path,
                    root_path=root, world_offset=offset, only_chunks=[(cx, cy)])
        if reset_sim:
            player_state = self.player.data.root_link_state_w.detach().clone()
            self.sim.reset(soft=False)
            self.player.reset()
            ids = torch.arange(self.count, dtype=torch.int32, device=self.sim.device)
            self.player.write_root_link_state_to_sim(player_state, env_ids=ids)
            self.player.write_data_to_sim()
            self.sim.play()
            self.player.update(self.cfg.physics_dt)

    def step(self, action_dict):
        actions = self._decode_actions(action_dict)
        if len(actions) != self.count: raise ValueError("action decoder must return one action per environment")
        for i, action in enumerate(actions): self.controllers[i].set_action(action)
        for _ in range(self.cfg.action_decimation): self._physics_tick(actions)
        rewards = np.zeros(self.count, dtype=np.float32)
        terminated = np.zeros(self.count, dtype=bool)
        truncated = np.zeros(self.count, dtype=bool)
        terms: dict[str, np.ndarray] = {}
        reasons = [None] * self.count
        for i in range(self.count):
            self.elapsed_steps[i] += 1
            current = self._body_state(i)
            terminated[i] = abs(current["x"]) > 512 or abs(current["y"]) > 512
            truncated[i] = self.elapsed_steps[i] >= self.cfg.episode_length_steps
            reward, item_terms = self.reward_function(self.last_states[i], current,
                                                       bool(terminated[i]), bool(truncated[i]))
            rewards[i] = reward
            for key, value in item_terms.items(): terms.setdefault(key, np.zeros(self.count, np.float32))[i] = value
            self.last_states[i] = current
            reasons[i] = "world_bounds" if terminated[i] else "time_limit" if truncated[i] else None
        return self._format_observation(), rewards, terminated, truncated, {
            "reward_terms": terms, "termination_reason": reasons,
            "physics_ticks": self.cfg.action_decimation * self.count,
            "player": [self._body_state(i) for i in range(self.count)],
            "canonical_content_id": self.reference.manifest["content_id"]}

    def snapshot(self):
        states = []
        for i in range(self.count):
            states.append({"grid": self.grids[i].snapshot(),
                           "player": self.player.data.root_link_state_w[i].detach().cpu().clone(),
                           "controller": copy.deepcopy(self.controllers[i].state),
                           "elapsed_steps": int(self.elapsed_steps[i]),
                           "physics_ticks": int(self.physics_ticks[i]),
                           "last_state": copy.deepcopy(self.last_states[i])})
        return {"schema": "casu.isaac-vector-state.v1", "slots": states,
                "canonical_content_id": self.reference.manifest["content_id"]}

    def restore(self, snapshot):
        if snapshot.get("schema") != "casu.isaac-vector-state.v1" or len(snapshot.get("slots", [])) != self.count:
            raise ValueError("snapshot has wrong vector schema or environment count")
        if snapshot.get("canonical_content_id") != self.reference.manifest["content_id"]:
            raise ValueError("snapshot belongs to another canonical world")
        from .state import SeededDecisionStream
        ids = list(range(self.count))
        env_ids = torch.arange(self.count, dtype=torch.int32, device=self.sim.device)
        roots = []
        for i, state in enumerate(snapshot["slots"]):
            before = self.grids[i].blocks.copy()
            self.grids[i].restore(state["grid"])
            self._sync_chunks(i, before, self.grids[i].blocks, reset_sim=False)
            self.controllers[i] = StandingPlayerController(self.controller_config,
                                                            copy.deepcopy(state["controller"]))
            self.player_shape_standing[i] = bool(state["controller"].crouch_amount <= 1e-5)
            from pxr import UsdPhysics
            standing_path, crouch_path = self.player_shape_paths[i]
            standing_enabled = bool(self.player_shape_standing[i])
            UsdPhysics.CollisionAPI(self.stage.GetPrimAtPath(standing_path)).GetCollisionEnabledAttr().Set(standing_enabled)
            UsdPhysics.CollisionAPI(self.stage.GetPrimAtPath(crouch_path)).GetCollisionEnabledAttr().Set(not standing_enabled)
            self.elapsed_steps[i] = state["elapsed_steps"]
            self.physics_ticks[i] = state["physics_ticks"]
            self.last_states[i] = copy.deepcopy(state["last_state"])
            roots.append(state["player"])
        root_state = torch.stack(roots).to(self.sim.device)
        self.sim.reset(soft=False)
        self.player.reset(env_ids=env_ids)
        self.player.write_root_link_state_to_sim(root_state, env_ids=env_ids)
        self.player.set_external_force_and_torque(
            torch.zeros((self.count, 1, 3), device=self.sim.device),
            torch.zeros((self.count, 1, 3), device=self.sim.device))
        self.player.write_data_to_sim()
        self.sim.play()
        self.player.update(self.cfg.physics_dt)

    def close(self):
        if self.closed: return
        self.closed = True
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()
