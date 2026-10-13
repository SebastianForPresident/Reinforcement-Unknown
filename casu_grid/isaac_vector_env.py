"""Batched Isaac Lab adapter: N independent canonical CasU worlds in one PhysX scene.

Every world is an :class:`~casu_grid.env_slot.EnvSlot` (own grid, RNG, body model,
runtime, scripted doors, counters). PhysX is advanced once per source fixed tick for
the whole batch; only the rigid-body step is shared.

Terrain modes
  * ``shared``      (default) one static terrain copy for all worlds. Worlds cannot
                    see each other: players and doors are placed in per-world collision
                    groups that are mutually filtered. Terrain edits are not supported.
  * ``independent`` each world gets its own terrain copy at a spatial offset, so
                    ``set_block`` / terrain restore work per world. Costs memory and
                    build time proportional to the world count.
"""
from __future__ import annotations

import copy
import time
from collections import defaultdict
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from .body_model import BodyConfig, PolicyAction
from .env_slot import EnvSlot
from .isaac_navigation_env import IsaacLayer1EnvConfig
from .player_controller import StandingAction
from .player_runtime import post_physics_phase, pre_physics_phase
from .reference import load_layer1_reference
from .world_objects import ground_polygons, load_physics_audit


class _SlotView:
    """What a single-world observation encoder expects from ``env`` (bound to one slot)."""

    def __init__(self, parent: "BatchedIsaacLayer1Env", index: int):
        self._parent, self._index = parent, index
        slot = parent.slots[index]
        self.slot = slot
        self.cfg, self.reference = parent.cfg, parent.reference
        self.controller_config = parent.controller_config

    grid = property(lambda self: self.slot.grid)
    model = property(lambda self: self.slot.model)
    elapsed_steps = property(lambda self: self.slot.elapsed_steps)

    def _body_state(self):
        return self.slot.body_dict()

    def _support_info(self, body=None):
        return self.slot.support_info()


class BatchedIsaacLayer1Env(gym.Env):
    """Single SimulationContext, N independently controlled CasU worlds."""

    metadata = {"render_modes": []}

    def __init__(self, config: IsaacLayer1EnvConfig, *,
                 action_decoder: Callable[[dict[str, Any]], StandingAction] | None = None,
                 observation_encoder: Callable[[Any], dict[str, Any]] | None = None,
                 reward_function: Callable[..., tuple[float, dict[str, float]]] | None = None,
                 terrain_mode: str = "shared"):
        if config.num_envs < 1:
            raise ValueError("BatchedIsaacLayer1Env requires num_envs >= 1")
        if config.seed != 391728405 or config.physics_dt != 0.02 or config.action_decimation != 10:
            raise ValueError("batched worlds require canonical seed, 20 ms ticks, and ten-tick actions")
        if terrain_mode not in {"shared", "independent"}:
            raise ValueError("terrain_mode must be 'shared' or 'independent'")
        if config.geometry_scope != "full":
            raise ValueError("the batched environment builds full canonical geometry")
        self.cfg = config
        self.terrain_mode = terrain_mode
        self.count = int(config.num_envs)
        self.reference = load_layer1_reference()
        self.controller_config = BodyConfig(step_up_clearance=config.step_up_clearance)
        self.action_decoder = action_decoder or self._default_action_decoder
        self.observation_encoder = observation_encoder or self._default_observation
        self.reward_function = reward_function or (lambda *_: (0.0, {}))
        self.closed = False
        self.timers: dict[str, float] = defaultdict(float)   # cumulative seconds per tick phase
        self.world_spacing = 1040.0
        self.offsets = np.zeros((self.count, 3), dtype=np.float32)
        if terrain_mode == "independent":
            self.offsets[:, 0] = np.arange(self.count, dtype=np.float32) * self.world_spacing

        self.world_objects = None
        object_polygons, door_defs = [], []
        if config.include_world_objects:
            self.world_objects, _ = load_physics_audit()
            door_defs = [c for c in self.world_objects.colliders if "shuttledoor" in c.name]
            object_polygons = ground_polygons(
                [c for c in self.world_objects.colliders if "shuttledoor" not in c.name])
        self.slots = [EnvSlot(config, self.reference, object_polygons=object_polygons, door_defs=door_defs,
                              fluid_seed=config.fluid_rng_seed + i, body_config=self.controller_config)
                      for i in range(self.count)]
        for i, slot in enumerate(self.slots):
            slot.on_doors_changed = lambda i=i: self._apply_door_state([i])
        self.views = [_SlotView(self, i) for i in range(self.count)]
        self.geometry_build_seconds = 0.0
        self._construct_simulation()
        self.action_space = spaces.Dict({
            "move_x": spaces.MultiDiscrete([3]), "jump": spaces.MultiDiscrete([2]),
            "crouch": spaces.MultiDiscrete([2]),
        })
        self.observation_space = spaces.Dict({
            "player_state": spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32),
        })

    # ------------------------------------------------------------ construction
    def _construct_simulation(self):
        import isaaclab.sim as sim_utils
        import isaacsim.core.utils.stage as stage_utils
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics
        from .collision_usd import author_layer1_box_collision, author_static_object_colliders
        from .isaac_player_body import RoundedBoxCollider

        if sim_utils.SimulationContext.instance() is not None:
            raise RuntimeError("Isaac SimulationContext already exists; close the previous environment first")
        device = "cuda:0" if self.cfg.use_cuda and torch.cuda.is_available() else "cpu"
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=self.cfg.physics_dt, device=device, gravity=(0.0, 0.0, -55.0),
            physx=sim_utils.PhysxCfg(enable_ccd=True, solver_type=self.cfg.physx_solver_type,
                                     enable_enhanced_determinism=self.cfg.physx_enhanced_determinism)))
        self.stage = stage_utils.get_current_stage()
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        self.material = sim_utils.RigidBodyMaterialCfg(
            static_friction=self.cfg.contact_friction,
            dynamic_friction=self.cfg.contact_friction, restitution=0.0)
        self.material_path = "/World/CasuPlayerMaterial"
        self.material.func(self.material_path, self.material)

        started = time.perf_counter()
        half = self.cfg.terrain_extrusion_depth / 2
        self.geometry_by_env = []
        static_objects = ([c for c in self.world_objects.colliders if "shuttledoor" not in c.name]
                          if self.world_objects is not None else [])
        terrain_roots = ([f"/World/Env_{i:05d}/CasUTerrain" for i in range(self.count)]
                         if self.terrain_mode == "independent" else ["/World/CasUTerrain"])
        for index, root in enumerate(terrain_roots):
            offset = tuple(float(v) for v in self.offsets[index])
            self.geometry_by_env.append(author_layer1_box_collision(
                self.stage, self.reference.terrain, half_thickness=half,
                physics_material=self.material, physics_material_path=self.material_path,
                root_path=root, world_offset=offset))
            if static_objects:
                objects_root = (f"/World/Env_{index:05d}/CasUObjects" if self.terrain_mode == "independent"
                                else "/World/CasUObjects")
                self.object_geometry = author_static_object_colliders(
                    self.stage, [self._shifted(c, offset) for c in static_objects], half_thickness=half,
                    physics_material_path=self.material_path, root_path=objects_root)
        self.geometry_build_seconds = time.perf_counter() - started

        rigid = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False, disable_gravity=False,
            max_depenetration_velocity=self.cfg.max_depenetration_velocity,
            solver_position_iteration_count=self.cfg.solver_position_iterations,
            solver_velocity_iteration_count=self.cfg.solver_velocity_iterations)
        body_cfg = sim_utils.CuboidCfg(
            size=(0.1, self.cfg.player_depth, 0.1), rigid_props=rigid,
            mass_props=sim_utils.MassPropertiesCfg(mass=self.controller_config.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            physics_material=self.material, physics_material_path=self.material_path)
        door_cfg = None
        if self.slots[0].door_defs:
            d0 = self.slots[0].door_defs[0]
            door_cfg = sim_utils.CuboidCfg(
                size=(d0.size[0], 2 * half, d0.size[1]),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                physics_material=self.material, physics_material_path=self.material_path)
        self.colliders = []
        self.group_paths = []
        for i, slot in enumerate(self.slots):
            local = slot.spawn_world_xy()
            position = (local[0] + float(self.offsets[i, 0]), float(self.offsets[i, 1]),
                        local[1] + float(self.offsets[i, 2]))
            path = f"/World/Env_{i:05d}/CasuPlayer"
            spawn_cfg = copy.deepcopy(body_cfg)
            spawn_cfg.func(path, spawn_cfg, translation=position)
            PhysxSchema.PhysxRigidBodyAPI.Apply(self.stage.GetPrimAtPath(path)).CreateEnableCCDAttr().Set(True)
            self.colliders.append(RoundedBoxCollider(
                self.stage, path, width=self.controller_config.orig_col_size[0],
                standing_height=self.controller_config.orig_col_size[1],
                edge_radius=self.controller_config.edge_radius, depth=self.cfg.player_depth,
                material_path=self.material_path, levels=self.cfg.crouch_levels,
                rest_offset=self.cfg.player_rest_offset, contact_offset=self.cfg.player_contact_offset))
            member_paths = [path]
            for k, door in enumerate(slot.door_defs):
                door_path = f"/World/Env_{i:05d}/ShuttleDoor{'LR'[k]}"
                dcfg = copy.deepcopy(door_cfg)
                dcfg.func(door_path, dcfg, translation=(door.center[0] + float(self.offsets[i, 0]),
                                                        0.0, door.center[1] + float(self.offsets[i, 2])))
                member_paths.append(door_path)
            self.group_paths.append(member_paths)
            anchor_path = f"/World/Env_{i:05d}/CasuPlanarAnchor"
            anchor = UsdGeom.Xform.Define(self.stage, anchor_path)
            UsdGeom.Xformable(anchor).AddTranslateOp().Set(Gf.Vec3d(*position))
            UsdPhysics.RigidBodyAPI.Apply(anchor.GetPrim()).CreateKinematicEnabledAttr(True)
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
        if self.terrain_mode == "shared":
            self._author_collision_groups(UsdPhysics)
        self.player = RigidObject(cfg=RigidObjectCfg(
            prim_path="/World/Env_[0-9]{5}/CasuPlayer", spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg()))
        self.doors = []
        if door_cfg is not None:
            for k in range(len(self.slots[0].door_defs)):
                self.doors.append(RigidObject(cfg=RigidObjectCfg(
                    prim_path=f"/World/Env_[0-9]{{5}}/ShuttleDoor{'LR'[k]}", spawn=None,
                    init_state=RigidObjectCfg.InitialStateCfg())))
        stage_utils.update_stage()
        self.sim.reset()
        self.sim.play()
        self.player.update(self.cfg.physics_dt)
        if self.player.num_instances != self.count:
            raise RuntimeError(f"expected {self.count} indexed player bodies, found {self.player.num_instances}")

    @staticmethod
    def _shifted(collider, offset):
        from dataclasses import replace
        if offset == (0.0, 0.0, 0.0):
            return collider
        return replace(collider, center=(collider.center[0] + offset[0], collider.center[1] + offset[2]))

    def _author_collision_groups(self, UsdPhysics):
        """Per-world collision groups, mutually filtered, so worlds share terrain but not each other."""
        group_paths = []
        for i, members in enumerate(self.group_paths):
            path = f"/World/CollisionGroups/World_{i:05d}"
            group = UsdPhysics.CollisionGroup.Define(self.stage, path)
            collection = group.GetCollidersCollectionAPI()
            collection.CreateIncludesRel().SetTargets(members)
            group_paths.append(path)
        for i, path in enumerate(group_paths):
            group = UsdPhysics.CollisionGroup(self.stage.GetPrimAtPath(path))
            group.CreateFilteredGroupsRel().SetTargets([p for j, p in enumerate(group_paths) if j != i])

    # --------------------------------------------------------------- physics I/O
    @property
    def episode_step_seconds(self):
        return self.cfg.physics_dt * self.cfg.action_decimation

    def _default_action_decoder(self, actions):
        def head(name, default):
            return np.asarray(actions.get(name, np.full(self.count, default))).reshape(self.count)
        move, jump, crouch = head("move_x", 1), head("jump", 0), head("crouch", 0)
        return [StandingAction(move_x=int(move[i]) - 1, jump=bool(jump[i]), crouch=bool(crouch[i]))
                for i in range(self.count)]

    def _decode_actions(self, actions):
        if self.action_decoder == self._default_action_decoder:
            return self._default_action_decoder(actions)
        decoded = []
        for i in range(self.count):
            row = {key: np.asarray(value)[i:i + 1] for key, value in actions.items()}
            decoded.append(self.action_decoder(row))
        return decoded

    def _default_observation(self, view):
        return {"player_state": view.slot.default_observation_vector()}

    def _body_state(self, i):
        return self.slots[i].body_dict()

    def _format_observation(self):
        rows = [self.observation_encoder(view) for view in self.views]
        return {key: np.stack([np.asarray(row[key], dtype=np.float32) for row in rows]) for key in rows[0]}

    def _apply_door_state(self, indices):
        for k, door_body in enumerate(self.doors):
            ids = torch.as_tensor(indices, dtype=torch.int32, device=self.sim.device)
            poses = torch.zeros((len(indices), 7), dtype=torch.float32, device=self.sim.device)
            for row, i in enumerate(indices):
                cx, cy = self.slots[i].shuttle_doors.centers()[k]
                poses[row] = torch.tensor([cx + float(self.offsets[i, 0]), 0.0, cy + float(self.offsets[i, 2]),
                                           1.0, 0.0, 0.0, 0.0])
            door_body.write_root_pose_to_sim(poses, env_ids=ids)
            door_body.write_data_to_sim()

    def _read_states(self):
        pos = self.player.data.root_pos_w
        vel = self.player.data.root_lin_vel_w
        out = []
        for i in range(self.count):
            out.append((float(pos[i, 0]) - float(self.offsets[i, 0]), float(pos[i, 2]) - float(self.offsets[i, 2]),
                        float(vel[i, 0]), float(vel[i, 2])))
        return out

    def _physics_tick(self, completed: int):
        clock = time.perf_counter
        timers = self.timers
        count, device = self.count, self.sim.device
        t0 = clock()
        poses = torch.zeros((count, 7), dtype=torch.float32, device=device)
        poses[:, 3] = 1.0
        vels = torch.zeros((count, 6), dtype=torch.float32, device=device)
        forces = torch.zeros((count, 1, 3), dtype=torch.float32, device=device)
        for i, slot in enumerate(self.slots):
            (fx, fy), _ = pre_physics_phase(slot.model, pre_tick_hook=slot.pre_tick, post_fixed_hook=slot.post_fixed)
            s = slot.model.s
            poses[i, 0], poses[i, 2] = s.x + float(self.offsets[i, 0]), s.y + float(self.offsets[i, 2])
            vels[i, 0], vels[i, 2] = s.vx, s.vy
            forces[i, 0, 0], forces[i, 0, 2] = fx, fy
        t1 = clock()
        current = self.player.data.root_pos_w
        moved = [i for i in range(count)
                 if abs(float(current[i, 0]) - float(poses[i, 0])) > 1e-6
                 or abs(float(current[i, 2]) - float(poses[i, 2])) > 1e-6]
        if moved:
            ids = torch.as_tensor(moved, dtype=torch.int32, device=device)
            self.player.write_root_pose_to_sim(poses[moved], env_ids=ids)
        self.player.write_root_velocity_to_sim(vels)
        self.player.set_external_force_and_torque(forces, torch.zeros_like(forces), is_global=True)
        self.player.write_data_to_sim()
        t2 = clock()
        self.sim.step(render=False)
        t3 = clock()
        self.player.update(self.cfg.physics_dt)
        states = self._read_states()
        t4 = clock()
        for slot, state, collider in zip(self.slots, states, self.colliders):
            s = slot.model.s
            s.x, s.y, s.vx, s.vy = state
            post_physics_phase(slot.model, slot.ground_world, slot.runtime, completed,
                               self.cfg.action_decimation, post_tick_hook=slot.post_tick)
            collider.set_level(collider.level_for_crouch(s.crouch_amount))
        t5 = clock()
        timers["controller_and_fluid_pre"] += t1 - t0
        timers["tensor_write"] += t2 - t1
        timers["physx_step"] += t3 - t2
        timers["tensor_read"] += t4 - t3
        timers["controller_post_and_shapes"] += t5 - t4

    # -------------------------------------------------------------------- lifecycle
    @staticmethod
    def _policy_action(action: StandingAction) -> PolicyAction:
        return PolicyAction(move=int(action.move_x), jump=int(bool(action.jump)),
                            vertical=int(action.move_y), crouch=int(bool(action.crouch)))

    def _write_slot_states(self, ids):
        """Write the logical body state of the given slots to PhysX and clear their forces."""
        env_ids = torch.as_tensor(ids, dtype=torch.int32, device=self.sim.device)
        pose = torch.zeros((len(ids), 7), dtype=torch.float32, device=self.sim.device)
        pose[:, 3] = 1.0
        velocity = torch.zeros((len(ids), 6), dtype=torch.float32, device=self.sim.device)
        for row, i in enumerate(ids):
            s = self.slots[i].model.s
            pose[row, 0], pose[row, 2] = s.x + float(self.offsets[i, 0]), s.y + float(self.offsets[i, 2])
            velocity[row, 0], velocity[row, 2] = s.vx, s.vy
        self.player.write_root_pose_to_sim(pose, env_ids=env_ids)
        self.player.write_root_velocity_to_sim(velocity, env_ids=env_ids)
        self.player.set_external_force_and_torque(
            torch.zeros((len(ids), 1, 3), device=self.sim.device),
            torch.zeros((len(ids), 1, 3), device=self.sim.device), env_ids=env_ids)
        self.player.write_data_to_sim()
        for i in ids:
            self.colliders[i].set_level(0)   # standing-sized collider, as at the source's first tick
        if self.doors:
            self._apply_door_state(list(ids))

    def _reset_indices(self, indices):
        ids = [int(i) for i in indices]
        if not ids:
            return
        changed = [i for i in ids if self.slots[i].reset_logical()]
        if changed:
            self._rebuild_dirty(changed)
        self._write_slot_states(ids)
        self.player.update(self.cfg.physics_dt)
        for i in ids:
            self.slots[i].last_state = self.slots[i].body_dict()

    def reset(self, *, seed=None, options=None):
        self._reset_indices(range(self.count))
        return self._format_observation(), {"seed": seed, "canonical_content_id": self.reference.manifest["content_id"]}

    def reset_done(self, mask):
        mask = np.asarray(mask, dtype=bool).reshape(self.count)
        self._reset_indices(np.flatnonzero(mask))
        return self._format_observation()

    def set_block(self, env_index: int, x: int, y: int, block_id: int):
        """Edit one world's terrain and rebuild its changed collision chunk."""
        if self.terrain_mode != "independent":
            raise NotImplementedError("terrain edits need terrain_mode='independent' (shared terrain is static)")
        if not 0 <= env_index < self.count:
            raise IndexError("environment index outside vector batch")
        self.slots[env_index].grid.set_block(x, y, block_id)
        self._rebuild_dirty([env_index])

    def _rebuild_dirty(self, indices):
        if self.terrain_mode != "independent":
            return
        from .collision_usd import author_layer1_box_collision
        any_change = False
        for i in indices:
            grid = self.slots[i].grid
            root = f"/World/Env_{i:05d}/CasUTerrain"
            offset = tuple(float(v) for v in self.offsets[i])
            for cx, cy in sorted(grid.dirty_chunks):
                path = f"{root}/Chunk_{cx}_{cy}"
                prim = self.stage.GetPrimAtPath(path)
                if prim and prim.IsValid():
                    self.stage.RemovePrim(path)
                if np.count_nonzero(grid.blocks[cx*64:(cx+1)*64, cy*64:(cy+1)*64]):
                    author_layer1_box_collision(
                        self.stage, grid.blocks, half_thickness=self.cfg.terrain_extrusion_depth / 2,
                        physics_material=self.material, physics_material_path=self.material_path,
                        root_path=root, world_offset=offset, only_chunks=[(cx, cy)])
                any_change = True
            grid.dirty_chunks.clear()
        if any_change:
            root_state = self.player.data.root_link_state_w.detach().clone()
            self.sim.reset(soft=False)
            self.player.reset()
            ids = torch.arange(self.count, dtype=torch.int32, device=self.sim.device)
            self.player.write_root_link_state_to_sim(root_state, env_ids=ids)
            self.player.write_data_to_sim()
            self.sim.play()
            self.player.update(self.cfg.physics_dt)

    def step(self, action_dict):
        actions = self._decode_actions(action_dict)
        if len(actions) != self.count:
            raise ValueError("action decoder must return one action per environment")
        for slot, action in zip(self.slots, actions):
            slot.model.apply_actions(latch=self._policy_action(action))
        for tick in range(self.cfg.action_decimation):
            self._physics_tick(tick + 1)
        rewards = np.zeros(self.count, dtype=np.float32)
        terminated = np.zeros(self.count, dtype=bool)
        truncated = np.zeros(self.count, dtype=bool)
        terms: dict[str, np.ndarray] = {}
        reasons: list = [None] * self.count
        players = []
        for i, slot in enumerate(self.slots):
            reward, item_terms, term, trunc, reason, current = slot.finish_step(self.reward_function)
            rewards[i], terminated[i], truncated[i], reasons[i] = reward, term, trunc, reason
            players.append(current)
            for key, value in item_terms.items():
                terms.setdefault(key, np.zeros(self.count, np.float32))[i] = value
        return self._format_observation(), rewards, terminated, truncated, {
            "reward_terms": terms, "termination_reason": reasons,
            "physics_ticks": self.cfg.action_decimation * self.count, "player": players,
            "canonical_content_id": self.reference.manifest["content_id"]}

    # ---------------------------------------------------------------- snapshot
    def snapshot(self):
        roots = self.player.data.root_link_state_w.detach().cpu().clone()
        slots = []
        for i, slot in enumerate(self.slots):
            snap = slot.snapshot_logical()
            snap["player_root_state_wxyz"] = roots[i].clone()
            snap["last_state"] = copy.deepcopy(slot.last_state)
            slots.append(snap)
        return {"schema": "casu.isaac-vector-state.v2", "slots": slots, "terrain_mode": self.terrain_mode,
                "canonical_content_id": self.reference.manifest["content_id"]}

    def restore(self, snapshot):
        if snapshot.get("schema") != "casu.isaac-vector-state.v2" or len(snapshot.get("slots", [])) != self.count:
            raise ValueError("snapshot has wrong vector schema or environment count")
        if snapshot.get("canonical_content_id") != self.reference.manifest["content_id"]:
            raise ValueError("snapshot belongs to another canonical world")
        changed = []
        for i, (slot, snap) in enumerate(zip(self.slots, snapshot["slots"])):
            if slot.restore_logical(snap):
                changed.append(i)
            slot.last_state = copy.deepcopy(snap["last_state"])
        if changed:
            self._rebuild_dirty(changed)
        roots = torch.stack([snap["player_root_state_wxyz"] for snap in snapshot["slots"]]).to(self.sim.device)
        env_ids = torch.arange(self.count, dtype=torch.int32, device=self.sim.device)
        # reset PhysX contact caches so a restored branch does not inherit warm-start impulses
        self.sim.reset(soft=False)
        self.player.reset(env_ids=env_ids)
        self._flush_contact_history()
        self.player.write_root_link_state_to_sim(roots, env_ids=env_ids)
        self.player.set_external_force_and_torque(
            torch.zeros((self.count, 1, 3), device=self.sim.device),
            torch.zeros((self.count, 1, 3), device=self.sim.device))
        self.player.write_data_to_sim()
        if self.doors:
            self._apply_door_state(list(range(self.count)))
        for slot, collider in zip(self.slots, self.colliders):
            collider.set_level(collider.level_for_crouch(slot.model.s.crouch_amount))
        self.sim.play()
        self.player.update(self.cfg.physics_dt)

    def _flush_contact_history(self):
        """Drop persistent PhysX contact pairs that survive ``sim.reset``.

        Without this, a restored branch's floating-point trajectory can depend on what the
        *other* worlds did before the restore (PhysX pair ordering), which breaks exact
        branch reproducibility. Bodies are parked far from any geometry for two steps.
        """
        count, device = self.count, self.sim.device
        far = torch.zeros((count, 7), dtype=torch.float32, device=device)
        far[:, 3] = 1.0
        far[:, 0] = 1.0e5 + torch.arange(count, dtype=torch.float32, device=device) * 100.0
        far[:, 2] = 1.0e5
        ids = torch.arange(count, dtype=torch.int32, device=device)
        for _ in range(2):
            self.player.write_root_pose_to_sim(far, env_ids=ids)
            self.player.write_root_velocity_to_sim(torch.zeros((count, 6), device=device), env_ids=ids)
            self.player.set_external_force_and_torque(torch.zeros((count, 1, 3), device=device),
                                                      torch.zeros((count, 1, 3), device=device))
            self.player.write_data_to_sim()
            self.sim.step(render=False)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()
