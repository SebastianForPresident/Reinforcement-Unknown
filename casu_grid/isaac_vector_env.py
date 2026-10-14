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
import os
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
from .isaac_player_body import set_levels
from .player_controller import StandingAction
from .player_runtime import post_physics_phase, pre_physics_phase
from .reference import load_layer1_reference
from .xy1_mini_world import load_mini_world_reference
from .world_objects import ground_polygons, load_physics_audit


_INCREMENTAL_TERRAIN = os.environ.get("CASU_INCREMENTAL_TERRAIN", "1") != "0"   # diff-based collider updates after excavation


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

    def __new__(cls, config=None, *args, **kwargs):
        # ``config.num_workers > 1`` transparently returns the multi-process wrapper (same API),
        # so existing launchers need no change. Workers construct this class with num_workers=1.
        if cls is BatchedIsaacLayer1Env and config is not None and getattr(config, "num_workers", 1) != 1:
            from .isaac_sharded_env import ShardedBatchedEnv, resolve_worker_count
            if resolve_worker_count(config) > 1:
                return ShardedBatchedEnv(config, *args, **kwargs)
        return super().__new__(cls)    # one process (num_workers=1, or auto that resolved to 1)

    def __init__(self, config: IsaacLayer1EnvConfig, *,
                 action_decoder: Callable[[dict[str, Any]], StandingAction] | None = None,
                 observation_encoder: Callable[[Any], dict[str, Any]] | None = None,
                 reward_function: Callable[..., tuple[float, dict[str, float]]] | None = None,
                 terrain_mode: str = "shared", world_offset: int = 0):
        if config.num_envs < 1:
            raise ValueError("BatchedIsaacLayer1Env requires num_envs >= 1")
        if config.seed != 391728405 or config.physics_dt != 0.02 or config.action_decimation != 10:
            raise ValueError("batched worlds require canonical seed, 20 ms ticks, and ten-tick actions")
        if terrain_mode not in {"shared", "independent"}:
            raise ValueError("terrain_mode must be 'shared' or 'independent'")
        if config.geometry_scope != "full":
            raise ValueError("the batched environment builds full canonical geometry")
        if config.navigation_mode and terrain_mode != "independent":
            raise ValueError("XY1 navigation requires independent terrain for per-world excavation")
        self.cfg = config
        self.terrain_mode = terrain_mode
        self.world_offset = int(world_offset)   # global index of this batch's first world (fluid seeds)
        self.np_random = np.random.default_rng(config.seed + self.world_offset)
        self.count = int(config.num_envs)
        self.reference = (load_mini_world_reference(config.seed)
                          if config.world_variant == "xy1_mini_128x128" else load_layer1_reference())
        self.controller_config = BodyConfig(step_up_clearance=config.step_up_clearance)
        self.action_decoder = action_decoder or self._default_action_decoder
        self.observation_encoder = observation_encoder or self._default_observation
        self.reward_function = reward_function or (lambda *_: (0.0, {}))
        self.closed = False
        self.timers: dict[str, float] = defaultdict(float)   # cumulative seconds per tick phase
        self.counters: dict[str, int] = defaultdict(int)     # diagnostic event counts (e.g. crouch-shape level changes)
        self.world_spacing = 1040.0
        self.offsets = np.zeros((self.count, 3), dtype=np.float32)
        if terrain_mode == "independent":
            self.offsets[:, 0] = np.arange(self.count, dtype=np.float32) * self.world_spacing

        self.world_objects = None
        object_polygons, door_defs = [], []
        if config.include_world_objects and config.world_variant == "canonical":
            self.world_objects, _ = load_physics_audit()
            door_defs = [c for c in self.world_objects.colliders if "shuttledoor" in c.name]
            object_polygons = ground_polygons(
                [c for c in self.world_objects.colliders if "shuttledoor" not in c.name])
        self.slot_references = [
            (load_mini_world_reference(config.seed + self.world_offset + i)
             if config.world_variant == "xy1_mini_128x128" else self.reference)
            for i in range(self.count)]
        self.slots = [EnvSlot(config, ref, object_polygons=object_polygons, door_defs=door_defs,
                              fluid_seed=config.fluid_rng_seed + self.world_offset + i,
                              body_config=self.controller_config)
                      for i, ref in enumerate(self.slot_references)]
        for i, slot in enumerate(self.slots):
            slot.on_doors_changed = lambda i=i: self._apply_door_state([i])
        self.views = [_SlotView(self, i) for i in range(self.count)]
        self._fast = self._build_fast_runtime()
        self.geometry_build_seconds = 0.0
        self._construct_simulation()
        self.action_space = spaces.Dict({"move_x": spaces.Discrete(3), "jump": spaces.Discrete(2),
            "crouch": spaces.Discrete(2)})
        self.observation_space = spaces.Dict({
            "player_state": spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32),
        })
        if self.cfg.navigation_mode:
            self.action_space=spaces.Dict({"move_x":spaces.Discrete(3),"jump":spaces.Discrete(2),
                "move_y":spaces.Discrete(3),"crouch":spaces.Discrete(2),"look_dx":spaces.Discrete(9),
                "look_dy":spaces.Discrete(11),"attack":spaces.Discrete(2)})
            window = tuple(self.cfg.observation_window_tiles)
            self.observation_space=spaces.Dict({"spatial":spaces.Box(0,1,shape=(16,*window),dtype=np.float32),
                "goal_relative":spaces.Box(-np.inf,np.inf,shape=(2,),dtype=np.float32)})

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
            physx=sim_utils.PhysxCfg(enable_ccd=self.cfg.physx_ccd, solver_type=self.cfg.physx_solver_type,
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
            if self.terrain_mode == "independent":
                # Terrain copies differ only by a spatial offset: author world 0 once in local coordinates
                # and Sdf-copy the finished prim tree for the others; each copy's root Xform carries its
                # offset (USD adds it to the same local translations, so colliders land on identical doubles).
                if self.cfg.world_variant == "xy1_mini_128x128":
                    started_world = time.perf_counter()
                    stats = author_layer1_box_collision(
                        self.stage, self.slot_references[index].terrain, half_thickness=half,
                        physics_material=self.material, physics_material_path=self.material_path,
                        root_path=root, contact_offset=self.cfg.terrain_contact_offset)
                    stats["build_seconds"] = time.perf_counter() - started_world
                elif index == 0:
                    stats = author_layer1_box_collision(
                        self.stage, self.reference.terrain, half_thickness=half,
                        physics_material=self.material, physics_material_path=self.material_path,
                        root_path=root, contact_offset=self.cfg.terrain_contact_offset)
                else:
                    started_copy = time.perf_counter()
                    self._copy_terrain(terrain_roots[0], root)
                    stats = dict(self.geometry_by_env[0], build_seconds=time.perf_counter() - started_copy,
                                 copied_from=terrain_roots[0])
                self._set_root_offset(root, offset)
                self.geometry_by_env.append(stats)
            else:
                self.geometry_by_env.append(author_layer1_box_collision(
                    self.stage, self.slots[index].physical_blocks, half_thickness=half,
                    physics_material=self.material, physics_material_path=self.material_path,
                    root_path=root, world_offset=offset, contact_offset=self.cfg.terrain_contact_offset))
            if static_objects:
                objects_root = (f"/World/Env_{index:05d}/CasUObjects" if self.terrain_mode == "independent"
                                else "/World/CasUObjects")
                self.object_geometry = author_static_object_colliders(
                    self.stage, [self._shifted(c, offset) for c in static_objects], half_thickness=half,
                    physics_material_path=self.material_path, root_path=objects_root,
                    contact_offset=self.cfg.terrain_contact_offset)
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
            PhysxSchema.PhysxRigidBodyAPI.Apply(self.stage.GetPrimAtPath(path)).CreateEnableCCDAttr().Set(bool(self.cfg.physx_ccd))
            self.colliders.append(RoundedBoxCollider(
                self.stage, path, width=self.controller_config.orig_col_size[0],
                standing_height=self.controller_config.orig_col_size[1],
                edge_radius=self.controller_config.edge_radius, depth=self.cfg.player_depth,
                material_path=self.material_path, levels=self.cfg.crouch_levels,
                rest_offset=self.cfg.player_rest_offset, contact_offset=self.cfg.player_contact_offset,
                representation=self.cfg.player_collider, hull_segments=self.cfg.player_hull_segments))
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

    def _copy_terrain(self, source_root: str, destination_root: str) -> None:
        from pxr import Sdf
        layer = self.stage.GetEditTarget().GetLayer()
        parent = destination_root.rsplit("/", 1)[0]
        self.stage.DefinePrim(parent)
        Sdf.CopySpec(layer, Sdf.Path(source_root), layer, Sdf.Path(destination_root))

    def _set_root_offset(self, root: str, offset) -> None:
        if offset == (0.0, 0.0, 0.0):
            return
        from pxr import Gf, UsdGeom
        xform = UsdGeom.Xformable(self.stage.GetPrimAtPath(root))
        xform.AddTranslateOp().Set(Gf.Vec3d(*offset))

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

    def slot_summaries(self) -> list[dict]:
        """Compact per-world discrete state (crouch, grounded, fluid tick and digest) for validation tools."""
        import hashlib
        return [{"crouch": float(s.model.s.crouch_amount), "grounded": bool(s.model.s.grounded),
                 "fluid_tick": int(s.grid.tick),
                 "fluid_digest": hashlib.sha256(s.grid.fluids.tobytes()).hexdigest()[:16],
                 "shuttle_doors": None if s.shuttle_doors is None else s.shuttle_doors.snapshot(),
                 "elapsed_steps": int(s.elapsed_steps)} for s in self.slots]

    def _build_fast_runtime(self):
        """Compiled per-world tick runtime (``fast_runtime.FastSlot``), or ``None`` for the reference path."""
        from . import backend
        controller = backend.controller_for()
        if controller is None:
            return None
        from .fast_runtime import FastSlot
        self._state_rows = np.zeros((self.count, controller.N_STATE))
        self._result_rows = np.zeros((self.count, 4))
        vector = controller.pack_config(self.controller_config, self.slots[0].ground_world.blocks.shape)
        return [FastSlot(slot, self._state_rows[i], vector, self.cfg.action_decimation, self._result_rows[i])
                for i, slot in enumerate(self.slots)]

    def performance_info(self) -> dict:
        """Execution layout and compute backend, for run metadata and benchmark results."""
        from . import backend
        return {"backend": backend.describe(), "num_workers": 1, "worlds": self.count,
                "compiled_controller": self._fast is not None,
                "terrain_mode": self.terrain_mode, "worlds_per_worker": [self.count]}

    # --------------------------------------------------------------- physics I/O
    @property
    def episode_step_seconds(self):
        return self.cfg.physics_dt * self.cfg.action_decimation

    def _default_action_decoder(self, actions):
        def head(name, default):
            return np.asarray(actions.get(name, np.full(self.count, default))).reshape(self.count)
        move, jump, crouch = head("move_x", 1), head("jump", 0), head("crouch", 0)
        vertical, attack = head("move_y", 1), head("attack", 0)
        return [StandingAction(move_x=int(move[i]) - 1, jump=bool(jump[i]), crouch=bool(crouch[i]),
                               move_y=int(vertical[i]) - 1, attack=bool(attack[i]),
                               look_dx=int(head("look_dx", 4)[i])-4,
                               look_dy=int(head("look_dy", 5)[i])-5)
                for i in range(self.count)]

    def _decode_actions(self, actions):
        if self.action_decoder == self._default_action_decoder:
            return self._default_action_decoder(actions)
        decoded = []
        for i in range(self.count):
            row = {key: np.asarray(value).reshape(self.count,-1)[i,0:1] for key, value in actions.items()}
            decoded.append(self.action_decoder(row))
        return decoded

    def _default_observation(self, view):
        if self.cfg.navigation_mode:
            from .xy1 import encode_xy1_observation
            return encode_xy1_observation(view)
        return {"player_state": view.slot.default_observation_vector()}

    def _body_state(self, i):
        return self.slots[i].body_dict()

    def _format_observation(self):
        rows = [self.observation_encoder(view) for view in self.views]
        out={key: np.stack([np.asarray(row[key], dtype=np.float32) for row in rows]) for key in rows[0]}
        if self.cfg.navigation_mode:
            for i,slot in enumerate(self.slots):
                px,py=slot.grid.world_to_cell(slot.model.s.x,slot.model.s.y,slot.grid.width,slot.grid.height)
                out['goal_relative'][i]=np.asarray([slot.goal_cell[0]-px,slot.goal_cell[1]-py],dtype=np.float32)
        return out

    # ----------------------------------------------------- compact observation (XY1)
    @property
    def has_stock_xy1_observation(self) -> bool:
        """True when the observation encoder is the unmodified XY1 encoder, so the compact id-window
        representation (``xy1.expand_compact_observation``) reproduces the observation bit for bit."""
        encoder = self.observation_encoder
        while getattr(encoder, "casu_wrapped", None) is not None:
            encoder = encoder.casu_wrapped
        return (self.cfg.navigation_mode and getattr(encoder, "__module__", "") == "casu_grid.xy1_callbacks"
                and getattr(encoder, "__name__", "") == "observation_encoder")

    def local_rollout_setup(self, actions, observations, model, device="cuda"):
        from .workbench.local_rollout import LocalRollout
        self._local_rollout = LocalRollout(self, actions, observations, model, device)

    def local_rollout_run(self, weights, steps, seeds, deterministic, episode_returns, episode_lengths):
        self._local_rollout.load_weights(weights)
        result = self._local_rollout.run(steps, seeds[0], deterministic, episode_returns, episode_lengths)
        result["seconds"] = {k: [v] for k, v in result["seconds"].items()}
        return result

    def compact_observation(self):
        """Per-world (block ids, fluid ids, cursor cell, goal offsets) windows; see :mod:`casu_grid.xy1`."""
        from .xy1 import encode_xy1_compact
        parts = [encode_xy1_compact(view) for view in self.views]
        return {"block_ids": np.stack([p[0] for p in parts]), "fluid_ids": np.stack([p[1] for p in parts]),
                "cursor": np.asarray([p[2] for p in parts], dtype=np.int16),
                "goal": np.asarray([p[3] for p in parts], dtype=np.float32)}

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

    def _np(self, tensor):
        """Host NumPy view of a (possibly GPU) tensor; zero-copy on the CPU pipeline."""
        return tensor.numpy() if tensor.device.type == "cpu" else tensor.cpu().numpy()

    def _dev(self, tensor):
        return tensor if self.sim.device == "cpu" else tensor.to(self.sim.device)

    def _read_states(self):
        pos = self._np(self.player.data.root_pos_w)
        vel = self._np(self.player.data.root_lin_vel_w)
        x = pos[:, 0].astype(np.float64) - self.offsets[:, 0].astype(np.float64)
        y = pos[:, 2].astype(np.float64) - self.offsets[:, 2].astype(np.float64)
        return list(zip(x.tolist(), y.tolist(), vel[:, 0].astype(np.float64).tolist(),
                        vel[:, 2].astype(np.float64).tolist()))

    def _staging(self):
        """Persistent CPU staging buffers (numpy views over torch tensors) for the per-tick writes."""
        if getattr(self, "_stage_pose", None) is None:
            n = self.count   # staging lives on the host (NumPy views); GPU pipelines copy per tick via _dev()
            self._direct = (self._fast is not None and self.sim.device == "cpu"
                            and os.environ.get("CASU_DIRECT_TENSOR_IO", "1") != "0")
            self._stage_pose = torch.zeros((n, 7), dtype=torch.float32)
            self._stage_pose[:, 6 if self._direct else 3] = 1.0     # identity quaternion: xyzw (PhysX view) or wxyz (Isaac Lab API)
            self._stage_vel = torch.zeros((n, 6), dtype=torch.float32)
            self._stage_force = torch.zeros((n, 1, 3), dtype=torch.float32)
            self._stage_torque = torch.zeros((n, 1, 3), dtype=torch.float32)
            self._physx_iface = None
            if self._direct and os.environ.get("CASU_DIRECT_STEP", "1") != "0":
                # ``sim.step(render=False)`` = Isaac Lab/Isaac Sim wrappers (stage and timeline queries, a USD read of the physics dt)
                # around ``simulate(dt, time)`` + ``fetch_results()``; the fast path issues those two calls itself
                import omni.physx
                self._physx_iface = omni.physx.get_physx_simulation_interface()
                self._physics_dt_exact = self.sim.get_physics_dt()
            if self._direct:
                self._pview = self.player.root_physx_view
                self._idx_all = torch.arange(n, dtype=torch.int32)
                self._force_flat = self._stage_force.view(n, 3)
                self._torque_flat = self._stage_torque.view(n, 3)
            self._np_pose, self._np_vel, self._np_force = (
                self._stage_pose.numpy(), self._stage_vel.numpy(), self._stage_force.numpy())
            self._off_x = self.offsets[:, 0].astype(np.float64)
            self._off_z = self.offsets[:, 2].astype(np.float64)
        return self._stage_pose

    def _physics_tick(self, completed: int):
        if self._fast is None:
            return self._physics_tick_reference(completed)
        from . import controller_kernels as CK
        clock = time.perf_counter
        timers = self.timers
        self._staging()
        S, R = self._state_rows, self._result_rows
        t0 = clock()
        for fast in self._fast:
            fast.pre_tick()
        pose, vel, force = self._np_pose, self._np_vel, self._np_force
        pose[:, 0] = np.add(S[:, CK.X], self._off_x)          # float64 sum, rounded to float32 on store
        pose[:, 2] = np.add(S[:, CK.Y], self._off_z)
        vel[:, 0], vel[:, 2] = S[:, CK.VX], S[:, CK.VY]
        force[:, 0, 0], force[:, 0, 2] = R[:, 0], R[:, 1]
        t1 = clock()
        if self.terrain_mode == "independent":
            stale = [i for i, slot in enumerate(self.slots) if slot.physical_dirty]
            if stale:
                self._rebuild_dirty(stale)
        if self._direct:
            # same PhysX tensor-view calls Isaac Lab's RigidObject wrappers issue, without their per-call bookkeeping
            view = self._pview
            current = view.get_transforms().numpy()
            moved = np.flatnonzero((np.abs(current[:, 0].astype(np.float64) - pose[:, 0].astype(np.float64)) > 1e-6)
                                   | (np.abs(current[:, 2].astype(np.float64) - pose[:, 2].astype(np.float64)) > 1e-6))
            if moved.size:
                view.set_transforms(self._stage_pose, indices=torch.as_tensor(moved, dtype=torch.int32))
            view.set_velocities(self._stage_vel, indices=self._idx_all)
            view.apply_forces_and_torques_at_position(force_data=self._force_flat, torque_data=self._torque_flat,
                                                      position_data=None, indices=self._idx_all, is_global=True)
            t2 = clock()
            if self._physx_iface is not None and self.sim.is_playing():
                self._physx_iface.simulate(self._physics_dt_exact, self.sim.current_time)
                self._physx_iface.fetch_results()
            else:
                self.sim.step(render=False)
            t3 = clock()
            self.player.update(self.cfg.physics_dt)
            pos = view.get_transforms().numpy()
            lin = view.get_velocities().numpy()
        else:
            current = self._np(self.player.data.root_pos_w)
            moved = np.flatnonzero((np.abs(current[:, 0].astype(np.float64) - pose[:, 0].astype(np.float64)) > 1e-6)
                                   | (np.abs(current[:, 2].astype(np.float64) - pose[:, 2].astype(np.float64)) > 1e-6))
            if moved.size:
                ids = torch.as_tensor(moved, dtype=torch.int32, device=self.sim.device)
                self.player.write_root_pose_to_sim(self._dev(self._stage_pose[torch.as_tensor(moved, dtype=torch.long)]), env_ids=ids)
            self.player.write_root_velocity_to_sim(self._dev(self._stage_vel))
            self.player.set_external_force_and_torque(self._dev(self._stage_force), self._dev(self._stage_torque), is_global=True)
            self.player.write_data_to_sim()
            t2 = clock()
            self.sim.step(render=False)
            t3 = clock()
            self.player.update(self.cfg.physics_dt)
            pos = self._np(self.player.data.root_pos_w)
            lin = self._np(self.player.data.root_lin_vel_w)
        S[:, CK.X] = pos[:, 0].astype(np.float64) - self._off_x
        S[:, CK.Y] = pos[:, 2].astype(np.float64) - self._off_z
        S[:, CK.VX] = lin[:, 0]
        S[:, CK.VY] = lin[:, 2]
        t4 = clock()
        for fast in self._fast:
            fast.post_tick(completed)
        t4b = clock()
        crouch = S[:, CK.CROUCH]
        levels = np.rint(np.clip(crouch, 0.0, 1.0) * self.cfg.crouch_levels).astype(np.int64).tolist()
        changes = sum(1 for c, lv in zip(self.colliders, levels) if lv != c.active_level)
        set_levels(self.colliders, levels)
        t5 = clock()
        # diagnostics (sub-intervals of controller_post_and_shapes, not additional wall time)
        timers["diag_controller_post_kernels"] += t4b - t4
        timers["diag_crouch_shape_switch"] += t5 - t4b
        self.counters["shape_changes"] += changes
        timers["controller_and_fluid_pre"] += t1 - t0
        timers["tensor_write"] += t2 - t1
        timers["physx_step"] += t3 - t2
        timers["tensor_read"] += t4 - t3
        timers["controller_post_and_shapes"] += t5 - t4

    def _physics_tick_reference(self, completed: int):
        clock = time.perf_counter
        timers = self.timers
        count = self.count
        self._staging()
        t0 = clock()
        xs, ys, vxs, vys, fxs, fys = [], [], [], [], [], []
        for slot in self.slots:
            (fx, fy), _ = pre_physics_phase(slot.model, pre_tick_hook=slot.pre_tick, post_fixed_hook=slot.post_fixed)
            s = slot.model.s
            xs.append(s.x); ys.append(s.y); vxs.append(s.vx); vys.append(s.vy); fxs.append(fx); fys.append(fy)
        pose, vel, force = self._np_pose, self._np_vel, self._np_force
        pose[:, 0] = np.add(xs, self._off_x)          # float64 sum, rounded to float32 on store
        pose[:, 2] = np.add(ys, self._off_z)
        vel[:, 0], vel[:, 2] = vxs, vys
        force[:, 0, 0], force[:, 0, 2] = fxs, fys
        t1 = clock()
        if self.terrain_mode == "independent":
            stale = [i for i, slot in enumerate(self.slots) if slot.physical_dirty]
            if stale:
                self._rebuild_dirty(stale)
        current = self._np(self.player.data.root_pos_w)
        moved = np.flatnonzero((np.abs(current[:, 0].astype(np.float64) - pose[:, 0].astype(np.float64)) > 1e-6)
                               | (np.abs(current[:, 2].astype(np.float64) - pose[:, 2].astype(np.float64)) > 1e-6))
        if moved.size:
            ids = torch.as_tensor(moved, dtype=torch.int32, device=self.sim.device)
            self.player.write_root_pose_to_sim(self._dev(self._stage_pose[torch.as_tensor(moved, dtype=torch.long)]), env_ids=ids)
        self.player.write_root_velocity_to_sim(self._dev(self._stage_vel))
        self.player.set_external_force_and_torque(self._dev(self._stage_force), self._dev(self._stage_torque), is_global=True)
        self.player.write_data_to_sim()
        t2 = clock()
        self.sim.step(render=False)
        t3 = clock()
        self.player.update(self.cfg.physics_dt)
        states = self._read_states()
        t4 = clock()
        levels = []
        for slot, state, collider in zip(self.slots, states, self.colliders):
            s = slot.model.s
            s.x, s.y, s.vx, s.vy = state
            post_physics_phase(slot.model, slot.ground_world, slot.runtime, completed,
                               self.cfg.action_decimation, post_tick_hook=slot.post_tick)
            levels.append(collider.level_for_crouch(s.crouch_amount))
        set_levels(self.colliders, levels)
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
                            vertical=int(action.move_y), crouch=int(bool(action.crouch)),
                            look_dx=action.look_dx, look_dy=action.look_dy)

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
            self.colliders[i].set_level(self.colliders[i].level_for_crouch(self.slots[i].model.s.crouch_amount))
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

    def reset(self, *, seed=None, options=None, observe=True):
        self._reset_indices(range(self.count))
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        if self.cfg.navigation_mode:
            for i,slot in enumerate(self.slots):
                rng = np.random.default_rng(int(self.np_random.integers(0, 2**32-1)))
                spawn = self._mini_center_spawn(slot, i) if self.cfg.world_variant == "xy1_mini_128x128" else self._sample_clear_spawn(slot, rng)
                slot.spawn_xy = spawn
                slot.goal_cell, slot.goal_world_xy = self._sample_goal(slot, rng, spawn)
                local_spawn=(spawn[0]-float(self.offsets[i,0]),spawn[1]-float(self.offsets[i,2]))
                slot.goal_world_xy=(slot.goal_world_xy[0]-float(self.offsets[i,0]),slot.goal_world_xy[1]-float(self.offsets[i,2]))
                slot.best_goal_distance = self._goal_distance(local_spawn, slot.goal_world_xy)
                from .xy1 import BestDistanceReward
                slot.xy1_reward=BestDistanceReward(slot.best_goal_distance)
                s = slot.model.s
                s.x, s.y, s.vx, s.vy = *local_spawn, 0.0, 0.0
                s.grounded = False; s.crouch_amount = 0.0; s.ragdoll_event = None
                slot.last_state=slot.body_dict()
            self._write_slot_states(range(self.count))
        return (self._format_observation() if observe else None), {"seed": seed, "canonical_content_id": self.reference.manifest["content_id"]}

    def _sample_clear_spawn(self, slot, rng, index=None):
        hw, hh = self.controller_config.orig_col_size[0]/2, self.controller_config.orig_col_size[1]/2
        index=self.slots.index(slot) if index is None else int(index)
        ox=float(self.offsets[index,0]);oy=float(self.offsets[index,2])
        for _ in range(self.cfg.spawn_clearance_attempts):
            gx, gy = int(rng.integers(0, slot.grid.width)), int(rng.integers(0, slot.grid.height))
            xy = (gx-slot.grid.width/2+0.5+ox, gy-slot.grid.height/2+0.5+oy)
            if not slot.ground_world.overlap_box(xy, (2*hw-0.02, 2*hh-0.02)):
                return tuple(map(float, xy))
        raise RuntimeError("could not find a full-body-clear randomized spawn")

    def _sample_goal(self, slot, rng, spawn):
        if self.cfg.world_variant == "xy1_mini_128x128":
            px, py = slot.reference.spawn_grid_xy
            window_w, window_h = self.cfg.observation_window_tiles
            x0, y0 = px - window_w // 2, py - window_h // 2
            gx = int(rng.integers(x0, x0 + window_w))
            gy = int(rng.integers(y0, y0 + window_h))
            return (gx,gy),(gx-slot.grid.width/2+0.5,gy-slot.grid.height/2+0.5)
        px, py = slot.grid.world_to_cell(*spawn, slot.grid.width, slot.grid.height)
        gx = min(max(px + int(rng.integers(-self.cfg.goal_radius_x,self.cfg.goal_radius_x+1)),0),slot.grid.width-1)
        gy = min(max(py + int(rng.integers(-self.cfg.goal_radius_y,self.cfg.goal_radius_y+1)),0),slot.grid.height-1)
        return (gx,gy),(gx-slot.grid.width/2+0.5,gy-slot.grid.height/2+0.5)

    def _mini_center_spawn(self, slot, index):
        x, y = slot.reference.spawn_world_xy
        return float(x), float(y)

    @staticmethod
    def _goal_distance(position, goal):
        return float(np.hypot(position[0]-goal[0],position[1]-goal[1]))

    def reset_done(self, mask, observe=True):
        mask = np.asarray(mask, dtype=bool).reshape(self.count)
        ids=np.flatnonzero(mask);self._reset_indices(ids)
        if self.cfg.navigation_mode:
            for i in ids:
                slot=self.slots[int(i)];rng=np.random.default_rng(int(self.np_random.integers(0,2**32-1)))
                spawn=(self._mini_center_spawn(slot,int(i)) if self.cfg.world_variant == "xy1_mini_128x128"
                       else self._sample_clear_spawn(slot,rng,index=i));slot.spawn_xy=spawn
                slot.goal_cell,slot.goal_world_xy=self._sample_goal(slot,rng,spawn)
                local_spawn=(spawn[0]-float(self.offsets[int(i),0]),spawn[1]-float(self.offsets[int(i),2]))
                slot.goal_world_xy=(slot.goal_world_xy[0]-float(self.offsets[int(i),0]),slot.goal_world_xy[1]-float(self.offsets[int(i),2]))
                slot.best_goal_distance=self._goal_distance(local_spawn,slot.goal_world_xy)
                from .xy1 import BestDistanceReward
                slot.xy1_reward=BestDistanceReward(slot.best_goal_distance)
                s=slot.model.s;s.x,s.y=local_spawn;s.vx=s.vy=0.0;s.grounded=False;s.crouch_amount=0.0;s.ragdoll_event=None
                slot.last_state=slot.body_dict()
            self._write_slot_states(ids)
        return self._format_observation() if observe else None

    def set_block(self, env_index: int, x: int, y: int, block_id: int):
        """Edit one world's terrain and rebuild its changed collision chunk."""
        if self.terrain_mode != "independent":
            raise NotImplementedError("terrain edits need terrain_mode='independent' (shared terrain is static)")
        if not 0 <= env_index < self.count:
            raise IndexError("environment index outside vector batch")
        self.slots[env_index].edit_block(x, y, block_id)
        self._rebuild_dirty([env_index])

    def _rebuild_dirty(self, indices):
        if self.terrain_mode != "independent":
            return
        from .collision_usd import author_layer1_box_collision
        any_change = False
        for i in indices:
            slot = self.slots[i]
            root = f"/World/Env_{i:05d}/CasUTerrain"
            offset = tuple(float(v) for v in self.offsets[i])
            for cx, cy in sorted(slot.physical_dirty):
                path = f"{root}/Chunk_{cx}_{cy}"
                if _INCREMENTAL_TERRAIN:
                    # only the rectangles whose cells changed are removed/added (same final collider set as a full rebuild)
                    from .collision_usd import update_chunk_cuboids
                    update_chunk_cuboids(
                        self.stage, path, slot.physical_blocks, cx, cy, half_thickness=self.cfg.terrain_extrusion_depth / 2,
                        material_path=self.material_path, contact_offset=self.cfg.terrain_contact_offset)
                    any_change = True
                    continue
                prim = self.stage.GetPrimAtPath(path)
                if prim and prim.IsValid():
                    self.stage.RemovePrim(path)
                x1=min((cx+1)*64,slot.physical_blocks.shape[0]);y1=min((cy+1)*64,slot.physical_blocks.shape[1])
                if np.count_nonzero(slot.physical_blocks[cx*64:x1, cy*64:y1]):
                    author_layer1_box_collision(
                        self.stage, slot.physical_blocks, half_thickness=self.cfg.terrain_extrusion_depth / 2,
                        physics_material=self.material, physics_material_path=self.material_path,
                        root_path=root, only_chunks=[(cx, cy)],   # local coordinates: the root carries the offset
                        contact_offset=self.cfg.terrain_contact_offset)
                any_change = True
            slot.physical_dirty.clear()
        if any_change and self.cfg.terrain_edit_scene_reset:
            root_state = self.player.data.root_link_state_w.detach().clone()
            self.sim.reset(soft=False)
            self.player.reset()
            ids = torch.arange(self.count, dtype=torch.int32, device=self.sim.device)
            self.player.write_root_link_state_to_sim(root_state, env_ids=ids)
            self.player.write_data_to_sim()
            self.sim.play()
            self.player.update(self.cfg.physics_dt)

    def step(self, action_dict, observe=True):
        actions = self._decode_actions(action_dict)
        self._last_actions = actions
        if len(actions) != self.count:
            raise ValueError("action decoder must return one action per environment")
        if self._fast is not None:
            for fast, action in zip(self._fast, actions):
                fast.begin_macrostep(self._policy_action(action))
        else:
            for slot, action in zip(self.slots, actions):
                slot.model.apply_actions(latch=self._policy_action(action))
        if self.cfg.navigation_mode:
            for i, (slot, action) in enumerate(zip(self.slots, actions)):
                # navigation mode latches the action a second time (the historical XY1 behaviour); compiled
                # worlds repeat the kernel latch and refresh the dataclass the attack raycast reads
                if self._fast is not None:
                    self._fast[i].relatch(self._policy_action(action))
                else:
                    slot.model.apply_actions(latch=self._policy_action(action))
                if action.attack:
                    s=slot.model.s; hit=slot.ground_world.raycast((s.x,s.y),(s.look_x-s.x,s.look_y-s.y),8.0)
                    if hit is not None:
                        gx=int(np.floor(hit.point[0]+slot.grid.width/2-1e-6));gy=int(np.floor(hit.point[1]+slot.grid.height/2-1e-6))
                        bid=slot.grid.get_block(gx,gy)
                        if bid and slot.grid.block_info(bid):
                            if slot.grid.damage_block(gx,gy,25.0,ignore_loot=True):
                                slot.physical_blocks[gx,gy]=0;slot.physical_dirty.add((gx//64,gy//64))
            if self.terrain_mode == "independent":
                dirty=[i for i,s in enumerate(self.slots) if s.physical_dirty]
                if dirty:self._rebuild_dirty(dirty)
        for tick in range(self.cfg.action_decimation):
            self._physics_tick(tick + 1)
        if self._fast is not None:
            for fast in self._fast:
                fast.end_macrostep()
        rewards = np.zeros(self.count, dtype=np.float32)
        terminated = np.zeros(self.count, dtype=bool)
        truncated = np.zeros(self.count, dtype=bool)
        terms: dict[str, np.ndarray] = {}
        reasons: list = [None] * self.count
        players = []
        actions = getattr(self, "_last_actions", actions)
        for i, slot in enumerate(self.slots):
            if self.cfg.navigation_mode:
                current=slot.body_dict();gx,gy=slot.goal_cell
                px,py=slot.grid.world_to_cell(current['x'],current['y'],slot.grid.width,slot.grid.height)
                arrived=abs(px-gx)<=1 and abs(py-gy)<=1
                d=self._goal_distance((current['x'],current['y']),slot.goal_world_xy)
                reward,item_terms=slot.xy1_reward((current['x'],current['y']),slot.goal_world_xy,arrived)
                slot.best_goal_distance=slot.xy1_reward.best_distance
                slot.elapsed_steps+=1;term=arrived;trunc=slot.elapsed_steps>=self.cfg.episode_length_steps and not term
                reason='arrival' if term else 'time_limit' if trunc else None;slot.last_state=current
            else:
                reward, item_terms, term, trunc, reason, current = slot.finish_step(self.reward_function)
            rewards[i], terminated[i], truncated[i], reasons[i] = reward, term, trunc, reason
            players.append(current)
            for key, value in item_terms.items():
                terms.setdefault(key, np.zeros(self.count, np.float32))[i] = value
        return (self._format_observation() if observe else None), rewards, terminated, truncated, {
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
