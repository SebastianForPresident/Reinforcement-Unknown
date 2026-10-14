"""Compiled controller on the XY1 mini-world (128 x 128, half-size 64): bit-identical to the pure-Python reference."""
import copy
import unittest

import numpy as np

from casu_grid import backend
from casu_grid.body_model import BodyConfig, BodyState, PolicyAction
from casu_grid.env_slot import EnvSlot
from casu_grid.isaac_navigation_env import IsaacLayer1EnvConfig
from casu_grid.player_runtime import post_physics_phase, pre_physics_phase
from casu_grid.xy1_mini_world import load_mini_world_reference

try:
    CK = backend.controller_for("numba")
    HAVE_NUMBA = CK is not None
except Exception:
    CK, HAVE_NUMBA = None, False

if HAVE_NUMBA:
    from casu_grid.fast_runtime import FastSlot


def random_state(rng, cells, half):
    cx, cy = cells[int(rng.integers(len(cells)))]
    x = cx - half + rng.uniform(-2.5, 3.5)
    y = cy - half + rng.uniform(-1.0, 4.0)
    crouch = float(rng.choice([0.0, 0.0, rng.uniform(0, 1), 1.0]))
    orig_y = 5.0
    action = lambda: PolicyAction(*(int(v) for v in (rng.integers(-1, 2), rng.integers(2), rng.integers(-1, 2),
                                                     rng.integers(2), rng.integers(-4, 5), rng.integers(-5, 6))))
    return BodyState(
        x=float(x), y=float(y), vx=float(rng.normal(0, 9)), vy=float(rng.normal(0, 9)),
        grounded=bool(rng.integers(2)), move_x=float(rng.integers(-1, 2)), move_y=float(rng.integers(-1, 2)),
        crouching=bool(rng.integers(2)), crouch_amount=crouch,
        col_size=(1.6, orig_y - orig_y * 0.5 * crouch), col_offset=(0.0, -orig_y * 0.25 * crouch),
        ended_jump=bool(rng.integers(2)), wants_jump_input=bool(rng.integers(2)),
        jump_cooldown=float(rng.choice([-1.0, 0.0, 0.1, 0.25])),
        time_since_grounded=float(rng.uniform(0, 0.3)), time_since_slid_left=float(rng.uniform(0, 0.4)),
        time_since_slid_right=float(rng.uniform(0, 0.4)), time_slid_for=float(rng.uniform(0, 1.5)),
        sliding_left=bool(rng.integers(2)), sliding_right=bool(rng.integers(2)),
        first_wall_jump=bool(rng.integers(2)), last_jumped_on_right_wall=bool(rng.integers(2)),
        ground_normal=(float(rng.choice([0.0, 0.0, 0.6, -0.6])), float(rng.choice([0.0, 1.0, 0.8]))),
        last_time_step_vx=float(rng.normal(0, 9)), last_time_step_vy=float(rng.normal(0, 25)),
        slippery=bool(rng.integers(2)), standing_on=int(rng.choice([0, 1, 11, 27, 28])),
        gravity_scale=float(rng.choice([1.0, 2.5])),
        look_x=float(x + rng.uniform(-3, 3)), look_y=float(y + rng.uniform(-3, 3)),
        ragdoll_event=None, last_action=action(), current_action=action())


@unittest.skipUnless(HAVE_NUMBA, "compiled controller not available")
class MiniControllerEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = IsaacLayer1EnvConfig(num_envs=1, navigation_mode=True, world_variant="xy1_mini_128x128",
                                   observation_window_tiles=(64, 64))
        ref = load_mini_world_reference(391728405)
        cls.slot = EnvSlot(cfg, ref, fluid_seed=cfg.fluid_rng_seed, body_config=BodyConfig(step_up_clearance=cfg.step_up_clearance))
        cls.world = cls.slot.ground_world
        cls.fast = FastSlot(cls.slot, np.zeros(CK.N_STATE), CK.pack_config(cls.slot.body_config, cls.world.blocks.shape), 10)
        solid = np.argwhere(cls.world.blocks != 0)
        rng = np.random.default_rng(5)
        cls.cells = [tuple(int(v) for v in solid[i]) for i in rng.choice(len(solid), 300, replace=False)] + [(64, 64), (3, 3), (120, 60)]
        cls.half = cls.world.blocks.shape[0] / 2.0

    def reference(self):
        """Run reference phases with the pure-Python world queries (the oracle)."""
        class _Oracle:
            def __enter__(s):
                s.saved, self.world._k = self.world._k, None

            def __exit__(s, *exc):
                self.world._k = s.saved
        return _Oracle()

    def test_world_is_mini(self):
        self.assertEqual(self.world.blocks.shape, (128, 128))
        self.assertEqual(self.fast.C[CK.C_HALF_W], 64.0)

    def test_fixed_update_matches_reference_exactly(self):
        rng = np.random.default_rng(31)
        compared = fallbacks = 0
        for _ in range(2500):
            state = random_state(rng, self.cells, self.half)
            model = self.slot.model
            model.s = copy.deepcopy(state)
            with self.reference():
                (fx, fy), _ = pre_physics_phase(model)
            expected = model.s
            model.s = copy.deepcopy(state)
            self.fast.pack()
            flag = CK.fixed_update(self.fast.S, self.fast.C, self.world.blocks, self.world.poly_pack, self.fast.R)
            if flag != CK.OK:
                fallbacks += 1
                continue
            compared += 1
            self.fast.unpack()
            self.assertEqual(model.s, expected)
            self.assertEqual((self.fast.R[0], self.fast.R[1]), (fx, fy))
        self.assertGreater(compared, 1200)
        self.assertLess(fallbacks, compared // 20)

    def test_post_physics_matches_reference_exactly(self):
        rng = np.random.default_rng(32)
        runtime = self.slot.runtime
        compared = 0
        for _ in range(2500):
            state = random_state(rng, self.cells, self.half)
            completed = int(rng.integers(1, 11))
            touching = frozenset(("chunk", int(rng.integers(0, 2)), int(rng.integers(0, 2))) for _ in range(int(rng.integers(0, 3))))
            model = self.slot.model
            model.s = copy.deepcopy(state)
            runtime.touching = touching
            with self.reference():
                post_physics_phase(model, self.world, runtime, completed, 10)
            expected, expected_touching = model.s, runtime.touching
            model.s = copy.deepcopy(state)
            self.fast.pack()
            ids = [(t[1] << 16) | t[2] for t in touching]
            self.fast.prev[:len(ids)] = ids
            flag = CK.post_physics(self.fast.S, self.fast.C, self.world.blocks, self.world.poly_pack,
                                   self.fast.prev, len(ids), self.fast.out, self.fast.R, completed, 10)
            if flag != CK.OK:
                continue
            compared += 1
            self.fast.unpack()
            self.assertEqual(model.s, expected)
            got = frozenset(("chunk", int(c) >> 16, int(c) & 0xFFFF) for c in self.fast.out[:int(self.fast.R[0])])
            self.assertEqual(got, expected_touching)
        self.assertGreater(compared, 1200)

    def test_latch_matches_reference_including_look_target(self):
        rng = np.random.default_rng(33)
        for _ in range(1500):
            state = random_state(rng, self.cells, self.half)
            action = PolicyAction(*(int(v) for v in (rng.integers(-1, 2), rng.integers(2), rng.integers(-1, 2),
                                                     rng.integers(2), rng.integers(-4, 5), rng.integers(-5, 6))))
            model = self.slot.model
            model.s = copy.deepcopy(state)
            model.apply_actions(latch=action)
            expected = model.s
            model.s = copy.deepcopy(state)
            self.fast.begin_macrostep(action)
            self.fast.unpack()
            self.assertEqual(model.s, expected)
            # the navigation-mode second latch
            model.s = copy.deepcopy(expected)
            model.apply_actions(latch=action)
            expected2 = model.s
            self.fast.pack()
            self.fast.relatch(action)
            self.assertEqual(self.slot.model.s, expected2)

    def test_tile_queries_match_python_reference(self):
        from casu_grid.ground_query import GroundWorld
        oracle = GroundWorld(self.world.blocks, (), backend="reference-python")
        rng = np.random.default_rng(34)
        for _ in range(4000):
            cx, cy = self.cells[int(rng.integers(len(self.cells)))]
            c = (cx - 64.0 + rng.uniform(-3, 4), cy - 64.0 + rng.uniform(-3, 4))
            size = (float(rng.choice([1.6, 0.8])), float(rng.choice([5.0, 2.5, 1.0])))
            d = [(0.0, -1.0), (0.0, 1.0), (1.0, 0.0), (-1.0, 0.0)][int(rng.integers(4))]
            dist = float(rng.choice([0.3, 0.1, 1.5, 4.0]))
            ref = oracle.box_cast(c, size, d, dist)
            got = self.world.box_cast(c, size, d, dist)
            self.assertEqual(ref is None, got is None)
            if ref is not None:
                self.assertEqual((ref.distance, ref.normal, ref.point), (got.distance, got.normal, got.point))
            self.assertEqual(oracle.overlap_box(c, size), self.world.overlap_box(c, size))
            rd = [(0.0, -1.0), (0.0, 1.0), (1.0, 0.0), (-1.0, 0.0), (0.7, 0.4)][int(rng.integers(5))]
            r1, r2 = oracle.raycast(c, rd, dist), self.world.raycast(c, rd, dist)
            self.assertEqual(r1 is None, r2 is None)
            if r1 is not None:
                self.assertEqual((r1.distance, r1.normal, r1.point), (r2.distance, r2.normal, r2.point))


if __name__ == "__main__":
    unittest.main()
