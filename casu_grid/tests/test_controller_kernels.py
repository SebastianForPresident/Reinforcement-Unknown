"""Compiled controller tick vs the reference ``StandingBodyModel`` / ``player_runtime`` (bit for bit)."""
import copy
import unittest

import numpy as np

from casu_grid import backend
from casu_grid.body_model import BodyState, PolicyAction
from casu_grid.perf import controller_harness as H
from casu_grid.player_runtime import post_physics_phase, pre_physics_phase

try:
    CK = backend.controller_for("numba")
    HAVE_NUMBA = CK is not None
except Exception:
    CK, HAVE_NUMBA = None, False

if HAVE_NUMBA:
    from casu_grid.fast_runtime import FastSlot


def _random_state(rng, cells, slot):
    cx, cy = cells[int(rng.integers(len(cells)))]
    x = cx - 512.0 + rng.uniform(-2.5, 3.5)
    y = cy - 512.0 + rng.uniform(-1.0, 4.0)
    crouch = float(rng.choice([0.0, 0.0, rng.uniform(0, 1), 1.0]))
    orig_y = 5.0
    s = BodyState(
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
        ragdoll_event=None,
        last_action=PolicyAction(*(int(v) for v in (rng.integers(-1, 2), rng.integers(2), rng.integers(-1, 2),
                                                    rng.integers(2), rng.integers(-4, 5), rng.integers(-5, 6)))),
        current_action=PolicyAction(*(int(v) for v in (rng.integers(-1, 2), rng.integers(2), rng.integers(-1, 2),
                                                       rng.integers(2), rng.integers(-4, 5), rng.integers(-5, 6)))))
    return s


@unittest.skipUnless(HAVE_NUMBA, "compiled controller not available")
class ControllerKernelEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg, slots = H.build_slots(1)
        cls.slot = slots[0]
        cls.fast = FastSlot(cls.slot, np.zeros(CK.N_STATE), CK.pack_config(cls.slot.body_config), 10)
        blocks = cls.slot.ground_world.blocks
        solid = np.argwhere(blocks != 0)
        rng = np.random.default_rng(7)
        pick = rng.choice(len(solid), 400, replace=False)
        near_objects = [(int(512 + p[0]), int(512 + p[1])) for p in
                        [(np.floor(poly[:, 0].mean()), np.floor(poly[:, 1].mean())) for _, poly in cls.slot.ground_world.polygons[:120]]]
        cls.cells = [tuple(int(v) for v in solid[i]) for i in pick] + near_objects + [(483, 1010), (7, 372), (512, 1019)]

    def _reference_pre(self, state):
        model = self.slot.model
        model.s = copy.deepcopy(state)
        (fx, fy), _ = pre_physics_phase(model)
        return model.s, (fx, fy)

    def test_fixed_update_matches_reference_exactly(self):
        rng = np.random.default_rng(11)
        world = self.slot.ground_world
        fallbacks = compared = 0
        for _ in range(3000):
            state = _random_state(rng, self.cells, self.slot)
            expected, (fx, fy) = self._reference_pre(state)
            self.slot.model.s = copy.deepcopy(state)
            self.fast.pack()
            S = self.fast.S.copy()
            flag = CK.fixed_update(self.fast.S, self.fast.C, world.blocks, world.poly_pack, self.fast.R)
            if flag != CK.OK:
                fallbacks += 1
                continue
            compared += 1
            self.fast.unpack()
            self.assertEqual(self.slot.model.s, expected, "state diverged")
            self.assertEqual((self.fast.R[0], self.fast.R[1]), (fx, fy))
        self.assertGreater(compared, 1500)
        self.assertLess(fallbacks, compared // 20)    # only the rare hypot-boundary fallback remains

    def test_post_physics_matches_reference_exactly(self):
        rng = np.random.default_rng(12)
        world, runtime = self.slot.ground_world, self.slot.runtime
        compared = fallbacks = 0
        for _ in range(3000):
            state = _random_state(rng, self.cells, self.slot)
            completed = int(rng.integers(1, 11))
            touching = frozenset([("chunk", int(rng.integers(0, 16)), int(rng.integers(0, 16))) for _ in range(int(rng.integers(0, 3)))]
                                 + [("object", int(rng.integers(0, 597))) for _ in range(int(rng.integers(0, 3)))])
            # reference
            model = self.slot.model
            model.s = copy.deepcopy(state)
            runtime.touching = touching
            post_physics_phase(model, world, runtime, completed, 10)
            expected, expected_touching = model.s, runtime.touching
            # compiled
            model.s = copy.deepcopy(state)
            self.fast.pack()
            ids = [(t[1] << 16) | t[2] if t[0] == "chunk" else CK.OBJECT_BASE + t[1] for t in touching]
            self.fast.prev[:len(ids)] = ids
            flag = CK.post_physics(self.fast.S, self.fast.C, world.blocks, world.poly_pack,
                                   self.fast.prev, len(ids), self.fast.out, self.fast.R, completed, 10)
            if flag != CK.OK:
                fallbacks += 1
                continue
            compared += 1
            self.fast.unpack()
            self.assertEqual(model.s, expected, "state diverged")
            got = frozenset(("chunk", int(c) >> 16, int(c) & 0xFFFF) if c < CK.OBJECT_BASE
                            else ("object", int(c) - CK.OBJECT_BASE) for c in self.fast.out[:int(self.fast.R[0])])
            self.assertEqual(got, expected_touching)
        self.assertGreater(compared, 1500)

    def test_latch_matches_reference(self):
        rng = np.random.default_rng(13)
        for _ in range(2000):
            state = _random_state(rng, self.cells, self.slot)
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

    def test_pack_unpack_roundtrip(self):
        rng = np.random.default_rng(14)
        for _ in range(200):
            state = _random_state(rng, self.cells, self.slot)
            self.slot.model.s = copy.deepcopy(state)
            self.fast.pack()
            self.fast.unpack()
            self.assertEqual(self.slot.model.s, state)


@unittest.skipUnless(HAVE_NUMBA, "compiled controller not available")
class PolygonKernelEquivalence(unittest.TestCase):
    """Polygon queries of the compiled controller vs the reference ``GroundWorld`` on the real objects."""

    @classmethod
    def setUpClass(cls):
        cfg, slots = H.build_slots(1)
        cls.world = slots[0].ground_world
        from casu_grid.ground_query import GroundWorld
        cls.reference = GroundWorld(cls.world.blocks, cls.world.polygons, backend="reference-python")
        cls.centers = [poly.mean(axis=0) for _, poly in cls.world.polygons]

    def _query_center(self, rng):
        c = self.centers[int(rng.integers(len(self.centers)))]
        return (float(c[0] + rng.uniform(-6, 6)), float(c[1] + rng.uniform(-6, 6)))

    def test_round12_matches_python_round(self):
        from casu_grid import kernels
        rng = np.random.default_rng(3)
        values = list(rng.uniform(-60, 60, 20000)) + [0.5, 1.5, -2.5e-13, 5e-13, 4.9999999999e-13, 1e-20, -1e-20, 12.3456789012345]
        values += list(np.round(rng.uniform(-30, 30, 5000), 4) + rng.choice([0.0, 5e-13, -5e-13], 5000))
        for v in values:
            self.assertEqual(kernels.round12(float(v)), round(float(v), 12), repr(v))

    def test_box_cast_ray_overlap_match_reference_near_objects(self):
        rng = np.random.default_rng(21)
        flag = np.zeros(1, dtype=np.int64)
        hits = 0
        for _ in range(6000):
            c = self._query_center(rng)
            size = (float(rng.choice([1.6, 0.8])), float(rng.choice([5.0, 2.5, 1.0])))
            d = [(0.0, -1.0), (0.0, 1.0), (1.0, 0.0), (-1.0, 0.0)][int(rng.integers(4))]
            dist = float(rng.choice([0.3, 0.2, 0.1, 1.5, 4.0]))
            ref = self.reference.box_cast(c, size, d, dist)
            got = CK._box_cast(self.world.blocks, self.world.poly_pack, flag, c[0], c[1], size[0] / 2.0, size[1] / 2.0,
                               d[0], d[1], dist)
            if ref is None:
                self.assertFalse(got[0], (c, size, d, dist))
            else:
                hits += 1
                self.assertEqual((ref.distance, ref.normal, ref.point), (got[1], (got[2], got[3]), (got[4], got[5])), (c, size, d, dist))
            rd = [(0.0, -1.0), (0.0, 1.0), (1.0, 0.0), (-1.0, 0.0), (0.7, 0.4)][int(rng.integers(5))]
            ref = self.reference.raycast(c, rd, dist)
            got = CK._raycast(self.world.blocks, self.world.poly_pack, flag, c[0], c[1], rd[0], rd[1], dist)
            if ref is None:
                self.assertFalse(got[0])
            else:
                self.assertEqual((ref.distance, ref.point), (got[1], (got[4], got[5])), (c, rd, dist))
                self.assertEqual(ref.normal, (got[2], got[3]))
            self.assertEqual(self.reference.overlap_box(c, size), bool(CK._overlap_box(
                self.world.blocks, self.world.poly_pack, flag, c[0], c[1], size[0], size[1])), (c, size))
        self.assertGreater(hits, 300)

    def test_moving_door_polygon_updates_are_seen(self):
        slot_cfg, slots = H.build_slots(1)
        slot = slots[0]
        world = slot.ground_world
        index = slot.door_polygon_slots[0]
        door = slot.door_defs[0]
        slot.shuttle_doors.restore({"activated": True, "progress": 3.0, "destroyed": False})
        slot.apply_door_polygons()
        c = slot.shuttle_doors.centers()[0]
        flag = np.zeros(1, dtype=np.int64)
        got = CK._box_cast(world.blocks, world.poly_pack, flag, c[0], c[1] + 6.0, 0.8, 0.5, 0.0, -1.0, 8.0)
        from casu_grid.ground_query import GroundWorld
        ref = GroundWorld(world.blocks, world.polygons, backend="reference-python").box_cast(
            (c[0], c[1] + 6.0), (1.6, 1.0), (0.0, -1.0), 8.0)
        self.assertEqual((ref.distance, ref.normal, ref.point), (got[1], (got[2], got[3]), (got[4], got[5])))


@unittest.skipUnless(HAVE_NUMBA, "compiled controller not available")
class CompiledRuntimeTrajectory(unittest.TestCase):
    def test_full_macrostep_loop_matches_reference_backend(self):
        digests = {}
        for compiled in (False, True):
            cfg, slots = H.build_slots(3, grid_xy=(483, 1010))
            digests[compiled] = H.digest_of_run(slots, macrosteps=12, compiled=compiled)
        self.assertEqual(digests[False], digests[True])


if __name__ == "__main__":
    unittest.main()
