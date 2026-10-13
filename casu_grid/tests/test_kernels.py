"""Compiled-backend equivalence: numba kernels vs the readable reference implementations."""
import unittest

import numpy as np

from casu_grid import backend
from casu_grid.ground_query import GroundWorld
from casu_grid.reference import load_layer1_reference
from casu_grid.state import CasUGridState, SeededDecisionStream

try:
    backend.kernels_for("numba")
    HAVE_NUMBA = True
except Exception:   # numba unavailable: the reference backend is the only one
    HAVE_NUMBA = False


def _grid(backend_name, blocks, fluids, camera, seed, push):
    grid = CasUGridState(blocks.shape[0], blocks.shape[1], camera_x=camera[0], camera_y=camera[1],
                         backend=backend_name)
    grid.blocks[...] = blocks
    grid.fluids[...] = fluids
    grid.rng = SeededDecisionStream(seed)
    grid.fluid_push_enabled = push
    return grid


def _fluid_state(grid):
    return (grid.fluids.tobytes(), grid.tile_cooldown, grid.water_move_count, grid.push_events,
            grid.sim_index, grid.tick, grid.rng.position, repr(grid.rng.rng.bit_generator.state))


@unittest.skipUnless(HAVE_NUMBA, "numba backend not available")
class FluidKernelTests(unittest.TestCase):
    def test_random_fixtures_match_the_literal_oracle_byte_for_byte(self):
        rng = np.random.default_rng(1234)
        for trial in range(12):
            w, h = int(rng.integers(40, 200)), int(rng.integers(40, 200))
            blocks = (rng.random((w, h)) < rng.uniform(0.0, 0.5)).astype(np.uint16) * rng.integers(1, 20)
            fluids = np.where(rng.random((w, h)) < rng.uniform(0.0, 0.6), rng.integers(1, 7, (w, h)), 0).astype(np.uint8)
            camera = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            push = bool(trial % 2)
            ref = _grid("reference-python", blocks, fluids, camera, 7 + trial, push)
            fast = _grid("numba", blocks, fluids, camera, 7 + trial, push)
            for step in range(40):
                ref.simulation_step_reference()
                fast.simulation_step()
                self.assertEqual(_fluid_state(ref), _fluid_state(fast), f"trial {trial} step {step}")
            self.assertIsNotNone(fast._k)

    def test_canonical_world_matches_oracle_over_many_ticks(self):
        reference = load_layer1_reference()
        for camera in ((483, 1010), (958, 178), (300, 500)):
            states = []
            for name, oracle in (("reference-python", True), ("numba", False)):
                grid = reference.new_state()
                grid._k = backend.kernels_for(name)
                grid.rng = SeededDecisionStream(391728405)
                grid.camera_x, grid.camera_y = camera
                for _ in range(112):   # two full scheduler cycles plus
                    grid.simulation_step_reference() if oracle else grid.simulation_step()
                states.append(_fluid_state(grid))
            self.assertEqual(states[0], states[1], camera)

    def test_decision_budget_path_still_uses_reference(self):
        grid = _grid("numba", np.zeros((70, 70), np.uint16), np.ones((70, 70), np.uint8), (35, 35), 1, False)
        grid.simulation_step(decision_budget=0)   # must not raise; falls back to the Python sweep
        self.assertEqual(grid.tick, 1)


@unittest.skipUnless(HAVE_NUMBA, "numba backend not available")
class QueryKernelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(99)
        self.blocks = np.zeros((1024, 1024), dtype=np.uint16)
        # clustered solids around (0,0) world: random walls, floors and pits
        sub = (rng.random((120, 120)) < 0.35)
        self.blocks[452:572, 452:572] = sub.astype(np.uint16) * 3
        self.polygons = [
            ("tri", np.array([(1.0, 1.0), (3.0, 1.0), (2.0, 3.5)])),
            ("box", np.array([(-4.0, -2.0), (-2.0, -2.0), (-2.0, 0.5), (-4.0, 0.5)])),
            ("far", np.array([(300.0, 300.0), (301.0, 300.0), (301.0, 301.0)])),
        ]
        self.ref = GroundWorld(self.blocks, self.polygons, backend="reference-python")
        self.fast = GroundWorld(self.blocks, self.polygons, backend="numba")
        self.rng = rng

    def _random_center(self):
        grid = self.rng.random() < 0.4
        base = self.rng.uniform(-45, 45, 2)
        if grid:   # land exactly on or within 1e-9 of cell edges (the delicate cases)
            base = np.round(base) + self.rng.choice([0.0, 1e-9, -1e-9, 0.5, 0.0055, -0.0055], 2)
        return float(base[0]), float(base[1])

    def test_box_cast_matches_reference_exactly(self):
        for _ in range(4000):
            c = self._random_center()
            size = (float(self.rng.choice([1.6, 1.6, 0.8, 2.0])), float(self.rng.choice([5.0, 2.5, 1.0])))
            d = [(0.0, -1.0), (0.0, 1.0), (1.0, 0.0), (-1.0, 0.0)][int(self.rng.integers(4))]
            dist = float(self.rng.choice([0.3, 0.2, 0.1, 0.0, 1.5]))
            self.assertEqual(self.ref.box_cast(c, size, d, dist), self.fast.box_cast(c, size, d, dist), (c, size, d, dist))

    def test_raycast_matches_reference_exactly(self):
        for _ in range(3000):
            o = self._random_center()
            d = [(0.0, -1.0), (0.0, 1.0), (1.0, 0.0), (-1.0, 0.0), (1.0, 1.0), (0.3, -0.7)][int(self.rng.integers(6))]
            dist = float(self.rng.choice([0.5, 1.0, 3.3, 8.8]))
            self.assertEqual(self.ref.raycast(o, d, dist), self.fast.raycast(o, d, dist), (o, d, dist))

    def test_overlap_and_touching_match_reference(self):
        for _ in range(4000):
            c = self._random_center()
            size = (float(self.rng.choice([1.6, 0.8])), float(self.rng.choice([5.0, 2.5])))
            self.assertEqual(self.ref.overlap_box(c, size), self.fast.overlap_box(c, size), (c, size))
            reach = float(self.rng.choice([0.125, 0.13, 0.1]))
            self.assertEqual(self.ref.touching_colliders(c, size, reach),
                             self.fast.touching_colliders(c, size, reach), (c, size, reach))

    def test_moving_polygon_update_is_seen_by_the_fast_path(self):
        poly = np.array([(10.0, 10.0), (12.0, 10.0), (12.0, 12.0), (10.0, 12.0)])
        for world in (self.ref, self.fast):
            world.set_polygon(2, "far", poly)
        for pos in ((11.0, 8.0), (11.0, 13.5), (30.0, 30.0)):
            for direction in ((0.0, 1.0), (0.0, -1.0)):
                self.assertEqual(self.ref.box_cast(pos, (1.6, 5.0), direction, 4.0),
                                 self.fast.box_cast(pos, (1.6, 5.0), direction, 4.0))


class BackendSelectionTests(unittest.TestCase):
    def test_reference_backend_always_available_and_described(self):
        self.assertIsNone(backend.kernels_for("reference-python"))
        self.assertEqual(backend.active_name("reference-python"), "reference-python")
        self.assertIn(backend.describe()["active"], ("numba", "reference-python"))

    def test_unknown_backend_name_is_rejected(self):
        import os
        old = os.environ.get("CASU_BACKEND")
        os.environ["CASU_BACKEND"] = "turbo"
        try:
            with self.assertRaises(ValueError):
                backend.requested()
        finally:
            if old is None:
                os.environ.pop("CASU_BACKEND")
            else:
                os.environ["CASU_BACKEND"] = old


if __name__ == "__main__":
    unittest.main()
