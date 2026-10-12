"""Source-derived 1024x1024 world correctness and scaling fixtures."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from casu_grid import CasUGridState


class FullWorldTests(unittest.TestCase):
    def make_world(self, camera_x=512, camera_y=512, decisions=()):
        return CasUGridState(1024, 1024, camera_x=camera_x,
                             camera_y=camera_y, decisions=decisions)

    def test_all_corners_all_chunk_seams_and_storage_types(self):
        s = self.make_world()
        self.assertEqual(s.blocks.shape, (1024, 1024))
        self.assertEqual(s.fluids.shape, (1024, 1024))
        self.assertEqual(s.blocks.dtype, np.uint16)
        self.assertEqual(s.fluids.dtype, np.uint8)
        for x, y, chunk in ((0, 0, (0, 0)), (1023, 0, (15, 0)),
                            (0, 1023, (0, 15)), (1023, 1023, (15, 15)),
                            (63, 63, (0, 0)), (64, 64, (1, 1)),
                            (511, 512, (7, 8)), (512, 511, (8, 7))):
            s.set_block(x, y, 5)
            self.assertEqual(s.get_block(x, y), 5)
            self.assertEqual(s.last_chunk_update, chunk)
        self.assertEqual(int(np.count_nonzero(s.blocks)), 8)

    def test_clamped_block_access_at_corners(self):
        s = self.make_world()
        s.set_block(-100, -100, 1)
        s.set_block(5000, 5000, 2)
        self.assertEqual(s.get_block(0, 0), 1)
        self.assertEqual(s.get_block(1023, 1023), 2)
        self.assertEqual(s.get_block(-1, -1), 1)
        self.assertEqual(s.get_block(9999, 9999), 2)

    def test_damage_destruction_and_metal_bonus_interior(self):
        s = self.make_world()
        s.set_block(511, 511, 1)
        self.assertFalse(s.damage_block(511, 511, 99.5))
        self.assertTrue(s.damage_block(511, 511, 0.5))
        self.assertEqual(s.get_block(511, 511), 0)
        s.set_block(512, 512, 6)
        self.assertTrue(s.damage_block(512, 512, 500, bonus_metal=True))
        self.assertEqual(s.get_block(512, 512), 0)

    def test_damage_record_eviction_order_full_map(self):
        s = self.make_world()
        for x in range(129):
            s.set_block(100 + x, 400, 1)
            s.damage_block(100 + x, 400, 1)
        self.assertEqual(len(s.block_damage), 128)
        self.assertIsNone(s.get_damage(100, 400))
        self.assertEqual((s.block_damage[0].x, s.block_damage[-1].x), (101, 228))

    def test_ice_destruction_random_decisions_on_full_world(self):
        s = self.make_world(decisions=[False, True])
        s.set_block(300, 700, 27)
        s.damage_block(300, 700, 50)
        self.assertEqual(s.get_liquid(300, 700), 1)
        s.set_block(301, 700, 27)
        s.damage_block(301, 700, 50)
        self.assertEqual(s.get_liquid(301, 700), 0)
        self.assertEqual(s.rng.position, 2)

    def test_interior_fall_and_vertical_fluid_conversion(self):
        s = self.make_world(camera_x=512, camera_y=512)
        s.set_liquid(512, 520, 3)
        # Band 4 is y=[512,528); first four calls advance phases without touching it.
        for _ in range(4):
            s.simulation_step()
        s.simulation_step()
        self.assertEqual((s.get_liquid(512, 520), s.get_liquid(512, 519)), (0, 3))

        t = self.make_world(camera_x=512, camera_y=512)
        t.set_block(512, 518, 1)
        t.set_block(511, 519, 1); t.set_block(513, 519, 1)
        t.set_block(511, 520, 1); t.set_block(513, 520, 1)
        t.set_liquid(512, 520, 1); t.set_liquid(512, 519, 2)
        for _ in range(5):
            t.simulation_step()
        self.assertEqual((t.get_liquid(512, 520), t.get_liquid(512, 519)), (2, 2))

    def test_camera_region_scheduler_and_wrap(self):
        s = self.make_world()
        expected = [(448 + 16 * i, 464 + 16 * i) for i in range(7)]
        for low, high in expected:
            _, yr = s.simulation_range_index()
            self.assertEqual(yr, (low, high))
        self.assertEqual(s.sim_index, 0)
        self.assertEqual(s.simulation_range_index()[1], (448, 464))

    def test_world_edge_scan_excludes_outermost_indices(self):
        # Source scan keeps a one-cell border. At x=1, left movement can enter
        # x=0; x=1022/1023 are never scan origins.
        left = self.make_world(camera_x=64, camera_y=64, decisions=[False])
        left.set_block(1, 9, 1)
        left.set_liquid(1, 10, 3)
        left.simulation_step()
        self.assertEqual(left.get_liquid(0, 10), 3)

        top = self.make_world(camera_x=512, camera_y=1085)
        top.set_liquid(512, 1022, 3)
        top.set_liquid(512, 1023, 4)
        top.simulation_step()
        self.assertEqual(top.get_liquid(512, 1022), 3)
        self.assertEqual(top.get_liquid(512, 1023), 4)

        right = self.make_world(camera_x=960, camera_y=64, decisions=[True])
        right.set_block(1021, 9, 1)
        right.set_liquid(1021, 10, 3)
        right.simulation_step()
        self.assertEqual(right.get_liquid(1022, 10), 3)

    def test_bottom_boundary_loss_at_full_scale(self):
        s = self.make_world(camera_x=512, camera_y=65)
        s.set_liquid(512, 2, 4)
        s.simulation_step()
        self.assertEqual(s.get_liquid(512, 2), 0)
        self.assertEqual(s.get_liquid(512, 1), 0)

    def test_full_snapshot_restores_all_state_exactly(self):
        s = self.make_world(decisions=[False, True] * 1000)
        s.set_block(500, 500, 6)
        s.damage_block(500, 500, 125, bonus_metal=True)
        s.set_liquid(512, 512, 3)
        snap = s.snapshot()
        for _ in range(12):
            s.simulation_step()
        s.damage_block(500, 500, 4875, bonus_metal=True)
        expected = s.snapshot()
        s.restore(snap)
        for _ in range(12):
            s.simulation_step()
        s.damage_block(500, 500, 4875, bonus_metal=True)
        self.assertTrue(s.snapshots_equal(expected, s.snapshot()))

    def test_repeat_tape_and_environment_count_independence(self):
        def build():
            x = self.make_world(camera_x=512, camera_y=512,
                                decisions=[False, True] * 1000)
            x.set_block(512, 500, 1)
            x.set_liquid(512, 520, 3)
            x.set_liquid(511, 520, 2)
            return x

        solo = build()
        multi = [build() for _ in range(4)]
        for _ in range(14):
            solo.simulation_step()
            for env in multi:
                env.simulation_step()
        for env in multi:
            self.assertTrue(solo.snapshots_equal(solo.snapshot(), env.snapshot()))

    def test_identical_tape_matches_across_fresh_processes(self):
        root = Path(__file__).resolve().parents[2]
        code = r'''import hashlib, json, sys
sys.path.insert(0, sys.argv[1])
from casu_grid import CasUGridState
s = CasUGridState(1024, 1024, camera_x=512, camera_y=512,
                  decisions=[False, True] * 1000)
s.set_block(512, 519, 1)
s.set_liquid(512, 520, 3)
s.set_block(400, 400, 27)
for _ in range(14): s.simulation_step()
s.damage_block(400, 400, 50)
snap = s.snapshot()
h = hashlib.sha256(snap['blocks'].tobytes() + snap['fluids'].tobytes() +
                   json.dumps({k: v for k, v in snap.items() if k not in ('blocks','fluids')},
                              sort_keys=True).encode()).hexdigest()
print(h)
'''
        fingerprints = []
        for _ in range(2):
            run = subprocess.run([sys.executable, "-c", code, str(root)],
                                 cwd=root, env=os.environ.copy(), check=True,
                                 text=True, capture_output=True)
            fingerprints.append(run.stdout.strip())
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_65x64_case_embedded_in_full_world(self):
        small = CasUGridState(65, 64, camera_x=32, camera_y=64,
                              decisions=[False] * 100)
        full = self.make_world(camera_x=64, camera_y=64, decisions=[False] * 100)
        small.set_block(10, 9, 1); small.set_liquid(10, 10, 3)
        full.set_block(10, 9, 1); full.set_liquid(10, 10, 3)
        for _ in range(14):
            small.simulation_step()
            full.simulation_step()
        np.testing.assert_array_equal(small.blocks, full.blocks[:65, :64])
        np.testing.assert_array_equal(small.fluids, full.fluids[:65, :64])
        self.assertEqual(small.sim_index, full.sim_index)
        self.assertEqual(small.rng.position, full.rng.position)


if __name__ == "__main__":
    unittest.main()
