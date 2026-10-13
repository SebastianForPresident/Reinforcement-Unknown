import copy
import unittest

import numpy as np

from casu_grid import CasUGridState


def grid(**kwargs):
    return CasUGridState(65, 64, camera_x=32, camera_y=64, **kwargs)


class GridStateTests(unittest.TestCase):
    def test_set_block_tracks_dirty_source_chunk_through_snapshot_restore(self):
        s = grid()
        s.set_block(64, 10, 4)
        self.assertEqual(s.dirty_chunks, {(1, 0)})
        snap = s.snapshot()
        s.dirty_chunks.clear()
        s.restore(snap)
        self.assertEqual(s.dirty_chunks, {(1, 0)})

    def test_seeded_decision_stream_snapshot_continuation(self):
        from casu_grid.state import SeededDecisionStream
        stream = SeededDecisionStream(9123)
        prefix = [stream.next_bool() for _ in range(11)]
        snapshot = stream.snapshot()
        expected = [stream.next_bool() for _ in range(17)]
        restored = SeededDecisionStream.from_snapshot(snapshot)
        self.assertEqual([restored.next_bool() for _ in range(17)], expected)
        self.assertEqual(restored.position, len(prefix) + 17)

    def test_empty_and_isolated_blocks(self):
        s = grid()
        self.assertEqual(s.blocks.dtype, np.uint16)
        self.assertEqual(s.fluids.dtype, np.uint8)
        self.assertTrue(s.empty(10, 10))
        s.set_block(10, 10, 1)
        self.assertEqual(s.get_block(10, 10), 1)
        self.assertFalse(s.empty(10, 10))
        self.assertEqual(s.last_chunk_update, (0, 0))

    def test_adjacent_cells_and_chunk_seam_indexing(self):
        s = grid()
        for x in (62, 63, 64):
            s.set_block(x, 10, 5)
        self.assertEqual([s.get_block(x, 10) for x in (62, 63, 64)], [5, 5, 5])
        self.assertEqual(s.last_chunk_update, (1, 0))

    def test_partial_damage_and_exact_threshold(self):
        s = grid()
        s.set_block(10, 10, 1)
        self.assertFalse(s.damage_block(10, 10, 99))
        self.assertEqual(s.get_damage(10, 10).damage, 99)
        self.assertTrue(s.damage_block(10, 10, 1))
        self.assertEqual(s.get_block(10, 10), 0)
        self.assertIsNone(s.get_damage(10, 10))

    def test_metal_bonus(self):
        s = grid()
        s.set_block(10, 10, 6)
        self.assertTrue(s.damage_block(10, 10, 500, bonus_metal=True))

    def test_damage_record_eviction_keeps_newest_129th(self):
        s = grid()
        for x in range(1, 130):
            s.set_block(x % 65, (x // 65) + 1, 1)
            s.damage_block(x % 65, (x // 65) + 1, 1)
        self.assertEqual(len(s.block_damage), 128)
        self.assertIsNone(s.get_damage(1, 1))
        self.assertIsNotNone(s.get_damage(64, 2))

    def test_ice_destruction_both_outcomes(self):
        wet = grid(); wet.set_block(10, 10, 27)
        self.assertTrue(wet.damage_block(10, 10, 50, ice_drop=True))
        self.assertEqual(wet.get_liquid(10, 10), 1)
        dry = grid(); dry.set_block(10, 10, 27)
        dry.damage_block(10, 10, 50, ice_drop=False)
        self.assertEqual(dry.get_liquid(10, 10), 0)

    def test_ice_random_stream_and_ignore_loot(self):
        s = grid(decisions=[False, True])
        s.set_block(10, 10, 27)
        s.damage_block(10, 10, 50)  # value <= .5 represented by False => water
        self.assertEqual(s.get_liquid(10, 10), 1)
        self.assertEqual(s.rng.position, 1)
        s.set_block(11, 10, 27)
        s.damage_block(11, 10, 50, ignore_loot=True)
        self.assertEqual(s.get_liquid(11, 10), 0)
        self.assertEqual(s.rng.position, 1)

    def test_fluid_blocked_on_all_sides(self):
        s = grid(decisions=[False] * 4)
        s.set_liquid(10, 10, 3)
        for x, y in ((10, 9), (9, 10), (11, 10)):
            s.set_block(x, y, 1)
        s.simulation_step()
        self.assertEqual(s.get_liquid(10, 10), 3)

    def test_fluid_falls(self):
        s = grid()
        s.set_liquid(10, 10, 1)
        s.simulation_step()
        self.assertEqual(s.get_liquid(10, 9), 1)
        self.assertEqual(s.get_liquid(10, 10), 0)

    def test_one_sided_horizontal_move(self):
        s = grid()
        s.set_liquid(62, 10, 3)
        s.set_block(62, 9, 1); s.set_block(61, 10, 1)
        s.simulation_step()
        self.assertEqual(s.get_liquid(63, 10), 3)

    def test_two_sided_random_horizontal_choice(self):
        s = grid(decisions=[True])
        s.set_liquid(62, 10, 3)
        s.set_block(62, 9, 1)
        s.simulation_step()
        self.assertEqual(s.get_liquid(63, 10), 3)

    def test_in_place_scan_order_allows_repeat_rightward_processing(self):
        s = grid(decisions=[True] * 60)
        s.set_liquid(60, 10, 3)
        # The source scans x upward. At each successive x the moved cell is
        # visited again, until the horizontal target is blocked.
        for x in range(60, 63):
            s.set_block(x, 9, 1)
        s.simulation_step()
        self.assertEqual(s.get_liquid(63, 10), 3)
        self.assertEqual(int(np.count_nonzero(s.fluids == 3)), 1)

    def test_vertical_type_1_2_conversion(self):
        s = grid()
        s.set_block(10, 9, 1); s.set_block(10, 8, 1)
        s.set_liquid(10, 10, 1); s.set_liquid(10, 9, 2)
        # Lower cell is outside scan only if its row is not scheduled; arrange
        # it as the below neighbor and block further horizontal escape.
        s.set_block(9, 10, 1); s.set_block(11, 10, 1)
        s.set_block(9, 9, 1); s.set_block(11, 9, 1)
        s.simulation_step()
        self.assertEqual((s.get_liquid(10, 10), s.get_liquid(10, 9)), (2, 2))

    def test_bottom_edge_fluid_loss(self):
        s = grid()
        s.camera_y = 65  # first active band includes y=1..16
        s.set_liquid(10, 2, 4)
        s.simulation_step()
        self.assertEqual(s.get_liquid(10, 2), 0)
        self.assertEqual(s.get_liquid(10, 1), 0)

    def test_seven_band_index_wrap(self):
        s = grid()
        indices = []
        for _ in range(7):
            s.simulation_range_index()
            indices.append(s.sim_index)
        self.assertEqual(indices, [16, 32, 48, 64, 80, 96, 0])

    def test_camera_relative_region_and_small_fixture_clamps(self):
        s = grid()
        xr, yr = s.simulation_range_index()
        self.assertEqual(xr, (1, 63))  # exclusive end: scanned x=1..62
        self.assertEqual(yr, (1, 16))
        s.camera_x = -100
        xr2, _ = s.simulation_range_index()
        self.assertEqual(xr2, (1, 1))  # original clamps collapse the range

    def test_destroyed_terrain_opens_fluid_path(self):
        s = grid()
        s.set_block(10, 9, 1); s.set_liquid(10, 10, 1)
        s.damage_block(10, 9, 100)
        self.assertEqual(s.get_block(10, 9), 0)
        s.simulation_step()
        self.assertEqual(s.get_liquid(10, 9), 1)

    def test_snapshot_restore_replays_exactly(self):
        s = grid(decisions=[True, False, True, False, True] * 100)
        s.set_block(5, 5, 6); s.damage_block(5, 5, 120, bonus_metal=True)
        s.set_block(12, 9, 1); s.set_liquid(12, 10, 3)
        saved = s.snapshot()
        for _ in range(5): s.simulation_step()
        s.damage_block(5, 5, 4880, bonus_metal=True)
        expected = s.snapshot()
        s.restore(saved)
        for _ in range(5): s.simulation_step()
        s.damage_block(5, 5, 4880, bonus_metal=True)
        self.assertTrue(s.snapshots_equal(expected, s.snapshot()))

    def test_controlled_input_change_diverges(self):
        left = grid(decisions=[True]); right = grid(decisions=[False])
        for s in (left, right):
            s.set_block(10, 9, 1); s.set_liquid(10, 10, 3)
        left.simulation_step(); right.simulation_step()
        self.assertNotEqual(left.fluids.tolist(), right.fluids.tolist())

    def test_repeatability_across_independent_instances(self):
        a, b = grid(decisions=[True] * 500), grid(decisions=[True] * 500)
        for s in (a, b):
            s.set_block(10, 9, 1); s.set_liquid(10, 10, 3)
            for _ in range(3): s.simulation_step()
        self.assertTrue(a.snapshots_equal(a.snapshot(), b.snapshot()))

    def test_snapshot_captures_future_inputs_and_state(self):
        s = grid(decisions=[True, False])
        s.simulation_step()
        saved = s.snapshot()
        clone = grid()
        clone.restore(saved)
        self.assertTrue(s.snapshots_equal(saved, clone.snapshot()))
        self.assertEqual(clone.rng.next_bool(), s.rng.next_bool())

    def test_world_dimensions_scale_without_rule_changes(self):
        s = CasUGridState(1024, 1024, camera_x=512, camera_y=512)
        self.assertEqual(s.blocks.shape, (1024, 1024))
        self.assertEqual(s.fluids.shape, (1024, 1024))


if __name__ == "__main__":
    unittest.main()
