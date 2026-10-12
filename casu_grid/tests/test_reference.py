"""Regression coverage for the frozen canonical Layer 1 import."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from casu_grid.reference import (
    FIXTURE_DIR, load_layer1_reference, grid_to_world, world_to_grid,
)


class CanonicalReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = load_layer1_reference()

    def test_manifest_and_full_array_payload_integrity(self):
        self.assertEqual(self.ref.manifest["seed"], 391728405)
        for label, dtype in (("terrain", np.dtype("<u2")), ("fluid", np.dtype("u1"))):
            filename = self.ref.manifest["payloads"][label]["file"]
            payload = FIXTURE_DIR / filename
            source_array = np.fromfile(payload, dtype=dtype).reshape((1024, 1024))
            imported = getattr(self.ref, label)
            self.assertEqual(imported.dtype, dtype)
            self.assertEqual(imported.shape, (1024, 1024))
            mismatch = np.argwhere(imported != source_array)
            if mismatch.size:
                x, y = map(int, mismatch[0])
                self.fail(f"{label} differs from fixture payload first at ({x},{y}): "
                          f"source={source_array[x,y]}, imported={imported[x,y]}")
            self.assertEqual(hashlib.sha256(payload.read_bytes()).hexdigest(),
                             self.ref.manifest["payloads"][label]["sha256"])

    def test_corner_and_chunk_addressing(self):
        state = self.ref.new_state()
        for x, y in ((0, 0), (1023, 0), (0, 1023), (1023, 1023),
                     (63, 63), (64, 64), (511, 512), (512, 511)):
            self.assertEqual(state.get_block(x, y), int(self.ref.terrain[x, y]))
            self.assertEqual(state.get_liquid(x, y), int(self.ref.fluid[x, y]))
        for seam in range(64, 1024, 64):
            for x, y in ((seam, 17), (17, seam)):
                state.set_block(x, y, state.get_block(x, y))
                self.assertEqual(state.last_chunk_update, (x // 64, y // 64))

    def test_coordinate_spawn_scheduler_and_transition(self):
        self.assertEqual(world_to_grid(*self.ref.spawn_world_xy), self.ref.spawn_grid_xy)
        self.assertEqual(self.ref.spawn_grid_xy, (512, 1020))
        self.assertEqual(grid_to_world(*self.ref.spawn_grid_xy), (0.5, 508.5))
        self.assertEqual(self.ref.transition_y_exclusive, -508.9)
        state = self.ref.new_state()
        self.assertEqual((state.camera_x, state.camera_y), (512, 1020))
        self.assertEqual((state.sim_index, state.tile_cooldown, state.tick), (0, 0, 0))
        self.assertEqual(state.block_damage, [])

    def test_fixture_instances_are_independent_and_restore_replays(self):
        first = self.ref.new_state(decisions=[False, True] * 10000)
        second = self.ref.new_state(decisions=[False, True] * 10000)
        original = self.ref.terrain.copy()
        initial = first.snapshot()
        tape = [("step", None)] * 8
        for _ in tape:
            first.simulation_step()
        expected = first.snapshot()
        first.restore(initial)
        for _ in tape:
            first.simulation_step()
        self.assertTrue(first.snapshots_equal(expected, first.snapshot()))
        first.set_block(0, 0, 0)
        self.assertEqual(second.get_block(0, 0), int(self.ref.terrain[0, 0]))
        np.testing.assert_array_equal(self.ref.terrain, original)

    def test_manifest_rejects_checksum_shape_and_schema_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            clone = Path(temp) / FIXTURE_DIR.name
            shutil.copytree(FIXTURE_DIR, clone)
            terrain = clone / "terrain_u16le_x_y.bin"
            terrain.write_bytes(terrain.read_bytes()[:-1] + b"X")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_layer1_reference(clone)
            shutil.rmtree(clone)
            shutil.copytree(FIXTURE_DIR, clone)
            manifest_path = clone / "canonical_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["schema"] = "casu.layer1-reference.v999"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "unsupported Layer 1 manifest schema"):
                load_layer1_reference(clone)
            shutil.rmtree(clone)
            shutil.copytree(FIXTURE_DIR, clone)
            manifest_path = clone / "canonical_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["payloads"]["terrain"]["shape"] = [1024, 1023]
            manifest.pop("content_id")
            manifest["content_id"] = "sha256:" + hashlib.sha256(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False).encode("utf-8")).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "unsupported dtype or shape"):
                load_layer1_reference(clone)


if __name__ == "__main__":
    unittest.main()
