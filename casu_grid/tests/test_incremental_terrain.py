"""Diff-based chunk updates must produce exactly the collider set of a full chunk rebuild."""
import unittest

import numpy as np

try:
    from pxr import Sdf, Usd, UsdGeom
except Exception:           # pragma: no cover - USD only exists in the Isaac Python
    Usd = None

from casu_grid.xy1_mini_world import generate_mini_terrain


def snapshot(stage, chunk_path):
    prim = stage.GetPrimAtPath(chunk_path)
    if not prim or not prim.IsValid():
        return None
    result = {}
    for child in prim.GetChildren():
        attrs = {a.GetName(): a.Get() for a in child.GetAttributes()}
        rel = {r.GetName(): [str(t) for t in r.GetTargets()] for r in child.GetRelationships()}
        result[child.GetName()] = (child.GetTypeName(), attrs, rel, list(child.GetAppliedSchemas()))
    return result


@unittest.skipIf(Usd is None, "pxr not available")
class IncrementalTerrain(unittest.TestCase):
    def author_full(self, stage, blocks, cx, cy):
        from casu_grid.collision_usd import author_layer1_box_collision
        path = f"/W/Chunk_{cx}_{cy}"
        existing = stage.GetPrimAtPath(path)
        if existing and existing.IsValid():
            stage.RemovePrim(path)
        if np.count_nonzero(blocks[cx * 64:(cx + 1) * 64, cy * 64:(cy + 1) * 64]):
            author_layer1_box_collision(stage, blocks, half_thickness=0.5, physics_material=object(),
                                        physics_material_path="/World/M", root_path="/W", only_chunks=[(cx, cy)],
                                        contact_offset=0.001)

    def test_matches_full_rebuild_over_random_edits(self):
        from casu_grid.collision_usd import update_chunk_cuboids
        rng = np.random.default_rng(3)
        blocks = generate_mini_terrain(17)
        a, b = Usd.Stage.CreateInMemory(), Usd.Stage.CreateInMemory()
        for stage in (a, b):
            UsdGeom.Xform.Define(stage, "/W")
        for cx in range(2):
            for cy in range(2):
                self.author_full(a, blocks, cx, cy)
                self.author_full(b, blocks, cx, cy)
        for step in range(60):
            cells = rng.integers(0, 128, size=(int(rng.integers(1, 4)), 2))
            for x, y in cells:
                blocks[x, y] = 0 if blocks[x, y] else int(rng.integers(1, 5))
            if step % 15 == 14:                       # empty an entire chunk once
                blocks[:64, :64] = 0
            for cx in range(2):
                for cy in range(2):
                    update_chunk_cuboids(a, f"/W/Chunk_{cx}_{cy}", blocks, cx, cy, half_thickness=0.5,
                                         material_path="/World/M", contact_offset=0.001)
                    self.author_full(b, blocks, cx, cy)
                    self.assertEqual(snapshot(a, f"/W/Chunk_{cx}_{cy}"), snapshot(b, f"/W/Chunk_{cx}_{cy}"), (step, cx, cy))
        blocks[:64, :64] = rng.integers(0, 3, (64, 64))      # re-populate an emptied chunk
        update_chunk_cuboids(a, "/W/Chunk_0_0", blocks, 0, 0, half_thickness=0.5, material_path="/World/M", contact_offset=0.001)
        self.author_full(b, blocks, 0, 0)
        self.assertEqual(snapshot(a, "/W/Chunk_0_0"), snapshot(b, "/W/Chunk_0_0"))


if __name__ == "__main__":
    unittest.main()
