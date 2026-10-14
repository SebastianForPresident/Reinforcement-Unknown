"""The compact observation must reproduce encode_xy1_observation exactly."""
import unittest

import numpy as np
import torch

from casu_grid.state import BLOCK_HEALTH
from casu_grid.xy1 import encode_xy1_compact, encode_xy1_observation, expand_compact_observation
from casu_grid.xy1_mini_world import generate_mini_terrain


class _Grid:
    def __init__(self, blocks, fluids):
        self.blocks, self.fluids = blocks, fluids
        self.width, self.height = blocks.shape

    @staticmethod
    def world_to_cell(x, y, w, h):
        return int(x + w // 2), int(y + h // 2)


def make_env(seed, window, x, y, look, goal):
    rng = np.random.default_rng(seed)
    blocks = generate_mini_terrain(seed).astype(np.uint16)
    blocks[rng.random(blocks.shape) < 0.05] = rng.choice(list(BLOCK_HEALTH), size=int((rng.random(blocks.shape) < 0.05).sum()) or 1)[0] if False else 22
    fluids = rng.integers(0, 7, blocks.shape).astype(np.uint8)
    grid = _Grid(blocks, fluids)
    state = type("S", (), {"x": x, "y": y, "current_action": type("A", (), {"look_dx": look[0], "look_dy": look[1]})()})()
    slot = type("Slot", (), {"grid": grid, "goal_cell": goal})()
    model = type("M", (), {"s": state})()
    cfg = type("C", (), {"observation_window_tiles": window})()
    return type("Env", (), {"slot": slot, "grid": grid, "model": model, "cfg": cfg})()


class CompactObservationTests(unittest.TestCase):
    def test_matches_reference_encoder_everywhere(self):
        cases = [(1, (64, 64), 0.4, 0.6, (0, 0), (70, 60)), (2, (64, 64), -62.0, -60.0, (4, 5), (3, 3)),
                 (3, (64, 64), 60.5, 62.5, (-4, -5), (120, 100)), (4, (85, 49), 0.5, 0.5, (2, -1), (55, 47)),
                 (5, (64, 64), -3.2, 30.0, (-4, 5), (64, 64)), (6, (64, 64), 10.7, -50.2, (4, -5), (0, 0))]
        for seed, window, x, y, look, goal in cases:
            with self.subTest(seed=seed, x=x, y=y):
                env = make_env(seed, window, x, y, look, goal)
                reference = encode_xy1_observation(env)
                b, f, cursor, g = encode_xy1_compact(env)
                spatial = expand_compact_observation(b[None], f[None], np.asarray([cursor]))[0].numpy()
                self.assertTrue(np.array_equal(spatial, reference["spatial"]))
                self.assertEqual(spatial.dtype, reference["spatial"].dtype)
                self.assertEqual(tuple(np.asarray(g, np.float32)), tuple(reference["goal_relative"]))

    def test_batched_expansion(self):
        envs = [make_env(7 + k, (64, 64), float(k), float(-k), (k % 9 - 4, k % 11 - 5), (k, k)) for k in range(5)]
        parts = [encode_xy1_compact(e) for e in envs]
        out = expand_compact_observation(np.stack([p[0] for p in parts]), np.stack([p[1] for p in parts]),
                                         np.asarray([p[2] for p in parts]))
        for k, e in enumerate(envs):
            self.assertTrue(np.array_equal(out[k].numpy(), encode_xy1_observation(e)["spatial"]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_expansion_is_identical(self):
        env = make_env(9, (64, 64), 5.0, 5.0, (1, 1), (9, 9))
        b, f, cursor, _ = encode_xy1_compact(env)
        cpu = expand_compact_observation(b[None], f[None], np.asarray([cursor]))
        gpu = expand_compact_observation(b[None], f[None], np.asarray([cursor]), device="cuda")
        self.assertTrue(torch.equal(cpu, gpu.cpu()))


if __name__ == "__main__":
    unittest.main()
