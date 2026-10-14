"""The graph-captured Gumbel-max sampler draws from the model's categorical distributions and reports Categorical log-probs."""
import unittest

import numpy as np
import torch

from casu_grid.workbench.experiments import xy1_point_navigation_mini as cfg
from casu_grid.workbench.local_rollout import LocalRollout, _PolicyStep


class _Env:
    count = 256


class SamplerTests(unittest.TestCase):
    def check(self, device):
        torch.manual_seed(0)
        owner = LocalRollout(_Env(), cfg.ACTION, cfg.OBSERVATION, cfg.MODEL, device)
        for p in owner.model.parameters():          # make the logits clearly non-uniform
            p.data.mul_(4.0)
        n = 256
        step = _PolicyStep(owner, n, (64, 64))
        rng = np.random.default_rng(1)
        one = {"block_ids": rng.integers(0, 17, (1, 64, 64)).astype(np.uint8), "fluid_ids": rng.integers(0, 8, (1, 64, 64)).astype(np.uint8),
               "cursor": np.asarray([[5, 7]], np.int16), "goal": np.asarray([[12.0, -30.0]], np.float32)}
        compact = {k: np.repeat(v, n, axis=0) for k, v in one.items()}
        generator = torch.Generator(device=owner.device)
        generator.manual_seed(5)
        draws = []
        for _ in range(120):
            latent, log_prob, value = step.run(compact, generator, False)
            draws.append(latent.numpy())
        draws = np.concatenate(draws)                                       # 30,720 samples of the same state
        spatial = None
        with torch.no_grad():
            from casu_grid.xy1 import expand_compact_observation
            obs = {"spatial": expand_compact_observation(one["block_ids"], one["fluid_ids"], one["cursor"], owner.device),
                   "goal_relative": torch.as_tensor(one["goal"], device=owner.device)}
            dists, ref_value = owner.model.distributions(obs)
        total = len(draws)
        for k, name in enumerate(owner.head_names):
            probs = dists[name].probs[0].cpu().numpy()
            freq = np.bincount(draws[:, k], minlength=len(probs)) / total
            sigma = np.sqrt(probs * (1 - probs) / total)
            self.assertTrue(np.all(np.abs(freq - probs) <= 5 * sigma + 1e-4), (name, freq, probs))
        # log-prob of the last draw equals the Categorical log_prob of the sampled indices
        expected = sum(dists[name].log_prob(torch.as_tensor(latent[:, k], device=owner.device).long()[:1])[0].item()
                       for k, name in enumerate(owner.head_names))
        self.assertLess(abs(float(log_prob[0]) - expected), 2e-3)
        self.assertLess(abs(float(value[0]) - float(ref_value[0])), 1e-3 * max(1.0, abs(float(ref_value[0]))))   # TF32 batch-shape noise

    def test_cpu(self):
        self.check("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_graph(self):
        self.check("cuda")


if __name__ == "__main__":
    unittest.main()
