"""CPU and CUDA PPO updates agree within predeclared tolerances (same weights, rollout, minibatch order)."""
import copy
import dataclasses
import unittest

import numpy as np
import torch

from casu_grid.workbench.contracts import ObservationContract, ObservationField
from casu_grid.workbench.experiments import xy1_point_navigation_mini as cfg
from casu_grid.workbench.ppo import PPOTrainer

# predeclared tolerances: relative on logged statistics, absolute on parameters (Adam moves each weight by <= lr per step)
REL_METRIC, ABS_PARAM = 5e-3, 3e-3
OBS = ObservationContract(spatial=(ObservationField("spatial", (16, 16, 16), "t", "none"),),
                          scalar=cfg.OBSERVATION.scalar, coordinate_convention="t")


class FakeEnv:
    def __init__(self, n):
        self.n, self.rng = n, np.random.default_rng(0)

    def _obs(self):
        return {"spatial": (self.rng.random((self.n, 16, 16, 16)) < 0.2).astype(np.float32),
                "goal_relative": self.rng.integers(-30, 30, (self.n, 2)).astype(np.float32)}

    def reset(self, seed=None):
        return self._obs(), {}

    def reset_done(self, mask):
        return self._obs()

    def step(self, action):
        term = self.rng.random(self.n) < 0.03
        info = {"reward_terms": {"distance_improvement": self.rng.random(self.n).astype(np.float32),
                                 "arrival_bonus": term.astype(np.float32)},
                "termination_reason": [None] * self.n, "physics_ticks": self.n}
        return self._obs(), self.rng.random(self.n).astype(np.float32), term, np.zeros(self.n, bool), info


def move(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move(v, device) for v in obj]
    return obj


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class DeviceEquivalence(unittest.TestCase):
    def test_update_matches(self):
        ppo = dataclasses.replace(cfg.PPO, rollout_steps=16)
        def build(device):
            torch.manual_seed(5)
            return PPOTrainer(FakeEnv(8), cfg.ACTION, OBS, cfg.MODEL, ppo, device=device, seed=1)
        cpu, gpu = build("cpu"), build("cuda")
        gpu.model.load_state_dict(cpu.model.state_dict())
        gpu.optimizer.load_state_dict(copy.deepcopy(cpu.optimizer.state_dict()))
        gpu.adopt_optimizer_state()
        records, last_value, *_ = cpu._collect(16)
        real = torch.randperm
        try:
            outputs = []
            for trainer, device in ((cpu, "cpu"), (gpu, "cuda")):
                gen = torch.Generator().manual_seed(9)
                torch.randperm = lambda n, device=None, **kw: real(n, generator=gen).to(device or "cpu")
                outputs.append(trainer._update(move(records, device), move(last_value, device)))
        finally:
            torch.randperm = real
        for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "grad_norm"):
            a, b = outputs[0][key], outputs[1][key]
            self.assertLess(abs(a - b), REL_METRIC * max(abs(a), 1e-3), (key, a, b))
        worst = max(float((p.cpu() - q.cpu()).abs().max()) for p, q in zip(cpu.model.parameters(), gpu.model.parameters()))
        self.assertLess(worst, ABS_PARAM)
        self.assertEqual(outputs[0]["optimizer_batches"], outputs[1]["optimizer_batches"])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class GraphEquivalence(unittest.TestCase):
    """CUDA-graph minibatch execution vs the eager GPU update: same operations, so near-identical results."""

    def run_mode(self, graph, records, last_value, initial, updates=3):
        import os
        from unittest import mock
        ppo = dataclasses.replace(cfg.PPO, rollout_steps=16)
        with mock.patch.dict(os.environ, {"CASU_PPO_GRAPH": "1" if graph else "0"}):
            torch.manual_seed(5)
            trainer = PPOTrainer(FakeEnv(8), cfg.ACTION, OBS, cfg.MODEL, ppo, device="cuda", seed=1)
            trainer.model.load_state_dict(initial)
            outputs = []
            for u in range(updates):
                torch.manual_seed(100 + u)
                outputs.append(trainer._update(copy.deepcopy(records), last_value))
        return outputs, [p.detach().clone() for p in trainer.model.parameters()], trainer

    def test_graph_matches_eager(self):
        ppo = dataclasses.replace(cfg.PPO, rollout_steps=16)
        torch.manual_seed(5)
        base = PPOTrainer(FakeEnv(8), cfg.ACTION, OBS, cfg.MODEL, ppo, device="cuda", seed=1)
        records, last_value, *_ = base._collect(16)
        initial = copy.deepcopy(base.model.state_dict())
        eager, p_eager, _ = self.run_mode(False, records, last_value, initial)
        graph, p_graph, trainer = self.run_mode(True, records, last_value, initial)
        self.assertIsNotNone(trainer._mb_graph and trainer._mb_graph.graph)
        for a, b in zip(eager, graph):
            for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "grad_norm"):
                self.assertLess(abs(a[key] - b[key]), 1e-4 * max(abs(a[key]), 1e-3), key)
            self.assertEqual(a["optimizer_batches"], b["optimizer_batches"])
        self.assertLess(max(float((x - y).abs().max()) for x, y in zip(p_eager, p_graph)), 1e-4)   # 0.3 lr after three updates


if __name__ == "__main__":
    unittest.main()
