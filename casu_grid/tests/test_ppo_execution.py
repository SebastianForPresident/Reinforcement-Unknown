"""Execution-only PPO optimizations must reproduce the reference update bit for bit."""
import copy
import os
import unittest
from unittest import mock

import numpy as np
import torch

from casu_grid.workbench.contracts import ModelContract, PPOContract
from casu_grid.workbench.engineering_smoke import ACTION, MODEL, OBSERVATION, PPO, REWARD
from casu_grid.workbench.ppo import PPOTrainer


class FakeVectorEnv:
    def __init__(self, n, seed=0):
        self.n, self.rng = n, np.random.default_rng(seed)

    def _obs(self):
        return {"player_state": self.rng.normal(size=(self.n, 8)).astype(np.float32)}

    def reset(self, seed=None):
        return self._obs(), {}

    def reset_done(self, mask):
        return self._obs()

    def step(self, action):
        terminated = self.rng.random(self.n) < 0.05
        truncated = (self.rng.random(self.n) < 0.03) & ~terminated
        info = {"reward_terms": {t.name: self.rng.normal(size=self.n).astype(np.float32) for t in REWARD.terms},
                "termination_reason": [None] * self.n, "physics_ticks": 10 * self.n}
        return self._obs(), self.rng.normal(size=self.n).astype(np.float32), terminated, truncated, info


def build(mode, ppo=PPO, model=MODEL, envs=12):
    with mock.patch.dict(os.environ, {"CASU_PPO_EXECUTION": mode}):
        torch.manual_seed(5)
        return PPOTrainer(FakeVectorEnv(envs), ACTION, OBSERVATION, model, ppo, device="cpu", seed=3)


def update_with(mode, records, last_value, initial, ppo=PPO, model=MODEL):
    trainer = build(mode, ppo, model)
    trainer.model.load_state_dict(initial[0])
    trainer.optimizer.load_state_dict(copy.deepcopy(initial[1]))
    torch.manual_seed(77)
    with mock.patch.dict(os.environ, {"CASU_PPO_EXECUTION": mode}):
        metrics = trainer._update(records, last_value)
    return metrics, [p.detach().clone() for p in trainer.model.parameters()], trainer.optimizer.state_dict(), torch.get_rng_state()


class PPOExecutionEquivalence(unittest.TestCase):
    def check(self, ppo=PPO, model=MODEL):
        base = build("reference", ppo, model)
        records, last_value, *_ = base._collect(ppo.rollout_steps)
        initial = (copy.deepcopy(base.model.state_dict()), copy.deepcopy(base.optimizer.state_dict()))
        ref = update_with("reference", records, last_value, initial, ppo, model)
        opt = update_with("optimized", records, last_value, initial, ppo, model)
        self.assertEqual(ref[0], opt[0])                                     # every metric
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(ref[1], opt[1])))   # parameters
        for key, state in ref[2]["state"].items():                           # Adam moments and step counts
            for name, value in state.items():
                self.assertTrue(torch.equal(value, opt[2]["state"][key][name]), (key, name))
        self.assertTrue(torch.equal(ref[3], opt[3]))                          # RNG consumption
        return ref[0]

    def test_default_contract(self):
        metrics = self.check()
        self.assertGreater(metrics["optimizer_batches"], 0)

    def test_early_stop_on_target_kl(self):
        ppo = PPOContract(learning_rate=5e-2, rollout_steps=16, minibatch_size=24, epochs=4, target_kl=1e-6)
        metrics = self.check(ppo)
        self.assertLess(metrics["epochs_completed"], 4)

    def test_uneven_minibatches_and_larger_network(self):
        ppo = PPOContract(learning_rate=1e-3, rollout_steps=10, minibatch_size=7, epochs=3, target_kl=None)
        model = ModelContract(spatial_channels=(), spatial_kernels=(), scalar_layers=(32, 32), fusion_layers=(32,))
        self.check(ppo, model)

    def test_recurrent_models_use_the_reference_path(self):
        ppo = PPOContract(learning_rate=1e-3, rollout_steps=8, minibatch_size=4, epochs=2, target_kl=None)
        model = ModelContract(spatial_channels=(), spatial_kernels=(), scalar_layers=(16,), fusion_layers=(16,),
                              recurrent=True, recurrent_size=8)
        trainer = build("optimized", ppo, model)
        with mock.patch.object(trainer, "_update_reference", return_value="reference") as ref, \
                mock.patch.object(trainer, "_update_optimized", return_value="optimized") as opt:
            self.assertEqual(trainer._update({}, None), "reference")
        ref.assert_called_once()
        opt.assert_not_called()

    def test_foreach_adam_matches_the_default_loop(self):
        ref, opt = build("reference"), build("optimized")
        self.assertFalse(ref.optimizer.defaults["foreach"] or False)
        self.assertTrue(opt.optimizer.defaults["foreach"])
        for t in (ref, opt):
            t.model.load_state_dict(copy.deepcopy(ref.model.state_dict()))
        gen = torch.Generator().manual_seed(1)
        for _ in range(40):
            grads = [torch.randn(p.shape, generator=gen) for p in ref.model.parameters()]
            for t in (ref, opt):
                for p, g in zip(t.model.parameters(), grads):
                    p.grad = g.clone()
                t.optimizer.step()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(ref.model.parameters(), opt.model.parameters())))


if __name__ == "__main__":
    unittest.main()
