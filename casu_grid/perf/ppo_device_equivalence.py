"""CPU-vs-CUDA PPO update equivalence on the real XY1 mini-world contract (no simulator needed).

    python casu_grid/perf/ppo_device_equivalence.py --envs 16 --updates 3 [--output out.json]

A recorded rollout (sparse binary 16x64x64 observations like the XY1 encoder produces, random seven-head
actions, rewards and terminations) is fed to two trainers that start from identical weights and optimizer state, one on CPU
and one on CUDA, with identical minibatch permutations. Reports the relative difference of every logged
statistic and the largest parameter difference after each update, plus update time on both devices.
"""
from __future__ import annotations
import argparse, copy, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from casu_grid.workbench import ppo as ppo_module
from casu_grid.workbench.ppo import PPOTrainer
from casu_grid.workbench.experiments import xy1_point_navigation_mini as cfg

p = argparse.ArgumentParser()
p.add_argument("--envs", type=int, default=16)
p.add_argument("--updates", type=int, default=3)
p.add_argument("--output", type=Path)
args = p.parse_args()
H = cfg.ACTION.heads


class FakeEnv:
    def __init__(self, n, seed=0):
        self.n, self.rng = n, np.random.default_rng(seed)

    def _obs(self):
        spatial = (self.rng.random((self.n, 16, 64, 64)) < 0.15).astype(np.float32) * self.rng.random((self.n, 16, 64, 64)).astype(np.float32)
        return {"spatial": spatial, "goal_relative": self.rng.integers(-40, 40, (self.n, 2)).astype(np.float32)}

    def reset(self, seed=None):
        return self._obs(), {}

    def reset_done(self, mask):
        return self._obs()

    def step(self, action):
        term = self.rng.random(self.n) < 0.01
        trunc = (self.rng.random(self.n) < 0.005) & ~term
        info = {"reward_terms": {"distance_improvement": self.rng.random(self.n).astype(np.float32) * 0.1,
                                 "arrival_bonus": term.astype(np.float32) * 10.0},
                "termination_reason": [None] * self.n, "physics_ticks": 10 * self.n}
        return self._obs(), (self.rng.random(self.n) * 0.1 + term * 10.0).astype(np.float32), term, trunc, info


def build(device):
    torch.manual_seed(1234)
    return PPOTrainer(FakeEnv(args.envs), cfg.ACTION, cfg.OBSERVATION, cfg.MODEL, cfg.PPO, device=device, seed=7)


cpu = build("cpu")
records, last_value, *_ = cpu._collect(cfg.PPO.rollout_steps)


def to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    return obj


results = {"envs": args.envs, "samples": cfg.PPO.rollout_steps * args.envs, "updates": []}
has_cuda = torch.cuda.is_available()
cuda = build("cuda") if has_cuda else None
if cuda is not None:
    cuda.model.load_state_dict(cpu.model.state_dict())
    cuda.optimizer.load_state_dict(copy.deepcopy(cpu.optimizer.state_dict()))
real_randperm = torch.randperm
for u in range(args.updates):
    seed = 99 + u
    row = {}
    for name, trainer, dev in (("cpu", cpu, "cpu"), ("cuda", cuda, "cuda")):
        if trainer is None:
            continue
        gen = torch.Generator().manual_seed(seed)
        torch.randperm = lambda n, device=None, **kw: real_randperm(n, generator=gen).to(device or "cpu")
        recs = to_device(records, dev)
        if dev == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        metrics = trainer._update(recs, to_device(last_value, dev))
        if dev == "cuda": torch.cuda.synchronize()
        row[name] = {"metrics": metrics, "seconds": time.perf_counter() - t0}
        torch.randperm = real_randperm
    if cuda is not None:
        diffs = {k: abs(row["cpu"]["metrics"][k] - row["cuda"]["metrics"][k]) / max(1e-12, abs(row["cpu"]["metrics"][k]))
                 for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "grad_norm")}
        pd = max(float((a.cpu() - b.cpu()).abs().max()) for a, b in zip(cpu.model.parameters(), cuda.model.parameters()))
        pr = max(float(((a.cpu() - b.cpu()).abs() / a.detach().cpu().abs().clamp_min(1e-6)).max()) for a, b in zip(cpu.model.parameters(), cuda.model.parameters()))
        row["relative_metric_difference"] = diffs
        row["max_abs_parameter_difference"] = pd
        row["max_rel_parameter_difference"] = pr
    results["updates"].append(row)
    print(json.dumps({k: v for k, v in row.items() if k.startswith(("relative", "max"))}), "cpu %.3fs cuda %s" % (
        row["cpu"]["seconds"], "%.3fs" % row["cuda"]["seconds"] if "cuda" in row else "n/a"), flush=True)
if args.output:
    args.output.write_text(json.dumps(results, indent=1, default=float))
