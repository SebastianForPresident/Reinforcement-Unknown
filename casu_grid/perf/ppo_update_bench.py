"""PPO update microbenchmark and execution-equivalence check (no simulator, no Kit).

    python casu_grid/perf/ppo_update_bench.py --envs 64 --repeats 5 --output out.json
    python casu_grid/perf/ppo_update_bench.py --config casu_grid/perf/synthetic_large_model.py --envs 64   # SYNTHETIC net

Builds the real ``PPOTrainer`` with the engineering-only contract, collects one rollout from a fake vector
environment (random observations, rewards and terminations; the PPO maths does not care where they came
from) and runs the update with the reference implementation (``CASU_PPO_EXECUTION=reference``) and with
the optimized execution path from identical model / optimizer / RNG state. It reports time per update and
compares the results exactly: parameters, optimizer state, metrics and RNG state after the update.
"""
from __future__ import annotations
import argparse, copy, importlib.util, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--envs", type=int, default=64)
p.add_argument("--repeats", type=int, default=5)
p.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "workbench/engineering_smoke.py")
p.add_argument("--threads", type=int, default=None, help="torch.set_num_threads for the measurement")
p.add_argument("--profile", action="store_true")
p.add_argument("--output", type=Path, default=None)
args = p.parse_args()
if args.threads:
    torch.set_num_threads(args.threads)

from casu_grid.workbench.ppo import PPOTrainer

spec = importlib.util.spec_from_file_location("bench_config", args.config)
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)


class FakeVectorEnv:
    """Random-observation vector environment with the engineering observation contract."""

    def __init__(self, n, seed=0):
        self.n, self.rng = n, np.random.default_rng(seed)

    def _obs(self):
        return {"player_state": self.rng.normal(size=(self.n, 8)).astype(np.float32)}

    def reset(self, seed=None):
        return self._obs(), {}

    def reset_done(self, mask):
        return self._obs()

    def step(self, action):
        terminated = self.rng.random(self.n) < 0.03
        truncated = self.rng.random(self.n) < 0.02
        info = {"reward_terms": {t.name: self.rng.normal(size=self.n).astype(np.float32) for t in cfg.REWARD.terms},
                "termination_reason": [None] * self.n, "physics_ticks": 10 * self.n}
        return self._obs(), self.rng.normal(size=self.n).astype(np.float32), terminated, truncated & ~terminated, info


def build():
    torch.manual_seed(1234)
    return PPOTrainer(FakeVectorEnv(args.envs), cfg.ACTION, cfg.OBSERVATION, cfg.MODEL, cfg.PPO, device="cpu", seed=7)


os.environ["CASU_PPO_EXECUTION"] = "reference"
trainer = build()
records, last_value, *_ = trainer._collect(cfg.PPO.rollout_steps)
initial_model = copy.deepcopy(trainer.model.state_dict())
initial_optimizer = copy.deepcopy(trainer.optimizer.state_dict())


def run(mode):
    os.environ["CASU_PPO_EXECUTION"] = mode
    trainer = build()          # optimizer/distribution settings are decided per mode at construction
    times, last = [], None
    for repeat in range(args.repeats + 1):
        trainer.model.load_state_dict(initial_model)
        trainer.optimizer.load_state_dict(copy.deepcopy(initial_optimizer))
        trainer.update_count = 0
        torch.manual_seed(99)
        started = time.perf_counter()
        metrics = trainer._update(records, last_value)
        elapsed = time.perf_counter() - started
        if repeat:                     # first run warms caches
            times.append(elapsed)
        last = {"metrics": metrics,
                "params": [v.detach().clone() for v in trainer.model.state_dict().values()],
                "opt_state": copy.deepcopy(trainer.optimizer.state_dict()),
                "rng": torch.get_rng_state().clone()}
    return times, last


def equal_state(a, b):
    out = {"metrics_identical": a["metrics"] == b["metrics"],
           "parameters_bit_identical": all(torch.equal(x, y) for x, y in zip(a["params"], b["params"])),
           "rng_state_identical": bool(torch.equal(a["rng"], b["rng"]))}
    sa, sb = a["opt_state"]["state"], b["opt_state"]["state"]
    out["optimizer_state_bit_identical"] = (sa.keys() == sb.keys() and all(
        all(torch.equal(sa[k][name], sb[k][name]) if torch.is_tensor(sa[k][name]) else sa[k][name] == sb[k][name]
            for name in sa[k]) for k in sa))
    out["max_abs_parameter_difference"] = max(float((x - y).abs().max()) for x, y in zip(a["params"], b["params"]))
    return out


results = {"schema": "casu.ppo-update-bench.v1", "envs": args.envs, "config": str(args.config), "threads": torch.get_num_threads(),
           "samples_per_update": cfg.PPO.rollout_steps * args.envs, "minibatch": cfg.PPO.minibatch_size, "epochs": cfg.PPO.epochs}
for mode in ("reference", "optimized"):
    if args.profile and mode == "reference":
        import cProfile, pstats
        pr = cProfile.Profile(); pr.enable()
        times, state = run(mode)
        pr.disable(); pstats.Stats(pr).sort_stats("tottime").print_stats(14)
    else:
        times, state = run(mode)
    results[mode] = {"update_seconds": {"mean": float(np.mean(times)), "std": float(np.std(times)), "min": float(np.min(times)),
                                        "samples": times}, "optimizer_batches": state["metrics"]["optimizer_batches"]}
    results[f"_state_{mode}"] = state
results["equivalence"] = equal_state(results.pop("_state_reference"), results.pop("_state_optimized"))
results["speedup"] = results["reference"]["update_seconds"]["mean"] / results["optimized"]["update_seconds"]["mean"]
print(json.dumps({k: v for k, v in results.items() if k not in ("reference", "optimized")}, indent=1))
print("reference %.1f ms/update, optimized %.1f ms/update (%d minibatches)" % (
    1e3 * results["reference"]["update_seconds"]["mean"], 1e3 * results["optimized"]["update_seconds"]["mean"],
    results["reference"]["optimizer_batches"]))
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
