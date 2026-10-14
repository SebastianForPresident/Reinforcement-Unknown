"""Worker-local rollout collection (policy inference inside the simulation workers).

The trainer normally samples every decision centrally: all workers step, the parent waits for the slowest, runs the
policy for all worlds, scatters the actions, and receives full 16 x 64 x 64 float observations through a pipe. Here each
worker owns a copy of the policy (weights are broadcast once per update) and runs a complete rollout of its shard on its own:

  * no per-decision synchronisation between workers (straggler waits average out over the rollout),
  * observations never cross a process boundary per decision: they are stored as compact id windows
    (``casu_grid.xy1``, 8 KB per world and decision) and expanded to channels on the GPU, bit for bit,
  * the value of the next observation (needed for GAE bootstrapping) is computed once and reused as the next decision's
    policy input when no world was reset, halving the number of network evaluations.

The mathematics of ``PPOTrainer._collect`` is unchanged: same sampling distribution (a per-update seeded generator replaces the
trainer's global RNG), same bootstrap rule, same episode bookkeeping, same reset order. ``tests/test_local_rollout.py`` compares
the two paths with deterministic action selection.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .model import ConfigurableActorCritic


class _PolicyStep:
    """expand -> network -> sample, captured as one CUDA graph for a fixed number of worlds.

    Sampling uses the Gumbel-max trick on externally generated uniform noise (exactly categorical sampling), so the graph
    contains no RNG state. ``deterministic`` multiplies the noise by zero (argmax). The log-probabilities are the
    normalised logits that ``torch.distributions.Categorical`` stores, gathered at the sampled index, so they equal the
    values the PPO update recomputes.
    """

    def __init__(self, owner, count, window):
        dev = owner.device
        self.owner, self.dev, self.n = owner, dev, count
        w, h = window
        total = sum(owner.head_sizes)
        self.static = {
            "block_ids": torch.zeros((count, w, h), dtype=torch.uint8, device=dev),
            "fluid_ids": torch.zeros((count, w, h), dtype=torch.uint8, device=dev),
            "cursor": torch.full((count, 2), -1, dtype=torch.int16, device=dev),
            "goal": torch.zeros((count, 2), dtype=torch.float32, device=dev),
            "noise": torch.full((count, total), 0.5, dtype=torch.float32, device=dev),
            "scale": torch.ones((1,), dtype=torch.float32, device=dev)}
        self.graph = None
        if dev.type == "cuda":
            torch.backends.cudnn.benchmark = True
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._compute()
            torch.cuda.current_stream().wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = self._compute()
        else:
            self.out = None

    def _compute(self):
        from ..xy1 import expand_compact_observation
        st, owner = self.static, self.owner
        with torch.no_grad():
            obs = {owner.spatial_name: expand_compact_observation(st["block_ids"], st["fluid_ids"], st["cursor"], self.dev),
                   owner.goal_name: st["goal"]}
            dists, value = owner.model.distributions(obs)
            latents, log_prob, offset = [], None, 0
            for name, size in zip(owner.head_names, owner.head_sizes):
                logits = dists[name].logits                                   # normalised log-probabilities
                u = st["noise"][:, offset:offset + size].clamp(1e-20, 1.0 - 1e-7)
                latent = (logits + st["scale"] * (-torch.log(-torch.log(u)))).argmax(dim=-1)
                lp = logits.gather(1, latent.unsqueeze(1)).squeeze(1)
                log_prob = lp if log_prob is None else log_prob + lp
                latents.append(latent)
                offset += size
            return torch.stack(latents, dim=1), log_prob, value

    def run(self, compact, generator, deterministic):
        st = self.static
        for key in ("block_ids", "fluid_ids", "cursor", "goal"):
            st[key].copy_(torch.from_numpy(np.ascontiguousarray(compact[key])), non_blocking=True)
        st["noise"].uniform_(generator=generator)
        st["scale"].fill_(0.0 if deterministic else 1.0)
        if self.graph is not None:
            self.graph.replay()
            latent, log_prob, value = self.out
        else:
            latent, log_prob, value = self._compute()
        return latent.cpu(), log_prob.cpu(), value.cpu()


class LocalRollout:
    def __init__(self, env, actions, observations, model_contract, device: str = "cuda"):
        if len(observations.spatial) != 1 or len(observations.scalar) != 1:
            raise ValueError("local rollouts support one spatial and one scalar observation field")
        if any(head.kind != "categorical" for head in actions.heads):
            raise ValueError("local rollouts support categorical action heads only")
        self.env = env
        self.actions = actions
        self.spatial_name = observations.spatial[0].name
        self.goal_name = observations.scalar[0].name
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.model = ConfigurableActorCritic(actions, observations, model_contract).to(self.device).eval()
        self.head_names = [h.name for h in actions.heads]
        self.head_sizes = [h.size for h in actions.heads]
        self._step_fn = None

    def load_weights(self, state) -> None:
        self.model.load_state_dict({k: torch.as_tensor(v) for k, v in state.items()})

    # ------------------------------------------------------------------ helpers
    def _expand(self, compact):
        from ..xy1 import expand_compact_observation
        spatial = expand_compact_observation(compact["block_ids"], compact["fluid_ids"], compact["cursor"], self.device)
        goal = torch.as_tensor(compact["goal"], dtype=torch.float32, device=self.device)
        return {self.spatial_name: spatial, self.goal_name: goal}

    def _forward(self, compact):
        with torch.no_grad():
            dists, value = self.model.distributions(self._expand(compact))
        return dists, value

    def _sample(self, dists, generator, deterministic):
        latents = []
        log_prob = None
        for name in self.head_names:
            d = dists[name]
            latent = d.probs.argmax(dim=-1) if deterministic else torch.multinomial(d.probs, 1, True, generator=generator).squeeze(-1)
            lp = d.log_prob(latent)
            log_prob = lp if log_prob is None else log_prob + lp
            latents.append(latent)
        return torch.stack(latents, dim=1), log_prob

    # --------------------------------------------------------------------- run
    def run(self, steps: int, seed: int, deterministic: bool, episode_returns, episode_lengths):
        env = self.env
        n = env.count
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        returns = torch.as_tensor(episode_returns, dtype=torch.float32).clone()
        lengths = torch.as_tensor(episode_lengths, dtype=torch.long).clone()
        out = {k: [] for k in ("block_ids", "fluid_ids", "cursor", "goal", "latent", "log_prob", "value", "next_value",
                               "reward", "done", "terminated", "truncated", "players", "reasons")}
        terms: dict[str, list] = {}
        completed_returns, completed_lengths = [], []
        termination_counts: dict[str, int] = {}
        # XY1 observability (read-only: nothing here feeds back into the policy, reward or environment)
        episodes = {k: [] for k in ("world", "reason", "length", "initial_distance", "best_distance", "final_distance", "blocks_destroyed", "return")}
        blocks_prev = getattr(self, "_blocks_prev", None)
        if blocks_prev is None or len(blocks_prev) != n:
            blocks_prev = np.zeros(n, np.float32)
        blocks_total = 0.0
        xy1 = None
        action_counts = [torch.zeros(size, dtype=torch.long) for size in self.head_sizes]
        env_seconds = infer_seconds = reset_seconds = 0.0
        clock = time.perf_counter
        compact = env.compact_observation()
        if self._step_fn is None or self._step_fn.n != n:
            self._step_fn = _PolicyStep(self, n, compact["block_ids"].shape[1:])
        step_fn = self._step_fn
        latent_cpu, log_prob, value = step_fn.run(compact, generator, deterministic)
        for _ in range(steps):
            t1 = clock()
            for k, count in enumerate(action_counts):
                count += torch.bincount(latent_cpu[:, k], minlength=len(count))
            env_action = {name: latent_cpu[:, k].numpy() for k, name in enumerate(self.head_names)}
            _, reward, terminated, truncated, info = env.step(env_action, observe=False)
            t2 = clock()
            env_seconds += t2 - t1
            reward_t = torch.as_tensor(reward, dtype=torch.float32)
            done = torch.as_tensor(np.logical_or(terminated, truncated), dtype=torch.float32)
            terminated_t = torch.as_tensor(terminated, dtype=torch.float32)
            next_compact = env.compact_observation()
            # one network evaluation of the state reached (before any reset): its value bootstraps GAE, and its sample is the
            # next decision's action when no world was reset
            next_latent, next_log_prob, next_value = step_fn.run(next_compact, generator, deterministic)
            t3 = clock()
            infer_seconds += t3 - t2
            out["block_ids"].append(compact["block_ids"]); out["fluid_ids"].append(compact["fluid_ids"])
            out["cursor"].append(compact["cursor"]); out["goal"].append(compact["goal"])
            out["latent"].append(latent_cpu); out["log_prob"].append(log_prob); out["value"].append(value)
            out["next_value"].append(next_value); out["reward"].append(reward_t); out["done"].append(done)
            out["terminated"].append(terminated_t); out["truncated"].append(torch.as_tensor(truncated))
            out["players"].append([(p["x"], p["y"], p["vx"], p["vy"]) for p in info["player"]])
            out["reasons"].append(list(info["termination_reason"]))
            for name, values in info["reward_terms"].items():
                terms.setdefault(name, []).append(torch.as_tensor(values, dtype=torch.float32))
            xy1 = info.get("xy1")
            if xy1 is not None:
                blocks_total += float((xy1["blocks_destroyed"] - blocks_prev).sum())
                blocks_prev = xy1["blocks_destroyed"].copy()
            returns += reward_t
            lengths += 1
            if bool(done.any()):
                mask = done.bool()
                completed_returns.extend(returns[mask].tolist())
                completed_lengths.extend(lengths[mask].tolist())
                for reason, finished in zip(info["termination_reason"], mask.tolist()):
                    if finished:
                        termination_counts[str(reason)] = termination_counts.get(str(reason), 0) + 1
                if xy1 is not None:
                    for j in np.flatnonzero(mask.numpy()):
                        episodes["world"].append(int(j))
                        episodes["reason"].append(str(info["termination_reason"][j]))
                        episodes["length"].append(int(lengths[j]))
                        episodes["initial_distance"].append(float(xy1["initial_distance"][j]))
                        episodes["best_distance"].append(float(xy1["best_distance"][j]))
                        episodes["final_distance"].append(float(xy1["goal_distance"][j]))
                        episodes["blocks_destroyed"].append(float(xy1["blocks_destroyed"][j]))
                        episodes["return"].append(float(returns[j]))
                    blocks_prev[mask.numpy()] = 0
                returns[mask] = 0
                lengths[mask] = 0
                t4 = clock()
                env.reset_done(mask.numpy(), observe=False)
                compact = env.compact_observation()
                latent_cpu, log_prob, value = step_fn.run(compact, generator, deterministic)
                reset_seconds += clock() - t4
            else:
                compact, latent_cpu, log_prob, value = next_compact, next_latent, next_log_prob, next_value
        # the policy input of the first decision of the next rollout is recomputed there (weights will have changed)
        self._blocks_prev = blocks_prev
        active = None
        if xy1 is not None:       # episodes still in progress at the end of the rollout (censored: they have not ended yet)
            keep = ~done.bool().numpy()
            active = {"decisions": lengths.numpy()[keep].astype(np.int64), "goal_distance": xy1["goal_distance"][keep],
                      "best_distance": xy1["best_distance"][keep], "initial_distance": xy1["initial_distance"][keep],
                      "blocks_destroyed": xy1["blocks_destroyed"][keep], "outside_world": xy1["outside_world"][keep]}
        stacked = {k: np.stack(v) for k, v in out.items() if k in ("block_ids", "fluid_ids", "cursor", "goal", "players")}
        result = {
            **stacked,
            "latent": torch.stack(out["latent"]).numpy(), "log_prob": torch.stack(out["log_prob"]).numpy(),
            "value": torch.stack(out["value"]).numpy(), "next_value": torch.stack(out["next_value"]).numpy(),
            "reward": torch.stack(out["reward"]).numpy(), "done": torch.stack(out["done"]).numpy(),
            "terminated": torch.stack(out["terminated"]).numpy(), "truncated": torch.stack(out["truncated"]).numpy(),
            "reasons": out["reasons"], "reward_terms": {k: torch.stack(v).numpy() for k, v in terms.items()},
            "completed_returns": completed_returns, "completed_lengths": completed_lengths,
            "termination_counts": termination_counts,
            "nav_episodes": episodes if xy1 is not None else None, "nav_active": active, "nav_blocks_destroyed": blocks_total if xy1 is not None else None,
            "action_counts": [c.numpy() for c in action_counts],
            "episode_returns": returns.numpy(), "episode_lengths": lengths.numpy(),
            "seconds": {"env_step": env_seconds, "inference": infer_seconds, "reset": reset_seconds},
            "physics_ticks": int(env.cfg.action_decimation) * n * steps,
        }
        return result
