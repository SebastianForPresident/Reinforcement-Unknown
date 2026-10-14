"""Small PPO implementation with explicit rollouts, diagnostics, and checkpoints."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .contracts import ActionContract, ModelContract, ObservationContract, PPOContract
from .model import ConfigurableActorCritic, count_parameters


def _as_tensor_obs(observation: dict[str, Any], device: torch.device):
    return {key: torch.as_tensor(value, dtype=torch.float32, device=device)
            for key, value in observation.items()}


def _generalized_advantage_estimate(rewards: torch.Tensor, values: torch.Tensor,
                                   dones: torch.Tensor, terminated: torch.Tensor,
                                   next_values: torch.Tensor, gamma: float,
                                   gae_lambda: float):
    """Compute GAE with time-limit truncations bootstrapped from final observations.

    ``dones`` stops advantage recursion at either terminal or truncation, while
    ``terminated`` alone disables the value bootstrap. This distinction is
    required when a finite rollout horizon is used.
    """
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(values[0])
    for t in reversed(range(len(rewards))):
        bootstrap_mask = 1.0 - terminated[t]
        continuation_mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * bootstrap_mask - values[t]
        gae = delta + gamma * gae_lambda * continuation_mask * gae
        advantages[t] = gae
    return advantages, advantages + values


def _restore_final_observations(next_obs: dict[str, torch.Tensor], info: Any,
                                device: torch.device):
    """Replace auto-reset observations with per-env final observations when given.

    Gymnasium vector wrappers may return reset observations from ``step`` and
    place the actual terminal/truncated state in ``info['final_observation']``.
    The Isaac adapter does not auto-reset, so its returned next observation is
    already the state to bootstrap from.
    """
    if not isinstance(info, dict) or "final_observation" not in info:
        return next_obs
    final = info["final_observation"]
    if final is None:
        return next_obs
    batch = next(iter(next_obs.values())).shape[0]
    if isinstance(final, dict):
        final_rows = [final] if batch == 1 else [None] * batch
    else:
        final_rows = list(final)
        if len(final_rows) != batch:
            raise ValueError("final_observation batch does not match environment batch")
    mask = info.get("_final_observation")
    if mask is None:
        valid = [row is not None for row in final_rows]
    else:
        valid = np.asarray(mask, dtype=bool).reshape(batch).tolist()
    result = {key: value.clone() for key, value in next_obs.items()}
    for env_index, (row, enabled) in enumerate(zip(final_rows, valid)):
        if not enabled:
            continue
        if not isinstance(row, dict) or set(row) != set(result):
            raise ValueError("each final_observation must contain the configured observation keys")
        for key in result:
            value = torch.as_tensor(row[key], dtype=result[key].dtype, device=device)
            if tuple(value.shape) != tuple(result[key][env_index].shape):
                raise ValueError(f"final observation {key} has shape {tuple(value.shape)}; "
                                 f"expected {tuple(result[key][env_index].shape)}")
            result[key][env_index] = value
    return result


class PPOTrainer:
    """Gymnasium-compatible PPO trainer; environment owns simulator semantics."""

    def __init__(self, env, actions: ActionContract, observations: ObservationContract,
                 model: ModelContract, config: PPOContract,
                 device: str = "cpu", seed: int = 0, *, log_dir: str | Path | None = None):
        self.env = env
        self.actions = actions
        self.observations = observations
        self.config = config
        self.model_contract = model
        self._local_mode = None          # None = undecided, False = central rollouts, True = worker-local rollouts
        self.device = torch.device(device)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = ConfigurableActorCritic(actions, observations, model).to(self.device)
        # foreach Adam applies the same elementwise operations as the default single-tensor loop (checked
        # bit-identical, perf/ppo_update_bench.py); CASU_PPO_EXECUTION=reference keeps the default loop
        self._mb_graph = None
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate,
                                          foreach=os.environ.get("CASU_PPO_EXECUTION", "optimized") != "reference",
                                          capturable=self._graph_enabled())      # device-resident step counter (CUDA graphs)
        self.update_count = 0
        self.total_steps = 0
        self.rng = np.random.default_rng(seed)
        self._observation = None
        self._recurrent_state = None
        self._episode_returns = None
        self._episode_lengths = None
        self.log_dir = None if log_dir is None else Path(log_dir)
        self.writer = None
        self.trajectory_path = None
        if self.log_dir is not None:
            from torch.utils.tensorboard import SummaryWriter
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(self.log_dir))
            self.trajectory_path = self.log_dir.parent / "trajectories.jsonl"
        contract_payload = json.dumps({
            "actions": asdict(actions), "observations": asdict(observations),
            "model": asdict(model), "ppo": asdict(config),
        }, sort_keys=True, separators=(",", ":"), default=list).encode("utf-8")
        self.contract_fingerprint = hashlib.sha256(contract_payload).hexdigest()

    def adopt_optimizer_state(self):
        """After ``optimizer.load_state_dict`` (possibly from a checkpoint written on another device): restore this trainer's
        ``capturable`` setting and put the Adam step counters where the update expects them."""
        capturable = self._graph_enabled()
        for group in self.optimizer.param_groups:
            group["capturable"] = capturable
        for p, state in self.optimizer.state.items():
            step = state.get("step")
            if torch.is_tensor(step):
                state["step"] = step.to(device=p.device if capturable else "cpu", dtype=torch.float32)
        self._mb_graph = None

    def _graph_enabled(self) -> bool:
        """CUDA-graph minibatch execution (``workbench.ppo_graph``); ``CASU_PPO_GRAPH=0`` selects eager execution."""
        return (self.device.type == "cuda" and os.environ.get("CASU_PPO_GRAPH", "1") != "0"
                and os.environ.get("CASU_PPO_EXECUTION", "optimized") != "reference" and not self.model.recurrent)

    # ------------------------------------------------------ worker-local rollouts
    def _use_local_rollout(self) -> bool:
        """Worker-local policy inference (see :mod:`casu_grid.workbench.local_rollout`) when the environment supports it.

        ``CASU_LOCAL_ROLLOUT=0`` forces the central path (policy evaluated by the trainer after every environment step)."""
        if self._local_mode is None:
            mode = os.environ.get("CASU_LOCAL_ROLLOUT", "auto")
            supported = (mode != "0" and hasattr(self.env, "local_rollout_run") and getattr(self.env, "has_stock_xy1_observation", False)
                         and not self.model.recurrent and len(self.observations.spatial) == 1 and len(self.observations.scalar) == 1
                         and all(h.kind == "categorical" for h in self.actions.heads)
                         and (self.device.type == "cuda" or mode == "1"))
            if supported:
                self.env.local_rollout_setup(self.actions, self.observations, self.model_contract,
                                             "cuda" if self.device.type == "cuda" else "cpu")
            self._local_mode = bool(supported)
        return self._local_mode

    def _expand_compact_batch(self, batch):
        from ..xy1 import expand_compact_observation
        spatial = expand_compact_observation(batch["block_ids"], batch["fluid_ids"], batch["cursor"], self.device)
        return {self.observations.spatial[0].name: spatial, self.observations.scalar[0].name: batch["goal"].float()}

    def _collect_local(self, steps: int):
        started = time.perf_counter()
        env = self.env
        if self._observation is None:
            env.reset(seed=int(self.rng.integers(0, 2**31 - 1)), observe=False)
            if self.trajectory_path is not None:
                with self.trajectory_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"schema": "casu.trajectory-reset.v1", "global_step": self.total_steps}) + "\n")
            self._observation = {self.observations.scalar[0].name: torch.zeros((env.count, 2), device=self.device)}
        n = env.count
        if self._episode_returns is None:
            self._episode_returns = torch.zeros(n, device=self.device)
            self._episode_lengths = torch.zeros(n, dtype=torch.long, device=self.device)
        workers = getattr(env, "num_workers", 1)
        seeds = [int(self.rng.integers(0, 2**31 - 1)) for _ in range(workers)]
        weights = {k: v.detach().cpu().numpy() for k, v in self.model.state_dict().items()}
        res = env.local_rollout_run(weights, steps, seeds, False, self._episode_returns.cpu().numpy(),
                                    self._episode_lengths.cpu().numpy())
        dev = self.device

        def t(array, dtype=None):
            return torch.as_tensor(array, dtype=dtype, device=dev)
        compact = {k: t(res[k]) for k in ("block_ids", "fluid_ids", "cursor", "goal")}
        latent, log_prob, value = t(res["latent"]), t(res["log_prob"]), t(res["value"])
        names = [h.name for h in self.actions.heads]
        records = {
            "obs": None, "obs_compact": compact,
            "actions": [{name: latent[i, :, k] for k, name in enumerate(names)} for i in range(steps)],
            "log_prob": list(log_prob.unbind(0)), "value": list(value.unbind(0)),
            "reward": list(t(res["reward"]).unbind(0)), "done": list(t(res["done"]).unbind(0)),
            "terminated": list(t(res["terminated"]).unbind(0)), "next_value": list(t(res["next_value"]).unbind(0)),
            "reward_terms": [{name: t(arr[i]) for name, arr in res["reward_terms"].items()} for i in range(steps)],
            "initial_state": [],
        }
        self._episode_returns = t(res["episode_returns"], torch.float32)
        self._episode_lengths = t(res["episode_lengths"], torch.long)
        if self.trajectory_path is not None:
            self._write_local_trajectory(res, names, steps, n)
        self.total_steps += n * steps
        total_rows = max(1, steps * n)
        frequencies = {}
        for head, counts in zip(self.actions.heads, res["action_counts"]):
            for choice, count in enumerate(counts.tolist()):
                frequencies[f"action/{head.name}/choice_{choice}_fraction"] = count / total_rows
        physics_step_seconds = max(res["seconds"]["env_step"]) + max(res["seconds"]["reset"])
        self._last_local_seconds = res["seconds"]
        self._last_nav = (res.get("nav_episodes"), res.get("nav_active"), res.get("nav_blocks_destroyed"))
        return (records, torch.zeros(n, device=dev), res["completed_returns"], res["completed_lengths"],
                res["termination_counts"], frequencies, time.perf_counter() - started, physics_step_seconds, res["physics_ticks"])

    def _write_local_trajectory(self, res, names, steps, n):
        """Append the same per-step rows the central path writes (``casu.trajectory-step.v1``)."""
        content_id = None
        rows = []
        for i in range(steps):
            info = {"reward_terms": {k: v[i] for k, v in res["reward_terms"].items()},
                    "termination_reason": res["reasons"][i], "physics_ticks": int(res["physics_ticks"] // steps),
                    "player": [dict(x=float(p[0]), y=float(p[1]), vx=float(p[2]), vy=float(p[3])) for p in res["players"][i]],
                    "canonical_content_id": getattr(self.env, "reference").manifest["content_id"]}
            row = {"schema": "casu.trajectory-step.v1", "global_step": self.total_steps + i * n,
                   "actions": {name: res["latent"][i, :, k].tolist() for k, name in enumerate(names)},
                   "reward": res["reward"][i].tolist(), "terminated": np.asarray(res["terminated"][i] > 0).tolist(),
                   "truncated": np.asarray(res["truncated"][i]).tolist(), "info": info}
            rows.append(json.dumps(row, sort_keys=True, default=_json_default, allow_nan=False))
        with self.trajectory_path.open("a", encoding="utf-8") as stream:
            stream.write("\n".join(rows) + "\n")

    def _collect(self, steps: int):
        if self._use_local_rollout():
            return self._collect_local(steps)
        collection_started = time.perf_counter()
        physics_step_seconds = 0.0
        physics_ticks = 0
        recurrent_state = self._recurrent_state
        if self._observation is None:
            obs, _ = self.env.reset(seed=int(self.rng.integers(0, 2**31 - 1)))
            self._observation = _as_tensor_obs(obs, self.device)
            if self.trajectory_path is not None:   # replay restarts its world here
                with self.trajectory_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"schema": "casu.trajectory-reset.v1",
                                             "global_step": self.total_steps}) + "\n")
        obs = self._observation
        env_count = next(iter(obs.values())).shape[0]
        expected = {field.name: field.shape for field in
                    self.observations.spatial + self.observations.scalar}
        if set(obs) != set(expected):
            raise ValueError(f"environment observation keys {sorted(obs)} do not match contract {sorted(expected)}")
        for name, shape in expected.items():
            if tuple(obs[name].shape[1:]) != tuple(shape):
                raise ValueError(f"observation {name} shape {tuple(obs[name].shape[1:])} != contract {shape}")
        records = {"obs": [], "actions": [], "log_prob": [], "value": [],
                   "reward": [], "done": [], "terminated": [], "next_value": [],
                   "reward_terms": [], "initial_state": []}
        if self._episode_returns is None:
            self._episode_returns = torch.zeros(env_count, device=self.device)
            self._episode_lengths = torch.zeros(env_count, dtype=torch.long, device=self.device)
        episode_returns = self._episode_returns
        episode_lengths = self._episode_lengths
        completed_returns = []
        completed_lengths = []
        termination_counts: dict[str, int] = {}
        action_counts = {head.name: torch.zeros(head.size, dtype=torch.long)
                         for head in self.actions.heads if head.kind == "categorical"}
        for _ in range(steps):
            with torch.no_grad():
                if self.model.recurrent:
                    initial_state = (torch.zeros(1, env_count, self.model.recurrent_size,
                                                  device=self.device)
                                     if recurrent_state is None else recurrent_state)
                    action, log_prob, value, latent_action, recurrent_state = self.model.act(
                        obs, return_latent=True, return_state=True, sequence_length=1,
                        initial_state=initial_state)
                    records["initial_state"].append(initial_state.detach())
                else:
                    action, log_prob, value, latent_action = self.model.act(obs, return_latent=True)
            env_action = {name: tensor.detach().cpu().numpy() for name, tensor in action.items()}
            for head in self.actions.heads:
                values = env_action[head.name]
                if values.shape[0] != env_count or not np.all(np.isfinite(values)):
                    raise ValueError(f"action {head.name} has invalid batch or non-finite values")
                if head.kind == "categorical" and (np.any(values < 0) or np.any(values >= head.size)):
                    raise ValueError(f"categorical action {head.name} is outside [0, {head.size})")
                if head.kind == "continuous" and (np.any(values < head.low - 1e-6) or
                                                    np.any(values > head.high + 1e-6)):
                    raise ValueError(f"continuous action {head.name} exceeds configured bounds")
            for name, latent in latent_action.items():
                if name in action_counts:
                    action_counts[name] += torch.bincount(latent.reshape(-1).cpu(),
                                                          minlength=len(action_counts[name]))
            step_started = time.perf_counter()
            next_obs, reward, terminated, truncated, info = self.env.step(env_action)
            physics_step_seconds += time.perf_counter() - step_started
            if isinstance(info, dict):
                tick_count = info.get("physics_ticks", 0)
                physics_ticks += int(tick_count) if np.isscalar(tick_count) else int(np.asarray(tick_count).sum())
            reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device).reshape(env_count)
            done = torch.as_tensor(np.logical_or(terminated, truncated), dtype=torch.float32,
                                   device=self.device).reshape(env_count)
            next_obs = _as_tensor_obs(next_obs, self.device)
            terminated_tensor = torch.as_tensor(terminated, dtype=torch.float32,
                                                device=self.device).reshape(env_count)
            # Value the final state before a done environment is reset. This
            # bootstraps time-limit truncations, but never true terminals.
            bootstrap_obs = _restore_final_observations(next_obs, info, self.device)
            with torch.no_grad():
                if self.model.recurrent:
                    _, next_value = self.model.distributions(
                        bootstrap_obs, sequence_length=1,
                        initial_state=initial_state if recurrent_state is None else recurrent_state)
                else:
                    _, next_value = self.model.distributions(bootstrap_obs)
            records["obs"].append({k: v.detach() for k, v in obs.items()})
            records["actions"].append({k: v.detach() for k, v in latent_action.items()})
            records["log_prob"].append(log_prob.detach())
            records["value"].append(value.detach())
            records["reward"].append(reward)
            records["done"].append(done)
            records["terminated"].append(terminated_tensor)
            records["next_value"].append(next_value.detach())
            terms = info.get("reward_terms", {}) if isinstance(info, dict) else {}
            if isinstance(terms, dict):
                records["reward_terms"].append({
                    name: torch.as_tensor(value, dtype=torch.float32, device=self.device)
                    for name, value in terms.items()
                })
            if self.trajectory_path is not None:
                row = {"schema": "casu.trajectory-step.v1",
                       "global_step": self.total_steps,
                       "actions": {key: value.tolist() for key, value in env_action.items()},
                       "reward": reward.detach().cpu().tolist(),
                       "terminated": np.asarray(terminated).tolist(),
                       "truncated": np.asarray(truncated).tolist(),
                       "info": info}
                with self.trajectory_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, sort_keys=True, default=_json_default,
                                            allow_nan=False) + "\n")
            episode_returns += reward
            episode_lengths += 1
            if done.any():
                completed_returns.extend(episode_returns[done.bool()].cpu().tolist())
                completed_lengths.extend(episode_lengths[done.bool()].cpu().tolist())
                reasons = info.get("termination_reason") if isinstance(info, dict) else None
                if reasons is not None:
                    if isinstance(reasons, (list, tuple, np.ndarray)):
                        # one reason per world; only worlds that finished this step count
                        for reason, finished in zip(reasons, done.bool().tolist()):
                            if finished:
                                termination_counts[str(reason)] = termination_counts.get(str(reason), 0) + 1
                    else:
                        termination_counts[str(reasons)] = termination_counts.get(str(reasons), 0) + int(done.sum())
                episode_returns[done.bool()] = 0
                episode_lengths[done.bool()] = 0
                if self.model.recurrent and done.any():
                    recurrent_state[:, done.bool(), :] = 0
                done_numpy = done.bool().detach().cpu().numpy()
                if hasattr(self.env, "reset_done"):
                    reset_obs = self.env.reset_done(done_numpy)
                    if reset_obs is not None:
                        next_obs = _as_tensor_obs(reset_obs, self.device)
                elif bool(done.all()):
                    reset_obs, _ = self.env.reset()
                    next_obs = _as_tensor_obs(reset_obs, self.device)
                else:
                    raise RuntimeError(
                        "vector environment terminated only some worlds and has no reset_done() method")
            obs = next_obs
            self.total_steps += env_count
        self._observation = obs
        self._recurrent_state = recurrent_state
        with torch.no_grad():
            if self.model.recurrent:
                _, last_value = self.model.distributions(
                    obs, sequence_length=1, initial_state=recurrent_state)
            else:
                _, last_value = self.model.distributions(obs)
        action_frequencies = {}
        total_action_rows = max(1, steps * env_count)
        for name, counts in action_counts.items():
            for choice, count in enumerate(counts.tolist()):
                action_frequencies[f"action/{name}/choice_{choice}_fraction"] = count / total_action_rows
        return (records, last_value, completed_returns, completed_lengths,
                termination_counts, action_frequencies, time.perf_counter() - collection_started,
                physics_step_seconds, physics_ticks)

    def _update(self, records, last_value):
        """PPO update. ``CASU_PPO_EXECUTION=reference`` selects the original literal implementation;
        the default executes the same mathematics with fewer, larger tensor operations (feed-forward
        models only; recurrent models always use the reference path). Equivalence is tested bit for bit
        by ``casu_grid/perf/ppo_update_bench.py`` and ``tests/test_ppo_execution.py``."""
        if os.environ.get("CASU_PPO_EXECUTION", "optimized") == "reference" or self.model.recurrent:
            if records.get("obs") is None and records.get("obs_compact") is not None:      # reference path wants dense observations
                dense = self._expand_compact_batch({k: v.flatten(0, 1) for k, v in records["obs_compact"].items()})
                steps = records["obs_compact"]["goal"].shape[0]
                records = dict(records, obs=[{k: v.reshape(steps, -1, *v.shape[1:])[i] for k, v in dense.items()} for i in range(steps)])
            return self._update_reference(records, last_value)
        return self._update_optimized(records, last_value)

    def _update_optimized(self, records, last_value):
        cfg = self.config
        rewards = torch.stack(records["reward"])
        dones = torch.stack(records["done"])
        terminated = torch.stack(records["terminated"])
        next_values = torch.stack(records["next_value"])
        values = torch.stack(records["value"])
        log_probs = torch.stack(records["log_prob"])
        advantages, returns = _generalized_advantage_estimate(
            rewards, values, dones, terminated, next_values, cfg.gamma, cfg.gae_lambda)
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        expand = None
        if records.get("obs_compact") is not None:      # worker-local rollouts: id windows, expanded per minibatch (exact)
            obs = {k: v.flatten(0, 1) for k, v in records["obs_compact"].items()}
            expand = self._expand_compact_batch
        else:
            obs = {k: torch.stack([row[k] for row in records["obs"]]).flatten(0, 1) for k in records["obs"][0]}
        actions = {k: torch.stack([row[k] for row in records["actions"]]).flatten(0, 1) for k in records["actions"][0]}
        old_log_prob, advantage_flat, return_flat = log_probs.flatten(), advantages.flatten(), returns.flatten()
        count = len(old_log_prob)
        minibatch = min(cfg.minibatch_size, count)
        batches, completed_epochs = 0, 0
        stats = []          # one (5,) float32 row per minibatch; converted once, summed in the reference order
        clip_fractions = []  # observability only: share of samples whose probability ratio left the clip range
        graphed = self._graph_enabled()
        if graphed:
            from .ppo_graph import GraphedMinibatch
            if self._mb_graph is None or self._mb_graph.minibatch != minibatch:
                self._mb_graph = GraphedMinibatch(self, minibatch)
            self._mb_graph.prepare(obs, actions, old_log_prob, advantage_flat, return_flat, expand)
        for _ in range(cfg.epochs):
            permutation = torch.randperm(count, device=self.device)
            if graphed:
                stop_epoch = False
                for start in range(0, count, minibatch):
                    row = self._mb_graph.step(permutation[start:start + minibatch])
                    policy_v, value_v, entropy_v, kl_v, grad_v, loss_v, clip_v = row.tolist()
                    if not (np.isfinite(loss_v) and np.isfinite(grad_v)):
                        raise FloatingPointError("non-finite PPO loss or gradient norm")
                    stats.append(row[:5])
                    clip_fractions.append(clip_v)
                    batches += 1
                    if cfg.target_kl is not None and kl_v > cfg.target_kl:
                        stop_epoch = True
                        break
                completed_epochs += 1
                if stop_epoch:
                    break
                continue
            # one gather per epoch; minibatches are contiguous slices of the permuted copies (same values
            # and memory layout as the reference's per-minibatch ``v[idx]`` gathers)
            p_obs = {k: v[permutation] for k, v in obs.items()}
            p_actions = {k: v[permutation] for k, v in actions.items()}
            p_old, p_adv, p_ret = old_log_prob[permutation], advantage_flat[permutation], return_flat[permutation]
            stop_epoch = False
            for start in range(0, count, minibatch):
                part = slice(start, start + minibatch)
                batch_obs = {k: v[part] for k, v in p_obs.items()}
                if expand is not None:
                    batch_obs = expand(batch_obs)
                new_log_prob, entropy, new_value = self.model.evaluate_actions(
                    batch_obs, {k: v[part] for k, v in p_actions.items()})
                old_log, adv, ret = p_old[part], p_adv[part], p_ret[part]
                ratio = (new_log_prob - old_log).exp()
                policy = -torch.minimum(
                    ratio * adv,
                    ratio.clamp(1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * adv,
                ).mean()
                value_loss = 0.5 * (new_value - ret).square().mean()
                entropy_mean = entropy.mean()
                loss = policy + cfg.value_coefficient * value_loss - cfg.entropy_coefficient * entropy_mean
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
                    raise FloatingPointError("non-finite PPO loss or gradient norm")
                self.optimizer.step()
                approx_kl = ((ratio - 1.0) - (new_log_prob - old_log)).mean()
                stats.append(torch.stack((policy, value_loss, entropy_mean, approx_kl, grad_norm)).detach())
                clip_fractions.append(float(((ratio - 1.0).abs() > cfg.clip_range).float().mean()))
                batches += 1
                if cfg.target_kl is not None and float(approx_kl.detach()) > cfg.target_kl:
                    stop_epoch = True
                    break
            completed_epochs += 1
            if stop_epoch:
                break
        totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "grad_norm": 0.0}
        for row in torch.stack(stats).tolist():
            for key, value in zip(totals, row):
                totals[key] += value
        self.update_count += 1
        # observability only: how well the critic explained the returns of this rollout (1 = perfectly, <= 0 = no better than the mean)
        return_var = float(return_flat.var(unbiased=False))
        explained = (1.0 - float((return_flat - values.flatten()).var(unbiased=False)) / return_var) if return_var > 1e-12 else None
        return {key: value / max(1, batches) for key, value in totals.items()} | {
            "clip_fraction": (sum(clip_fractions) / len(clip_fractions)) if clip_fractions else None,
            "explained_variance": explained, "learning_rate": float(cfg.learning_rate),
            "optimizer_batches": batches, "epochs_completed": completed_epochs,
            "parameters": count_parameters(self.model),
        }

    def _update_reference(self, records, last_value):
        cfg = self.config
        rewards = torch.stack(records["reward"])
        dones = torch.stack(records["done"])
        terminated = torch.stack(records["terminated"])
        next_values = torch.stack(records["next_value"])
        values = torch.stack(records["value"])
        log_probs = torch.stack(records["log_prob"])
        advantages, returns = _generalized_advantage_estimate(
            rewards, values, dones, terminated, next_values, cfg.gamma, cfg.gae_lambda)
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        obs = {k: torch.stack([row[k] for row in records["obs"]]).flatten(0, 1)
               for k in records["obs"][0]}
        actions = {k: torch.stack([row[k] for row in records["actions"]]).flatten(0, 1)
                   for k in records["actions"][0]}
        old_log_prob = log_probs.flatten()
        advantage_flat, return_flat = advantages.flatten(), returns.flatten()
        old_value = values.flatten()
        count = len(old_log_prob)
        env_count = rewards.shape[1]
        sequence_length = len(records["obs"]) if self.model.recurrent else None
        recurrent_initial_state = None
        if self.model.recurrent:
            recurrent_initial_state = records["initial_state"][0].detach()
            # Preserve per-env rollout sequences when shuffling minibatches.
            obs = {k: torch.stack([row[k] for row in records["obs"]]).transpose(0, 1)
                   for k in records["obs"][0]}
            actions = {k: torch.stack([row[k] for row in records["actions"]]).transpose(0, 1)
                       for k in records["actions"][0]}
            old_log_prob = log_probs.transpose(0, 1)
            advantage_flat, return_flat = advantages.transpose(0, 1), returns.transpose(0, 1)
            old_value = values.transpose(0, 1)
            count = env_count
        minibatch = min(cfg.minibatch_size, count)
        totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                  "approx_kl": 0.0, "grad_norm": 0.0}
        batches, completed_epochs = 0, 0
        clip_total = 0.0       # observability only
        for _ in range(cfg.epochs):
            permutation = torch.randperm(count, device=self.device)
            stop_epoch = False
            for start in range(0, count, minibatch):
                idx = permutation[start:start + minibatch]
                if self.model.recurrent:
                    batch_obs = {k: v[idx].reshape(-1, *v.shape[2:]) for k, v in obs.items()}
                    batch_actions = {k: v[idx].reshape(-1, *v.shape[2:])
                                     for k, v in actions.items()}
                    batch_initial = recurrent_initial_state[:, idx]
                    new_log_prob, entropy, new_value = self.model.evaluate_actions(
                        batch_obs, batch_actions, sequence_length=sequence_length,
                        initial_state=batch_initial)
                    old_log = old_log_prob[idx].reshape(-1)
                    adv = advantage_flat[idx].reshape(-1)
                    ret = return_flat[idx].reshape(-1)
                else:
                    batch_obs = {k: v[idx] for k, v in obs.items()}
                    batch_actions = {k: v[idx] for k, v in actions.items()}
                    new_log_prob, entropy, new_value = self.model.evaluate_actions(batch_obs, batch_actions)
                    old_log, adv, ret = old_log_prob[idx], advantage_flat[idx], return_flat[idx]
                ratio = (new_log_prob - old_log).exp()
                policy = -torch.minimum(
                    ratio * adv,
                    ratio.clamp(1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * adv,
                ).mean()
                value_loss = 0.5 * (new_value - ret).square().mean()
                entropy_mean = entropy.mean()
                loss = policy + cfg.value_coefficient * value_loss - cfg.entropy_coefficient * entropy_mean
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
                    raise FloatingPointError("non-finite PPO loss or gradient norm")
                self.optimizer.step()
                approx_kl = ((ratio - 1.0) - (new_log_prob - old_log_prob[idx])).mean()
                clip_total += float(((ratio - 1.0).abs() > cfg.clip_range).float().mean().detach().cpu())
                for key, value in (("policy_loss", policy), ("value_loss", value_loss),
                                   ("entropy", entropy_mean), ("approx_kl", approx_kl),
                                   ("grad_norm", grad_norm)):
                    totals[key] += float(value.detach().cpu())
                batches += 1
                if cfg.target_kl is not None and float(approx_kl.detach()) > cfg.target_kl:
                    stop_epoch = True
                    break
            completed_epochs += 1
            if stop_epoch:
                break
        self.update_count += 1
        flat_returns = returns.flatten()
        return_var = float(flat_returns.var(unbiased=False))
        explained = (1.0 - float((flat_returns - values.flatten()).var(unbiased=False)) / return_var) if return_var > 1e-12 else None
        return {key: value / max(1, batches) for key, value in totals.items()} | {
            "clip_fraction": (clip_total / batches) if batches else None,
            "explained_variance": explained, "learning_rate": float(cfg.learning_rate),
            "optimizer_batches": batches, "epochs_completed": completed_epochs,
            "parameters": count_parameters(self.model),
        }

    def train_update(self, rollout_steps: int | None = None):
        started = time.perf_counter()
        (records, last_value, episode_returns, episode_lengths, termination_counts,
         action_frequencies, collection_seconds, physics_step_seconds, physics_ticks) = \
            self._collect(rollout_steps or self.config.rollout_steps)
        update_started = time.perf_counter()
        metrics = self._update(records, last_value)
        optimization_seconds = time.perf_counter() - update_started
        metrics.update({
            "update": self.update_count,
            "environment_steps": self.total_steps,
            "completed_episodes": len(episode_returns),
            "mean_completed_return": float(np.mean(episode_returns)) if episode_returns else None,
            "mean_completed_episode_length": float(np.mean(episode_lengths)) if episode_lengths else None,
            "rollout_seconds": time.perf_counter() - started,
            "collection_seconds": collection_seconds,
            "optimization_seconds": optimization_seconds,
            "physics_step_seconds": physics_step_seconds,
            "physics_ticks": physics_ticks,
            "physics_ticks_per_second": physics_ticks / max(physics_step_seconds, 1e-9),
            "environment_count": int(next(iter(self._observation.values())).shape[0]),
            "batch_calls_per_second": (rollout_steps or self.config.rollout_steps) /
                                      max(collection_seconds, 1e-9),
            "policy_decisions_per_second": ((rollout_steps or self.config.rollout_steps) *
                                              int(next(iter(self._observation.values())).shape[0])) /
                                             max(collection_seconds, 1e-9),
            "policy_decisions_per_env_second": (rollout_steps or self.config.rollout_steps) /
                                                max(collection_seconds, 1e-9),
        })
        metrics.update(action_frequencies)
        worker_seconds = getattr(self, "_last_local_seconds", None) if self._use_local_rollout() else None
        if worker_seconds:               # per-worker diagnostics of the local rollout (straggler = max / mean)
            for kind, values in worker_seconds.items():
                metrics[f"worker_{kind}_seconds_mean"] = float(np.mean(values))
                metrics[f"worker_{kind}_seconds_max"] = float(np.max(values))
        for reason, count in termination_counts.items():
            metrics[f"termination/{reason}"] = count
        nav = getattr(self, "_last_nav", None) if self._use_local_rollout() else None
        if nav and nav[0] is not None:
            decisions = (rollout_steps or self.config.rollout_steps) * metrics["environment_count"]
            metrics.update(navigation_metrics(nav[0], nav[1], nav[2], decisions))
        if records["reward_terms"]:
            names = sorted({name for step in records["reward_terms"] for name in step})
            for name in names:
                samples = [values[name].reshape(-1) for values in records["reward_terms"]
                           if name in values]
                if samples:
                    metrics[f"reward_term/{name}"] = float(torch.cat(samples).mean().cpu())
        if self.writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and value is not None:
                    self.writer.add_scalar(key, value, self.update_count)
            self.writer.flush()
        return metrics

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "schema": "casu.ppo-checkpoint.v1",
            "contract_fingerprint": self.contract_fingerprint,
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count, "total_steps": self.total_steps,
            "rng_state": self.rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "observation": self._observation,
            "recurrent_state": self._recurrent_state,
            "episode_returns": self._episode_returns,
            "episode_lengths": self._episode_lengths,
            "environment_state": self.env.snapshot() if hasattr(self.env, "snapshot") else None,
            "metadata": metadata or {},
        }
        # A process interruption during serialization must leave the previous
        # checkpoint intact. The temporary is on the same filesystem so replace
        # is atomic on supported local filesystems.
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            torch.save(checkpoint, temporary)
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        self._write_checkpoint_sidecar(path, checkpoint)
        return path

    @staticmethod
    def _write_checkpoint_sidecar(path: Path, checkpoint: dict) -> None:
        """``<name>.pt.json``: what the Studio needs to judge a checkpoint without importing torch (best effort; never fails a save)."""
        try:
            environment = checkpoint.get("environment_state")
            sizes = environment.get("sizes") if isinstance(environment, dict) else None
            info = {"schema": "casu.checkpoint-info.v1", "checkpoint_schema": checkpoint["schema"],
                    "contract_fingerprint": checkpoint["contract_fingerprint"], "update_count": int(checkpoint["update_count"]),
                    "total_steps": int(checkpoint["total_steps"]), "world_partition": list(sizes) if sizes is not None else None,
                    "terrain_mode": environment.get("terrain_mode") if isinstance(environment, dict) else None,
                    "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "size_bytes": path.stat().st_size,
                    "metadata": checkpoint.get("metadata", {})}
            sidecar = path.with_name(path.name + ".json")
            temporary = sidecar.with_name(sidecar.name + ".tmp")
            temporary.write_text(json.dumps(info, indent=1) + "\n")
            os.replace(temporary, sidecar)
        except Exception:
            pass

    def load_policy(self, path: str | Path) -> dict[str, Any]:
        """Load only the network weights of a checkpoint (evaluation): same schema and learning-contract checks as :meth:`load`,
        but no optimizer, RNG or environment state, so the checkpoint may come from a run with a different world count."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("schema") != "casu.ppo-checkpoint.v1":
            raise ValueError("unsupported PPO checkpoint schema")
        if checkpoint.get("contract_fingerprint") != self.contract_fingerprint:
            raise ValueError("PPO checkpoint learning contract is incompatible with this run")
        self.model.load_state_dict(checkpoint["model"])
        return {"update_count": int(checkpoint["update_count"]), "total_steps": int(checkpoint["total_steps"]),
                "metadata": checkpoint.get("metadata", {})}

    def load(self, path: str | Path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("schema") != "casu.ppo-checkpoint.v1":
            raise ValueError("unsupported PPO checkpoint schema")
        if checkpoint.get("contract_fingerprint") != self.contract_fingerprint:
            raise ValueError("PPO checkpoint learning contract is incompatible with this run")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.adopt_optimizer_state()
        self._mb_graph = None             # captured graphs reference the old optimizer-state tensors
        self.update_count = int(checkpoint["update_count"])
        self.total_steps = int(checkpoint["total_steps"])
        self.rng.bit_generator.state = checkpoint["rng_state"]
        # ``map_location`` moved the generator states to the training device; the generators want CPU ByteTensors
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if checkpoint.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state"]])
        self._observation = checkpoint.get("observation")
        self._recurrent_state = checkpoint.get("recurrent_state")
        self._episode_returns = checkpoint.get("episode_returns")
        self._episode_lengths = checkpoint.get("episode_lengths")
        if checkpoint.get("environment_state") is not None and hasattr(self.env, "restore"):
            self.env.restore(checkpoint["environment_state"])
        return checkpoint.get("metadata", {})


def navigation_metrics(episodes: dict, active: dict | None, blocks_destroyed: float | None, decisions: int) -> dict[str, Any]:
    """XY1 task statistics of one rollout window (``nav/*``), for the Studio and for the learning-readiness analysis.

    Pure observability: computed from episode summaries the worker-local rollout already collected; nothing feeds back into learning.
    ``None`` means *no sample* (e.g. no episode finished in this window), never zero. Distances are in tiles. Episodes still running at
    the end of the window are reported separately (``nav/active_*``) because completed episodes alone are a biased, censored sample.
    """
    reasons = episodes["reason"]
    completed = len(reasons)
    arrivals = [i for i, r in enumerate(reasons) if r == "arrival"]
    mean = lambda values: (float(sum(values) / len(values)) if len(values) else None)
    out: dict[str, Any] = {
        "nav/completed_episodes": completed,
        "nav/arrivals": len(arrivals),
        "nav/timeouts": sum(1 for r in reasons if r == "time_limit"),
        "nav/arrival_rate": (len(arrivals) / completed) if completed else None,
        "nav/arrivals_per_1k_decisions": 1000.0 * len(arrivals) / max(1, decisions),
        "nav/decisions_to_arrival_mean": mean([episodes["length"][i] for i in arrivals]),
        "nav/decisions_to_arrival_median": (float(sorted(episodes["length"][i] for i in arrivals)[len(arrivals) // 2]) if arrivals else None),
        "nav/completed_initial_distance_mean": mean(episodes["initial_distance"]),
        "nav/completed_best_distance_mean": mean(episodes["best_distance"]),
        "nav/completed_final_distance_mean": mean(episodes["final_distance"]),
        "nav/completed_progress_mean": mean([1.0 - b / i for b, i in zip(episodes["best_distance"], episodes["initial_distance"]) if i > 0]),
    }
    if active is not None:
        count = len(active["goal_distance"])
        outside = active["outside_world"] > 0.5
        inside = ~outside
        progress = [1.0 - float(b) / float(i) for b, i in zip(active["best_distance"], active["initial_distance"]) if i > 0]
        out.update({
            "nav/active_episodes": count,
            # agents that left the 128 x 128 world (the floor can be dug through and nothing ends such an episode) are reported on their own: a
            # free-falling body would otherwise dominate every distance average
            "nav/active_outside_world_fraction": float(outside.mean()) if count else None,
            "nav/active_in_world_episodes": int(inside.sum()),
            "nav/active_progress_mean": mean(progress),
            "nav/active_distance_mean": mean(active["goal_distance"][inside].tolist()),
            "nav/active_best_distance_mean": mean(active["best_distance"].tolist()),
            "nav/active_initial_distance_mean": mean(active["initial_distance"].tolist()),
            "nav/active_decisions_mean": mean(active["decisions"].tolist()),
        })
    if blocks_destroyed is not None:
        out["nav/blocks_destroyed"] = float(blocks_destroyed)
        out["nav/blocks_destroyed_per_1k_decisions"] = 1000.0 * float(blocks_destroyed) / max(1, decisions)
    return out


def write_metrics(path: str | Path, metrics: dict[str, Any]):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(metrics, sort_keys=True, allow_nan=False) + "\n")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"cannot JSON encode {type(value).__name__}")
