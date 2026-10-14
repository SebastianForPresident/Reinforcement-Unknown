"""CUDA-graph execution of one PPO minibatch (gather -> forward -> loss -> backward -> clip -> Adam step).

The minibatch loop of the PPO update launches roughly 400 small CUDA kernels per minibatch from Python (about 4.6 ms of
launch-bound work for 2.5 ms of GPU time at XY1 sizes). Capturing the identical sequence once and replaying it removes
the launch overhead. The mathematics is unchanged -- the same tensor operations on the same data in the same order --
including the optimizer (``Adam(capturable=True)`` keeps its step counter on the device).

Requirements handled here:
  * capture needs static tensors: the rollout (compact observations, actions, old log-probs, advantages, returns) is
    copied into persistent buffers at each update; each replay is driven by a persistent index tensor;
  * warm-up iterations change weights and optimizer state, so both are saved before warm-up and restored after capture;
  * hyper-parameters baked into the graph (clip range, coefficients, learning rate) are constant; a changed learning rate
    or contract triggers a re-capture.
"""
from __future__ import annotations

import os

import torch
from torch.nn.utils import clip_grad_norm_

_CLIP_STAT = os.environ.get("CASU_PPO_CLIP_STAT", "1") != "0"   # clip-fraction statistic inside the captured graph (diagnostic switch)


class GraphedMinibatch:
    def __init__(self, trainer, minibatch: int):
        self.trainer = trainer
        self.minibatch = int(minibatch)
        self.graph = None
        self.key = None
        self.sources = None

    # ----------------------------------------------------------------- the step body (captured and eager)
    def _body(self, idx, record_stats):
        t, cfg = self.trainer, self.trainer.config
        src = self.sources
        batch_obs = {k: v[idx] for k, v in src["obs"].items()}
        if src["expand"] is not None:
            batch_obs = src["expand"](batch_obs)
        batch_actions = {k: v[idx] for k, v in src["actions"].items()}
        new_log_prob, entropy, new_value = t.model.evaluate_actions(batch_obs, batch_actions)
        old_log, adv, ret = src["old"][idx], src["adv"][idx], src["ret"][idx]
        ratio = (new_log_prob - old_log).exp()
        policy = -torch.minimum(ratio * adv, ratio.clamp(1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * adv).mean()
        value_loss = 0.5 * (new_value - ret).square().mean()
        entropy_mean = entropy.mean()
        loss = policy + cfg.value_coefficient * value_loss - cfg.entropy_coefficient * entropy_mean
        loss.backward()
        grad_norm = clip_grad_norm_(t.model.parameters(), cfg.max_grad_norm)
        t.optimizer.step()
        approx_kl = ((ratio - 1.0) - (new_log_prob - old_log)).mean()
        clip_fraction = (((ratio - 1.0).abs() > cfg.clip_range).float().mean() if _CLIP_STAT
                         else new_log_prob.new_zeros(()))     # observability only (not part of the loss)
        return torch.stack((policy, value_loss, entropy_mean, approx_kl, grad_norm, loss, clip_fraction)).detach()

    # ---------------------------------------------------------------------------- capture
    def prepare(self, obs, actions, old, adv, ret, expand):
        """Bind this update's rollout tensors; (re)capture when shapes, learning rate or contract changed."""
        t = self.trainer
        key = (tuple((k, tuple(v.shape), v.dtype) for k, v in obs.items()), tuple((k, tuple(v.shape)) for k, v in actions.items()),
               tuple(old.shape), expand is not None, t.optimizer.param_groups[0]["lr"], id(t.optimizer), id(t.model))
        if self.sources is None or key != self.key:
            self.graph = None
            self.sources = {"obs": {k: torch.empty_like(v) for k, v in obs.items()},
                            "actions": {k: torch.empty_like(v) for k, v in actions.items()},
                            "old": torch.empty_like(old), "adv": torch.empty_like(adv), "ret": torch.empty_like(ret),
                            "expand": expand}
            self.idx = torch.zeros(self.minibatch, dtype=torch.long, device=old.device)
            self.key = key
        s = self.sources
        for k, v in obs.items():
            s["obs"][k].copy_(v)
        for k, v in actions.items():
            s["actions"][k].copy_(v)
        s["old"].copy_(old); s["adv"].copy_(adv); s["ret"].copy_(ret)
        if self.graph is None:
            self._capture(old.shape[0])

    def _capture(self, count):
        t = self.trainer
        torch.backends.cudnn.benchmark = True       # convolution algorithm search runs in the (eager) warm-up below, ~13% faster steps
        params = list(t.model.parameters())
        saved_params = [p.detach().clone() for p in params]
        saved_state = {id(p): {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in t.optimizer.state.get(p, {}).items()}
                       for p in params}
        self.idx.copy_(torch.arange(self.minibatch, device=self.idx.device) % count)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):                      # warm-up: cuDNN/cuBLAS handles, lazily created optimizer state
                t.optimizer.zero_grad(set_to_none=True)
                self._body(self.idx, record_stats=False)
        torch.cuda.current_stream().wait_stream(side)
        t.optimizer.zero_grad(set_to_none=True)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.stats = self._body(self.idx, record_stats=True)
        # undo the warm-up's effect on the weights and the Adam state (in place: the graph keeps its addresses)
        with torch.no_grad():
            for p, saved in zip(params, saved_params):
                p.copy_(saved)
            for p in params:
                state = t.optimizer.state.get(p, {})
                for k, v in state.items():
                    if torch.is_tensor(v):
                        original = saved_state[id(p)].get(k)
                        if original is None:
                            v.zero_()
                        else:
                            v.copy_(original)
        torch.cuda.synchronize()

    # ------------------------------------------------------------------------------ run
    def step(self, index_slice):
        """One minibatch for the given 1-D index tensor; returns the (7,) statistics tensor on the device (policy, value, entropy, KL, grad norm, loss, clip fraction)."""
        if index_slice.shape[0] == self.minibatch and self.graph is not None:
            self.idx.copy_(index_slice)
            self.graph.replay()
            return self.stats.clone()
        return self._body_eager(index_slice)

    def _body_eager(self, index_slice):
        self.trainer.optimizer.zero_grad(set_to_none=True)
        return self._body(index_slice, record_stats=False)
