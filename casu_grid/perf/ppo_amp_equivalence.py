"""Differential test of the PPO minibatch step under different numerics (CASU_PPO_AMP unset / fp16 / bf16).

    CASU_PPO_AMP=fp16 python casu_grid/perf/ppo_amp_equivalence.py run OUT.npz      # one mode
    python casu_grid/perf/ppo_amp_equivalence.py compare REF.npz CAND.npz            # error distribution

Identical initial weights, optimizer state, rollout (real mini-world id windows, model-sampled actions, N(0,1) advantages) and
minibatch order; 128 graphed minibatches (2 epochs of a 2048-sample update). Per minibatch the six logged statistics are recorded;
at the end the parameters.
"""
import os, sys
import numpy as np
import torch

mode = sys.argv[1]
if mode == "compare":
    a, b = np.load(sys.argv[2]), np.load(sys.argv[3])
    sa, sb = a["stats"], b["stats"]
    names = ["policy_loss", "value_loss", "entropy", "approx_kl", "grad_norm", "loss"]
    print(f"{'statistic':12s} {'median rel':>11s} {'p95 rel':>11s} {'max rel':>11s}   (rel = |cand-ref| / max(|ref|, 1e-3 * scale))")
    for k, n in enumerate(names):
        scale = np.abs(sa[:, k]).mean()
        rel = np.abs(sb[:, k] - sa[:, k]) / np.maximum(np.abs(sa[:, k]), 1e-3 * max(scale, 1e-12))
        print(f"{n:12s} {np.median(rel):11.2e} {np.percentile(rel, 95):11.2e} {rel.max():11.2e}   mean ref {sa[:, k].mean():+.4e} cand {sb[:, k].mean():+.4e}")
    pa, pb = a["params"], b["params"]
    d = np.abs(pb - pa)
    print(f"parameters: max |d| {d.max():.2e}  rms {np.sqrt((d ** 2).mean()):.2e}  (lr 3e-4: one Adam step moves a weight by <= 3e-4);"
          f" rms |ref - init| {np.sqrt(((pa - a['init']) ** 2).mean()):.2e}; relative update error {np.linalg.norm(pb - pa) / max(np.linalg.norm(pa - a['init']), 1e-30):.3f}")
    sys.exit(0)

sys.path.insert(0, ".")
from casu_grid.workbench.ppo import PPOTrainer
from casu_grid.workbench.ppo_graph import GraphedMinibatch
from casu_grid.workbench.experiments import xy1_point_navigation_mini as cfg
from casu_grid.xy1 import expand_compact_observation
from casu_grid.xy1_mini_world import generate_mini_terrain
torch.backends.cudnn.benchmark = True
if os.environ.get('CASU_TF32_OFF') == '1':
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False       # true fp32 reference


class E:
    pass


torch.manual_seed(0)
tr = PPOTrainer(E(), cfg.ACTION, cfg.OBSERVATION, cfg.MODEL, cfg.PPO, device="cuda", seed=1)
rng = np.random.default_rng(0)
N = 2048
terrain = generate_mini_terrain(3)
bid, fid, cur, goal = [], [], [], []
for _ in range(N):
    px, py = int(rng.integers(33, 95)), int(rng.integers(33, 95))
    bid.append(terrain[px - 32:px + 32, py - 32:py + 32].astype(np.uint8)); fid.append(np.zeros((64, 64), np.uint8))
    cur.append((int(rng.integers(0, 64)), int(rng.integers(0, 64)))); goal.append(rng.integers(-40, 40, 2))
obs = {"block_ids": torch.as_tensor(np.stack(bid), device="cuda"), "fluid_ids": torch.as_tensor(np.stack(fid), device="cuda"),
       "cursor": torch.as_tensor(np.array(cur, np.int16), device="cuda"), "goal": torch.as_tensor(np.array(goal, np.float32), device="cuda")}
with torch.no_grad():
    dense = {tr.observations.spatial[0].name: expand_compact_observation(obs["block_ids"], obs["fluid_ids"], obs["cursor"], "cuda"),
             tr.observations.scalar[0].name: obs["goal"]}
    dists, _ = tr.model.distributions(dense)
    acts = {h.name: dists[h.name].sample() for h in cfg.ACTION.heads}
    old = sum(dists[h.name].log_prob(acts[h.name]) for h in cfg.ACTION.heads)
adv = torch.as_tensor(rng.standard_normal(N).astype(np.float32), device="cuda")
ret = torch.as_tensor(rng.standard_normal(N).astype(np.float32) * 0.5, device="cuda")
init = torch.cat([p.detach().flatten().cpu() for p in tr.model.parameters()]).numpy()
mb = GraphedMinibatch(tr, 32)
mb.prepare(obs, acts, old, adv, ret, tr._expand_compact_batch)
torch.manual_seed(1)
rows = []
for epoch in range(2):
    perm = torch.randperm(N, device="cuda")
    for s in range(0, N, 32):
        rows.append(mb.step(perm[s:s + 32]).cpu().numpy())
params = torch.cat([p.detach().flatten().cpu() for p in tr.model.parameters()]).numpy()
np.savez(sys.argv[2], stats=np.stack(rows), params=params, init=init)
print("saved", sys.argv[2], "AMP =", os.environ.get("CASU_PPO_AMP", "0"), flush=True)
