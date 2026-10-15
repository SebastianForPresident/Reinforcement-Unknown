# XY1 v2 learning validation: three independent seeds

Branch `xy1-layer-exit`, experiment `xy1_point_navigation_mini_layer_exit.py`, termination contract **`xy1-termination-v2-layer-exit`** (bottom-of-layer exit is a true terminal event,
no extra reward or penalty; arrival bonus, progress reward, 100,000-decision truncation, network and PPO unchanged). Raw data: `casu_grid/results/xy1_v2_validation/seed{7,17,27}/`
(training metrics, run manifests, system samples, initial and final checkpoints, full per-episode evaluation JSON), `analysis.json`, `campaign.log`. Everything below was produced by
`casu_grid/perf/xy1_v2_campaign.sh` and `casu_grid/perf/xy1_v2_analyze.py`; nothing was tuned between seeds and the decision rule below was committed before any seed finished.

## Verdict: **ENCOURAGING**

All three seeds, each from an untrained network, improved held-out navigation by a large and statistically clear margin in the stochastic mode (+25.8, +37.5, +33.6 points; every cluster-bootstrap
95 % interval excludes zero) and by a smaller but also clear margin in the deterministic mode (+9.4, +14.8, +7.0 points; all three intervals exclude zero). No numerical instability, no failed
worker, no environment collapse. The qualifications that matter are in §6 (one held-out scenario set, one horizon, 250 updates, and layer exits are still ~38–40 % of finished training episodes).

Pre-registered rule (`xy1_v2_analyze.py`): *Encouraging* = stochastic paired improvement > 0 in all seeds, with the interval excluding 0 in ≥ 2 of 3; deterministic ≥ 0 in ≥ 2; all losses finite and no
failed run; layer exits not dominating (> 50 %) finished episodes. *Concerning* = stochastic improvement ≤ 0 in ≥ 2 seeds, or instability, or layer-exit dominance. Otherwise *inconclusive*.
Result: stochastic positive 3/3, intervals excluding 0 in 3/3, deterministic ≥ 0 in 3/3, instability none, layer-exit share 0.38–0.40 (< 0.5) ⇒ ENCOURAGING.

## 1. Experiment verification (run manifests)

| | seed 7 | seed 17 | seed 27 |
|---|---|---|---|
| experiment id | `xy1-point-navigation-mini-128x128-layer-exit-v2` | same | same |
| `termination_contract` / `bottom_exit_terminates` | `xy1-termination-v2-layer-exit` / true | same | same |
| worlds · workers · PhysX threads · PPO device | 64 · 8 · 2 · CUDA | same | same |
| episode ceiling · ticks per decision · terrain | 100,000 · 10 · independent | same | same |
| `CASU_LEGACY_MINI_SPAWN` | off (`legacy_spawn: false`) | off | off |
| updates · decisions | 250 · 1,024,000 | 250 · 1,024,000 | 250 · 1,024,000 |
| launch commit | `d493b75` (+1 modified harness file) | `c0b3951` | `24d76f1` (+ result files of earlier seeds) |

Trainer, environment and PPO code are identical across the three launches (`git diff d493b75 HEAD` over them is empty; later commits only add the runner, analysis script and results). Seeds started from
untrained networks (`checkpoints/initial.pt` preserved); no v1 or guarded-variant checkpoint was resumed. All 250 updates completed for every seed, `COMPLETE` marker present, no `failure.json`, no worker loss
(8 workers alive in every system sample). Throughput through the Studio launch path: **1,594 decisions/s end-to-end** in each seed (reference ≈ 1,700), peak unique memory 16.3 GiB (PSS 17.1 GiB), minimum available
memory 8.0–8.3 GiB, GPU 3.6 GiB, about 10.7 min of rollout time per seed (≈ 11.5 min wall including start-up).

## 2. Held-out evaluation design

`workbench/evaluate.py` (contract-aware: it ran under `xy1-termination-v2-layer-exit`, recorded in every result), **16 held-out terrains** (seed offset 1,000,000; training used offsets 0–63), **8 goal draws per terrain =
128 cases per policy and mode**, horizon 1,000 decisions, success = arrival within one tile within the horizon; a layer exit or running out of the horizon is a failure. Initial (`initial.pt`) and final
(`update_000250.pt`, 1,024,000 decisions) policies play **identical cases** (same terrain, same goal draws, per mode). The same 128 cases are used for all three seeds. Stochastic and deterministic modes are reported separately.
Because the 8 goal draws of a terrain share that terrain, episodes are not independent: the interval is a **cluster bootstrap over the 16 terrains** (10,000 resamples) on the paired difference; the exact McNemar p on discordant
cases is shown only as a secondary, optimistic figure.

## 3. Held-out results per seed (paired)

Stochastic (primary: the behaviour PPO trains):

| seed | initial → final success | paired improvement (cluster 95 %) | discordant: only final / only initial | McNemar p | outcomes initial → final (arrival / layer exit / horizon) |
|---|---|---|---|---|---|
| 7 | 29/128 (22.7 %) → 62/128 (48.4 %) | **+25.8 pts (+16.4, +36.7)** | 37 / 4 | < 0.001 | 29/3/96 → 62/2/64 |
| 17 | 17/128 (13.3 %) → 65/128 (50.8 %) | **+37.5 pts (+28.1, +46.1)** | 51 / 3 | < 0.001 | 17/2/109 → 65/0/63 |
| 27 | 19/128 (14.8 %) → 62/128 (48.4 %) | **+33.6 pts (+26.6, +41.4)** | 43 / 0 | < 0.001 | 19/2/107 → 62/2/64 |

Deterministic (most likely action):

| seed | initial → final success | paired improvement (cluster 95 %) | only final / only initial | McNemar p | outcomes initial → final |
|---|---|---|---|---|---|
| 7 | 2/128 (1.6 %) → 14/128 (10.9 %) | **+9.4 pts (+3.9, +15.6)** | 13 / 1 | 0.002 | 2/0/126 → 14/0/114 |
| 17 | 0/128 (0 %) → 19/128 (14.8 %) | **+14.8 pts (+10.2, +18.8)** | 19 / 0 | < 0.001 | 0/0/128 → 19/0/109 |
| 27 | 2/128 (1.6 %) → 11/128 (8.6 %) | **+7.0 pts (+3.1, +10.9)** | 9 / 0 | 0.004 | 2/0/126 → 11/0/117 |

Across seeds (descriptive; three runs, so no pooled significance claim): mean paired improvement **+32.3 points stochastic (range 25.8–37.5)**, **+10.4 points deterministic (range 7.0–14.8)**; the sign is positive in 3 of 3 seeds and every
one of the six intervals excludes zero.

Other held-out measures (initial → final):

| | stochastic: progress toward goal · mean best distance (tiles) · median decisions to arrival · blocks destroyed per episode | deterministic: same |
|---|---|---|
| seed 7 | 54 % → 72 % · 11.9 → 7.1 · 109 → 100 · 33.1 → 29.7 | 15 % → 34 % · 22.0 → 17.1 · 9 → 12 · 0.2 → 3.7 |
| seed 17 | 52 % → 71 % · 12.1 → 7.5 · 174 → 98 · 33.5 → 24.3 | 9 % → 35 % · 23.4 → 17.2 · n/a (no initial arrival) → 11 · 0.5 → 0.5 |
| seed 27 | 43 % → 73 % · 15.1 → 6.6 · 74 → 126 · 36.8 → 24.4 | 5 % → 28 % · 24.7 → 18.8 · 1 → 3 · 0.8 → 0.4 |

Decisions-to-arrival medians are over arrivals only and are therefore not comparable between policies with very different numbers of arrivals (deterministic initial policies arrive in 0–2 cases, mostly trivially near goals). The deterministic
policies rarely dig (0.4–3.7 blocks per episode) while stochastic policies dig 24–30 blocks per episode (initially 33–37): training shifted the stochastic policy towards less excavation per episode, not towards none.

## 4. Training-side measurements (not evidence of generalisation)

| | seed 7 | seed 17 | seed 27 |
|---|---|---|---|
| decisions · end-to-end speed | 1,024,000 · 1,594 /s | 1,024,000 · 1,594 /s | 1,024,000 · 1,594 /s |
| finished episodes: arrivals · layer exits · timeouts | 357 · 215 · 0 | 359 · 237 · 0 | 394 · 242 · 0 |
| layer exits as share of finished episodes | 37.6 % | 39.8 % | 38.1 % |
| goals per 1,000 decisions, blocks of 50 updates | 0.20 · 0.29 · 0.36 · 0.42 · 0.47 | 0.60 · 0.26 · 0.20 · 0.21 · 0.48 | 0.32 · 0.47 · 0.27 · 0.35 · 0.51 |
| layer exits per 1,000 decisions, same blocks | 0.12 · 0.23 · 0.21 · 0.26 · 0.23 | 0.11 · 0.23 · 0.30 · 0.26 · 0.25 | 0.16 · 0.29 · 0.23 · 0.27 · 0.24 |
| progress toward goal of running episodes | 47 → 50 % (flat) | 58 → 46 % | 51 → 55 % (flat) |
| entropy first / last 10 updates | 8.784 → 8.730 | 8.751 → 8.729 | 8.790 → 8.664 |
| approx. KL mean / max (target 0.03) | 0.0059 / 0.0093 | 0.0058 / 0.0093 | 0.0060 / 0.0102 |
| updates with KL above target · clip fraction mean | 0 · 0.071 | 0 · 0.068 | 0 · 0.072 |
| max gradient norm (clip 0.5) · all losses finite | 1.30 · yes | 1.49 · yes | 1.36 · yes |
| explained variance first 50 → last 50 updates | 0.44 → 0.56 | 0.27 → 0.54 | 0.31 → 0.46 |
| value loss first 10 → last 10 | 0.313 → 0.217 | 0.626 → 0.224 | 0.388 → 0.317 |
| attack frequency first 10 → last 10 updates | 49.2 → 50.8 % | 46.0 → 51.7 % | 47.2 → 48.3 % |
| blocks destroyed per 1,000 decisions, first / last 50 | 41.5 → 41.1 | 41.2 → 38.4 | 46.4 → 39.8 |
| agents outside the world (max) | 0 | 0 | 0 |

Stability warnings: none. KL stays an order of magnitude below the target, clip fraction ≈ 7 %, gradient norms are above the 0.5 clip limit at times (as in every earlier run; max 1.3–1.5), explained variance rises, nothing non-finite.
Reward-trend vs actual navigation: the training-time goal rate is a sparse, censored statistic (about 1–2 arrivals per update); it rises clearly only in seed 7 (0.20 → 0.47), modestly in seed 27 and non-monotonically in seed 17 (0.60 in the first
block, 0.21 at the trough, 0.48 at the end), and training-time progress is flat. The held-out evaluation, which controls terrain, goals and horizon, shows the effect far more clearly than the training curves do. Neither a rising training reward nor
these curves were used as evidence; the verdict rests on the paired held-out comparison.

## 5. Context only: the earlier guarded-variant diagnostic

The single-seed Research Studio V2 diagnostic used the *superseded* guarded variant (`xy1-termination-v1x-out-of-world-truncation`: truncation, not a terminal, at |y| > half the world) and a different evaluation set. It is not the same termination contract
as v2 and is not evidence for v2. It is mentioned only because it motivated the contract decision; the v2 evidence is §3 alone.

## 6. Qualifications and open questions

* **One held-out scenario set** (16 terrains × 8 goal draws, horizon 1,000) shared by all seeds: the intervals describe sampling uncertainty over those terrains and goals, not over other evaluation protocols. Longer horizons, other goal distributions and more terrains are untested.
* **Early training only**: 250 updates (1.02 M decisions). Held-out success of the stochastic policy is about 48–51 % (about 9–15 % deterministic); nothing is known about the plateau or about the 49 M-decision campaign.
* **Layer exits remain frequent in training**: 38–40 % of finished training episodes end by leaving through the bottom (≈ 0.25 per 1,000 decisions after the first 50 updates, up from ≈ 0.12), and excavation stays near 40 blocks per 1,000 decisions. Training episodes are long (100,000-decision ceiling), so exits are a major way episodes end
  even though only 0–3 of 128 held-out episodes (horizon 1,000) ended in a layer exit. Whether policies learn to avoid digging through the floor, or whether the layer exit is itself a useful shortcut that competes with navigation, is an open question for a longer run.
* The deterministic policy improves less and from a near-zero base; its median decisions to arrival are computed from very few arrivals.
* Three seeds support "repeatable early learning", not a precise effect size; the cross-seed range (+25.8 to +37.5 stochastic) is the honest spread.
* Seed 7 was launched with one modified harness file in the working tree and seed 27 with result files of earlier seeds untracked/modified; no trainer, environment or PPO code differed (verified by diff).
* Multiple comparisons (6 cells) were not corrected; the pre-registered rule needed only the sign and interval pattern, which holds in all six.

## 7. Commands and resumption

All three seeds completed training and evaluation; nothing is pending. To rerun (markers prevent rerunning finished seeds): `CASU_SCRATCH=<dir with st/env_le.sh> casu_grid/perf/xy1_v2_campaign.sh 7 17 27`, then `python3 casu_grid/perf/xy1_v2_analyze.py`.
Run folders (full logs, all checkpoints, trajectories): `experiments/v2-seed{7,17,27}-*` (git-ignored); preserved copies of the training metrics, manifests and the initial/final checkpoints are under `casu_grid/results/xy1_v2_validation/seed*/`.
Validated revision of this campaign: the commit containing this report (`git log -1 -- casu_grid/perf/XY1_V2_LEARNING_VALIDATION.md`); scientific code identical to `d493b75`.
