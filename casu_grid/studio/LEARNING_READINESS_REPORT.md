# XY1 learning-readiness report

Branch `studio-v2` (worktree `Reinforcement-Unknown-Isaac-studio`), starting revision Optimization IV `95640c9`; log `casu_grid/studio/STUDIO_LOG.md`.
Machine: i7-11700K, RTX 3070, Isaac Sim 4.5.0 / Isaac Lab 2.1.1. All numbers below come from real simulator runs started through the Studio.

## 1. Verdict

| question | answer |
|---|---|
| Is the XY1 environment *implemented correctly* (placement, goals, observation, arrival, reward, timeout, resets, isolation)? | **Yes.** 19 of 19 independent checks pass (`perf/xy1_readiness_audit.py`, §3). |
| Is the instrumentation sufficient to tell learning from finite losses? | **Yes** (§4), after adding read-only statistics; the Studio keeps training-time curves and held-out evaluation strictly apart. |
| Is **XY1 as defined** ready for the first long learning campaign? | **No — one decision is needed first.** Agents dig through the floor, fall out of the world and are never reset; in a 64-world run 10 % of the population is gone after 31 updates, half after 62, 90 % after 107 and all after about 200. The sample stream of a long campaign would be empty after the first ~10 minutes (§5). |
| What does the early learning signal look like? | With XY1 **as defined**: *concerning* (37 goals in 1,024,000 decisions, curves collapse with the population) and held-out *inconclusive* (+11.5 points, p = 0.06). With the **exploratory variant** that merely ends an episode when the agent leaves the world: *encouraging* — held-out success 16.7 % → 43.8 % (+27.1 points, 95 % interval +14.2 to +38.8, p < 0.001) after 1,024,000 decisions (§6). Single seed; confirm before relying on it. |
| Is everything else ready? | Yes: launch, monitoring, graceful stop, resume, held-out evaluation, run comparison all work against the real simulator (UX report, test evidence). |

**Decision needed from you before the long campaign** (nothing was changed in XY1 itself): how out-of-world agents are to be handled. The options and a recommendation are in §7.

## 2. Source revision and launch configuration

* Code: `studio-v2` on top of Optimization IV `95640c9`. The Studio's validation list checks, at every launch, that `a7ebcab` (spawn/goal placement), `e6e3838` (Layer-1 reset),
  `dafc767` (boundary dig) and `1f5d68b` (vectorised terrain) are ancestors of HEAD and that the checkout the Studio runs from is the one that will run (it records the commit and
  warns about uncommitted files; one tracked file was modified when diagnostic 1 started (from the edit history a Studio analysis module, not trainer or environment code; the launch record notes it) and the tree was clean for diagnostic 2).
* `CASU_LEGACY_MINI_SPAWN` is refused by the Studio and by `train.py` (the latter needs `CASU_REPRODUCE_LEGACY=1` to override); every run record states `legacy_spawn: false`.
* Configuration of both diagnostics (`results/studio/diagnostics/*/studio_launch.json`): 64 worlds × 8 workers × 2 PhysX threads, PPO on CUDA, seed 7, 250 updates = 1,024,000 decisions,
  10 physics ticks per decision, independent terrain, 100,000-decision episode ceiling, initial checkpoint + a numbered checkpoint every 25 updates.
  Measured speed through the Studio: **1,633 decisions/s end-to-end** (XY1 as defined) and **1,564** (variant) against the Optimization IV reference of about 1,700; peak unique memory
  16.2–16.3 GiB (PSS 17.0–17.1 GiB), minimum available memory 8.2 GiB, GPU 3.5 GiB, no failures.
* Environment check: the experiment contracts of the loaded file are compared with the contract the Studio describes (heads `[3,2,3,2,9,11,2]`, 10 ticks, 16×64×64 + 2 goal offsets,
  rollout 64, minibatch 32, 2 epochs); the mini-world file has no mismatches.

## 3. Scientific validation (independent recomputation, `results/studio/audit/audit_xy1.json`)

| check | result |
|---|---|
| every world has its own terrain; terrain seeds are 391728405 + world index | pass (4 distinct hashes) |
| every body starts at its own world's centre; stage positions 0.5 / 1040.5 / 2080.5 / 3120.5 (no stacked bodies) | pass (deviation 0.0) |
| every observation window contains its own terrain (1,024–1,239 non-empty cells) | pass |
| goals lie in the 64 × 64 viewport; relative-goal observation = goal tile − player tile (recomputed); goal world coordinates in the local frame | pass |
| goal sampling is uniform (20,000 draws: x, y ∈ [32, 95], mean 63.3 / 63.4) | pass |
| arrival: on the goal tile and diagonal neighbour arrive (+10 bonus), two tiles away does not | pass |
| reward: 0.1 × improvement of the all-time best distance only (five teleports checked against the formula, regress and repeat pay 0) | pass |
| timeout: truncation (not termination) exactly at the step limit | pass |
| terrain edits stay in their world (logical and collider terrain); `reset_done` re-draws the goal and restores terrain of the finished world only | pass |
| **leaving the world: the episode does not end; the body keeps falling (y −71 → −88 in 4 ticks)** | documented behaviour (§5) |

Other validation carried over and re-run on this branch: XY1 trajectory gate `284e8dde…` unchanged, 8 of 8 trajectory gates bit-identical (`perf/run_perf_check.py --quick`), 151 trainer-side
tests and the Kit test OK, Studio tests 80 OK; fresh-run PPO statistics with the new instrumentation differ from the baseline by 4.5e-5 relative, less than two identical Optimization IV
runs differ from each other (6.3e-4; GPU kernel selection), trajectories identical. PPO rollout, GAE, optimisation and checkpointing are unchanged (differential tests in `tests/`); resume was
validated in Optimization IV and again through the Studio (stop → resume continues at update N+1 with a valid contract/layout check).

## 4. What the trainer records now (`metrics.jsonl`, additive)

PPO health: policy/value loss, entropy, approximate KL, **clip fraction**, gradient norm, **explained variance**, learning rate, optimiser batches/epochs. Throughput: end-to-end,
rollout, per-phase seconds, physics ticks/s, per-worker seconds. Task: `nav/completed_episodes`, `nav/arrivals`, `nav/timeouts`, `nav/arrival_rate` (**`null` — never zero — when no episode ended**),
`nav/arrivals_per_1k_decisions` (the censoring-safe rate), `nav/decisions_to_arrival_{mean,median}`, initial / best / final distance, progress toward the goal for finished **and still-running**
episodes, `nav/active_outside_world_fraction`, `nav/active_in_world_episodes`, blocks destroyed (total and per 1,000 decisions), `nav/out_of_world_resets` (variant), action fractions, reward terms.
The Studio distinguishes: no episode finished (gap) vs zero arrivals among finished (0 %) vs timeouts vs genuine arrivals vs distance reduction without arrival (progress). Distance averages exclude
agents outside the world, which would otherwise dominate every mean (one agent at y = −10⁷ gave an average distance of 3 million tiles).
**Unresolved / not measured:** per-episode trajectories are not viewable in the Studio (the raw `trajectories.jsonl` exists); CPU utilisation and per-worker timing are recorded in the metrics but not
charted; crouch/jump usage only appears in the raw action fractions; training-time goals per update are sparse (about one arrival per update), so the held-out evaluation is the sensitive instrument.

## 5. Finding: agents leave the world and are never reset

How it was found: the first live chart showed "distance to goal" climbing to 3 × 10⁶ tiles. The trajectory records of that run show it unambiguously (free fall, y → −10⁷). The floor of a mini-world
(two rows of ordinary, destructible blocks) can be dug through by the random attack action; nothing terminates the episode of a body outside the world, and the only other terminations
are arrival and the 100,000-decision ceiling.

| XY1 as defined, 64 worlds, seed 7 | updates 1–50 | 51–100 | 101–150 | 151–200 | 201–250 |
|---|---|---|---|---|---|
| agents outside the world (end of update) | 9 % | 68 % | 92 % | 98 % | 100 % |
| goals reached (total in the block) | 33 | 4 | 0 | 0 | 0 |
| goals per 1,000 decisions | 0.16 | 0.02 | 0.00 | 0.00 | 0.00 |
| entropy | 8.749 | 8.755 | 8.806 | 8.797 | 8.753 |
| explained variance | 0.39 | 0.57 | −2.07 | −1.33 | −0.25 |

10 % of the population was gone at update 31, 50 % at update 62, 90 % at update 107. After that the optimiser trains on episodes that cannot succeed (explained variance negative).
This is a property of the experiment as defined, independent of PPO, and it dwarfs any learning signal in a long run. The Studio reports it (chart *Agents that left the world*, a warning on
the Experiments and Training pages, a "drained" verdict instead of "learning" once half of the agents are gone) and does not change the experiment.

## 6. Bounded learning diagnostics (1,024,000 decisions each; total 2.05 M, within the 1–2 M allowance)

Both runs: identical configuration and seed (7), through the Studio, followed by a held-out evaluation of the **untrained network (same initial weights)** and the **final checkpoint**.
Evaluation (`workbench/evaluate.py`): 16 terrains from seed offset 1,000,000 (training used 0–63), 6 goal draws per world = 96 episodes per cell, identical goals for every policy, horizon 1,000
decisions, success = arrival within the horizon; deterministic (most likely action) and stochastic (sampled) policy. Re-running the evaluation reproduced every number exactly.

| | untrained | trained (update 250) | difference (95 % interval, p) | reading |
|---|---|---|---|---|
| **XY1 as defined**, stochastic | 17/96 = 17.7 % | 28/96 = 29.2 % | +11.5 points (−0.6 to +23.1, p = 0.061) | inconclusive |
| XY1 as defined, deterministic | 2/96 = 2.1 % | 5/96 = 5.2 % | +3.1 points (−2.9 to +9.7, p = 0.25) | inconclusive |
| **Variant** (episode ends when the agent leaves the world), stochastic | 16/96 = 16.7 % | 42/96 = 43.8 % | **+27.1 points (+14.2 to +38.8, p < 0.001)** | **encouraging** |
| Variant, deterministic | 2/96 = 2.1 % | 12/96 = 12.5 % | **+10.4 points (+3.1 to +18.6, p = 0.006)** | **encouraging** |

Training-time view of the variant (same rules as training, 64 worlds): goals per 1,000 decisions 0.22 / 0.21 / 0.19 / 0.48 / 0.35 over the five 50-update blocks (297 goals in total against 37
for XY1 as defined), progress toward the goal 49 → 57 %, entropy 8.758 → 8.696, explained variance 0.32–0.53, KL ≈ 0.0055, clip fraction ≈ 0.065, 197 out-of-world resets in 1.02 M decisions
(about 0.8 per update). The training-curve statistic alone is *not* conclusive (the Studio's own test says "progress increasing, goals within the noise"); the held-out evaluation is what shows the effect.

Caveats, stated plainly: one seed and one run per arm; 96 episodes per cell; four comparisons were made (no correction applied); the variant differs from XY1 in the termination rule and was
evaluated under its own rule (the success definition is the same); an increasing reward or progress was not used as evidence.

**Reading: XY1 as defined — concerning (the experiment drains its own sample stream); with the out-of-world rule — encouraging, not yet confirmed (single seed).**

## 7. Is the first long campaign ready? What to decide

Operationally yes; scientifically **not until the out-of-world question is decided**. Options (none was applied to XY1):

1. **Adopt the variant as the experiment** (end the episode, as a truncation with no reward, when the agent has left the world): smallest change, preserves world distribution, reward and arrival rule;
   changes the termination contract, so it needs your explicit approval. Implemented as an opt-in, separately named experiment file (`xy1_point_navigation_mini_guarded.py`, flag
   `terminate_out_of_world`, default off) with its own audit (20 of 20) so the choice is reversible.
2. Make the bottom of the world indestructible (changes the procedural world distribution; the agent could still leave sideways/upwards only if open — the top is open).
3. Run XY1 as defined and accept degeneration after about 100 updates (not recommended).

Recommended before the 8-hour campaign: decide §7; run **three seeds** of the 1 M-decision diagnostic under the chosen rule (about 10 minutes each) and compare held-out success of the final
checkpoints; schedule held-out evaluations of numbered checkpoints (the Studio keeps one every 25 updates) while no training is running (they compete for the CPU).

## 8. Reproduction

```bash
python3 -m casu_grid.studio                                    # Experiments → "Diagnostic: about 1 million decisions" → Start
"$ISAAC_SIM/python.sh" casu_grid/perf/xy1_readiness_audit.py --output audit.json [--guarded]
QT_QPA_PLATFORM=offscreen python3 -m casu_grid.studio.tests.journey diagnostic [--experiment guarded --tag _guarded]
QT_QPA_PLATFORM=offscreen python3 -m casu_grid.studio.tests.journey diag_eval [--tag _guarded]
```
Raw data: `casu_grid/results/studio/diagnostics/<run>/` (metrics, launch record, system samples, evaluation JSON incl. every episode), `results/studio/diagnostics_summary.json`, `results/studio/audit/`.
