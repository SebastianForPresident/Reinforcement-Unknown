# CB1 Experiment Log

This is the provisional record of experiments performed against the CB1/VB1
lineage. It exists so results are not lost while CB1 remains an active
experiment. It should be reconciled with the final checkpoints, traces, and
TensorBoard data when CB1 is declared complete.

The baseline throughout is ordinary CB1 PPO unless an entry explicitly says
otherwise. A shadow experiment observes an alternative without allowing it to
change the actor's advantages or updates.

## 1. PPO stabilization intervention at 209,976 steps

**Question:** Were CB1's default Stable-Baselines3 PPO updates moving the
policy too aggressively for the expensive, highly correlated CasU rollout
distribution?

**Boundary:** `casu_ppo_cb1_vb1_209976_steps.zip` is the last preserved
checkpoint with the original settings. The new settings are present by
`casu_ppo_cb1_vb1_219976_steps.zip`.

**Settings:**

| Parameter | Through 209,976 | After resume |
|---|---:|---:|
| Learning rate | 3e-4 | 1e-4 |
| Minibatch size | 64 | 256 |
| Epochs per rollout | 10 | 4 |
| Target KL | none | 0.05 |
| Rollout length (`n_steps`) | 2,048 | 2,048 |

This reduced optimizer steps per rollout from 320 (`2048 / 64 * 10`) to 32
(`2048 / 256 * 4`), while making each gradient use four times as many samples
and reducing the learning rate by three.

**Immediate diagnostic response:** Comparing the 150k–210k window before the
change with 220k–280k afterward:

- Mean approximate KL fell from approximately 0.364 to 0.053.
- Mean clip fraction fell from approximately 0.760 to 0.125.
- Mean policy-gradient-loss magnitude fell from approximately 0.091 to 0.035.
- Entropy remained substantial rather than collapsing.

The calmer update regime persisted in the following 280k–340k window: mean
clip fraction remained approximately 0.127, with mean approximate KL around
0.082.

**Observed result:** Training immediately became substantially less clipped
and more controlled, and the live policy's behavior was visibly improved. CB1
continued from the same actor lineage under these settings; they remain the
established configuration.

**Evidence limitation:** Four hyperparameters changed together, and the actor
continued maturing on new procedural caves. This intervention establishes that
the combined stabilization package improved PPO diagnostics and did not break
learning. It does not identify which individual setting deserves causal
credit, nor prove that this package is optimal.

## 2. Independent-critic shadow experiment

**Question:** Was CB1's shared PPO critic failing mainly because it needed more
and separate value training on recent experience?

**Method:**

- Began at 1,270,648 CB1 steps and ended at 1,317,752.
- Copied the current CB1 value path into an independent critic.
- Retained a rotating window of the newest 32 rollouts.
- Trained the independent critic on archived TD(lambda) targets.
- Evaluated against each current rollout before adding that rollout to its
  training window.
- Left the real CB1 actor and shared critic on ordinary PPO updates. The
  independent critic remained shadow-only.

**Observed result:**

- Independent held-out loss wins: 5/22.
- Independent held-out explained-variance wins: 5/22.
- Median independent/shared held-out-loss ratio: 1.18.
- Catastrophic explained variance below -10 occurred five times for the
  independent critic and once for the shared critic.
- On the three reward-rich rollouts, mean loss was effectively tied.

**Conclusion:** Additional recent-window critic training did not yield a more
reliable value function. It frequently fit one return regime and transferred
poorly to the next quiet rollout. The critic snapshot was rejected for actor
advantages and remains shadow-only.

Full implementation and result details are in
[`PPO_Python/INDEPENDENT_CRITIC.md`](PPO_Python/INDEPENDENT_CRITIC.md).

## 3. Critic-free observed-return/lookahead shadow experiment

**Question:** Would observed discounted reward-to-go provide a clearly better
actor-training signal than CB1's critic-backed GAE?

**Method:**

- Collected 2,506 decisions under one frozen policy.
- Treated the first 2,048 as the candidate training prefix.
- Used the final 458 decisions only as observed future-reward lookahead.
- Chose 458 because, at gamma 0.99, the first omitted reward has weight
  `0.99 ** 459 < 0.01`.
- Stopped return accumulation at episode boundaries.
- Ran in shadow mode: ordinary CB1 GAE continued to train the actor while the
  alternative return signal was logged for comparison.

**Observed result:** The shadow comparison did not establish that the critic
was the cause of CB1's behavioral plateaus or that critic-free reward-to-go
would be a superior actor signal. Sparse rewards left many prefixes with
little additional observed evidence, so disagreement with GAE was not itself
evidence that GAE was wrong.

**Conclusion:** The suspected critic was not convicted. Active critic-free
training was not adopted, and ordinary CB1 PPO resumed unchanged.

The implemented method is documented in
[`PPO_Python/REINFORCE_LOOKAHEAD.md`](PPO_Python/REINFORCE_LOOKAHEAD.md).

## 4. Deterministic-inference observation

**Question:** Was CB1's stochastic action sampling merely producing needless
hesitation, and would deterministic inference expose a faster competent
policy?

**Method:** Loaded a frozen CB1 checkpoint for deterministic inference, taking
the maximum-probability action from each categorical action branch instead of
sampling.

**Observed result:**

- At times the Sawian moved very swiftly and decisively.
- It also entered repeated attractors in which it remained stuck attacking in
  one direction.
- External disturbance was required on multiple occasions to break those
  attractors and restore movement.

**Conclusion:** Stochasticity is not merely visible sloppiness in CB1. It
currently contributes to escape from repetitive local behaviors. Faster
deterministic movement did not translate into a generally safer inference
mode, so ordinary stochastic inference remains behaviorally important.

**Evidence limitation:** Unlike the two shadow experiments, this was an
observed live inference test and does not yet have a dedicated quantitative
artifact or matched stochastic control. Its qualitative result is retained
here pending final CB1 documentation.

## 5. Acceleration failure and fixed-cadence repair

**Question:** Could CB1 be accelerated to 10x without changing the transition
dynamics learned by the strong pre-acceleration policy?

**Historical boundary:** `casu_ppo_cb1_vb1_2091300_steps.zip` is the preserved
pre-acceleration checkpoint. Commit `84e3658` introduced 4x operation, the
60-FPS/vsync-off cap, high-time-scale safety patches, and profiling. Commit
`bae455e` introduced 10x plus exact blocking at the post-Physics2D boundary.
The latter removed the old per-action `Time.timeScale = 0` pause and moved
observation publication to after the tenth completed physics tick.

The validated 10x protocol continued to execute exactly 2,500 physics ticks
and 250 policy macrosteps per 50 simulated seconds. It sustained approximately
44--47 steps/second without synchronization errors. However, the 60-FPS cap
allowed only approximately 240--291 ordinary `Body.Update` calls during the
same window. At 15x the count fell as low as 32--75. Raising the FPS cap to 120
or setting `Time.maximumDeltaTime = 0.2` did not make Unity leave an
already-running fixed-step batch to render an intervening frame.

This exposed a hidden environment variable: render scheduling. Grounding,
sliding, water, crouching, climbing, limb/arm state, and other gameplay state
were advanced in `Update`, while PPO actions and observations were clocked by
physics. At 1x the render and physics rates were close enough to conceal the
coupling. At 10x, the effective rate of essential player logic fell to about
one render update per policy macrostep.

### Frozen-policy matched-pairs evidence

Ten identical cave seeds were evaluated stochastically with common action RNG
seeds at both 1x and the original 10x harness. Each trial had a 54,000-step
(180 simulated minute) horizon.

| Checkpoint / speed | Completed | Capped median simulated time |
|---|---:|---:|
| pre-acceleration at 1x | 10/10 | 27.16 min |
| pre-acceleration at original 10x | 9/10 | 46.52 min |
| 3,238,180-step checkpoint at 1x | 10/10 | 62.13 min |
| 3,238,180-step checkpoint at original 10x | 10/10 | 74.40 min |

For the pre-acceleration checkpoint, 1x won 9/10 pairs, the median paired
1x/10x ratio was 0.603, and the exact two-sided sign-test p-value was 0.0215.
At 1x the pre-acceleration checkpoint also beat the later checkpoint on 9/10
pairs (later/pre median ratio 2.175, p=0.0215). The acceleration was therefore
not behaviorally equivalent, and continued training under it produced a
slower policy.

### Fixed-step diagnostic

On August 16, a guarded harness experiment replaced the PPO player's natural
render-driven `Body.Update` with one call after every 50-Hz physics tick. The
call uses `Time.fixedDeltaTime` internally so health, movement, and other
timers do not advance by a render-frame delta 50 times per simulated second.
The profiler then reported exactly 2,500 `Body.Update` calls per 2,500 physics
ticks while retaining 9.98x--10.02x execution. Body logic consumed roughly
5--7% of wall time and did not degrade over more than 1,200 profiler windows.

An initial single-seed invocation accidentally used deterministic inference;
it repeated one action for thousands of steps and is invalid as an
acceleration comparison. A new explicit stochastic inference mode was added
and seeded with the same `20260815 + world_seed` convention as the matched
test.

With stochastic inference on seed 391728405, the first fixed-body 10x episode
completed in 22,546 steps. The process then continued automatically and
completed 29 consecutive episodes on the same cave. Their median was 9,609
steps (32.03 simulated minutes), and the best was 4,771 steps (15.90 minutes),
slightly faster than the prior 4,864-step (16.21-minute) record on that seed.
Only the first episode reproduced the paired-test RNG initialization; the
later episodes are exploratory repeated-seed evidence, not 29 independent
matched trials.

### Adopted transition repair

CB1 remains the same experiment lineage, but continuation is gated on a new
ten-seed frozen-policy validation. The repaired harness:

- advances the PPO player's `Body.Update` exactly once per physics tick using
  fixed-step delta time and suppresses its natural render invocation;
- applies each PPO action only at the exact post-physics policy boundary,
  removing variable render-driven reapplication of jump, attack, interaction,
  and item-use behavior;
- suppresses render-driven player `Limb.Update` while functional godmode is
  active, because CB1 explicitly neutralizes physiology and repairs limb state
  every fixed tick;
- disables earthquakes during CB1 because onset and impulses are randomly
  sampled in `WorldGeneration.Update` and therefore change distribution with
  render cadence, like the already-disabled radiation deadline;
- leaves presentation-only UI, camera, audio, animation, and particle work on
  the render clock; and
- leaves item/building destruction callbacks render-driven because those
  inventories and visible-entity records are omitted from the active reduced
  observation, while forcing the callbacks would duplicate loot, destruction,
  and particle side effects.

The repair is enabled by default for PPO launches and can be disabled for
historical comparison with `PPO_FORCE_BODY_UPDATE=0`. Candidate DLL hash at
the time of this entry:
`b6fedd0a81e11cc2f39a706b5d627bcb55d8971f6b1f1698b409fc28ed1fcd96`.

**Continuation gate:** Re-run the pre-acceleration checkpoint on the same ten
seeds at repaired 10x and compare it with the archived 1x half. Do not resume
actor training until fixed cadence, completion performance, walls, fluids,
excavation, and borders pass. If accepted, continue CB1 from the strong
2,091,300-step checkpoint rather than the 3,238,180-step policy adapted to the
render-coupled environment.

**Boundary-only action repair rejected:** The first repair applied a newly
sampled action once at the 5-Hz policy boundary and removed all render-driven
reapplication. On the first eight matched seeds it beat archived 1x on 0/8,
had a median repaired/1x step ratio of 2.92, and completed only 6/8 within the
54,000-step horizon. Border occupancy reached 34,288, 38,414, 20,003, and
43,038 steps in four trials. This showed that boundary-only execution was
constant but did not preserve the pre-acceleration policy's learned action
semantics: at ordinary 1x, a held action had been processed approximately once
per render/physics interval, not only once per ten-tick macrostep. These failed
results remain in `matched_pairs_repaired_10x_results.csv` under the
`repaired10x` label.

The successor repair samples actions at 5 Hz but executes the held action at a
fixed 50 Hz. It applies a new action before tick one, reapplies held effects
after ticks one through nine for the following intervals, and does not reapply
the old action after tick ten. Edge-triggered controls are consumed by the
first application, while held movement, jump, attack, and tool behavior receive
exactly ten executions per macrostep. Profiler output includes
`fixedActions`; settled windows must report 2,500 applications alongside 2,500
physics and body ticks. Its separate continuation gate writes
`matched_pairs_fixed_action_10x_results.csv` under `fixedAction10x`.

After the first four fixed-50-Hz trials, that repair beat original 10x on 4/4
seeds and cut the median time relative to original 10x to 0.528. It beat the
archived capped-1x control on 1/4 seeds, but its median fixed50/1x ratio
remained 1.50. This monotonic improvement over boundary-only execution
supports action/body cadence as causal while showing that 50 Hz is not yet an
equivalent replacement for the capped-1x render path.

The next isolated ablation schedules gameplay at deterministic 60 Hz over
unchanged 50-Hz physics: every ten-tick macrostep receives twelve body updates
and twelve held-action executions, and manual body methods use a 1/60-second
delta. PPO sampling remains stochastic at 5 Hz. The schedule performs one body
update after each physics tick plus a second after ticks five and ten. A newly
latched action supplies the first action execution; held effects are reapplied
after ticks one through nine, with a second execution after ticks four and
eight. The old action is never reapplied after tick ten. Settled profiler
windows must therefore report 3,000 `bodyUpdate` and 3,000 `fixedActions` calls
per 2,500 physics ticks. Results are isolated under `fixed60Hz10x` in
`matched_pairs_fixed_60hz_10x_results.csv`.

**Fixed-60-Hz continuation gate accepted:** All ten matched seeds completed on
August 16. Fixed-60-Hz 10x completed 9/10 versus 10/10 at archived capped 1x.
Pairwise wins were exactly 5--5; the median paired 1x/fixed60 ratio was 0.963,
the median paired difference was 1.19 simulated minutes favoring 1x, and the
two-sided sign-test p-value was 1.0. This is behaviorally unlike the original
10x failure, where 1x won 9/10 with ratio 0.603 and p=0.0215. Fixed 60 Hz also
set a new seed-391728405 record of 3,197 steps (10.66 simulated minutes),
compared with 4,864 steps at archived 1x.

The remaining failure was the known world-border tail: seed 723910864 reached
the 54,000-step horizon with 46,214 border steps. Seed 1047293851 completed
with 9,441 border steps. Because 1x no longer performed materially or
consistently better, the fixed-60-Hz transition contract passed the declared
continuation criterion. CB1 continuation is authorized from checkpoint
2,091,300. The render-coupled 2.10M--3.24M actor lineage is rejected for
continuation but retained as forensic evidence.

Full matched-pairs procedure and the acceleration Git reconstruction are in
[`PPO_Python/MATCHED_PAIRS.md`](PPO_Python/MATCHED_PAIRS.md).

## Planned: spatial counterfactual / Clever Hans audit

This has been designed but not yet run and is not counted as a completed CB1
experiment.

Take a recorded observation containing a clear passage and a visible
alternative. Hold the complete general observation fixed and evaluate the
frozen policy on controlled versions of only the spatial input:

1. Original open passage.
2. Passage filled with ordinary breakable material.
3. Passage filled with nearly destroyed breakable material.
4. Passage filled with correctly encoded infinirock or another effectively
   indestructible wall.

Compare the complete categorical action distributions, especially movement,
cursor direction, and attack probability. A mechanically meaningful response
would distinguish entering open space, excavating breakable material,
finishing a damaged obstruction, and abandoning an impossible obstruction for
the visible alternative.

Run the frozen checkpoint on CPU so the audit cannot interfere with active CB1
CUDA training. This is a causal representation-use probe, not proof of general
route understanding.
