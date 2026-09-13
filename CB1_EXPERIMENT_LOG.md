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

## 5. World-border runaway and training truncation

**Incident:** The accepted fixed-60-Hz continuation resumed from 2,091,300
steps and completed its first 16 caves. Episode 17, seed 1431186900, then
became trapped near the right world border. It reached best depth 271 at
episode step 38,238 but made no further best-depth progress. By shutdown it
had consumed 1,238,528 episode steps, 93.4% of them near a world border. The
policy reached approximately 3.70M global steps while TensorBoard's completed-
episode return and length summaries remained frozen. Entropy loss moved from
about -5.33 to -4.34 during the correlated experience, and critic explained
variance deteriorated to approximately -52.

**Branch decision:** The contaminated continuation beginning with episode 17
is rejected. The complete 34GB run, trace, grid history, checkpoints, and
TensorBoard evidence are preserved externally at
`Reinforcement-Unknown Archives/CB1_2026-08-17_border_trap_2471860_to_3709220`.
The active lineage was severed at `casu_ppo_cb1_vb1_2470180_steps.zip`, the
last completed PPO checkpoint before episode 17 began at global step
2,471,860. Its SHA-256 is
`56624302fc0265ec3a027ab1b6f5927ab76cfc604d1eb476d06348e897ed66bc`.
Active TensorBoard history was clipped to the same 2,470,180-step boundary.

**Fix:** Training now reports `truncated=True` after either 100,000 decisions
in one episode or 75,000 decisions without increasing `BestLayerDepth`.
Completion remains the only true termination. These thresholds preserve all
16 successful clean episodes: the longest completed episode used 77,836
steps, and its longest no-progress plateau was 62,305 steps. The rejected
episode would instead have truncated near step 113,238. Evaluation and matched-
pairs inference retain their explicit horizons and completion accounting;
truncation is never recorded as a win.

**Mature-policy acceleration recheck planned:** Training was frozen at the
complete 4,866,340-step checkpoint for a final matched 1x/fixed-60-Hz-10x
comparison. The test uses ten new world seeds, paired stochastic action seeds,
and the established 54,000-step capped horizon. Configuration and incremental
results are `PPO_Python/matched_pairs_mature_4866340.json` and
`PPO_Python/matched_pairs_mature_4866340_results.csv`. The checkpoint SHA-256
is `ed3a695d188a969eeac40cf541d485c2e05efc7bf8856ecbba45a061dc0ceeea`.

**TensorBoard chronology repair:** SB3 restart suffixes had split the clean
lineage across `PPO_1 -> PPO_0 -> PPO_1`, causing TensorBoard to draw a
misleading gap-spanning line. The raw files are preserved unchanged under
`tensorboard_raw_sb3_2026-08-19`. The active TensorBoard directory was rebuilt
from scalar events, retaining the latest wall-time value for each duplicated
step/tag and separating actual intervention boundaries:

- `PPO_0`: 2,048--208,120 (original PPO settings).
- `PPO_1`: 210,168--2,091,300 (stabilized PPO settings).
- `PPO_2`: 2,093,348--2,470,180 (fixed-60-Hz continuation).
- `PPO_3`: 2,472,228--4,868,388 (training truncation enabled).

Subsequent unchanged continuation begins in `PPO_4`. Numeric-only directory
names are intentional: SB3 recognizes them and will automatically allocate
the next number on any future restart.

The independent-critic and observed-return experiments remain within the
stabilized segment because they were shadow-only and did not alter actor
updates. This is a visualization repair only; checkpoints and policy data are
unchanged.

## 6. Post-peak policy collapse and recovery branch

**Branch point:** Matched stochastic inference identified
`casu_ppo_cb1_vb1_3862820_steps.zip` as the empirical peak. It completed all
10 test seeds at fixed-60-Hz 10x with a 40.38-simulated-minute median. Its
SHA-256 is
`1afe9a45a5946d87b7e62455b1f176bcf813c560299e5f234f91f8b4025c5cc1`.
The 4,866,340-step policy completed only 7/10 identical seeds and required
1.962 times the peak policy's paired median capped time. A matched 1x/10x
recheck exonerated acceleration: the mature policy completed 7/10 at 10x and
5/10 at 1x, with 10x winning seven pairs and 1x winning one.

**Observed degradation:** The mature policy first lost conditional
excavation behavior in ordinary interior pockets. On three seeds that the peak
policy completed, the mature policy revisited only 302--524 positions during
each final 10,000-step plateau. Its attack rate fell to 41.6%, 18.3%, and 8.6%
versus 56.7%, 73.2%, and 55.0% for the peak policy on the corresponding
successful episodes. These failures did not involve borders, fluids, or
climbing.

Unchanged training was deliberately continued as a failure-mode experiment.
The behavior subsequently condensed into a right-world-border attractor. In
failed episodes 20--30 of the final continuation, all eleven spent 63--100%
of their final 10,000 decisions at the right border. Eight occupied no more
than four distinct tile positions; four occupied exactly one. Representative
sidecars show the crouched player pressed against dense border-side terrain
with open space behind it. The policy continued to sample attack, jump, look,
and conflicting movement actions, but those actions produced almost no
trajectory diversity or depth reward.

By shutdown the continuation had reached checkpoint 7,538,980. Its final
trace contained 14 wins and 16 watchdog failures, ending with 11 consecutive
79,163--100,000-step failures. TensorBoard ended with entropy loss -2.03,
explained variance -28.55, rolling episode length 38,245.63, and rolling
reward 17.51. During the collapse, explained variance remained as low as
approximately -59 while absolute value loss was only a few millionths: the
border loops had made rollout returns nearly constant and deprived on-policy
PPO of successful counterexamples.

**Archive and lineage decision:** The 3,862,820-step checkpoint remains the
active branch point. All 359 later checkpoints, the overlapping training
trace and geometry sidecar, matched-pairs evidence, later continuation traces,
and original `PPO_3`/`PPO_4` TensorBoard runs are preserved externally at
`Reinforcement-Unknown Archives/CB1_2026-08-20_policy_collapse_3862820_to_7538980`
(approximately 79 GB). Active `PPO_3` was rebuilt from scalar events and
clipped to 3,862,820; active post-peak checkpoints were removed from resume
selection.

**Failure penalty:** CB1 remains a descent-only policy. Any watchdog-ended
episode is therefore an endogenous task failure rather than an exogenous time
limit. Both `stagnant_best_depth` and `max_episode_steps` now add a terminal
`-10.0` incomplete-episode penalty. They return `terminated=True` and
`truncated=False`, preventing SB3 from bootstrapping through failure. Genuine
completion remains the only `episode_complete` outcome. The reward ordering is
therefore shallow failure < deep failure < completion, while no action such as
attack, movement, crouch, or border avoidance receives a hand-authored reward.
The 100,000-step absolute and 75,000-step no-progress safeguards remain as
technical backstops.

**Credit-horizon correction:** At 5 Hz, the original `gamma=0.99` and
`gae_lambda=0.95` reduced direct GAE credit to 1% after approximately 75
decisions, or 15 simulated seconds. A 75,000-step penalty therefore could not
blame the actions that entered a trap. `gae_lambda` is now 0.98, extending the
1% GAE horizon to approximately 153 decisions or 31 simulated seconds and
reducing dependence on the demonstrably unstable critic. Resume loading
explicitly overrides the checkpoint's saved gamma and lambda with the current
contract values.

**Capable-inactivity termination:** A new 75-decision detector delivers the
same terminal `-10.0` penalty after 15 simulated seconds of complete physical
inactivity while the player is capable of acting. The timer advances only
when tile position is unchanged, both velocity components are within 0.05,
the full local block-health map is unchanged, depth did not improve, the
player is conscious, and the player is neither dead nor ragdolled. Movement,
excavation damage, depth progress, or incapacity resets the timer. This keeps
the trap-entry transition inside the new GAE horizon without treating slow but
productive excavation as failure. Analysis of all ten successful peak-policy
matched episodes rejected a simple 15-second no-depth rule: even the shortest
successful episode had a 198.6-simulated-second interval without increasing
integer `BestLayerDepth`.

Every inactivity gate is recorded on every scalar trace row: capable state,
unchanged tile, inactive velocity, unchanged local block health, unchanged
depth, whether the timer advanced, the resulting timer value, watchdog status,
failure reason, and the explicit incomplete-episode reward term. Together with
the full geometry sidecar and sampled action, this makes timer resets and
possible avoidance strategies auditable. In particular, later analysis must
check whether the policy learns to reset the timer through one-tile twitching,
irrelevant block damage, or deliberate incapacity while still failing to
descend; the unchanged 75,000-step failure backstop prevents such avoidance
from escaping penalty indefinitely.

## 7. Mobile shelf-surfing basin and continuous-progress termination

**Strong basin and failure continuation:** The `.98` continuation developed
two performance basins substantially better than its original 3,862,820-step
branch point. The later basin reached the global minimum rolling episode
length of 14,251.73 at step 10,258,724 and produced CB1's first sub-ten-minute
completion: 2,921 decisions, or 9m44.2s simulated. The rollout-aligned
`casu_ppo_cb1_vb1_10262820_steps.zip` is retained as the active branch point;
its SHA-256 is
`db6ecc26f992b4f8d901d788b82f0e4b34466e933e2a2c1f0eccd722633dc02b`.

Unchanged training was deliberately continued to observe the next failure
mode. Completion reliability remained high, but the rolling episode length
rose to 25,025.49 by step 13,605,156. Entropy loss ended at -2.73, critic
explained variance at -34.15, KL at 0.0648, and clip fraction at 0.185. The
last complete checkpoint was 13,601,060 with SHA-256
`72781fa83896e8524519aac22dded2595f1c1d40e22f646a4597f70eaa121d`.

**Observed behavior:** This was not the earlier stationary-pocket or
world-border collapse. Recent slow episodes usually occupied no border tiles,
visited hundreds to more than one thousand distinct positions, and eventually
completed. Across long plateau segments, attack probability fell to about
35.6%, jump probability rose to 56.3%, crouch remained near 61%, and grounded
observations fell to 32.8%. Geometry samples showed relatively open space
above the player and partial shelves below, with no consistent lateral wall,
fluid, or climbing obstruction. The policy had condensed around mobile
lateral traversal and intermittent excavation that eventually found another
downward breakthrough: shelf-surfing.

**Continuous-progress analysis:** A postmortem of 165 terminal episodes used
uninterrupted decisions with exactly zero increase in monotonic continuous
`LayerProgress`. The maximum no-progress span had median / 90th percentile /
maximum values of 1,660 / 2,656 / 3,450 for 31 sub-10k wins; 3,885 / 6,822 /
9,899 for 75 ordinary 10k--25k wins; and 9,758 / 21,846 / 36,195 for 46 slow
over-25k wins. A 2,500-step threshold would reject healthy record-class runs.
A 5,000-step threshold would preserve all fast wins but reject 21 ordinary
wins. A 7,500-step threshold would reject seven ordinary wins. A conservative
10,000-step threshold rejected none of the 106 observed fast or ordinary wins
while identifying 22 of 46 slow wins.

**Lineage and archive:** Checkpoints after 10,262,820 through 13,601,060,
complete raw `PPO_4` events, and both scalar/geometry trace pairs are preserved
externally at `Reinforcement-Unknown Archives/CB1_2026-08-22_shelf_surf_10262820_to_13601060`
(approximately 70 GB). Active checkpoints after the branch point were removed
from resume selection. Active `PPO_4` was rebuilt from its preserved scalar
events and clipped exactly at step 10,262,820. The new reward-contract
continuation begins as `PPO_5`, preserving a touching but non-overlapping
TensorBoard chronology.

**Fix:** A separate `mobile_stagnation` terminal now fires after 10,000
consecutive decisions with no increase in continuous monotonic best progress.
Any genuine new depth resets the timer; movement, velocity, jumping, attacks,
block damage, fluids, incapacity, and position changes do not. It applies the
existing `-10.0` incomplete-episode penalty and reports `terminated=True`,
`truncated=False`. The 75-decision exact capable-inactivity detector,
75,000-step integer-best-depth watchdog, and 100,000-step absolute backstop
remain unchanged. Trace rows record the timer, configured maximum, whether
continuous progress improved, whether the timer advanced, and the terminal
reason. This remains outcome-only reward shaping: no action or geometry class
is declared intrinsically bad.

## 8. Successful-episode length metric

Late in the `PPO_5` continuation, capable-inactivity failures exposed a
measurement ambiguity in SB3's standard `rollout/ep_len_mean`. The policy had
reached a genuine fast-completion basin, including a completion-only mean /
median of 6,026 / 4,971 decisions over one observed 11-win sample. It then
began mixing similarly fast wins with short capable-inactivity failures. Four
failures among 15 terminal episodes ended at 10,104, 1,235, 4,487, and 3,397
decisions. Because SB3 averages every terminal episode, those early failures
lowered `ep_len_mean` even as `ep_rew_mean` and completion reliability fell.
By global step 14,047,524, TensorBoard reported `ep_len_mean=7,655.01` and
`ep_rew_mean=17.39`; the last 15 terminal episodes were only 80% successful,
and their completion-only mean / median were 7,341 / 6,181. The displayed
length improvement therefore combined genuine faster wins with a substantial
failure-discounting artifact.

CB1 now additionally records `rollout/success_ep_len_mean`. Its window is the
last 100 terminal episodes, matching SB3's standard statistics window, but
the mean includes only genuine completions within that window. Failures still
affect `ep_rew_mean` and the denominator/window occupancy; they cannot make
the success-length curve look faster merely by terminating early. The metric
starts cleanly on the first resume after checkpoint 14,061,860. Historical
TensorBoard event files remain untouched.

A proposed historical backfill was tested in dry-run mode against the scalar
traces. As a mandatory safety check, the reconstructed ordinary
`ep_len_mean` first had to reproduce SB3's existing values. Only 2,614 of
6,797 points (38.46%) matched within 0.05 decisions, reflecting ambiguous
alignment across interrupted sessions and severed branches. The validator
refused to write; no event file was created or modified. Exact continuous
backfill was abandoned rather than publish a plausible but false curve.
Historical comparisons will instead use explicit completion-only summaries
at documented checkpoints and basins.

**Restart-window repair:** The first pause after introducing the metric exposed
that its Python callback window was process-local. The accepted pre-pause
value was 8,119.87 at checkpoint 16,458,020. After restart, a 7,320-step win
was incorrectly displayed as the complete rolling mean, followed by 10,656.5
after a 13,993-step win. Those episodes were genuine completions, but the
displayed statistic was not comparable with the preceding 100-terminal
window. The short continuation also contained a one-update KL spike to 0.316.
It was discarded at the user's direction rather than mix the discontinuous
metric segment into the accepted chronology.

Checkpoints 16,468,260 and 16,478,500, the restart trace and geometry sidecar,
and the complete original event sources are preserved externally at
`Reinforcement-Unknown Archives/CB1_2026-08-24_success_metric_restart_16458020_to_16480548`.
Active TensorBoard was clipped to 16,458,020 and active resume selection now
ends at `casu_ppo_cb1_vb1_16458020_steps.zip`.

`SuccessEpisodeLengthCallback` now persists each member of its 100-terminal
window, including terminal global step, to `success_ep_len_state.json`.
Resume loading discards entries newer than the selected checkpoint, so an
ordinary pause or a short rollback both recover the correct historical
window. The initial persisted state was streamed from the accepted trace
through exactly 2,396,160 post-checkpoint decisions. It contains 100 terminal
episodes, 90 successful, and reproduces the accepted 8,119.8667 mean exactly.

## 9. Fifty-million-step milestone and convergence extension

Training reached its complete-rollout boundary at 50,000,164 policy decisions
on 2026-09-03. The milestone is preserved as
`casu_ppo_cb1_vb1_50000164_steps.zip`; it remains an independently selectable
checkpoint even though CB1 continues.

The policy had established a mature common-case descent strategy, including
an all-time raw `rollout/success_ep_len_mean` of 1,137.5369 at global step
46,434,596 and a recorded 440-decision individual completion. It had not yet
shown full-distribution convergence: rolling reliability and the slow tail
continued to cycle, and recent `capable_inactivity` failures repeatedly ended
grounded and nearly motionless at undersized wall/ledge openings while the
policy attacked empty space beyond the actual blocker. At the 50,000,164
boundary, the final 100-terminal window contained 93 completions, with a
completion-only mean / median of 1,247.4 / 1,137 decisions.

Because million-step completion means and successful tail percentiles had
continued to improve over the long run, the target was extended from
50,000,000 to 60,000,000 decisions. The 50--60M interval is a continuation of
the same `PPO_5` lineage, reward, observation/action protocol, optimizer
configuration, and 10x simulation setup--not a new ablation. Its purpose is
to test whether repeated vanilla on-policy exposure durably reduces the rare
clearance-failure tail or merely produces recurring speed/reliability cycles.

## 10. PPO_6 exhaustion and incapacitated-inactivity intervention

**Lineage boundary:** `PPO_5` was stopped on 2026-09-04 after the complete
optimizer/checkpoint boundary at 51,843,364 decisions. The preserved source is
`casu_ppo_cb1_vb1_51843364_steps.zip`, SHA-256
`136399631a8cc9e79c3ae98a94bd75bbb3942bd10f9f4bad42730676ddd0b2be`.
`PPO_6` resumes from exactly this model and keeps the CB1 action/observation
schema, network, PPO hyperparameters, progress/completion reward, 10x
simulation, and all existing watchdog thresholds. This is an intentional
environment/termination intervention rather than evidence of uninterrupted
training under one contract.

**Why the earlier exhaustion designs were not reused:** Reward V7 (commit
`e2e47af`) continuously charged `-0.001 * exhaustion^4`. Reward V8 (commit
`c102a7e`) instead rewarded the signed stamina delta at scale 0.001, thereby
charging stamina expenditure and rewarding recovery. Reward V10 (commit
`03a12a3`) retained that delta term and added an action-gated low-stamina stall
penalty below 10 stamina after a 30-second grace period, doubling every ten
seconds up to 0.01. Reward V15 (commit `0dc3efe`) retained the latter
low-stamina/action proxy. These formulations directly valued stamina state,
stamina conservation, recovery, or selected actions. They could therefore
discourage useful exertion or teach behavior around a hand-authored proxy.
Their historical coexistence with broader reward and environment changes also
prevents attributing old policy outcomes to the exhaustion term alone.

**PPO_6 formulation:** Exhaustion has no direct reward or penalty. Functional
godmode no longer resets `body.stamina` to 100, allowing the game's ordinary
stamina expenditure and recovery to occur; it continues resetting energy to
100 so this phase introduces exhaustion without simultaneously introducing
long-horizon energy depletion. Stamina and exhaustion remain observable
through the existing scalar observation. No observation or action dimension
changes.

The existing 75-decision inactivity watchdog is repurposed so capability is
context, not an exemption. While capable, its existing strict gates remain:
depth and tile position must not change, both velocity components must remain
within epsilon 0.05, and the complete local block-health map must remain
unchanged. While incapacitated, the timer advances unconditionally; physics
jitter, passive tumbling, passive depth change, and other motion the policy
cannot control do not excuse or reset the state. Because the 5-Hz policy makes
75 decisions in 15 simulated seconds, an incapacitated player can no longer
fill indefinitely many rollouts with unactionable states. At threshold, the
episode receives the unchanged `-10.0` incomplete-episode penalty and reports
`terminated=True`, `truncated=False`, under the new reason `inactivity`.

The credit-assignment hypothesis is that repeated terminal transitions into
self-induced incapacity will keep the controllable lead-up within the 0.98-GAE
horizon, while remaining completely neutral toward exhaustion that does not
produce prolonged physical inactivity. This is a hypothesis to test, not yet
an outcome. A healthy result would begin with some new physical-inactivity
failures and then reduce their frequency without sacrificing completion speed
or causing systematic stamina hoarding.

**Auditability:** Scalar traces retain raw stamina, energy, consciousness,
ragdoll time, death, velocity, tile/depth state, and block-health change. PPO_6
adds `inactivity_incapacitated`, `inactivity_physical_gates_met`, and
`inactivity_timer_advanced_while_incapacitated`, while retaining every prior
inactivity gate and counter. The run directory also receives
`intervention_ppo6.json`, which pins the source checkpoint and exact terminal,
reward, godmode, and threshold semantics. Analyses must distinguish ordinary
stationary clearance failures from low-stamina/incapacitated failures rather
than inferring exhaustion from the terminal reason alone.

**First-start validation correction:** The first PPO_6 launch connected and
loaded the source checkpoint but stopped before collecting a single decision:
the fail-fast Python validator still required stamina to be at least 99 as a
legacy godmode invariant and rejected the intentional first observation at
98.980. No rollout or optimizer update occurred. PPO_6's validator now permits
the full normalized 0--100 stamina range while retaining every other godmode,
clock, action-feedback, and finite-observation invariant. A regression test
explicitly validates a zero-stamina observation.

**Second-start timer correction:** The next launch exposed the exhaustion
state successfully but showed that the first implementation still measured
incapacitated *physical stillness*. In the first 3,584-decision episode,
3,478 observations were classified incapacitated and `TimeRagdolled` reached
approximately 695 simulated seconds, yet passive velocity jitter repeatedly
exceeded 0.05 and the inactivity counter never exceeded 13. This run was
stopped and rejected. No checkpoint beyond 51,843,364 was produced. Its event
files and complete scalar/geometry traces are archived as a false start; the
accepted PPO_6 chronology restarts from the source checkpoint. The corrected
rule counts consecutive incapacity itself, independent of passive physical
motion, while retaining the original physical-change gates for capable play.

**Checkpoint cadence during adaptation:** PPO_6's repeated short failed
episodes reduced throughput to approximately 16 decisions/s. At the user's
2026-09-04 transcript-processing pause, checkpoint cadence was increased from
one save per 100 complete PPO updates (204,800 decisions, roughly 3.5 hours in
this regime) to one per 25 updates (51,200 decisions, roughly 53 minutes).
Checkpoints remain complete post-optimizer boundaries; the model format and
retention policy are unchanged. With approximately 503 GiB free at the change,
the additional roughly 200 MiB per save is not an immediate disk risk, but the
cadence should be revisited if high-throughput long-episode training resumes.

**Passive-depth reset correction:** Later inspection of apparent long-episode
improvement found a second legacy interaction. Although `AdvanceInactivity`
correctly ignored depth while incapacitated, the surrounding environment step
still reset `inactive_steps` immediately whenever integer `BestLayerDepth`
increased. An unconscious/ragdolled body could therefore fall passively and
restart its 75-decision lease. In episode 938, for example, uninterrupted
incapacity began at step 100, passive depth reset the counter from 24 to 1 at
step 124, and termination was delayed until step 198. Other apparent long
episodes reached thousands of decisions through repeated short counter resets;
they are not accepted as evidence of learned stamina management.

Training was stopped when this was identified. All PPO_6 checkpoints, events,
success-window state, and traces produced under the passive-depth loophole are
archived as rejected experimental evidence. Accepted PPO_6 again begins from
the pristine 51,843,364 source. The pre-reset was removed: capable depth
progress still resets physical inactivity through `AdvanceInactivity`, whereas
incapacity now advances consecutively regardless of passive velocity, tile,
block-health, or depth changes. The unit regression explicitly covers an
incapacitated observation with all of those changes and `depth_improved=True`.

## 11. PPO_6 actual-swing / claw-contact shadow audit

The exhaustion intervention described above was ultimately rejected on
architectural grounds. NavHead's task contract assumes a conscious,
functional, sufficiently equipped body; medical state and authority to invoke
navigation belong to the medical/planner layer. Teaching NavHead to manage
exhaustion mixed those responsibilities. Every checkpoint and event/trace
artifact produced after the 51,843,364 source under that intervention is kept
only in the external archive and is not accepted into the CB1 lineage.

PPO_6 is reassigned to a narrower experiment from the unchanged
`casu_ppo_cb1_vb1_51843364_steps.zip` source. Historical reward code used loss
of `ClawHealth` as evidence of a successful unarmed contact (a full contact was
documented as costing up to 0.3 claw health), but that signal previously
coexisted with several other reward terms and cannot support a clean causal
claim. The new audit isolates the measurement before changing reward.

Wire protocol V2 adds two macrostep telemetry values: the number of actual
swings accepted by the game, detected by an increase in attack cooldown across
`UseItemInHand`, and accumulated claw-health loss captured immediately before
functional godmode repairs it. These fields are deliberately excluded from
the policy observation tensor, so the checkpoint's 139-scalar network input
and action space are unchanged. Functional godmode again restores stamina,
and the accepted capable-only inactivity watchdog is restored.

An executed macrostep swing with zero accumulated claw-health loss is recorded
as a candidate ineffective swing. An executed swing with any claw-health loss
is contact; holding attack during cooldown is not counted as a swing. The
candidate penalty is logged at `-0.0001` per miss, but is initially disabled:
the applied value is exactly zero and CB1's progress/completion reward is
unchanged. This shadow phase must establish plausible swing counts, contact
loss magnitudes, miss frequency, and cooldown behavior before any reward
intervention is enabled. Claw loss establishes physical contact, not useful
terrain damage; metal impacts and coincident unrelated claw injury are known
interpretive caveats.

## 12. PPO_6 binary attack permission (2026-09-05)

The user selected an explicit planner-controlled permission contract instead
of penalizing missed swings. A separate run was created non-destructively at
`checkpoints/CB1_PPO6_attack_permission_51843364`, from the preserved
51,843,364-step checkpoint. The previous run, checkpoints, TensorBoard events,
and shadow telemetry are retained in their existing locations. The shadow
run is not the parent of this new lineage.

An appended general scalar means 0 = attacks allowed, 1 = forbidden. Training
samples these two modes with equal probability once per episode. Reset options
can explicitly select `attack_forbidden` for evaluation/planner control.
The new policy masks the attack-on logit in sampling, inference, and PPO
likelihood/entropy evaluation. The environment independently clamps forbidden
attack commands before sending them to Unity. Traces record permission,
requested attack, executed attack, swings, and claw contact.

Allowed episodes retain the existing reward and watchdog -10 penalty.
Forbidden episodes retain progress/completion rewards, but watchdog endings
receive zero additional penalty. All watchdog endings remain unsuccessful true
terminals (no bootstrapping). Existing thresholds are retained, including
75 capable-inactive decisions and 10,000 decisions without new continuous
depth. Zero terminal penalty does not eliminate the opportunity cost of lost
future reward; it does not establish that an attack-free route was impossible.
No intermediate cost levels and no miss penalty are enabled.

The general input expands from 139 to 140; spatial input remains 16x85x49.
The new first-layer weight column and its Adam moments are zero-initialized;
all old weights, optimizer moments, and global step are preserved. The
extractor now gets its dimension from the observation space, allowing old and
new checkpoints to coexist. Model schema is 2; Unity wire protocol stays V2.
Permission mode is opt-in with `PPO_ATTACK_PERMISSION=1` and protected by the
run manifest. Existing default launch behavior remains the shadow contract.

The checkpoint preparation script checks allowed-mode action probabilities
and values on eight reproducible synthetic observations, exact forbidden
attack probability zero, sampling/evaluation log-probability agreement, and
finite masked backpropagation. The recorded maximum allowed probability
error was 0.0. These are compatibility checks, not live behavioral evidence.
The report is `surgery_validation.json` in the new run. Completion statistics
start fresh because permission mixing changes the task distribution; compare
allowed and forbidden episodes separately using trace permission fields.

### PPO_6 restricted-attempt timeout correction (2026-09-05)

The first live forbidden episode exposed an unsuitable inherited timeout.
At decision 6,144 it oscillated near x=477--480, y=910--912; changing tile
coordinates and the relative block map reset the strict inactivity counter.
It had gone 6,103 decisions without a new integer best depth. The old
10,000-decision continuous-depth watchdog allowed excessively long attempts.

Attack-forbidden episodes now terminate after 150 decisions (30 simulated
seconds) without a new integer BestLayerDepth, reason
`attack_forbidden_no_depth`, with zero added terminal penalty. Horizontal
movement, map changes, and sub-tile progress cannot reset this counter.
Successful completion takes precedence. Allowed episodes retain all previous
watchdogs. This is a planner handback heuristic, not proof of an impossible
route. The threshold deliberately gives lateral exploration a finite budget.
Prior manifests are preserved as `*_before_short_timeout.json`; prior traces
and events are retained. The running Python process requires restart to apply
this change. At preparation time the only saved checkpoint in this run was
the surgically expanded 51,843,364 source.

### Frozen attack-removal diagnostic prepared (2026-09-05)

`PPO_Python/frozen_attack_ablation.json` evaluates the original unmodified
51,843,364 checkpoint on ten fixed world seeds in two conditions: attacks
allowed and attacks blocked at execution. Both conditions sample the full
original stochastic action distribution with the same per-seed RNG seed;
blocking occurs after sampling. Trial order alternates by seed. No weights
or optimizer state change. Inference runs on CPU at the existing 10x game
speed. The original observation schema and ordinary watchdogs apply equally
to both conditions; the new restricted 30-second rule is not enabled.
Each attempt is capped at 10,000 decisions. A cap without a terminal is a
censored result, not a completion. This test measures the frozen policy's
dependence on attacking; it cannot prove an attack-free route is impossible.

Per-step scalar/geometry traces retain evaluation condition, seed, RNG seed,
proposed attack, executed action, and contact telemetry for divergence
analysis. Completed trials append to a separate results CSV and are skipped
on resume. Training artifacts are preserved. The first two seeds come from
recent observed episodes; the set is fixed for reproducibility rather than
claimed as an independently randomized benchmark.

### Completed frozen matched attack-removal test

All 20 trials finished. Exact results are in
`PPO_Python/evaluation_results/frozen_attack_ablation_51843364.csv`; scalar
and geometry records share timestamp `2026-09-05_15-34-28`.
Allowed attacks completed 9/10 worlds, with median final progress 100% and
mean final progress 94.5194%. Blocked attacks completed 0/10, with median
final progress 8.1186% and mean 13.4612%. Median attempt lengths were 1,301
and 193 decisions respectively; the latter is shortened by failure and is
not evidence of efficiency. Eight blocked trials ended via capable inactivity;
two reached the 10,000-decision evaluation cap (censored). No blocked trial
executed a swing. Best blocked progress was 39.6106%. Summed trial wall time
was 12.8868 minutes, excluding setup/reset overhead.

Every paired seed reached greater final progress with attacks allowed. This
establishes substantial dependence of this frozen policy on attacking under
these conditions. It does not establish unavoidable excavation, route
optimality, or inability to learn a different movement strategy. The two
conditions share world seeds and action RNG seeds, but their states and
resulting policy decisions diverge after intervention.

### Alternating permission curriculum

The next experiment is isolated in
`checkpoints/CB1_PPO6_alternating_permission_51843364`. It starts from the
zero-column expanded 51,843,364 model, preserving previous experiments.
Every episode starts allowed, then alternates allowed/forbidden every 150
decisions (30 simulated seconds). Permission for the next action is supplied
in the returned observation. Old-phase reward, watchdogs and trace are
processed before switching. A forbidden phase with no new integer best depth
for all 150 decisions ends with zero added penalty before attacks return.
Its counter starts fresh on phase entry and resets on new best tile depth.
The capable-inactivity watchdog remains active at 75 decisions; allowed-mode
watchdogs and penalties remain unchanged.

Traces record phase index, phase steps, accumulated new-depth progress,
near-zero-velocity steps, no-depth counter, and phase completion. Analyze
forbidden-phase progress and stationary fraction to distinguish learning from
waiting for attack availability. Episode-level completion alone is inadequate.
The input still exposes current permission only, not phase time remaining;
the deterministic schedule therefore introduces hidden temporal context.
No additional model surgery, reward term, or Unity wire change is made.
Activation requires both `PPO_ATTACK_PERMISSION=1` and `PPO_ATTACK_PHASES=1`.

### Alternating curriculum penalty revision before launch

The user explicitly revised the zero-cost handback rule: all watchdog
failures in the alternating curriculum now apply the existing -10 incomplete
episode penalty, including forbidden-phase no-depth and capable-inactivity
failures. The earlier zero-penalty description above records the initial
proposal and is superseded for this experiment. Completion still takes
precedence and retains its bonus. Termination still occurs before the
permission switch, with no value bootstrapping. There is no per-swing or
per-step penalty. This is intended to discourage waiting through restrictions;
some states may nevertheless require excavation and make failure unavoidable.
The older full-episode permission experiment retains its original semantics.
Both manifests for the separate alternating run record the new -10 value.

### Masking versus punishment lesson

The alternating experiment uses a hard action mask during forbidden phases.
This removes attack-on from the policy distribution before sampling and PPO
evaluation. Consequently, the learner never chooses a forbidden attack and
receives no direct negative credit for preferring it; it only learns from the
rewards produced by the remaining movement actions and from eventual watchdog
failure. This is behaviorally different from leaving attack available and
penalizing its execution. A punishment formulation would generate explicit
counterexamples in which selecting attack causes negative advantage, while a
mask expresses an external capability constraint and asks the conditional
policy to optimize only among legal actions. Future attack-control experiments
must choose between these semantics deliberately rather than treating them as
equivalent.

### Alternating permission outcome and archive (2026-09-06)

The alternating curriculum was stopped and rejected after approximately
991,511 decisions beyond its 51,843,364 source (about global step 52,834,875).
It produced 14 successes in 2,984 completed episodes (0.469%). No success
occurred in the final 500 episodes examined. In matched 100-phase windows,
forbidden stationary fraction rose from 60.5% initially to 79.4% at the end;
median forbidden progress remained effectively zero (0.2% in the final
window), and mean forbidden progress changed from 3.42% to 3.00%. Allowed
mean phase progress also declined from 13.72% to 10.59%. The final 100
episodes contained 84 capable-inactivity and 16 forbidden-no-depth failures.

The optimizer remained active: latest entropy loss was -1.1155, approximate
KL 0.0711, clip fraction 0.1797, and explained variance 0.7581. There were no
forbidden swings or synchronization violations. Thus the negative result is
not explained by a failed mask, collapsed optimizer, or timing error. Under
this objective and exposure scheme, the policy increasingly waited or became
pinned during forbidden phases and eventually lost completion reliability.
This supports rejecting the curriculum; it does not prove that conditional
attack control is unlearnable under a punishment or different task design.

The complete rejected run is preserved externally at
`Reinforcement-Unknown Archives/CB1_2026-09-06_ppo6_rejected_alternating_permission_51843364_to_52834875`.
That archive contains the full run directory (19 post-source checkpoints,
manifests, validation report, success state and PPO_6 events) plus both CSV
and compressed geometry traces. Its size at archival was about 5.5 GB.
The workspace contains no TensorBoard PPO_6 directory after archival. The
accepted source remains the original PPO_5 checkpoint at 51,843,364 with
SHA-256 `136399631a8cc9e79c3ae98a94bd75bbb3942bd10f9f4bad42730676ddd0b2be`.

### PPO_6 alternating attack-cost experiment (prepared 2026-09-06)

The next PPO_6 experiment implements the user's preferred punishment
semantics and begins independently from the accepted 51,843,364 PPO_5 source.
Its run directory is
`checkpoints/CB1_PPO6_alternating_attack_penalty_51843364`. It inherits no
optimizer updates from any earlier PPO_6 experiment.

The appended scalar is now attack cost: 0 means free and 1 means costly.
Every episode starts free and alternates between cost 0 and cost 1 every 150
decisions (30 simulated seconds). Attack remains present in the categorical
distribution and executable in both phases; there is no policy mask or
environment clamp. Each swing actually accepted by the game during cost 1
adds `-0.001` reward. Held attack during cooldown is not charged because no
swing executed. Contact and usefulness do not affect the price. Cost-0
behavior retains the original progress/completion reward exactly.

The 150-decision costly-phase no-new-integer-depth watchdog and every existing
watchdog apply the standard -10 incomplete-episode penalty. Termination is
processed before a scheduled return to the free phase. This creates direct
negative credit for both costly swings and failure to navigate; in terrain
that truly requires excavation, the learner must trade swing costs against
progress and avoiding failure. There is no miss penalty.

Model surgery expands the general input from 139 to 140, zero-initializing
the added column in the shared, actor and critic general encoders and the
corresponding Adam moments. All existing parameters, other optimizer state
and the global step remain intact. On eight reproducible synthetic
observations per cost value, both cost 0 and cost 1 initially reproduced the
PPO_5 action probabilities and values exactly (maximum recorded error 0.0).
The saved policy is the ordinary unmasked SB3 policy. Details are preserved
in the new run's `surgery_validation.json`; both manifests pin the semantics.

Traces record current cost, executed swings, the exact execution penalty,
phase progress, stationary steps, phase boundary, proposed action and actual
action. The principal learning evidence is whether swing rate separates
between cost-0 and cost-1 phases while progress and completion recover. A
falling global attack rate alone is insufficient. This binary experiment is
the precursor to a possible continuous planner-selected cost in [0, 1].

### PPO_6 per-swing `-0.001` outcome and archive (2026-09-06)

The alternating per-swing experiment was stopped and rejected after about
917,541 post-source decisions (approximately global step 52,760,905). Across
991 completed episodes it produced 607 successes. The final 100 contained 61
successes, 36 costly-phase no-depth failures and 3 capable-inactivity failures.
It avoided the catastrophic waiting behavior caused by hard masking, but did
not learn attack-cost conditioning: in the final 50,000-decision window the
free and costly executed-swing rates were 0.9071 and 0.9061 per decision, and
attack was commanded on 99.46% and 99.55% of decisions respectively.

The critic was healthy at the final inspection (explained variance 0.9259),
with entropy loss -0.8049, approximate KL 0.0605 and clip fraction 0.1463.
Penalty accounting and 10-tick synchronization had zero observed violations.
A typical 150-decision costly phase charged only about -0.136 while abstaining
could lead to a -10 watchdog failure and loss of progress/completion reward.
Continuing to attack was therefore compatible with the actual objective; the
failure is attributed to insufficient and internally dominated incentive,
not an inability of PPO to observe the cost scalar.

The complete run and traces are preserved externally at
`Reinforcement-Unknown Archives/CB1_2026-09-06_ppo6_rejected_attack_penalty_001_51843364_to_52760905`
(about 5.1 GB at archival). No checkpoint from it enters the next lineage.

### PPO_6 terminal prohibited-swing experiment (prepared 2026-09-06)

The replacement run is
`checkpoints/CB1_PPO6_alternating_attack_violation_51843364`, again derived
directly from the accepted 51,843,364 PPO_5 checkpoint. It retains 150-decision
alternating phases. Scalar 0 means attack allowed; scalar 1 means attack
prohibited. Attack remains in the sampled policy distribution and is sent to
Unity. When the game accepts an actual swing during scalar 1, the episode
terminates immediately with reason `attack_prohibited_swing` and receives the
standard -10 incomplete-episode penalty. Merely holding attack during cooldown
does not trigger the rule. There is no per-swing or miss penalty.

The prohibited-phase 150-decision no-new-best-depth watchdog remains and also
applies -10, as do all existing watchdog failures. Completion takes precedence
if it occurs on the same macrostep. This contract provides PPO with explicit
negative samples for choosing attack while prohibited, while preserving the
action as a learnable choice. Initially the inherited policy will almost
certainly terminate on its first prohibited-phase swing; reduced violation
frequency and longer prohibited phases are the primary adaptation measures.

The 139-to-140 input surgery again zero-initializes the new column and matching
Adam moments while retaining all existing weights and optimizer state. Eight
reproducible observations at each scalar value reproduced PPO_5 action
probabilities and values exactly (maximum error 0.0); the policy contains no
mask. Unit tests verify execution reaches Unity and the accepted-swing terminal
transition produces exactly the recorded -10 failure term. Manifests and
`surgery_validation.json` pin the new semantics.

### Terminal-violation saturation result and exploration repair (2026-09-06)

The first terminal prohibited-swing run was stopped after approximately
156,558 decisions (about global step 51,999,922). It produced 1,037 completed
episodes and zero successes. All 100 of the first, middle and last sampled
prohibited-phase windows terminated from `attack_prohibited_swing`; their
median duration was one decision, and none survived a complete prohibited
phase. The prohibited attack command rate remained 100%. The mechanism was
correct (zero timing or violation-classification discrepancies), while PPO
metrics remained active and explained variance was 0.6264 at final inspection.

The inherited attack distribution was too saturated to provide appreciable
no-attack counterexamples. Repeated terminal punishment therefore taught the
critic that prohibited-phase entry predicted failure without giving the actor
enough alternative-action experience. The run is rejected and fully archived
at `Reinforcement-Unknown Archives/CB1_2026-09-06_ppo6_rejected_terminal_attack_saturation_51843364_to_51999922`
(about 1.1 GB including checkpoints, events and both trace files).

The repaired experiment is separately prepared at
`checkpoints/CB1_PPO6_terminal_attack_exploration_51843364`, again from the
original 51,843,364 PPO_5 model. It retains the same observable binary cost,
150-decision phases, selectable attack action and immediate -10 accepted-swing
violation. Only the two attack logits are divided by temperature during
prohibited phases. Initial temperature is 0.1 and increases linearly to the
ordinary temperature 1.0 over 500,000 post-source decisions. The identical
transformed distribution is used for rollout sampling, likelihood evaluation
and entropy, preserving on-policy PPO accounting. Allowed phases are not
transformed.

This is targeted behavior-policy exploration, not masking: both attack values
remain possible and a sampled, accepted attack still produces the terminal
punishment. On 64 reproducible synthetic observations, initial allowed-mode
outputs remained identical to PPO_5; prohibited attack probability averaged
0.6476 (range 0.3233--0.8337) at temperature 0.1 instead of the inherited
allowed distribution's mean 0.9012 (range 0.0006--approximately 1.0). These
synthetic probabilities validate the mechanism rather than predict live state
frequencies. Temperature is saved with the policy, reconstructed from global
step on resume, and logged as `train/prohibited_attack_temperature`.

The experimental question is whether the actor learns raw cost-conditioned
abstinence before external temperature reaches 1. Track violation latency,
complete prohibited phases and raw behavior as temperature rises. If behavior
relapses with annealing, the learned policy did not internalize the rule.

### Fixed-duration exploration result and adaptive curriculum (2026-09-07)

The temperature-softened but fixed-150 experiment was stopped after about
189,594 decisions (approximately global step 52,032,958). Across 1,255
episodes it produced zero completions. Every sampled first, middle and final
200 prohibited-phase window ended through `attack_prohibited_swing`; median
duration remained one decision and no prohibited phase reached 150 decisions.
Initially softened prohibited attack commands occurred on 75.43% of sampled
steps, but climbed to 99.26% as the global-step temperature schedule reached
0.4355. Final explained variance was 0.6026 with active KL/clipping and zero
timing or violation discrepancies. The schedule restored saturation before
the actor learned abstinence, while the 150-consecutive-choice requirement
provided effectively no successful restricted trajectories.

The rejected run is fully preserved at
`Reinforcement-Unknown Archives/CB1_2026-09-07_ppo6_rejected_fixed150_exploration_51843364_to_52032958`
(about 1.1 GB with checkpoints, events and traces). No model from it enters
the replacement lineage.

The replacement is prepared independently at
`checkpoints/CB1_PPO6_adaptive_attack_curriculum_51843364`. It starts directly
from PPO_5 step 51,843,364 with the same verified zero-column surgery and
terminal prohibited-swing contract. Allowed phases remain 150 decisions.
Prohibited duration begins at one decision, so choosing no attack immediately
returns to allowed play without penalty, whereas an accepted swing terminates
with -10. Short curriculum phases do not require depth progress; the ordinary
watchdogs remain active. The prohibited no-depth failure applies when the
curriculum eventually reaches the full 150-decision duration.

Curriculum durations are `1, 2, 4, 8, 16, 32, 64, 100, 150`; prohibited attack
temperatures are `0.1, 0.25, 0.5, 1.0`. Each stage evaluates the latest 100
prohibited phases. At least 80% survival promotes temperature; after surviving
at temperature 1.0, duration advances and temperature returns to 0.1. Survival
below 40% demotes an advanced stage. Intermediate rates retain the stage with
a rolling window. Thus exploration does not anneal merely because unrelated
allowed decisions elapsed. Completion during a prohibited phase counts as
survival; another watchdog failure does not.

State is atomically persisted in `attack_curriculum_state.json`, including
level, recent outcomes, promotions and demotions, so restart cannot silently
reset difficulty. The policy's applied temperature is synchronized after PPO
updates and reconstructed from state before resumed rollout collection.
Traces retain current and next duration/temperature, applied temperature,
survival and promotion events; TensorBoard retains the applied temperature.
The experiment succeeds only if it advances stages while retaining completion
reliability, and ultimately abstains at ordinary temperature 1.0.

### Adaptive attack-curriculum outcome and archive (2026-09-07)

The adaptive terminal-punishment experiment was stopped and rejected at
approximately global step 52,264,130, after about 420,766 decisions and 2,102
terminal episodes without a completion. It generated 2,746 prohibited phases,
but none promoted beyond the initial one-decision duration and temperature
0.1. Survival was 33% in the first 100 phases, 24% in a middle sample and 26%
in the final 100; prohibited attack commands remained roughly 71--78%. Of the
final 100 prohibited phases, 91 ended in an attack violation and nine through
inactivity. The final recorded training sample still showed active learning
(entropy loss -1.0241, approximate KL 0.0574, clip fraction 0.1613, value loss
2.4475 and explained variance 0.3125), but the curriculum produced no usable
restricted behavior. Timing and requested/executed-action checks found no
mechanism violation.

The complete rejected lineage, TensorBoard data and trace/grid sidecars were
moved to
`Reinforcement-Unknown Archives/CB1_2026-09-07_ppo6_rejected_adaptive_attack_curriculum_51843364_to_52264130`
(approximately 2.5 GB). No checkpoint from that experiment enters the next
lineage.

### PPO_6 fixed missed-attack penalty (prepared 2026-09-07)

PPO_6 is restarted once more from the accepted PPO_5 checkpoint at exactly
51,843,364 steps. The policy remains the original schema-v1 model with its
139-value general observation, seven action components and no model surgery.
Attack permission, action masking/clamping, alternating phases, exploration
temperature and curriculum logic are all disabled.

The intervention applies `-0.005` for each game-confirmed executed attack swing
when accumulated claw-health loss for that policy macrostep is zero. A swing
with any recorded claw-health loss receives no attack penalty. Merely holding
attack during cooldown (zero executed swings) also receives no penalty. The
existing best-depth progress reward, completion bonus and `-10` incomplete
watchdog penalty remain unchanged.

This is intentionally an outcome proxy, not a reachability oracle. It can
penalize anticipatory swings that miss before later contact, and contact with a
strategically useless or indestructible metal block is exempt because claw
health changed. If multiple swings were ever aggregated into one macrostep,
any contact would conservatively exempt all of them; observed telemetry so far
has shown at most one executed swing per macrostep. The experiment tests
whether a small dense cost can desaturate the inherited near-always-attack
distribution enough for PPO to sample and retain useful abstinence without
dictating where excavation is desirable.

### `-0.005` missed-attack outcome and stronger retry (2026-09-07)

The `-0.005` run was stopped at its complete 52,150,564-step checkpoint and
rejected after roughly 307,200 decisions (150 PPO rollouts). The trace contained
222 completed episodes: 210 successes and 12 `capable_inactivity` failures, for
94.6% reliability. Successful episodes had mean 1,398.2, median 1,186, minimum
498 and 95th percentile approximately 2,925 decisions. Performance therefore
remained strong, but the intended attack correction did not emerge.

Across approximately 303,643 analyzed trace decisions, attack was commanded on
99.45% of decisions. There were 275,090 executed swings and 127,196 classified
misses (46.24%). Comparing completed-episode windows, attack-command frequency
rose from 99.27% in the first 100 to 99.64% in the last 100, while misses per
executed swing rose from 44.53% to 48.70%. The agent was paying an average
missed-attack cost of approximately `-0.00209` per decision without reducing
the saturated behavior. Recent explained variance was healthy (last-20 mean
approximately 0.805), so the absence of correction was not explained by an
obviously failed critic.

The rejected run, events, checkpoints and trace/grid sidecars are preserved at
`Reinforcement-Unknown Archives/CB1_2026-09-07_ppo6_rejected_missed_attack_minus0.005_51843364_to_52150564`
(approximately 1.9 GB). No trained checkpoint from it enters the retry.

The retry again starts from the byte-identical accepted PPO_5 checkpoint at
51,843,364 steps and changes only the missed executed-swing penalty from
`-0.005` to `-0.05`. All classification semantics and limitations above remain
the same. This produces a deliberately unmistakable signal: at the rejected
run's miss frequency it would cost roughly 21 reward per 1,000 decisions. The
first evaluation gate is 50,000--100,000 new decisions; evidence of success is
a decline in executed-miss and attack-command rates rather than reward alone.

### `-0.05` missed-attack outcome and archive (2026-09-07)

The stronger missed-attack experiment was stopped at its complete
52,560,164-step checkpoint after approximately 718,858 trace decisions. It
proved that the outcome signal was learnable but not that it induced
abstinence. Misses per executed swing declined from 46.21% in the first 50,000
decisions to 33.20% in the final 50,000, while attack commands increased from
99.12% to 99.83% and executed swings remained approximately 90.76% of all
decisions. Timmy learned to turn misses into registered contacts instead of
learning when not to swing.

The last 100 completed episodes had 94% reliability, median length 1,324, mean
1,892.6, 95th percentile approximately 5,222.5 and maximum 11,206 decisions.
The preceding 100 had 96% reliability, median 1,296, mean 1,462.7 and 95th
percentile approximately 2,669.9. Thus the success-time tail materially
worsened even as the literal miss metric improved. The critic remained
functional (latest explained variance approximately 0.941 and value loss
0.138), so critic collapse does not explain the result.

The complete rejected lineage is preserved at
`Reinforcement-Unknown Archives/CB1_2026-09-07_ppo6_rejected_missed_attack_minus0.05_51843364_to_52560164`
(approximately 4.1 GB). Its terminal source copy retains the accepted PPO_5
SHA-256 `136399631a8cc9e79c3ae98a94bd75bbb3942bd10f9f4bad42730676ddd0b2be`.
No learned model from the rejected run enters the next experiment.

### PPO_6 annealed attack-logit prior (prepared 2026-09-07)

The next experiment again begins from accepted PPO_5 step 51,843,364, without
model surgery or observation/action changes. It retains the `-0.05` executed
miss penalty and adds an optimizer-only binary cross-entropy prior on the
attack categorical branch. The target marginal attack probability is 0.75.
The prior coefficient begins at 0.1 and decreases linearly to zero over one
million new decisions.

Actions are neither masked nor overwritten, and rollout sampling uses the
ordinary policy distribution whose log probability PPO records. The added
loss therefore changes the learned logits directly while keeping PPO's sampled
actions and likelihood ratios internally consistent. Its purpose is to open a
temporary non-attack sampling channel; the state-dependent PPO advantage and
miss reward must determine where attacks remain useful. TensorBoard records
`train/attack_probability`, `train/attack_prior_loss`, and
`train/attack_prior_coefficient`. A falling sampled attack rate is only a
mechanism check; the experiment succeeds only if selective behavior survives
annealing without destroying completion reliability or the episode-length
tail.

**First-start memory correction:** The first launch collected one 2,048-step
rollout but failed before its first optimizer step or checkpoint. The initial
implementation computed the regularizer through `policy.get_distribution`
after PPO had already evaluated the same minibatch, creating a second complete
85x49 CNN activation graph. CUDA failed on an additional 132 MiB allocation
with only approximately 131.6 MiB free. This was an implementation-memory
error, not a policy result.

The optimizer now derives values, action log probabilities, entropy and attack
probability from one shared feature-extractor pass, matching standard PPO's
forward structure and adding only the small categorical BCE graph. A direct
forward/backward validation confirmed exactly one feature-extractor call,
finite gradients, an inherited attack probability of approximately 0.9961 on
the synthetic validation batch and the intended downward attack-logit gradient.
All 42 tests pass. The failed startup's tiny event file and 2,048-step trace
were archived separately at
`Reinforcement-Unknown Archives/CB1_2026-09-07_ppo6_attack_prior_failed_startup_oom`;
the accepted experiment still starts from 51,843,364 with clean TensorBoard and
no optimizer update from the failed launch.

### Attack-logit-prior outcome and archive (2026-09-08)

This experiment decisively demonstrated that the inherited saturated attack
head was recoverable. Sampled attack commands fell from 93.08% in the first
50,000 decisions to 73.43% around 250,000, 66.31% around 750,000, 36.66% in a
post-anneal sample and 21.07% in the final analyzed 50,000. TensorBoard's mean
policy attack probability likewise fell from a first-20-update mean of 0.9426
to a last-20 mean of 0.2224 (latest analyzed value 0.1680). Misses per executed
swing fell from 46.26% to 21.37%.

The prior coefficient reached zero after one million decisions, yet attack
probability continued downward. Thus the temporary optimizer prior succeeded
at opening non-attack sampling, after which the retained `-0.05` miss penalty
continued selecting abstinence. The result overshot useful selectivity. In the
latest analyzed 100 completed episodes, reliability fell to 85%, median length
rose to 2,671.5, mean to 3,471.6, 95th percentile to approximately 9,828 and
maximum to 18,558 decisions. Twelve ended through `capable_inactivity` and
three through `mobile_stagnation`. Recent mean explained variance fell to
approximately 0.044 during the distributional transition, although the latest
sample had recovered to 0.338.

The causal conclusion is narrow but strong: an attack-branch BCE prior can
recover sampling from a nearly deterministic inherited categorical head.
This run does not establish a good final attack rate, because the simultaneously
active miss penalty confounds the post-prior dynamics and drove continued
abstinence after annealing. Any retry should isolate or sharply weaken that
reward pressure and treat completion reliability and episode-length tails as
constraints, not optimize marginal attack rate alone.

Training was stopped at the complete 53,174,564-step checkpoint. The complete
7.4 GB lineage (27 checkpoints, TensorBoard, success state, CSV trace and grid
sidecar) is preserved at
`Reinforcement-Unknown Archives/CB1_2026-09-08_ppo6_attack_logit_prior_successful_mechanism_overabstinence_51843364_to_53174564`.
The active workspace contains no PPO_6 experiment directory after archival.

### PPO_6 isolated attack-logit-prior ablation (prepared 2026-09-08)

The next run isolates the mechanism that successfully reopened the saturated
attack head. It starts again from accepted PPO_5 step 51,843,364 and retains
the same attack-branch BCE target (0.75), initial coefficient (0.1), and linear
one-million-decision anneal. The `-0.05` missed-swing penalty is removed
entirely; missed swings, contact swings and non-attack decisions receive no
attack-specific reward or punishment. Original depth progress, completion and
watchdog rewards remain unchanged.

This ablation asks whether temporarily supplying approximately 25% non-attack
sampling lets ordinary state-dependent descent advantages discover useful
restraint. The key result is behavior after the coefficient reaches zero. A
return to near-100% attack means the task return prefers or cannot distinguish
the inherited saturation. Persistent context-dependent restraint with healthy
reliability and episode-length tails would show that the prior supplied
exploration while PPO learned a useful policy. The run is independent of all
previous PPO_6 optimizer updates.

### Isolated attack-logit-prior outcome and archive (2026-09-08)

The isolated run was stopped at its complete 53,686,564-step checkpoint after
1,843,200 new decisions. The prior reached zero at step 52,843,364, leaving
approximately 843,200 decisions of ordinary PPO training with no attack prior
and no missed-swing penalty. The mechanism again worked: attack probability
fell into the intended approximately 75% region while the coefficient was
active. After annealing, the natural task gradient pulled it upward and it
settled near 96% over the final 100 optimizer updates rather than returning to
the inherited approximately 99.5--99.8% saturation.

The final sampled 50,000-decision window commanded attack on approximately
96.04% of decisions and executed a swing on 88.86%; 47.36% of executed swings
had no recorded claw-health contact. Attack-specific reward remained exactly
zero. Thus the intervention left roughly 4% non-attack sampling, several times
the inherited rate, but did not preserve the much broader 20--25% abstinence
created while the prior was active.

Task performance recovered as the policy moved back toward frequent attacking.
The final 100 completed episodes had 95% reliability, median length 1,179.5,
mean 1,373.1, 90th percentile approximately 1,876.6 and 95th percentile
approximately 2,166.4, with one isolated 12,719-step success. Recent
TensorBoard means were approximately 1,374.7 for all episode lengths and
1,388.4 for successful lengths. The final training region was numerically
healthy: last-20 explained variance averaged approximately 0.857, approximate
KL 0.063 and clip fraction 0.117.

The isolated causal result is that the regularizer can reopen the saturated
head without reward shaping, but ordinary descent returns substantially prefer
frequent attack and only retained a modest persistent shift after removal.
The run did not clearly outperform accepted PPO_5, so it is preserved as an
experimental branch rather than promoted. The full 11 GB lineage (37
checkpoints, TensorBoard, success state, CSV trace and grid sidecar) is archived
at
`Reinforcement-Unknown Archives/CB1_2026-09-08_ppo6_isolated_attack_prior_51843364_to_53686564`.
The active workspace contains no PPO_6 experiment directory.

### PPO_6 budget-conditioned resource-efficient descent (prepared 2026-09-08)

This intervention targets one missing navhead capability directly: completing
navigation while respecting an externally assigned resource allowance. Because
functional godmode restores stamina before policy observations, cumulative
stamina expenditure is not presently a reliable observable training signal.
The contract therefore uses the existing game-confirmed executed-swing counter
as a precise excavation-resource unit; it must not be described as measured
physiological stamina consumption.

PPO_5-like trace calibration used the first 100 successful episodes of the
`-0.005` lineage before its behavior materially shifted. Executed swings had
mean 1,297.2, median 1,080.5, 75th percentile 1,427.25, 90th percentile
2,097.4, 95th percentile 2,719.95 and range 533--7,253. Initial episode budgets
are uniformly sampled as integers from 1,500 through 3,000. This begins with a
meaningful but generally achievable constraint spanning approximately the
historical 75th through 96th percentiles rather than immediately demanding an
unproven minimum.

One scalar is appended to the general observation: remaining executed swings
divided by 3,000. The initial value therefore lies from 0.5 through 1.0 and
decreases only when the game confirms a swing actually executed. Using exactly
the assigned budget is legal; the first swing beyond it terminates the episode
as `attack_budget_exceeded` and applies the standard `-10` incomplete-task
penalty. Completion beyond budget is a failure and its completion bonus is
removed. Unused budget earns no reward, missed swings carry no special penalty,
and ordinary depth progress and valid completion rewards are unchanged.

The temporary isolated attack-logit prior is reused solely to open non-attack
sampling: target 0.75, coefficient 0.1, linear anneal to zero over one million
decisions. The product hypothesis is not a target marginal rate. It is that
PPO will use the observable remaining allowance and ordinary completion return
to preserve attacks for obstructions worth clearing.

TensorBoard additionally records rolling 100-episode budget utilization,
budget-exceeded rate and successful-episode budget utilization. The full trace
records assigned, used and remaining budget plus the exact violation flag and
terminal reason on every decision.

The new run is `checkpoints/CB1_PPO6_resource_budget_51843364`, derived
non-destructively from accepted PPO_5 step 51,843,364. Zero-column expansion
changes the general input from 139 to 140 while preserving all existing actor
and critic outputs exactly at every tested budget value. Twenty-four synthetic
observations produced maximum action-probability and value errors of 0.0;
three serialized encoder weight aliases and two optimizer tensors were
expanded, with optimizer moments retained. Details are recorded in
`resource_budget_surgery.json`. This is model schema 2; it cannot be confused
with or resumed as schema-1 PPO_5.

### Resource-budget outcome and archive (2026-09-09)

The budget-conditioned run was stopped after 1,606,695 traced decisions
(approximately global step 53,450,059); the last complete saved optimizer
checkpoint is 53,430,564. The attack prior had been zero for roughly 587,000
decisions by the final analysis. Sampled attack had risen to 89.19% in the
latest 50,000 decisions and recent learned attack probability averaged about
91.66% over 100 optimizer updates.

The intervention did not produce measurable conditioning on the supplied
budget. Across 300 recent successful episodes, Pearson correlation between
assigned budget and swings used was only 0.0977, while correlation between
budget and per-step attack rate was -0.0263. In an ordinary least-squares check
controlling for episode length, the standardized budget coefficient on swings
used was -0.00193, effectively zero; episode length explained almost all
resource consumption. Timmy spent approximately 1,000--1,130 swings across all
four budget quartiles instead of conserving more under tighter allowances.

Consequently, tight budgets primarily produced failures. Reliability declined
from 89% in the preceding 100 completed episodes to 82% in the last 100.
Budget-exceeded failures increased from five to eleven, alongside seven
`capable_inactivity` failures. Rolling budget-exceeded rate averaged 9.25% over
the final 20 updates (latest 10%), versus 7.86% during the first 20. Successful
episode speed did improve to a recent mean near 1,244 decisions, but that is
not evidence of budget use because the randomized allowance had effectively
zero controlled influence on spending.

The causal result is that an observable hard budget plus sparse terminal
failure was insufficient to teach budget-conditioned excavation from this
initial policy. The temporary logit prior again changed the marginal action
distribution, but after its removal Timmy reverted toward frequent attack and
allowed low-budget episodes to fail rather than adapting expenditure. The run
is rejected as a product baseline.

The full 8.9 GB record (32 checkpoints, TensorBoard, manifests and surgery
report, success state, CSV trace and grid sidecar) is archived at
`Reinforcement-Unknown Archives/CB1_2026-09-09_ppo6_resource_budget_rejected_51843364_to_53430564_trace_to_53450059`.
The directory name distinguishes the last complete checkpoint from the later
uncheckpointed trace tail. No active PPO_6 experiment directory remains.

### PPO_6 dense resource-budget feedback (prepared 2026-09-09)

The hard-budget result showed that sparse terminal punishment did not make
resource use depend on the allowance. This retry keeps the same independently
sampled 1,500--3,000 executed-swing budgets, remaining-budget observation,
hard ceiling and temporary attack-logit prior, but adds immediate resource
feedback for every game-confirmed swing.

Each executed swing receives reward `-10 / assigned_budget`. Spending the
entire allowance therefore accumulates exactly `-10` under every budget:
approximately -0.006667 per swing at budget 1,500, -0.005 at budget 2,000 and
-0.003333 at budget 3,000. Smaller allowances make resource use immediately
more expensive without changing the total cost represented by exhausting the
budget. Contact, misses and material type are irrelevant because all consume
the same measured resource unit. The first swing beyond the allowance still
terminates as `attack_budget_exceeded` with the standard incomplete-task
penalty; unused allowance has no terminal bonus.

This combination tests a mechanism unavailable to the earlier immediate-swing
experiment: the proven logit regularizer supplies genuine non-attack samples
while the dense, budget-scaled cost supplies immediate blame. Success requires
resource use to respond to the randomized allowance after controlling for
episode length, not merely a lower global attack rate.

The fresh run is
`checkpoints/CB1_PPO6_dense_resource_budget_51843364`, created independently
from accepted PPO_5 step 51,843,364. The same zero-column 139-to-140 expansion
preserved all tested initial action probabilities and values exactly (maximum
error 0.0 across 24 observations/budget values), with optimizer moments
retained. No checkpoint from the rejected sparse-budget branch enters it.

### Dense resource-budget outcome and archive (2026-09-11)

The dense-budget retry was stopped after approximately 4.53 million traced
decisions. Its last complete checkpoint is 56,348,964 and its trace continues
to approximately 56,377,726. The mechanism improved the measured constraint
outcomes: late budget-exceeded rates were commonly near 5%, successful runs
used roughly 46% of their assigned allowance, successful episode lengths were
near 1,150 decisions, and the critic was numerically healthy. However, after
the one-million-decision prior reached zero, learned attack probability rose
back to approximately 95%. Timmy therefore satisfied the aggregate allowance
mainly by completing quickly while continuing to attack almost constantly; it
did not learn the intended selective abstinence. The experiment is rejected
for that product objective despite its real speed and budget-compliance gains.

The complete 25 GB lineage, TensorBoard record, scalar trace and geometry
sidecar is preserved non-destructively at
`Reinforcement-Unknown Archives/CB1_2026-09-11_ppo6_dense_resource_budget_51843364_to_56348964_trace_to_56377726`.

### PPO_6 natural exhaustion plus 25% attack prior (prepared 2026-09-11)

The next experiment again starts independently from accepted PPO_5 checkpoint
51,843,364 (SHA-256
`136399631a8cc9e79c3ae98a94bd75bbb3942bd10f9f4bad42730676ddd0b2be`).
It does not inherit a dense-budget checkpoint, append an observation, or use
model surgery. The copied source checkpoint is byte-identical.

Functional godmode now leaves stamina to the game's ordinary depletion and
recovery mechanics while continuing to restore energy and every other CB1
health invariant. Stamina was already part of the 139-scalar observation, so
the policy contract has no dimensional change. There is no synthetic swing
cost, missed-attack penalty, attack budget, masking, or action override.

To prevent the inherited near-constant attacker from filling rollouts with
unactionable ragdoll physics, 75 consecutive incapacitated decisions terminate
as `incapacitated_inactivity` with the existing `-10` incomplete-task penalty.
At 5 Hz this is 15 simulated seconds. Incapacitated motion, passive falling,
tile changes, depth changes, and block changes do not reset that counter; the
older strict physical-inactivity gates remain unchanged while capable.

An attack-branch-only BCE prior targets attack probability 0.25 with initial
coefficient 0.1, linearly annealing to zero over one million decisions. This
changes optimization only: actions remain sampled from the learned policy.
The hypothesis is that forced exposure to non-attack transitions plus genuine
stamina consequences can move the policy into a selective-use basin which
ordinary PPO retains after the prior disappears. Success requires a materially
sub-saturated attack probability, fewer incapacitation endings, and retained
completion reliability/speed after coefficient zero; merely surviving by
finishing before exhaustion is not sufficient.

The prepared directory is
`checkpoints/CB1_PPO6_natural_exhaustion_prior25_51843364` and records the
complete contract in `protocol_cb1.json`, `intervention_ppo6.json`, and
`natural_exhaustion_preparation.json`.

### Natural-exhaustion 25% prior outcome and 5% retry (2026-09-11)

The 25% run produced 1,377,911 traced decisions and 138 terminal episodes with
zero completions. Terminations comprised 116 `incapacitated_inactivity`, 12
`capable_inactivity`, nine `mobile_stagnation`, and one `max_episode_steps`.
Across the final 100,000 decisions, commanded attack rate was 30.884%, mean
stamina was 1.246/100, stamina was below 10 on 98.405% of observations, and
the player was classified incapacitated on 40.819%. The last 100 optimizer
updates reported mean episode length near 13,625 and mean reward near -8.33;
recent explained variance was severely negative. Although the prior opened
the attack head and remained effective after reaching coefficient zero, 25%
attack was not remotely sustainable under the game's stamina dynamics and
provided PPO no successful example of resource-managed descent.

The rejected 7.5 GB run is archived at
`Reinforcement-Unknown Archives/CB1_2026-09-11_ppo6_rejected_natural_exhaustion_prior25_51843364_to_53174564_trace_to_53221275`.
The last complete checkpoint is 53,174,564; the trace continues to estimated
global step 53,221,275. None of its learned weights enter the retry.

The clean retry again copies accepted PPO_5 step 51,843,364 byte-for-byte and
changes exactly one experimental hyperparameter: the temporary BCE target for
the attack branch is reduced from 0.25 to 0.05. Initial coefficient 0.1,
one-million-decision linear annealing, natural stamina, retained energy/health
godmode, 75-decision incapacitated-inactivity termination, and all PPO/reward/
observation/action settings remain unchanged. The purpose is to sample long
enough attack-free intervals for actual stamina recovery. A zero-like attack
collapse would present as capable-inactivity failures at blockers; continued
overuse would present as near-zero stamina and incapacitated-inactivity; the
desired result is the emergence of successful episodes followed by a
state-dependent learned attack rate after coefficient zero.

### Natural-exhaustion 5% prior outcome and archive (2026-09-11)

The 5% retry was stopped after 1,220,291 traced decisions. Its last complete
checkpoint is 53,020,964 and the trace continues to estimated global step
53,063,655. No successful episode occurred, so TensorBoard never emitted
`rollout/success_ep_len_mean`. Late `ep_len_mean` was approximately 11,697,
`ep_rew_mean` was -8.62, and the critic became meaningless on the nearly
uniform failure distribution (recent explained variance approximately -53).

The intervention did open and then strongly suppress the saturated attack
head. After the prior reached coefficient zero, learned attack probability
was approximately 1.6--1.8%. Nevertheless, a bounded late trace sample found
mean stamina only 0.91/100, stamina below 10 on 99.0% of observations, and
incapacity on 43.4%. Thus sustained attack abstinence was not sufficient to
restore functional stamina or produce descent.

A constant-memory scan of all within-episode stamina transitions identified
the missing confound. Attack commands correlated with mean stamina change
-0.160 per decision versus +0.0046 without attack, but locomotion also carried
large costs: non-neutral movement correlated with -0.0352 versus +0.1476 while
neutral, and jump with -0.0373 versus +0.0292 without jump. The cleanest
observed recovery behavior--no attack, neutral horizontal movement, no jump,
neutral vertical movement--recovered approximately +0.293 stamina per
decision. These are observational conditional means, not isolated causal
action effects, because policy state selection and the near-zero floor
confound them. They are still decisive evidence that attack suppression alone
does not solve resource management.

The environment contract also opposes the required recovery behavior: full
stillness is terminated after 75 decisions (15 simulated seconds), while
recovery near +0.293 per decision can restore only about 22 stamina before
that watchdog fires. Timmy was therefore asked both to rest long enough to
recover and never remain physically inactive for that long. Adding a stamina
penalty or the previous attack-budget observation would not resolve this
contradiction.

The complete 6.7 GB run is archived at
`Reinforcement-Unknown Archives/CB1_2026-09-11_ppo6_rejected_natural_exhaustion_prior05_51843364_to_53020964_trace_to_53063655`.
No active PPO_6 run directory remains. Any next design should first choose an
architectural contract: either NavHead assumes a functional body and stamina
management remains outside it, or NavHead receives resource authority and a
non-self-contradictory recovery/rest mechanism. No further intervention was
started.

### PPO_6 5% attack prior plus missed-swing penalty (prepared 2026-09-11)

After observing manually that ordinary traversal generally requires rest, the
natural-stamina experiment is not continued as the immediate NavHead training
contract. This isolated retry returns to functional stamina reset and the
accepted capable-only inactivity rule. It starts from the same byte-identical
PPO_5 checkpoint at 51,843,364 and retains the unchanged 139-scalar policy
observation and seven-branch action space; there is no model surgery.

The attack-branch BCE prior targets 0.05 with coefficient 0.1 linearly
annealed to zero over one million decisions. Each game-confirmed executed
swing with zero accumulated pre-godmode claw-health loss during its macrostep
receives -0.05. Any positive claw-health loss counts as contact and receives
no miss penalty, including contact with metal. Holding attack during cooldown
is not an executed swing and is not penalized. There is no attack budget,
natural stamina, direct command penalty, masking, or sampling override.

This shaping asserts only the narrow invariant that an executed swing which
contacts nothing is undesirable. It does not label a material, route, or
geometry as good or bad. The prior reopens non-attack sampling; the miss
penalty supplies immediate credit only when the game confirms an actual miss.
Success requires completions to survive the low-prior interval and, after the
prior reaches zero, attack probability to recover selectively while the miss
fraction falls. A low marginal attack rate alone is not success.

The prepared run is
`checkpoints/CB1_PPO6_prior05_missed_attack_51843364`. Its source and copied
checkpoint both have SHA-256
`136399631a8cc9e79c3ae98a94bd75bbb3942bd10f9f4bad42730676ddd0b2be`.
The exact contract is pinned in `protocol_cb1.json`,
`intervention_ppo6.json`, and `prior_missed_attack_preparation.json`.

### 5% attack prior plus missed-swing penalty outcome (2026-09-12)

The combined run was stopped after 1,787,777 traced decisions. Its last
complete checkpoint is 53,584,164 and its trace continues to estimated global
step 53,631,141. The prior had been at coefficient zero for approximately
788,000 decisions by the final evaluation, making the lack of rebound a
post-intervention learned result rather than an observation taken during the
forced-prior interval.

The implementation operated exactly as specified. In a bounded late sample,
every executed zero-contact swing received -0.05, every contact swing received
zero miss penalty, cooldown holds were not counted, and there were no computed
penalty mismatches. The prior and immediate miss signal nevertheless combined
into a one-way attack collapse: recent learned attack probability averaged
approximately 0.052%, with a latest value near 0.030%; sampled attack frequency
was approximately 0.041% (ten swings in 24,241 decisions, six misses and four
contacts). Useful contacts became too rare to supply enough counterevidence
for recovery after the prior disappeared.

Task performance failed with it. Late `ep_len_mean` was approximately 13,807,
`success_ep_len_mean` approximately 8,333, and `ep_rew_mean` approximately
-8.10. Recent sampled terminations were `mobile_stagnation`, consistent with a
policy that wandered without clearing obstructions. Explained variance was
severely unstable on the failure-heavy return distribution. The experiment
therefore verifies that the miss classifier is usable but rejects the 5%
prior plus -0.05 penalty combination as a training solution.

The complete 9.8 GB record is archived at
`Reinforcement-Unknown Archives/CB1_2026-09-12_ppo6_rejected_prior05_missed_attack_51843364_to_53584164_trace_to_53631141`.
No active PPO_6 experiment directory remains, and no successor intervention
was started during cleanup.

### PPO_6 25% attack prior plus missed-swing penalty (prepared 2026-09-12)

This controlled retry again starts from accepted PPO_5 checkpoint 51,843,364.
Relative to the rejected 5% prior plus -0.05 miss experiment, exactly one
experimental hyperparameter changes: the attack-branch BCE target increases
from 0.05 to 0.25. The prior coefficient remains 0.1 and anneals linearly to
zero over one million decisions. The game-confirmed missed-swing penalty
remains -0.05; contacts, including metal contact, remain unpenalized, and
cooldown holds remain excluded.

Functional stamina reset, the 139-scalar observation, action space, PPO
settings, progress/completion reward and watchdogs all remain unchanged.
There is no natural stamina, budget scalar, model surgery, masking, or action
override. The purpose is to distinguish inadequate useful-attack sampling
from an intrinsically overdominant miss penalty. Success requires attack to
settle into a non-saturated equilibrium with a lower miss fraction after the
prior reaches zero. If attack again collapses after coefficient zero, the
-0.05 penalty is rejected at this scale rather than increasing prior pressure
again.

The prepared run is
`checkpoints/CB1_PPO6_prior25_missed_attack_51843364`.

### 25% prior plus -0.05 missed-swing outcome; -0.005 retry (2026-09-12)

The 25%/-0.05 run was stopped after 1,210,287 traced decisions. Its last
complete checkpoint is 53,020,964 and the trace continues to estimated global
step 53,053,651. Approximately 200,000 decisions after the prior reached zero,
recent learned attack probability had fallen to approximately 0.46% (latest
approximately 0.50%) and was still declining from a 50-update mean near 2.05%.
A bounded late sample contained 128 swings in 24,904 decisions: 61 misses and
67 contacts, with no penalty-calculation errors. Recent sampled terminals were
`mobile_stagnation` and `capable_inactivity`; `success_ep_len_mean` had degraded
to approximately 5,548 and `ep_rew_mean` to approximately -0.41.

This controlled comparison rejects insufficient prior sampling as the primary
explanation for the earlier collapse. Raising the target from 5% to 25% did
not produce a stable post-prior equilibrium under a -0.05 miss penalty. The
complete 6.7 GB record is archived at
`Reinforcement-Unknown Archives/CB1_2026-09-12_ppo6_rejected_prior25_missed_minus0p05_51843364_to_53020964_trace_to_53053651`.

The next clean retry again starts from accepted PPO_5 step 51,843,364 and
changes exactly one experimental value: the zero-contact executed-swing
penalty is reduced tenfold from -0.05 to -0.005. The prior target remains
0.25, coefficient remains 0.1, and annealing remains one million decisions.
All contact classification, reward, PPO, watchdog, functional-godmode,
observation and action semantics remain fixed. This tests whether the weaker
immediate signal can suppress misses without overwhelming the sparse positive
evidence for useful excavation. The prepared directory is
`checkpoints/CB1_PPO6_prior25_missed005_51843364`.

### 25% prior plus -0.005 missed-swing outcome and archive (2026-09-13)

The weaker-penalty run was deliberately allowed to continue well beyond the
prior interval. It was stopped after 2,860,524 traced decisions; the last
complete checkpoint is 54,659,364 and the trace continues to estimated global
step 54,703,888. By then the prior had been at coefficient zero for roughly
1.86 million decisions.

The tenfold reduction delayed but did not prevent the same global attack
collapse. Late learned attack probability averaged approximately 0.069% over
ten updates, 0.084% over 50, and 0.122% over 100; realized attack frequency in
a bounded tail was approximately 0.179% (36 swings in 20,120 decisions).
Those rare swings split evenly between 18 misses and 18 contacts. Thus the
policy did not acquire increasing contact selectivity; it removed the action
that could miss and starved useful contacts of counterevidence.

Task performance degraded continuously with the collapsing head. Late
`success_ep_len_mean` averaged approximately 10,556, `ep_len_mean` 16,037,
and `ep_rew_mean` -1.05. The sampled terminal was a 24,803-decision
`mobile_stagnation` failure. Explained variance remained severely unstable
(recent mean approximately -35). This rejects -0.005 at the tested 25% prior:
the smaller penalty changes the time constant, not the qualitative outcome.

The trainer was already stopped at cleanup, but its Unity/Proton process was
still orphaned and was explicitly terminated before artifacts were moved. The
complete 16 GB record is archived at
`Reinforcement-Unknown Archives/CB1_2026-09-13_ppo6_rejected_prior25_missed_minus0p005_51843364_to_54659364_trace_to_54703888`.
No active PPO_6 directory or experiment process remains, and no successor was
started.

## PPO_5 spatial counterfactual / Clever Hans audit (2026-09-13)

The frozen accepted PPO_5 checkpoint at 51,843,364 steps was evaluated on CPU
using 50 deterministically sampled, sidecar-aligned trace observations. For
each observation, only a 3x3 spatial patch centered on the encoded cursor was
changed between open space, intact breakable material, nearly broken material,
and infinirock; all 139 scalar values and all other spatial cells were held
fixed. Complete distributions for all seven action branches were recorded.

Mean attack probability was 99.9788% for open space, 99.9719% for breakable
material, 99.9777% for damaged material, and 99.9636% for infinirock. Mean
attack-branch total variation versus open was at most 0.0191%, while infinirock
changed movement by 15.76%, horizontal look by 11.06%, and vertical look by
11.10%. The spatial representation therefore measurably influences navigation
and aiming, but PPO_5's attack output is effectively saturated and locally
insensitive in probability space. A follow-up dump of pre-softmax logits showed
that saturation hides material-state dependence: relative to open, the paired
attack-minus-no-attack logit gap changed by a mean -0.690 for breakable
material, -0.166 for damaged material, and -0.720 for infinirock. The decrease
occurred in 43/50, 41/50, and 40/50 pairs respectively. Thus the attack branch
receives geometry-conditioned signal, but not the desired selective rule: open
space had the largest mean attack gap (10.002), and every condition remained
overwhelmingly on the attack side of the boundary. This does not prove that
selective attack is architecturally impossible or that these counterfactuals
cover every state distribution.

Final-head decomposition ruled out a single enormous output-layer bias as the
main source. The literal attack-minus-no-attack bias difference is only +0.560;
the actor latent contributes +9.048 on average across the 200 evaluations.
Their sum reconstructs the observed gaps within 2.9e-6. In a frozen diagnostic
offset sweep, subtracting 9 logits reduced mean attack probability to 66.85%
for open, 64.59% for damaged, 56.45% for breakable, and 55.43% for infinirock.
Thus removing a global offset exposes variation, but not useful material
selectivity; it preserves the same open-first ordering. The sweep did not run
gameplay and is not evidence that an offset-modified policy navigates well.

The reproducible audit implementation is `PPO_Python/tests/test_clever_hans.py`;
the 200-evaluation distribution dump is
`PPO_Python/evaluation_results/clever_hans_attack_semantics_ppo5_51843364.json`.

### PPO_5 identical-block relocation follow-up (2026-09-13)

A second frozen audit used 50 matched states and a shared cleared-patch
baseline, then inserted the same 3x3 100-health breakable block at the cursor,
below the player, opposite the cursor, or at a distant control. The hypotheses
were recorded in the output before interpretation. Cursor placement reduced
the attack logit gap by mean -0.363 and median -0.205, with decreases in 39/50
pairs (two-sided exact sign test p=0.0000902). Below placement was weaker and
directionally inconsistent (mean -0.106, 30/50 decreases, p=0.203); opposite
placement had nearly zero mean direction (mean +0.047, 24/50 decreases,
p=0.888). The distant control produced a small but consistent increase (mean
+0.063, 36/50 increases, p=0.00260), roughly one-sixth the cursor effect by
absolute mean direction and with small action-distribution changes.

This favors a target-occupancy component over a travel-path obstruction or
simple opposite-side directional rule. It remains an anti-occupancy modulation
under an overwhelming attack habit, not useful excavation selectivity. The
test is an off-policy spatial counterfactual and cannot establish gameplay
quality or a temporal motor explanation. Full probabilities and logits for all
seven branches are in
`PPO_Python/evaluation_results/clever_hans_attack_location_ppo5_51843364.json`.

Within the same 50 cursor-occupancy pairs, a preregistered follow-up tested
whether the signed attack-gap suppression covaried with per-state movement and
cursor distribution changes (total-variation distance). Pearson/Spearman
correlations were -0.011/-0.010 for movement, 0.128/0.171 for horizontal look,
-0.130/0.030 for vertical look, and -0.014/0.070 for their mean. Deterministic
100,000-permutation two-sided p-values were all at least 0.237, and all eight
Holm-adjusted p-values were 1.0. Thus stronger attack suppression was not
associated with a stronger whole-motor-policy change in these samples. This
favors a relatively direct target-occupancy influence on attack over a single
shared motor-mode transition, but correlation cannot identify network
causality or exclude a shared feature whose downstream head sensitivities
differ.

### PPO_5 target-feature isolation and single-cell sweep (2026-09-13)

The PPO_5 protocol has no explicit occupancy, solidity, destructibility, or
material-ID channel. Block presence is represented indirectly by log-normalized
health, an extreme-health flag, toxicity, and four sleep-quality channels. A
50-state single-cursor-cell audit isolated those observable features. Raising
raw health from empty to 1, 100, 1,000, and 15,000 changed the mean attack gap
by -0.013, -0.085, -0.121, and -0.157 respectively; every health comparison
remained significant after Holm correction. At fixed maximum normalized
health, enabling the extreme flag instead increased the gap by +0.070
(Holm p=0.0198). Toxicity reduced it by -0.079 (Holm p=0.00655), while sleep
category 2 increased it by +0.136 (Holm p=0.000215); the other sleep categories
were weaker or inconsistent after correction. Thus the earlier effect is
primarily health-sensitive, with additional learned dependence on ancillary
block attributes, not a directly represented semantic material class.

An additional 9x9 sweep placed one identical raw-health-100 cell at each offset
around the cursor against a shared cleared-region baseline. The response was a
broad asymmetric field rather than a detector sharply centered on the cursor.
The strongest mean suppression was -0.176 one cell left and one cell below the
cursor (44/50 decreases; Holm p=2.63e-6); the exact cursor cell was only -0.052
(38/50; Holm p=0.0236). This is evidence for a learned positional health pattern,
not proof of collider geometry, intended target semantics, or gameplay utility.
All per-state action probabilities/logits and all 81 aggregate cells are in
`PPO_Python/evaluation_results/clever_hans_attack_spatial_features_ppo5_51843364.json`.

### CB1 observation-aliasing audit (2026-09-13)

The live PPO_5 terrain grid serializes only current block health, toxicity, and
sleep quality. Normalization derives continuous health and an extreme-health
flag, but block ID, base/max health, metallic, slippery, hit/step sound, and
other `BlockInfo` properties are absent. Five pairs/groups of pristine block
IDs therefore collide exactly: gravel/clay, scrap pile/mushroom cap,
plastic/wood, sand/snow, and granite/heavy rock. None of those pristine groups
differs in the mechanically important metallic or slippery flags, so their
existence alone does not prove an unsolvable CB1 attack decision. Damage can
create more consequential aliases because only remaining health is sent; for
example, intact slippery ice at health 50 can match a damaged non-slippery
Mediocre-sleep block at remaining health 50, despite different movement and
destruction consequences. Infinirock is not aliased with pristine ordinary
terrain because its 420,133,760 health activates the dedicated extreme flag.

There is a real but narrower alias in non-grid entity geometry. Commit
`fe684e2e56cb12ca760bb8851c04865d12cea7af` disabled the previously defined
visible-building and visible-item payload on 2026-08-01. The harness still
queries only the block grid and fluids for the live spatial tensor, while game
attacks raycast and damage `BuildingEntity` colliders separately. Consequently,
an otherwise identical state with a separate entity collider over an air tile
and one without it can alias. However, a subsequent WorldGeneration audit
corrected an initially overbroad interpretation: structure tilemaps are copied
directly into `worldBlocks` by `GenerateObjectAtPos`, so the large walls and
terrain geometry of pods, bridges, and similar structures remain visible in
the PPO health grid. Only separately instantiated `BuildingEntity` children
and distributed entities are omitted. Their spatial exposure is sparse relative
to the full terrain field, so this alias is not a credible standalone cause of
PPO_5's approximately 99% attack rate. It remains a localized limitation, not
the explanation for global saturation established by this audit.
