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
