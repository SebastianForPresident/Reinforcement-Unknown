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
