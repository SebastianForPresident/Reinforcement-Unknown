# CB1 matched-pairs acceleration test

This evaluates two frozen checkpoints at both 1x and 10x on the same ten cave
seeds. Each speed run alternates checkpoint order within each seed. Results
are appended after every trial and an interrupted run skips completed cells
when restarted.

The primary test uses stochastic inference because CB1 depends on stochastic
actions to escape attractors. Every checkpoint/speed trial for a cave receives
the same action RNG seed. The common random numbers reduce sampling noise but
do not force different policies to choose the same actions.

## Freeze the test inputs

Copy `matched_pairs.example.json` to `matched_pairs.json` and replace both
checkpoint placeholders with exact immutable ZIP paths. Do not use a moving
"latest" alias. The pre-acceleration checkpoint is fixed at 2,091,300 steps:
it was written at 16:56:59 on August 14, immediately before the recorded
"acceleration to 4x" change at 18:08:20. Checkpoint cadence was approximately
4.7 steps/second through that file and approximately 14.5 steps/second after
the acceleration restart, independently confirming the boundary.

Historical confounds reconstructed from Git:

- Commit `84e3658` introduced 4x, a 60-FPS/vsync-off cap, high-time-scale
  safety-threshold patches, and profiler instrumentation. It retained
  `PauseAtPolicyBoundary()` and continued setting `Time.timeScale = 0` after
  every published observation. The next action was latched from an ordinary
  Unity `Update`, which restored the configured time scale.
- Commit `bae455e` introduced 10x and removed that per-action time-scale
  pause. It installed a player-loop callback immediately after
  `Physics2DFixedUpdate`, published after tick ten, blocked Unity's main thread
  for Python, and applied the next action before returning for tick eleven.
  The missing visible zero-speed flicker therefore begins with this exact
  blocking protocol: the UI cannot render while the main thread is waiting.
- Moving from the `Body.FixedUpdate` postfix to a post-physics player-loop
  callback also changed the phase at which observations are collected. The
  current checkpoint has trained under that newer phase; the pre-acceleration
  checkpoint trained under the older phase.

Neither acceleration commit changed Python, PPO hyperparameters, reward code,
the observation schema, or the action schema. Nevertheless, the historical
checkpoint comparison combines continued PPO training, speed, frame policy,
and observation/action synchronization changes. The within-checkpoint
1x-versus-10x pairs under the current harness remain the clean test of current
execution-speed sensitivity; pre-versus-current is not a pure acceleration
ablation and may also expose adaptation to the new post-physics protocol.

The default 54,000-step horizon is 180 simulated minutes. A non-completion is
retained as a right-censored trial at that common horizon.

## Run the 10x half

In the server terminal:

```bash
./.venv/bin/python PPO_Python/Server.py matched-eval PPO_Python/matched_pairs.json 10x
```

In the game terminal:

```bash
PPO_TIME_SCALE=10 ./PPO_Harness/Deploy.sh
```

## Run the 1x half

After the 10x half exits, restart both processes:

```bash
./.venv/bin/python PPO_Python/Server.py matched-eval PPO_Python/matched_pairs.json 1x
PPO_TIME_SCALE=1 ./PPO_Harness/Deploy.sh
```

Never run `Deploy.sh` while training is active. It builds and deploys a new DLL
and starts a new game process. These commands are for the later frozen-policy
test only.

## Analyze

```bash
./.venv/bin/python PPO_Python/analyze_matched_pairs.py \
  PPO_Python/matched_pairs_results.csv
```

The report includes completion counts, capped median simulated time, paired
wins, median paired deltas and ratios, and an exact two-sided sign test. With
only ten pairs, 9-1 or 10-0 directional results are required for a two-sided
sign-test p-value below 0.05. Practical materiality should also be declared in
advance; a suggested threshold is a median 1x/10x time ratio at or below 0.80,
or any repeated completion advantage at the common horizon.

## Run the repaired-10x continuation gate

This configuration evaluates only the strong pre-acceleration checkpoint. Its
ten 1x controls are already preserved in `matched_pairs_results.csv`.

```bash
./.venv/bin/python PPO_Python/Server.py matched-eval \
  PPO_Python/matched_pairs_repaired_10x.json 10x
```

```bash
PPO_TIME_SCALE=10 PPO_FORCE_BODY_UPDATE=1 ./PPO_Harness/Deploy.sh
```

Results append to `PPO_Python/matched_pairs_repaired_10x_results.csv`, so an
interrupted run resumes by restarting the same two commands. Accept the repair
only after comparing every repaired 10x seed with the archived
`pre_acceleration`/`1x` row for that seed.

```bash
./.venv/bin/python PPO_Python/analyze_matched_pairs.py \
  PPO_Python/matched_pairs_results.csv \
  PPO_Python/matched_pairs_repaired_10x_results.csv
```

The boundary-only repaired run was rejected after losing all first eight
pairs. The successor fixed-action gate processes each held action once per
physics interval while retaining 5-Hz sampling:

```bash
./.venv/bin/python PPO_Python/Server.py matched-eval \
  PPO_Python/matched_pairs_fixed_action_10x.json 10x
```

```bash
PPO_TIME_SCALE=10 PPO_FORCE_BODY_UPDATE=1 ./PPO_Harness/Deploy.sh
```

Its results are isolated in
`PPO_Python/matched_pairs_fixed_action_10x_results.csv` with speed label
`fixedAction10x`.

The deterministic-60-Hz successor retains stochastic policy sampling but runs
twelve gameplay updates per ten physics ticks:

```bash
./.venv/bin/python PPO_Python/Server.py matched-eval \
  PPO_Python/matched_pairs_fixed_60hz_10x.json 10x
```

```bash
PPO_TIME_SCALE=10 PPO_FORCE_BODY_UPDATE=1 ./PPO_Harness/Deploy.sh
```

Its isolated result label is `fixed60Hz10x` and its output is
`PPO_Python/matched_pairs_fixed_60hz_10x_results.csv`.
