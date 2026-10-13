# CasU Isaac research workbench

## Start here

One entry point provides commands and a guided menu:

```bash
python -m casu_grid.workbench.lab guide
python -m casu_grid.workbench.lab --help
python -m casu_grid.workbench.lab menu
python -m casu_grid.workbench.lab configs
```

Use the Isaac bundled Python for commands that initialize Isaac Sim (setup in
[`../../ISAAC_LAB_SETUP.md`](../../ISAAC_LAB_SETUP.md)). Pure file operations,
configuration preflight, run discovery, comparison and plotting use the same
CLI and do not initialize Isaac.

### First session

```bash
python -m casu_grid.workbench.lab about
python -m casu_grid.workbench.lab new "Test a navigation idea" \
  --output casu_grid/workbench/experiments/navigation_idea.py \
  --hypothesis "Write the question being tested"
python -m casu_grid.workbench.lab validate casu_grid/workbench/experiments/navigation_idea.py
```

The generated Python file uses the existing action, observation, reward, model
and PPO contract APIs. Its values and callback come from the engineering smoke
and are explicitly examples, not approved Timmy choices. Review and edit them;
the tool will refuse to overwrite an existing destination. `callbacks()` is
Python code and should be treated as executable code you authored.
`EXPERIMENT` records a generated stable ID, human-readable name and creation
time, with optional hypothesis, notes and parent reference.

Preflight summarizes action heads and 10-physics-tick decision cadence,
observation shapes/normalization/masks, reward terms, model and PPO settings,
checkpoint compatibility and environment capabilities. It does not start Isaac
or execute your reward/observation callback. The callback adapter checks
observation keys/shapes/finiteness and reward scalar/term keys/values at
the environment boundary. If `callbacks()` is omitted, a non-empty reward
contract is rejected instead of silently training on zero reward.

## Configure and train

Read the generated config sections, then run the two-stage preview/launch:

```bash
ISAAC_ROOT="$HOME/Applications/IsaacLab-2.1.1-IsaacSim-4.5.0"
ISAAC_SIM="$ISAAC_ROOT/isaac-sim-4.5.0"
"$ISAAC_SIM/python.sh" -m casu_grid.workbench.lab launch \
  casu_grid/workbench/experiments/navigation_idea.py \
  --run-dir experiments/navigation-idea --updates 3 --num-envs 1 --preview
"$ISAAC_SIM/python.sh" -m casu_grid.workbench.lab launch \
  casu_grid/workbench/experiments/navigation_idea.py \
  --run-dir experiments/navigation-idea --updates 3 --num-envs 1
```

A three-update run is a bounded engineering check; choose your own budget after
reviewing the configuration. Launch preview works without Isaac startup and
prints a shell-quoted command (set `ISAAC_SIM` to choose the bundled `python.sh`
when invoking preview from system Python). The CLI always reports the effective command,
full terrain requirement, output directory and mode. It will not launch the
engineering smoke template as research unless `--engineering-smoke` is given.
Environment options include `--terrain-mode shared|independent` (independent is
more expensive and permits per-world terrain edits), `--episode-steps N`, and
`--seed N` for PPO sampling (the world fixture remains canonical seed 391728405).
The simulator contract remains canonical seed 391728405, 50 physics
ticks/s, and ten physics ticks per policy decision.

The run folder stores the exact contract JSON and source config path/hash,
source revision and dirty-tree state, environment manifest, per-update JSONL,
trajectory JSONL, checkpoint and TensorBoard files. A CLI-launched run writes
explicit running/completed/failed evidence. Ctrl-C reports the latest checkpoint
path when one exists. Resume writes to a new run directory and validates the
exact checkpoint fingerprint before Isaac starts:

```bash
python -m casu_grid.workbench.lab resume navigation-idea --updates 100 --preview
python -m casu_grid.workbench.lab resume navigation-idea --updates 100
```

Never overwrite the source run to continue it; a resumed run links to the source
checkpoint through its invocation and has its own output folder.

## Inspect and compare

```bash
python -m casu_grid.workbench.lab runs
python -m casu_grid.workbench.lab status navigation-idea
python -m casu_grid.workbench.lab watch navigation-idea
python -m casu_grid.workbench.lab plot navigation-idea
python -m casu_grid.workbench.lab compare navigation-idea other-run \
  --output experiments/comparison.json
python -m casu_grid.workbench.lab tensorboard navigation-idea
```

Run paths can be supplied instead of IDs. `runs` reconstructs its view from
artifacts; the index is not a database and never rewrites old runs. Missing
values remain unknown. Metrics without a completion marker are classified as
interrupted; old folders with no reliable process/artifact evidence are
unknown/stale. Trailing incomplete JSONL records are reported and ignored.
`status` and `watch` show elapsed time, latest-checkpoint age, action frequencies,
optimizer metrics, reward terms and termination counts when recorded.
Comparisons show contract/environment diffs and retain each run's numeric
metrics with the recorded `environment_steps` key. Reward-contract differences
produce a comparability warning. `plot` writes `analysis/training_metrics.png`
with decision-aligned x axes and labels. TensorBoard binds to loopback only.

Metrics distinguish optimizer updates, environment policy decisions across all
worlds, per-world decisions/s and aggregate physics ticks/s. A policy decision
advances ten physics ticks. Completed-episode return/length values are only
reported when episodes actually finish; a training reward is not evidence of
task success.

## Replay and reproducibility

```bash
python -m casu_grid.workbench.lab replay navigation-idea --preview
"$ISAAC_SIM/python.sh" -m casu_grid.workbench.lab replay navigation-idea
```

This runs a deterministic simulator replay of the recorded action tape, not a
rendered CasU gameplay video. It reuses the recorded config path if available;
use `--config /path/to/preserved_config.py` when that source file has moved.
It writes `replay/replay.csv`, `replay/paths.png`, and
`replay/summary.json`. The summary reports maximum pose/velocity deviation and
whether it is within tolerance; the detailed CSV has step/world and pose fields.
A path plot is a trajectory projection, not source-game visualization. For
CasU-versus-Isaac comparisons and fidelity caveats, see
[`../ACCEPTANCE_REPORT.md`](../ACCEPTANCE_REPORT.md).

`config.json` records config hash, source revision/working-tree status, seed,
environment settings, and contract snapshot; `latest.pt` embeds the learning
contract fingerprint, optimizer/RNG/environment state. Keep both with results.
Do not change the scientific contract when resuming: checkpoint loading rejects
mismatched fingerprints.
Each newly launched run also contains the exact Python research config as
`research_config.py`; replay/resume verifies the recorded hash and uses that
snapshot if the original path moved or changed. Preserve any separately imported
helper modules alongside it.

## Errors and recovery

- **Missing Isaac / package dependency:** use `ISAAC_LAB_SETUP.md` and the
  bundled Python. `validate`, `runs`, `status`, `compare`, and `plot` do not need
  Isaac startup (plotting requires Matplotlib).
- **Invalid config or shape:** `validate CONFIG` reports the failing field;
  trainer checks actual observation/action batch shapes before optimization.
- **Incompatible or missing checkpoint:** `resume RUN` validates the recorded
  `latest.pt`; select the exact checkpoint/config pair or restore the archived
  config. No fallback checkpoint is chosen.
- **Run failure/resource exhaustion:** inspect run `failure.json`, latest
  complete metrics row, console output and checkpoint. A checkpoint means
  resumable state may exist, not that the run completed successfully. Resume
  into a new folder after fixing the cause.
- **Missing metrics:** `status` reports unavailable metrics; it does not infer
  progress from a checkpoint. JSONL parse errors identify the affected line.
- **Replay divergence:** inspect `replay/summary.json` and first CSV rows; verify
  the exact config and source revision before interpreting errors.
- **Unsupported terrain:** training requires full canonical geometry. Shared
  terrain rejects edits; use independent mode only when the authored task needs
  per-world edits.

## Research contract reference

`contracts.py` defines action heads, observation fields, reward terms, model
layers and PPO settings; `model.py` constructs the actor-critic; `ppo.py` is the
existing supported trainer and checkpoint implementation. Action semantics,
observation representation, reward, architecture and optimization settings are
researcher-owned. Changes to action/observation/model/PPO contracts make a
checkpoint incompatible by design. The exact config bytes are copied into each
new run as `research_config.py`, alongside source path and hash; replay/resume
uses the hash-matching snapshot if the original moved or changed. Preserve any
separately imported helper modules too.
run folder as research data. Replay/resume checks the hash and uses the snapshot
if the original config path moved or changed; preserve imported helper modules too.

## Historical CasU research versus current Isaac laboratory

The current Layer 1 laboratory is `casu_grid/` and its researcher interface is
`casu_grid/workbench/`. It uses the canonical seed 391728405 in a source-informed
Isaac surrogate. It is not bit-exact CasU; consult the acceptance report before
making transfer claims.

`PPO_Python/` preserves the historical Unity/BepInEx research: prior REINFORCE
and PPO workflows, independent critics, counterfactual replay, attack studies,
and matched-pair analysis. Its focused guides (`REINFORCE_LOOKAHEAD.md`,
`INDEPENDENT_CRITIC.md`, `MATCHED_PAIRS.md`) document those workflows. They
remain useful for reproducing and interpreting historical work, but new Isaac
experiments should start through `casu_grid.workbench.lab`. Existing PPO harness
and old checkpoints are preserved.
