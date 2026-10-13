# Research workbench usability acceptance report

Updated 2026-10-13. Scope: researcher-facing usability around the existing Isaac
Layer 1 simulator and PPO implementation. The temporary smoke configuration is
engineering-only and is not an approved Timmy experiment.

## What changed

- Added one entry point: `python -m casu_grid.workbench.lab`, with help,
  scriptable subcommands and a guided interactive menu.
- Added experiment scaffolding with a unique stable ID, name, creation time,
  optional hypothesis/notes/parent, versioned metadata, and the existing editable
  Python contract/callback APIs. It refuses overwrite.
- Added preflight for actions and mappings, cadence, observation fields and
  normalization/masks, reward terms, model/PPO contracts, model parameter count
  when torch is available, and checkpoint fingerprint constraints. It does not
  initialize Isaac. Runtime callback adapters now check keys, shapes, finite
  values, reward scalar and declared reward-term names.
- Added safe launch preview/launch around the existing trainer, exact source
  config snapshots, completion/failure/interruption markers, exact-checkpoint
  resume into a new folder, and atomic checkpoint replacement.
- Added file-backed config/run discovery, conservative status classification,
  text and JSON status summaries, polling watch, metrics parsing with bounded
  tail reads for monitoring, config diffs, reward/objective comparability
  warnings, shared environment-decision alignment, JSON comparison reports,
  labeled plots, loopback TensorBoard launch and run-selected deterministic replay.
- Replay CSV now includes actions and logged/replayed rewards. Replay summary
  reports the first divergence, worst field errors, configuration hash match and
  source revision.
- Added root navigation separating the current Isaac lab from preserved Unity
  research, plus the quick-start and troubleshooting guide.

The existing `train.py`, `ppo.py`, simulator contracts and deterministic replay
remain the execution backend. The world seed remains 391728405, timestep 0.02 s,
and policy cadence ten physics ticks per decision.

## Main entry point

- CLI: `python -m casu_grid.workbench.lab --help`
- Guided menu: `python -m casu_grid.workbench.lab menu`
- First-session guide: [`README.md`](README.md)
- Project navigation: [`../../RESEARCH_WORKBENCH.md`](../../RESEARCH_WORKBENCH.md)

Use system Python for file-only operations if its standard library is available.
Use the Isaac bundled `python.sh` for model parameter counts and all simulator,
training, replay and TensorBoard operations (setup in `../../ISAAC_LAB_SETUP.md`).
The canonical simulator is a useful source-validated surrogate, not bit-exact
CasU; see [`../ACCEPTANCE_REPORT.md`](../ACCEPTANCE_REPORT.md).

## Tested end-to-end engineering workflow

Executed on the local Isaac Sim 4.5 / Isaac Lab 2.1.1 runtime with full canonical
terrain, one world, engineering-only callbacks and 20-decision episode limit:

1. Created and validated an editable Python config through `new`/`validate`.
2. Previewed and launched a bounded PPO run through `lab launch` from the newly scaffolded config.
3. Trained one update: 32 environment policy decisions, 320 physics ticks, four
   PPO optimizer minibatches, finite loss and gradient norms, checkpoint and
   TensorBoard output. One completed time-limit episode was recorded. This tests plumbing only; return is not a navigation success metric.
4. Discovered the run as `completed`, inspected reward terms, throughput, return,
   episode length, termination, checkpoint and metrics.
5. Generated a labeled metrics plot and compared it with a second bounded run.
   The comparison aligned at 32 environment decisions and warned that source
   config hashes differed despite matching declared reward contracts.
6. Resumed the exact checkpoint in a new run folder; update count continued from
   1 to 2 and total environment decisions from 32 to 64.
7. Replayed eight recorded decisions from the run-local config snapshot. Maximum
   pose/velocity deviation was 0.0; the config hash matched.
8. Confirmed invalid missing-run paths and spawn-chunk PPO attempts return clear
   errors without starting the training environment.

The replay is a deterministic simulator replay, not rendered gameplay or a
source-game comparison.

## Automated verification

- Usability CLI and existing PPO workbench tests: **24 passed** in the latest
  focused run, including scaffolding/no-overwrite, config failures, missing
  reward callback, run status and partial JSONL, comparison alignment/diffs,
  snapshot resolution, relative paths, machine-readable listing, runtime callback
  errors and atomic checkpoint preservation.
- Existing simulator/reference regression suite: **85 passed** (verified before
  final usability additions; rerun below after integration).
- Historical `PPO_Python/tests`: collection reached 17 tests, with 6 import
  errors because `stable_baselines3` is not installed in the Isaac runtime. No
  legacy scripts, results, checkpoints or the user's `matched_pairs*` deletions
  were changed.
- No matched throughput A/B against the pre-usability revision was run. The new
  validation executes at the policy observation/reward boundary, not inside the
  physics tick loop; atomic checkpoint handling occurs only during saves. The
  end-to-end engineering samples measured 792–930 aggregate physics ticks/s and
  73–84 aggregate decisions/s for one world, but those short PPO runs are not
  directly comparable to the existing 875 ticks/s simulator-only benchmark.

## Researcher five-minute workflow

1. Read [`../../RESEARCH_WORKBENCH.md`](../../RESEARCH_WORKBENCH.md), then check
   `ISAAC_LAB_SETUP.md` once for the local runtime.
2. Create a file:

   ```bash
   python -m casu_grid.workbench.lab new "My question" \
     --output casu_grid/workbench/experiments/my_question.py \
     --hypothesis "State the research question"
   ```

3. Edit the scientific contract and callback sections, then run
   `python -m casu_grid.workbench.lab validate casu_grid/workbench/experiments/my_question.py`.
   The generated values are examples that need researcher review.
4. Preview a bounded launch, then use Isaac Python to launch it:

   ```bash
   "$ISAAC_SIM/python.sh" -m casu_grid.workbench.lab launch \
     casu_grid/workbench/experiments/my_question.py --updates 3 \
     --run-dir experiments/my-question --preview
   "$ISAAC_SIM/python.sh" -m casu_grid.workbench.lab launch \
     casu_grid/workbench/experiments/my_question.py --updates 3 \
     --run-dir experiments/my-question
   ```

5. Inspect with `runs`, `status my-question`, `plot my-question`,
   `compare my-question another-run`, and `replay my-question`. Resume with
   `resume my-question --updates N`; it creates a new run directory.

## A–J usability self-audit

| Scenario | Result | Evidence / limit |
|---|---|---|
| A. Return after absence | Pass | Root README and `RESEARCH_WORKBENCH.md` point to the current entry point and five-minute guide. |
| B. Create an experiment | Pass | `new` creates stable identity metadata and editable existing contracts; `configs` lists unlaunched configs. Values are clearly marked examples. |
| C. Failed configuration | Pass | Missing members, missing reward callbacks, shape errors and reward-term errors have field-specific messages and tests. |
| D. Training | Pass | Preview plus bounded full-terrain engineering PPO launch created metrics/checkpoint artifacts. |
| E. Monitoring | Pass | `status`, `watch`, JSONL metrics, checkpoint age, TensorBoard and plots use recorded values. |
| F. Compare | Pass | Config diffs, objective warnings, raw records, shared decision budgets and JSON output were exercised. |
| G. Replay | Pass | Run-selected replay generated CSV/plot/summary and reported zero deviation in the tested segment. |
| H. Interrupted work | Pass with limit | Status distinguishes running/completed/interrupted/failed/unknown; exact checkpoint resume passed; atomic-save failure test preserves prior bytes. A live SIGINT was not injected into a full Isaac process. |
| I. Historical work | Pass | Historical directories remain untouched and the root navigation identifies their focused guides. |
| J. Reproducibility | Pass | Runs record revision, dirty tree, config hash/snapshot, seed/environment settings and checkpoint contract; replay verifies the config hash. |

## Files and repository state

Implementation files: `casu_grid/workbench/lab.py`,
`casu_grid/workbench/runtime_validation.py`, `casu_grid/workbench/train.py`,
`casu_grid/workbench/replay.py`, `casu_grid/workbench/ppo.py`, and lazy public
exports in `casu_grid/__init__.py`.

Tests: `casu_grid/workbench/test_lab.py` and additions to
`casu_grid/tests/test_workbench.py`.

Documentation/navigation: `README.md`, `RESEARCH_WORKBENCH.md`,
`casu_grid/workbench/README.md`, and this report. The worktree's pre-existing
`Instructions.txt` and `PPO_Python/matched_pairs*` deletions remain untouched and
unstaged. No historical artifacts were removed. A focused usability commit is included after final review.

Remaining limitations: callback code imported from external helper modules must
be preserved alongside the run-local config snapshot; Matplotlib and TensorBoard
must exist in the selected Python for those features; the legacy test suite needs
its separate `stable_baselines3` dependency; full source fidelity caveats remain
those in the engineering acceptance report.
