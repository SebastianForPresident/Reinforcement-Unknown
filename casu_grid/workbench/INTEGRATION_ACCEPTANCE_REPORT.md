# Isaac laboratory integration acceptance

## Researcher summary

The workbench now records the execution layout reported by a live environment, presents that information with run status, checks replay partition compatibility, and preserves failures as failures. A bounded engineering-only PPO update completed through the CLI in the single-process path. This does not approve the engineering smoke contracts as Timmy's experiment.

## Verified

- GUI and CLI share configuration and run services. Offscreen GUI construction, safe experiment discovery, typed model/PPO controls and plotting-view behavior are covered by automated tests. A manual desktop usability review remains necessary.
- Workbench tests: `python -m unittest discover -s casu_grid -p 'test_*.py'` — 26 passed, including injected early-launcher failure/log preservation.
- Syntax: `python -m compileall -q casu_grid/workbench` — passed.
- Formatting check: `git diff --check` — passed.
- Single-process CLI lifecycle: engineering smoke configuration, one PPO update, 32 environment decisions, real metrics, checkpoint, trajectories, completed status and inspectable environment metadata. Evidence: `/tmp/casu-integration-single-2`.
- Older GUI acceptance smoke artifact discovery: `/tmp/casu-gui-isaac-smoke-20261013`.
- Active performance backend settings are `CASU_BACKEND=auto|numba|reference-python` and `CASU_NUM_WORKERS=N|auto`. The requested values and live effective backend/layout are recorded separately.
- Failure records carry exception type, message and traceback. A pre-launch manifest and streamed `launcher.log` preserve failures that occur before `AppLauncher` can enter the trainer's handler. A nonzero trainer exit cannot receive a completion marker. Cancellation targets the dedicated launcher process group.
- Cleanup review found CRIU/counterfactual and matched-pair materials are tracked historical research artifacts, useful analysis, or scientific records. No archival or uncertain material was deleted.

## Not verified in this environment

An optimizer-owned simulator process was running while the live smoke was completed. To avoid simulator/GPU contention, this campaign did not run a second concurrent simulator job. Sharded PPO training, worker crash recovery, repeated cancellation/restart, resume, deterministic replay, explicit replay mismatch, injected Kit initialization failure, and GUI-launched training/replay therefore remain unverified live. Automated coverage is not a substitute for those runtime checks.

The bounded smoke used a single process. Its run manifest and metrics were produced by the real trainer. During that run, inspection found that the single-environment adapter lacked `performance_info()` and the launcher could leave a stale `RUNNING` marker; code fixes now address both. The already-created smoke's environment report was updated manually for accurate inspection after the evidence was captured. Re-run the smoke on an idle host to verify the new metadata path end to end.

## Run and inspect

Setup for the local Isaac Lab 2.1.1 / Isaac Sim 4.5.0 installation (adjust paths to the installation):

```bash
export ISAAC_LAB="$HOME/Applications/IsaacLab-2.1.1-IsaacSim-4.5.0/IsaacLab-2.1.1"
export ISAAC_ROOT="$HOME/Applications/IsaacLab-2.1.1-IsaacSim-4.5.0"
export ISAAC_SIM="$ISAAC_ROOT/isaac-sim-4.5.0"
export LD_LIBRARY_PATH="$ISAAC_ROOT/compat-libs/usr/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ISAAC_ROOT/isaaclab-deps:$ISAAC_LAB/source/isaaclab:$ISAAC_LAB/source/isaaclab_tasks:$ISAAC_LAB/source/isaaclab_assets:$ISAAC_LAB/source/isaaclab_rl:$ISAAC_LAB/source/isaaclab_mimic:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

Create and validate a draft without starting training:

```bash
python -m casu_grid.workbench.lab new "My experiment" \
  --output casu_grid/workbench/experiments/my_experiment.py
python -m casu_grid.workbench.lab validate \
  casu_grid/workbench/experiments/my_experiment.py
```

Launch the explicitly labeled engineering fixture as a bounded systems check:

```bash
"$ISAAC_LAB/isaaclab.sh" -p -m casu_grid.workbench.lab launch \
  casu_grid/workbench/engineering_smoke.py --engineering-smoke \
  --run-dir experiments/integration-smoke --updates 1 --num-envs 1
```

Browse, inspect, compare and visualize saved evidence:

```bash
python -m casu_grid.workbench.lab configs
python -m casu_grid.workbench.lab runs
python -m casu_grid.workbench.lab status integration-smoke
python -m casu_grid.workbench.lab compare RUN_A RUN_B
python -m casu_grid.workbench.lab replay integration-smoke --preview
```

Use the GUI with `python -m casu_grid.workbench.gui`. Continue a run through the `resume` command, which writes to a new run directory and keeps a link to the originating checkpoint. For deterministic replay, retain the recorded `CASU_NUM_WORKERS` partition; the replay summary records whether the current partition matches.

For full launch instructions, GUI workflow, configuration authoring, output artifacts and limitations, see [README.md](README.md) and [ISAAC_LAB_SETUP.md](../../ISAAC_LAB_SETUP.md). The engineering smoke is an infrastructure fixture, not a scientific recommendation.

## Repository and release status

Starting HEAD was `a075acc45bd59f345b33f08f44c1d1017e9b7cef`, on `isaac-lab` and aligned with `origin/isaac-lab`. GUI/workbench source was uncommitted at campaign start. The worktree also contained optimizer-owned changes and user deletions; these were preserved. See [INTEGRATION_LOG.md](INTEGRATION_LOG.md) for the file audit, process constraint, and detailed evidence.

The GUI/workbench integration checkpoint is commit `18887ad2a68989b24cde52737adb287f1bdd6dc6`. The checkpoint and documentation are pushed to `origin/isaac-lab`; follow-up lifecycle refinements are recorded in the subsequent commit history.
