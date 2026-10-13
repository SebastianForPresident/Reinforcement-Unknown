# Laboratory integration log

## Starting state

- Branch `isaac-lab`, HEAD `a075acc45bd59f345b33f08f44c1d1017e9b7cef`, matching `origin/isaac-lab` at reconnaissance.
- GUI/workbench work from the prior campaign was present in the worktree but uncommitted. It includes the PySide6 studio, reward component support, observation probe, user documentation, and GUI acceptance report.
- The current shared worktree also contains optimizer-owned edits in `casu_grid/isaac_vector_env.py`, `casu_grid/isaac_player_ab.py`, and `casu_grid/perf/gpu_probe.py`; user deletions in `Instructions.txt` and legacy matched-pair outputs; and campaign instruction documents. These were not reverted or staged.
- The archival checkout `../Reinforcement-Unknown` remains untouched.

## Integration changes

- Single-process and batched environments expose `performance_info()`. Training records its effective backend report, effective worker count, environment count, per-worker partition, requested backend/workers, terrain mode, timestep, action decimation, world fixture/content IDs, fluid seed and liquid phase in `environment.json`.
- The trainer records initialization/training exceptions as `failure.json` with type, message and traceback, marks keyboard interruption, and closes environment/trainer while preserving the original exception. The launcher supervises a dedicated process group, marks completion only for exit code zero, removes stale status markers, and terminates the owned process group on cancellation.
- The CLI writes a pre-launch manifest before Isaac startup and streams child output to `launcher.log`. Early Kit/AppLauncher failures remain discoverable and include a log tail in `failure.json`; launcher terminal state is atomically persisted in the manifest.
- Replay restores a recorded multi-worker count when no override is present, rejects explicit worker-count/layout conflicts before simulation startup, and includes recorded/current partition equality in its deterministic result.
- Run inspection exposes experiment identity and execution metadata. Empty `configs` output now gives the configuration-creation command.

## Validation evidence

- `python -m compileall -q casu_grid/workbench`: passed.
- `python -m unittest discover -s casu_grid -p 'test_*.py'`: 26 tests passed, including GUI offscreen construction, config lifecycle, metadata presentation, injected early launcher failure/log preservation, failure precedence, and replay partition helper coverage.
- Isaac-free simulator/workbench regression suite under the bundled Isaac Lab Python: 114 tests passed via `"$ISAAC_LAB/isaaclab.sh" -p -m unittest discover -s casu_grid/tests -p 'test_*.py'`. Running the same suite under system Python is unsupported here because it has no NumPy.
- `git diff --check`: passed.
- A real single-process engineering-only smoke completed one PPO update (32 environment decisions, one metrics row), with trajectory and `latest.pt` checkpoint. Artifacts were in `/tmp/casu-integration-single-2`; its `RUNNING` marker was removed after the run. The metadata was manually normalized after execution to reflect the effective backend because the single-env backend report was missing; the code now supplies that report directly.
- A prior Isaac engineering smoke artifact at `/tmp/casu-gui-isaac-smoke-20261013` is discoverable and reports completed, with checkpoint and metrics.
- The local Isaac runtime required the install-provided `compat-libs/usr/lib64`, `isaaclab-deps`, and Isaac Lab source paths. An optimizer-owned Isaac process was active during this campaign, so a second live simulator run (sharded, resume, replay, failure injection) was not started to avoid contention. The performed single-process smoke finished before its exact process group was signalled; no process from that group remains.

## Cleanup audit and disposition

| Candidate/material | Classification | Evidence and disposition |
|---|---|---|
| `PPO_Harness/CRIU_FEASIBILITY.md`, counterfactual harness sources/scripts | Historical and reusable research infrastructure | Tracked and preserve-worthy; counterfactual execution still has source harness and tests. Retain in place. |
| `PPO_Python/CounterfactualReplay.py`, `analyze_matched_pairs.py`, example/configs, results and tests | Historical scientific/reusable analysis | Referenced by historical docs/tests and contains data; retain in place. |
| Active-branch matched-pair JSON/CSV files currently marked deleted | Existing user changes | Do not restore, stage, or reinterpret as cleanup. The archival checkout was not modified. |
| Isaac performance regression, trajectory/parity and benchmark infrastructure | Required by active lab | Retain; referenced by performance documentation and acceptance reports. |
| Other inherited files | Uncertain unless individually classified above | No broad cleanup performed; unique scientific history is preserved. |

No files were removed as part of this integration campaign. No archival checkout content was changed.

## Remaining acceptance work

- Run sharded training, worker-failure/cancellation, checkpoint resume, exact replay, mismatch rejection, GUI process actions, and fresh restart discovery on an idle simulator host.
- Test environment-startup failure against the actual Isaac shutdown failure mode; unit tests currently cover status interpretation and pure-workbench paths, not Kit teardown.
- Inspect the full GUI manually on a desktop. Automated GUI coverage proves construction and selected widget behavior only, not day-to-day usability.
- Complete cleanup review of any uncertain inherited artifacts only when dependency/history evidence is available; no uncertain material was deleted.
- Commit and push this integration checkpoint without staging optimizer-owned changes or the user's existing deletions.
