# Research workflow usability audit

## Current strengths

- `contracts.py`, `model.py`, and `ppo.py` already provide the real configurable research contracts, strict checkpoint fingerprint, resumable PPO, TensorBoard scalars, JSONL metrics, and trajectories.
- `train.py` already creates a run manifest and supports environment count, terrain mode, update budget, episode limit and checkpoint load.
- `replay.py` already re-executes logged actions and emits a CSV, path plot and determinism summary.
- The canonical simulator and source-fidelity caveats are documented in `ACCEPTANCE_REPORT.md`.

## Friction and gaps observed

- Entry points required memorizing `train`, `replay`, TensorBoard, and manual file paths separately.
- There was no guided experiment scaffold, preflight summary, run browser, status vocabulary, or resume path selection.
- Users had to inspect raw JSONL for progress and metrics; comparison and plots were manual.
- Replay required supplying the config and run directory together.
- Run status was implicit: a checkpoint alone cannot prove successful completion.
- Documentation was spread across README files and engineering reports, with the historical Unity research distinction not prominent at repository root.
- Preflight validates dataclass contracts but cannot validate callback output shapes without an actual environment. This distinction must remain explicit.

## Planned interface and compatibility

- Add `python -m casu_grid.workbench.lab` as one scriptable CLI plus an interactive menu. Keep existing modules as the execution backend.
- Add file-backed discovery and analysis only; no database and no edits to old run folders.
- Mark runs started through the CLI and record explicit completion/failure/interruption evidence. Existing runs without those artifacts remain unknown/stale or interrupted, never guessed complete.
- Scaffold an ordinary Python contract file from the existing engineering example; preserve explicit engineering-only labeling and refuse overwrite.
- Add bounded synthetic tests for file operations. Keep Isaac-dependent checks as a separate bounded integration workflow.

## Risks and coverage

- Existing runs may have missing manifests or partial JSONL; discovery should keep them visible and explain unavailable fields.
- Checkpoint compatibility must match the trainer's exact fingerprint algorithm.
- Source configs can disappear; replay should require an explicit preserved copy rather than silently using another.
- New tests cover scaffolding, config errors, discovery/status, comparisons, metrics corruption, plots, CLI and historical-file preservation. The full existing suite remains the simulator regression gate.
