# XY1 operator guide

## Select and inspect

1. Open Research Studio with `python -m casu_grid.workbench.gui`.
2. Select **Experiments**, search for **XY1 Point Navigation**, and double-click the row.
3. In the editor select **Inspect contracts** to review the seven action branches, `(16,85,49)` spatial input, two relative goal offsets, reward terms, model, PPO settings and ten-tick cadence.

## Preview and launch

4. Select **Training** and browse to `casu_grid/workbench/experiments/xy1_point_navigation.py`.
5. Set output root to `experiments/` on the Environment page if needed. Choose a new run directory. The XY1 config supplies one environment, independent terrain, and a 100,000-decision horizon.
6. Preview the command. Confirm it includes `--terrain-mode independent --episode-steps 100000`.
7. For an infrastructure smoke use 2 updates. For a substantive run, use the proposed plan in the project handoff; do not infer learning from the smoke.

## Monitor, resume and compare

8. Training displays live output and metrics. **Runs & metrics** discovers completed runs and exposes `latest.pt`, metrics and trajectory artifacts. **Compare** aligns run contracts and metrics. Select **Resume** from the run actions only for a checkpoint created from the identical XY1 contract fingerprint.
9. The Replay page reports action-tape measurements and a static path plot; it is not a live gameplay viewport.

Code-only fields include the spatial channel semantics, spawn clearance attempts, goal radii, and attack damage/ray distance. The model/PPO editor can change PPO settings; the experiment source remains authoritative for action, observation, reward and environment contracts.
