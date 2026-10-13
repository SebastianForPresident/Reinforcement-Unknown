# XY1 implementation and bounded validation

## Implemented

XY1 is a selectable Research Studio configuration with seven categorical heads, the historical 16×85×49 local spatial shape and two signed tile-offset scalars, gamma 0.99 / GAE lambda 0.98, distance-best progress reward and nine-tile arrival termination. The research configuration explicitly sets the provisional 100,000 decision horizon. The environment samples full-body-clear tile-center spawns, zeros initial velocity, and samples an unfiltered destination in the local 85×49 window. Arrival is immediate; timeout is truncation so PPO bootstraps from the final observation.

The 16 channels preserve the historical grouping and normalization style: 3 block descriptors, 4 sleep-quality channels, 7 fluid categories, player-center marker and cursor marker. Sleep channels are all zero because the surrogate does not simulate sleep. Grid axes are x-first; scalar offsets are destination minus player, with positive y upward.

## Mechanics boundary

Horizontal movement, jump, vertical movement, crouch, look offsets and attack are passed through the simulator action pipeline at the existing ten-tick cadence. Attack raycasts toward the cursor and applies a 25-damage structural strike to terrain using the existing block-health/destruction/collider rebuild mechanisms. It does not reproduce CasU weapon inventory, attack cooldown/animations, limb/creature damage, building damage, loot drops or full attack physics. The controller surrogate does not simulate health systems; this experiment isolates navigation and uses the existing navigation body model rather than a medical/combat model. The accepted compound collider and physics settings remain unchanged. The known missing world-top cap remains unchanged.

The surrogate still has out-of-world/ragdoll mechanics in its general mode. XY1 navigation mode terminates on arrival and otherwise truncates at the configured horizon; this does not claim full CasU parity. Goals at map edges are clamped to valid world cells, so their distribution can be edge-weighted when a spawn is near a boundary.

## Validation evidence

- Contract preflight: passes, all seven heads and observation/reward/PPO contracts visible.
- Focused and adjacent workbench tests under Isaac Python: 33 passing in the final test run.
- Isaac bounded smoke: 2 updates, 1 environment, 128 policy decisions, 1,280 physics ticks, CUDA PhysX, 16,385 trainable parameters, optimizer updates and checkpoint creation; 65–74 decisions/s and about 992–1,061 physics ticks/s. No completed episodes and no arrival; this tests plumbing only.
- Run artifacts: `experiments/xy1-smoke-20261013-rerun/` (`COMPLETE`, `latest.pt`, `metrics.jsonl`, `trajectories.jsonl`, config and environment manifests).

## GUI workflow

See [XY1_OPERATOR_GUIDE.md](workbench/XY1_OPERATOR_GUIDE.md). Replay is an action-tape report and static plot, not a live gameplay viewport. The GUI smoke check could not import `PySide6` from either available Python environment during implementation; configuration listing, command preview, and run discovery were verified through the same GUI backend CLI.

## Proposed first substantive run (not executed)

Use XY1 config unchanged, 16 independent environments, 100,000-decision horizon, 10,000 PPO updates, seed 391728405, compound collider, full geometry, and independent terrain. At smoke throughput, one environment is roughly 65 decisions/s; 16-env throughput must be measured with a short bounded scaling check before committing the researcher to wall-clock expectations. A first substantive run should therefore be authorized separately after review of the bounded evidence and available compute budget.
