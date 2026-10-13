# Source (CasU) harness instrumentation

Snapshot of the opt-in, read-only diagnostics used to collect the CasU reference
traces under `casu_grid/results/` and `fixtures/*/physics_audit/`. They live in
the historical `Reinforcement-Unknown` checkout (uncommitted there) and run only
inside a disposable staged runtime (`PPO_Harness/StageCounterfactualRuntime.sh`).

- `CollisionProbe.cs` — per-tick player traces (pre-FixedUpdate, post-FixedUpdate,
  post-Physics2D), controller private state, body/rigidbody configuration, and the
  Physics2D occupancy/collider audit (`PPO_DIAG_OCC_*`).
- `Plugin.cs.diff-vs-main-HEAD` — the full uncommitted `Plugin.cs` diff (includes
  the `Body.FixedUpdate` prefix hook that calls `RecordPlayerPreFixed`).
- `RunCounterfactual.sh`, `NamespaceChild.sh` — launch scripts with the diagnostic
  environment-variable allowlist. `NamespaceChild.sh` must also be copied into the
  staged runtime's `launcher/` directory.
- `CounterfactualSmoke.py` — smoke client; `--skip-state` avoids the multi-second
  `CF_STATE` snapshot per step so long action tapes are practical.

Resync with `casu_grid/tools/sync_source_harness.sh`.
