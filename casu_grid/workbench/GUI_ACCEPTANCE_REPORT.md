# Graphical Research Studio acceptance report

Date: 2026-10-13. Implementation scope: local PySide6 desktop client and
subprocess-isolated reward registry integrated with the existing workbench.

## Framework and architecture

PySide6 was selected because it is installed in the local system Python and
provides native Linux desktop widgets, tables, forms, dialogs and background
workers without initializing Isaac. The application entry point is
`python -m casu_grid.workbench.gui`. The frontend delegates configuration
preflight, run discovery, training launch, monitoring artifacts, comparison,
plots, resume and replay to the existing `casu_grid.workbench.lab` CLI. Isaac
operations use the configured bundled `python.sh` with the setup environment;
lightweight browsing and editing use normal Python.

```text
PySide6 Research Studio
  ├── local read-only discovery and explicit Python editor
  ├── reward registry ── JSON discovery ── selected worker subprocess
  └── existing workbench CLI/services
         ├── experiment contracts and runtime callback checks
         ├── PPO training / checkpoint resume (Isaac runtime)
         └── run status, metrics, comparison and deterministic replay
```

Python configs and run manifests remain authoritative. The editor checks the
opened file hash before saving, parses proposed Python, shows a unified diff,
and requires an explicit save. Configuration browsing parses literal
`EXPERIMENT` metadata without executing every candidate Python file.
Preference persistence covers window geometry, selected runtime, and run root.

## Delivered workflow

- Home, experiments, editor, components, training, run/metrics, comparison,
  replay, environment and help pages.
- Experiment creation uses the existing no-overwrite scaffold. Configs can be
  opened, duplicated with a new identity/parent, and hidden/restored through a
  separate registry without deleting research files. The library shows file
  modification time, hash and associated runs when paths/IDs are recorded.
- Contract inspection reuses workbench validation and exposes action mappings,
  ten-tick cadence, declared observations, reward terms, model and PPO settings.
  Built-in observation source inventory identifies `player_state` from the
  default adapter. Custom callback sources remain code-defined and require a
  runtime probe; there is no fabricated sample observation.
- An explicit one-world observation probe performs canonical reset/step/reset,
  checks finite values and declared batch shapes, and reports observation
  summaries and reward outputs; it does not start automatically.
- Reward modules have a versioned JSON sidecar contract documented in
  `REWARD_COMPONENTS.md`. Discovery is passive, importing requires explicit
  selection, and execution/synthetic probes occur in a five-second worker.
  Source hashes and selected parameters are recorded in the experiment and
  exact component source is copied into every run. Training, replay and resume
  resolve the same reference while preserving action/observation callbacks.
- Training displays the command preview and requests launch confirmation.
  It uses `python.sh`, writes to a new run folder and supports graceful SIGINT
  delivery to the supervised process group.
  The run root is configurable and metrics refresh can be toggled while the
  process continues to be supervised.
  Historical run discovery and metrics use real manifests/artifacts.
- Comparison and replay reuse shared CLI services. Replay is simulator action
  tape replay, not rendered gameplay. Metrics and replay path plots open in an
  in-app image viewer; exact trajectory values remain in the replay CSV.

## Verification evidence

- `python -m compileall -q casu_grid/workbench`: passed.
- `python -m unittest casu_grid.workbench.test_components casu_grid.workbench.test_lab casu_grid.workbench.test_gui`:
  **21 passed**.
- Isaac bundled Python `python -m unittest casu_grid.tests.test_workbench`:
  **14 passed** (PPO and callback regression).
- `QT_QPA_PLATFORM=offscreen python -m casu_grid.workbench.gui --smoke`: passed;
  Qt application constructed all pages and completed read-only discovery.
- Existing CLI `--help`, launch preview and availability remain intact.
- Real Isaac bounded engineering smoke: one update, one environment, two-decision
  episode limit; completed, emitted 32 policy decisions / 320 physics ticks,
  metrics and `latest.pt`. Metrics reported 816.6 aggregate physics ticks/s in
  that run. This short sample is not a throughput comparison.
- Live observation probe: **passed** on the canonical fixture. Reset/step/reset
  observations were each float32 with runtime batch shape `(1, 8)` and all
  finite values; reset summaries matched and the probe's consistency assertion
  passed. It covered one 10-tick policy
  decision and returned finite reward/term values.
- Imported custom reward integration: selected custom callback trained one
  update on canonical Isaac Layer 1; its `progress_delta` and `goal` terms were
  emitted in PPO metrics. Run manifest contains source-hash provenance and a
  run-local source copy.
- Deterministic replay of that run with original component source moved aside:
  passed for two recorded decisions, max pose/velocity deviation 0.0 and exact
  config hash match.
- Exact-checkpoint resume from the component run while its original module was
  unavailable: passed; update advanced to 2 and custom reward terms remained in
  metrics. The resumed run also contains its own source snapshot.
- Metrics plot generated in Isaac Python. Two-run comparison of original and
  resumed runs completed and reported their environment configuration
  difference.
- GUI integration: the offscreen PySide6 app launched two-update runs with
  metric polling both disabled and enabled, discovered the monitored run,
  launched a replay through its page and opened the resulting path plot, and
  generated/opened the run's metrics chart in the in-app image viewer.

These are engineering-only checks, not an approved Timmy experiment. The
surrogate remains source-informed rather than bit-exact CasU.

## Limits and next work

- Synthetic callback validation uses the author's declared synthetic context;
  the live probe is a single generic observation/action/reward boundary check,
  not an exhaustive probe for stateful or multi-world components.
- Observation fields are inspectable from contracts; arbitrary custom encoders
  cannot be statically explained. There is no interactive spatial tensor
  heatmap or dedicated normalization/mask visualization yet.
- GUI component registry currently implements reward callbacks. Other plugin
  categories, richer component history across archived configs, and source-code
  recovery backups remain future work. Experiment hide/restore is implemented
  through a non-destructive registry.
- The editor provides typed controls for model and PPO settings and shows the
  resulting source diff. Action and observation definitions, reward contract
  term/source mappings, and custom callbacks remain Python-authored.
- Plotting needs Matplotlib in either the UI Python or configured Isaac Python.
  The host's system Python lacks Matplotlib; Isaac Python generated the plot.
- Matched throughput check used two runs per profile, each with two updates,
  one environment, fixed seed, engineering smoke config and the same Isaac
  runtime. GUI-run profiles used the PySide6 process supervisor offscreen;
  metric refresh was respectively disabled/enabled. Results are mean
  `physics_ticks_per_second` over the four updates per profile:

  | Profile | Update 1 mean | Update 2 mean | Overall mean | Mean collection seconds |
  |---|---:|---:|---:|---:|
  | CLI, no GUI | 847 | 1,048 | 947 | 0.500 |
  | GUI supervised, polling off | 476 | 1,446 | 961 | 0.474 |
  | GUI supervised, metrics polling on | 480 | 1,417 | 949 | 0.474 |

  The strong first/second-update difference shows warmup/noise dominates this
  short workload. There is no clear monitoring penalty in these samples, but
  the sample is too small to claim a precise overhead bound. Metrics polling
  reads artifact files and does not add simulator hot-path hooks. Artifacts
  are under `/tmp/casu-studio-overhead/` and are not repository outputs.
- The GUI itself was constructed and operated in Qt offscreen mode; manual
  interaction on a live desktop was not recorded. Large-scale session
  supervision and crash recovery remain less exercised than the CLI services.

## Five-minute start

1. Install PySide6 in the project/system Python if needed: `python -m pip install PySide6`.
2. Launch: `python -m casu_grid.workbench.gui`.
3. Create an experiment from the Experiments page; template values are
   engineering placeholders and need researcher review.
4. Inspect/edit its Python contracts. Import a trusted reward with a matching
   `REWARD.terms` contract; see `REWARD_COMPONENTS.md` for metadata.
5. Select the Isaac Sim `python.sh` on Environment, preview a bounded launch,
   then explicitly confirm it. Inspect the resulting run, compare, plot and
   replay from their pages.

## Repository state

No commit was created. The existing user deletions, untracked documents and
concurrent simulator/optimization edits remain untouched. GUI implementation
and docs are under `casu_grid/workbench/`; simulator internals were not changed
by the graphical campaign.
