# Research Studio V2 — UX report

Written for the researcher who runs XY1 experiments. Screenshots are in `casu_grid/results/studio/screenshots/`; each one is a render of the real running application (offscreen Qt for the
scripted journey, and the real Wayland desktop session for `80`–`83`).

## 1. Main design choices

* **One question per page, in the order of the work:** Overview → Experiments → Training → Metrics → Runs & checkpoints → Evaluation → Help. The sidebar always shows whether anything is
  running (a status chip with glyph, word and colour) and its button opens the monitor.
* **What is fixed vs what you choose is spelled out.** The Experiments page has a 🔒 "scientific contract" block (world, observation, seven action heads, decision cadence, reward, episode
  end, PPO settings) read from the experiment file, and a separate "training workload" block (budget, worlds, workers, PhysX threads, device, seed, checkpoints) with the validated
  64 × 8 × 2 CUDA layout as default. The Studio can only change the second block, and it checks the first against the contract it describes.
* **Validation is a visible checklist, not a surprise at launch.** About ten checks (experiment contract, validated layout, Isaac found, source revision contains the Optimization IV
  corrections, legacy-spawn mode off, GPU, memory, no stray simulator, disk, run folder free, resume compatibility) turn ✔ / ⚠ / ✖ as you edit; *Start* is disabled with the first
  blocking reason in its tooltip. There is no silent fallback (no GPU ⇒ error, not CPU).
* **Status is the real process state**, derived from the launcher's process and the artifacts it leaves (starting / running / stopping / completed / failed / interrupted), never from "the
  command returned". A crashed owner shows as *interrupted* (it left metrics) or *failed*.
* **Honest empty states and honest numbers.** A metric without a sample is a gap, never a zero; every empty chart says why; the Overview says plainly that no learning run exists yet and
  does not present speed measurements as learning. "Is it learning?" reports numbers and says "within the noise" unless the difference is clearly larger than its uncertainty, flags a
  drained sample stream, and always points to held-out evaluation.
* **Teaching at the point of use:** every chart, tile and setting that needs it has an (i) button with a short explanation (the same glossary is on the Help page): environment, decision,
  PPO update, rollout vs end-to-end speed, entropy, KL, clip fraction, explained variance, arrival, truncation, checkpoint, resume, held-out evaluation, "what does a learning curve establish?".
* **Charts are plots, not pictures** (`charts.py`, drawn with QPainter): separate widgets with titles, axis labels and units; hover shows the exact value of every series with the update and
  decision counts; wheel zoom, drag pan, Shift-drag box zoom, double-click reset; raw and/or smoothed (window you choose; gaps stay gaps); range presets and linked zoom; click legend
  entries to hide series; any number of runs overlaid; per-chart export to PNG and CSV, and "export all"; ⤢ opens a detail view (statistics for the visible range, how-to-read text, a list of
  the other charts so the overview is one click away). A 20,000-update run paints in ~70 ms per chart, pans at ~7 ms per repaint and is polled incrementally (0.4 ms for 10 new rows).
* **The interface never blocks on the simulator:** run discovery and the system monitor run on worker threads, metrics are read incrementally from a byte offset, partial or malformed
  lines are skipped and counted.
* **Closing the window does not stop training** (the simulator is a separate process); the Studio says so and picks the run up again when reopened.

## 2. How to navigate

1. **Overview** — readiness tiles (simulator, GPU, memory, source revision, legacy mode), the *What to do next* card (set up a run / training is running / the last run failed / evaluate this
   checkpoint / continue), the selected experiment, and the latest runs (click one to open it in Runs & checkpoints). Screenshots `01`, `50`.
2. **Experiments** — pick an experiment, read the contract, choose a budget preset and settings, watch the validation list. `02`, `51`.
3. **Training** — status badge, elapsed time, decisions, updates, current end-to-end speed, latest checkpoint, progress bar with ETA and launch phase, the "Is it learning?" card with three
   sparklines (click one to open it in Metrics), system health (memory of the run, free memory, GPU, simulation workers alive), the exact launch configuration, and a technical log on demand.
   `10`, `71_*`, `72`.
4. **Metrics** — three tabs: *Navigation & task* (goals per 1,000 decisions, **agents that left the world**, progress, distance, arrival rate, decisions to arrival, episode outcomes,
   excavation, attack frequency, reward components, return), *PPO health* (entropy, KL with the 0.03 target, clip fraction, value loss, explained variance, policy loss, gradient norm),
   *Simulator & system* (decisions/s, time per update, physics ticks, memory, GPU, workers alive). `11`–`16`, `53`, `64`, `65`.
5. **Runs & checkpoints** — table of all runs (filter by text and status, hide short smoke tests), details with artifacts, checkpoints with a compatibility verdict against the selected
   experiment, buttons to open metrics, compare, continue, evaluate, open the folder or the log. `17`, `18`, `21`, `54`.
6. **Evaluation** — form (checkpoint, untrained baseline, held-out worlds, goal draws, horizon, seed, modes) with a cost and uncertainty estimate, progress while running, results as bar
   chart with 95 % intervals, table, and a conservative written reading (difference, interval, p-value). `40`, `41`, `55`, `66`, `67`, `74*`.
7. **Help** — three-minute quick start, searchable glossary, troubleshooting, settings (Isaac folder, run folder).

## 3. How to launch XY1 training

*Experiments* → select **XY1 Point Navigation · Mini World 128 × 128** (★ recommended) → choose **Diagnostic: about 1 million decisions** (or Smoke test for a two-minute pipeline check) →
read the validation list (green = go; ⚠ lines are explained) → **Start training** → confirm the summary (experiment, folder, layout, seed, budget, checkpoints, RAM/disk) → the Training page
opens. The command it runs, and the revision, are shown on the Training page and stored in `studio_launch.json` next to the run.

## 4. How to inspect learning

* Training page: the *Is it learning?* card compares the first and latest windows of goals per 1,000 decisions **and** of progress toward the goal, with their uncertainty.
* Metrics: read goals per 1,000 decisions together with progress and — importantly — **agents that left the world**; use ⤢ for numbers and the explanation.
* Evaluation: the only statement about skill. Run the final checkpoint against the untrained baseline; trust a difference only when its interval excludes zero (the page says which case you are in).

## 5. How to compare runs

Runs & checkpoints: Ctrl/Shift-select several runs → *Compare selected runs* (table of final-window means, layout, seed, revision, with the caveat that different seeds differ by chance)
and the Metrics page opens with the runs overlaid (up to six; run colour, dashes for series; legend chips; linked zoom). Screenshots `16`, `17`, `65`.

## 6. How to resume

Runs & checkpoints: select a run, select a checkpoint (✔/✖ with the reason), **Continue from checkpoint…** → Experiments opens with the starting point chosen, worlds/workers locked to the
checkpoint's layout, learning-contract / layout / experiment-file checks in the validation list → Start. The new run lives in a new folder and its update counter continues (tested: stopped at
update 9 → resumed run starts at 10). Incompatible checkpoints are not offered; a changed experiment file is a warning you must read. `21`, `22`, `23`.
Stopping: *Stop gracefully* finishes the current update, keeps the checkpoint and marks the run interrupted (no pause is offered because none is supported); *Force stop* is the last resort.
Failures: the page names the cause in plain words (worker died, GPU/host out of memory, diverged loss, incompatible checkpoint, Isaac did not start), shows the technical text, and says
whether a checkpoint survived. `30`–`32`.

## 7. Defects found by inspecting screenshots and fixed

Stale widgets painted over the hero card (hide-before-delete); rich text shown as raw tags; "&" swallowed as a mnemonic ("Runs_checkpoints"); spin-box and combo arrows hidden by the
stylesheet (Fusion + palette instead); sidebar labels invisible when unchecked; legends forcing horizontal scrolling (flow layout, shortened run labels); one-column chart grid at 1,150 px;
runs table too tall and truncated; "no navigation statistics" shown while a run was still starting; system health showing zeros for a finished run (now peaks); evaluation table headers
truncated; an all-zero series hidden on the axis frame; misleading "improving" verdict while 100 % of agents were lost (now "drained").

## 8. Testing evidence

* Unit and widget tests (`casu_grid/studio/tests`, 80 tests, ~26 s, offscreen Qt): incremental metrics reading incl. partial/malformed lines and truncation, status from process state,
  launch validation (legacy spawn, layouts, existing folders, no-GPU, resume compatibility, shard partition parity with the trainer), chart maths (smoothing keeps gaps, decimation keeps spikes,
  runs are split at gaps, hover/zoom), catalog ↔ **real trainer keys** (fixture captured from a real run), learning-signal rules, statistics, every page with and without runs, theme switching,
  resume offers only compatible checkpoints, evaluation reading, help search.
* Real-simulator journey (`tests/journey.py`, logs `results/studio/journey_*.log`): launch from the optimized checkout, open XY1, validate, start (smoke 16 × 2 and the real 64 × 8 diagnostic),
  live status and metrics, zoom/pan with real mouse events, linked charts, focus view, graceful stop, checkpoint browse, **resume through CUDA** (update numbering continues), previous-run
  metrics, two-run overlay and comparison, **deliberate failure** (worker killed ⇒ failed, cause named, checkpoint offered), held-out evaluation.
* Trainer side: 151 tests + Kit test, XY1 trajectory gate unchanged, 8/8 trajectory gates, readiness audit 19/19 (20/20 for the variant).
* Real desktop: `80`–`83` (Wayland, 1920 × 1080) via `tests/real_desktop.py`, both Qt's own grab and a desktop screenshot of the actual window.
* Known environment fact: `workbench/test_gui.py` (the legacy window) needs PySide6, which the Isaac interpreter does not have; it passes under the system Python
  (`QT_QPA_PLATFORM=offscreen python3 -m unittest casu_grid.workbench.test_gui`). That is an interpreter issue, not a product failure.

## 9. Known limitations

* Loading a 20,000-update run takes ~0.7 s once (on the GUI thread); polls are instant.
* No pause (unsupported by the trainer); no live gameplay viewport or terrain preview — nothing in the Studio pretends to be one.
* CPU utilisation and per-worker timings are recorded in the metrics but not charted; crouch/jump usage is only in the raw action fractions.
* Evaluation and training compete for the CPU, so the Studio blocks one while the other runs; evaluation has about a minute of start-up.
* Comparison is overlay + table; cross-seed statistics are left to the researcher (the Studio only flags the caveat).
* The learning workflow targets XY1 navigation experiments; other experiment files load but are flagged. Single user, local machine; no keyboard-shortcut set; accessibility beyond non-colour
  status cues (glyph + word) is untested.
* The legacy V1 window (`python -m casu_grid.workbench.gui`) still works and is marked as superseded.
