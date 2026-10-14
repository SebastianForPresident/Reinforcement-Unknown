"""Training: what is the run doing right now, is it healthy, is it learning, what do I do if it is not?"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from .. import analysis, catalog, launch, theme
from ..charts import ChartSpec, ChartView
from ..runs import MetricsStore, RunRecord, human_bytes, human_count, human_duration, read_json
from ..widgets import (Banner, Card, FlowRow, InfoButton, PageHeader, StatTile, StatusBadge, make_label, open_path, page_container, scroll_page)


def parse_utc(text: str | None) -> float | None:
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


class Sparkline(QtWidgets.QFrame):
    clicked = QtCore.Signal(str)

    def __init__(self, chart_key: str, palette: theme.Palette):
        super().__init__()
        self.chart_key = chart_key
        self.chart = catalog.CHART_BY_KEY[chart_key]
        self.setObjectName("card")
        self.setCursor(QtCore.Qt.PointingHandCursor)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 6)
        layout.setSpacing(2)
        self.title = make_label(f"<b>{self.chart.spec.title}</b>", wrap=False)
        layout.addWidget(self.title)
        spec = ChartSpec(chart_key, self.chart.spec.title, "", empty="no data yet", y_min=self.chart.spec.y_min, y_max=self.chart.spec.y_max,
                         include_zero=self.chart.spec.include_zero, references=self.chart.spec.references, percent=self.chart.spec.percent)
        self.view = ChartView(spec, palette, compact=True)
        self.view.setMinimumHeight(110)
        self.view.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        layout.addWidget(self.view)
        self.last = make_label("", "hint", wrap=False)
        layout.addWidget(self.last)

    def mouseReleaseEvent(self, e):
        self.clicked.emit(self.chart_key)


class TrainingPage(QtWidgets.QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.run_path: Path | None = None
        self.store: catalog.RunData | None = None
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        content, self.layout_ = page_container()
        outer.addWidget(scroll_page(content))
        self.header = PageHeader("Training", "")
        self.badge = StatusBadge()
        self.header.actions.addWidget(self.badge)
        self.layout_.addWidget(self.header)
        # no active run
        self.empty = Card("No run is selected")
        self.empty.add(make_label("Nothing is training right now. Start a run from Experiments, or open a previous run from Runs & checkpoints."))
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Set up a run")
        b.setObjectName("primary")
        b.clicked.connect(lambda: ctx.navigate.emit("experiments", None))
        row.addWidget(b)
        b2 = QtWidgets.QPushButton("Browse runs")
        b2.clicked.connect(lambda: ctx.navigate.emit("runs", None))
        row.addWidget(b2)
        row.addStretch(1)
        self.empty.add(row)
        self.layout_.addWidget(self.empty)
        # failure / notices
        self.notice = Banner("info", "", "")
        self.layout_.addWidget(self.notice)
        self.notice.hide()
        self.failure_details = QtWidgets.QPlainTextEdit()
        self.failure_details.setReadOnly(True)
        self.failure_details.setMaximumHeight(220)
        self.failure_details.hide()
        self.layout_.addWidget(self.failure_details)
        self.health_banner = Banner("warn", "", "")
        self.health_banner.hide()
        self.layout_.addWidget(self.health_banner)
        # controls
        self.controls = QtWidgets.QHBoxLayout()
        self.stop_button = QtWidgets.QPushButton("Stop gracefully")
        self.stop_button.setToolTip("Finishes the current update, keeps the checkpoint and marks the run interrupted. Resumable.")
        self.stop_button.clicked.connect(self._stop)
        self.force_button = QtWidgets.QPushButton("Force stop")
        self.force_button.setObjectName("danger")
        self.force_button.setToolTip("Interrupts the simulator immediately. The checkpoint of the last finished update remains.")
        self.force_button.clicked.connect(self._force)
        self.resume_button = QtWidgets.QPushButton("Resume from latest checkpoint")
        self.resume_button.clicked.connect(self._resume)
        self.metrics_button = QtWidgets.QPushButton("Open all metrics")
        self.metrics_button.clicked.connect(lambda: self.run_path and ctx.navigate.emit("metrics", self.run_path))
        self.folder_button = QtWidgets.QPushButton("Open run folder")
        self.folder_button.clicked.connect(lambda: self.run_path and open_path(self.run_path))
        self.log_button = QtWidgets.QPushButton("Show technical log")
        self.log_button.setCheckable(True)
        self.log_button.toggled.connect(self._toggle_log)
        for w in (self.stop_button, self.force_button, self.resume_button, self.metrics_button, self.folder_button, self.log_button):
            self.controls.addWidget(w)
        self.controls.addStretch(1)
        self.layout_.addLayout(self.controls)
        # tiles
        self.tiles_row = FlowRow(170)
        names = (("elapsed", "Elapsed", ""), ("decisions", "Decisions", "decision"), ("updates", "PPO updates", "update"),
                 ("throughput", "End-to-end speed", "throughput"), ("checkpoint", "Latest checkpoint", "checkpoint"))
        self.tiles = {}
        for key, label, help_key in names:
            tile = StatTile(label, "–", help_key, ctx.show_help)
            self.tiles[key] = tile
            self.tiles_row.add(tile)
        self.layout_.addWidget(self.tiles_row)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(True)
        self.layout_.addWidget(self.progress)
        self.phase = make_label("", "hint")
        self.layout_.addWidget(self.phase)
        # learning
        self.learning = Card("Is it learning?", "Training-time signal on the training terrains. It is not held-out evidence; see Evaluation.", help_key="learning_curve",
                             help_callback=ctx.show_help)
        self.lost_banner = Banner("warn", "", "")
        self.lost_banner.hide()
        self.learning.add(self.lost_banner)
        self.signal_banner = Banner("info", "", "")
        self.learning.add(self.signal_banner)
        self.sparks_row = FlowRow(260)
        self.sparks = {}
        for key in ("nav_success", "nav_progress", "ppo_entropy"):
            s = Sparkline(key, ctx.palette)
            s.clicked.connect(lambda k: ctx.navigate.emit("metrics", {"run": str(self.run_path), "chart": k}))
            self.sparks[key] = s
            self.sparks_row.add(s)
        self.learning.add(self.sparks_row)
        self.layout_.addWidget(self.learning)
        # health
        self.health = Card("System health")
        self.health_grid = QtWidgets.QGridLayout()
        self.health_labels: dict[str, QtWidgets.QLabel] = {}
        for i, (key, label) in enumerate((("memory", "Memory of the run (unique)"), ("available", "Memory available"), ("gpu", "GPU memory"),
                                          ("workers", "Simulation workers alive"), ("swap", "Swap in use"), ("recent", "Recent problems"))):
            self.health_grid.addWidget(make_label(label, "hint", wrap=False), i, 0)
            lab = make_label("–")
            self.health_grid.addWidget(lab, i, 1)
            self.health_labels[key] = lab
        self.health_grid.setColumnStretch(1, 1)
        self.health.add(self.health_grid)
        self.layout_.addWidget(self.health)
        # launch configuration
        self.config_card = Card("Launch configuration", "What this run was started with.")
        self.config_text = make_label("", selectable=True)
        self.config_card.add(self.config_text)
        self.layout_.addWidget(self.config_card)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(400)
        self.log.setMinimumHeight(220)
        self.log.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.log.hide()
        self.layout_.addWidget(self.log)
        self.layout_.addStretch(1)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(2000)
        ctx.runsChanged.connect(self.refresh)
        ctx.monitorSample.connect(lambda _s: self.refresh())
        ctx.themeChanged.connect(lambda: self.badge.set_palette(ctx.palette))
        self._stop_requested_at: float | None = None
        self.refresh()

    # ------------------------------------------------------------------------------------------ selection
    def on_show(self, payload=None):
        if isinstance(payload, (str, Path)):
            self.set_run(Path(payload))
        elif self.run_path is None or not self.ctx.run_by_path(self.run_path):
            self.set_run(None)
        self.refresh()

    def set_run(self, path: Path | None):
        if path is None:
            active = self.ctx.active_run() or self.ctx.session_run() or (self.ctx.runs[0] if self.ctx.runs else None)
            path = active.path if active else None
        if path != self.run_path:
            self.run_path = path
            self.store = None
            self._stop_requested_at = None
            if path is not None:
                self.store = catalog.RunData(path.name, MetricsStore(path / "metrics.jsonl"), color=theme.SERIES_COLORS[0])

    # ------------------------------------------------------------------------------------------ actions
    def _run(self) -> RunRecord | None:
        return self.ctx.run_by_path(self.run_path) if self.run_path else None

    def _stop(self):
        run = self._run()
        if run is None:
            return
        answer = QtWidgets.QMessageBox.question(self, "Stop training?", "The trainer will finish the update it is working on (a few seconds), keep its checkpoint and mark the run "
                                                "“interrupted”. You can resume it later.\n\nStop now?")
        if answer == QtWidgets.QMessageBox.Yes:
            self._stop_requested_at = time.time()
            self.ctx.stop_training(run)

    def _force(self):
        run = self._run()
        if run is None:
            return
        answer = QtWidgets.QMessageBox.question(self, "Force stop?", "This interrupts the simulator immediately. The checkpoint of the last finished update stays valid, "
                                                "but the work of the current update is lost.\n\nForce stop?")
        if answer == QtWidgets.QMessageBox.Yes:
            self.ctx.stop_training(run, force=True)

    def _resume(self):
        run = self._run()
        if run and run.latest:
            self.ctx.navigate.emit("experiments", {"resume": str(run.latest.path), "run": str(run.path)})

    def _toggle_log(self, on: bool):
        self.log.setVisible(on)
        self.refresh()

    # ------------------------------------------------------------------------------------------ refresh
    def refresh(self):
        run = self._run()
        if self.run_path is None or run is None:
            if self.run_path is None:
                self.set_run(None)
                run = self._run()
        have = run is not None
        for w in (self.tiles_row, self.progress, self.phase, self.learning, self.health, self.config_card):
            w.setVisible(have)
        self.empty.setVisible(not have)
        for b in (self.stop_button, self.force_button, self.resume_button, self.metrics_button, self.folder_button, self.log_button):
            b.setVisible(have)
        if not have:
            self.badge.set_status("ready")
            self.header.subtitle.setText("No run is active")
            return
        self.badge.set_palette(self.ctx.palette)
        self.badge.set_status(run.status)
        self.header.title.setText("Training" if run.status in ("starting", "running", "stopping") else "Run")
        self.header.subtitle.setText(f"{run.experiment} · {run.run_id}")
        if self.store is None:
            self.store = catalog.RunData(run.path.name, MetricsStore(run.path / "metrics.jsonl"))
        try:
            self.store.refresh()
        except Exception:
            pass
        alive = run.status in ("starting", "running", "stopping")
        self._controls(run, alive)
        self._tiles(run, alive)
        self._learning(run)
        self._health(run, alive)
        self._config(run)
        self._notices(run, alive)
        if self.log.isVisible():
            text = self._log_text(run)
            if text != getattr(self, "_last_log", None):
                self._last_log = text
                self.log.setPlainText(text)
                self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _controls(self, run: RunRecord, alive: bool):
        self.stop_button.setVisible(run.status in ("running", "starting"))
        self.stop_button.setEnabled(run.status in ("running", "starting"))
        self.force_button.setVisible(alive)
        stuck = run.status == "stopping" and self._stop_requested_at and time.time() - self._stop_requested_at > 45
        self.force_button.setObjectName("danger")
        self.resume_button.setVisible((not alive) and run.latest is not None)
        self.metrics_button.setEnabled(run.has_metrics)
        if stuck:
            self.force_button.setText("Force stop (the graceful stop is taking long)")
        else:
            self.force_button.setText("Force stop")

    def _tiles(self, run: RunRecord, alive: bool):
        m = self.store.metrics if self.store else None
        created = parse_utc(run.created_utc)
        if alive and created:
            elapsed = time.time() - created
        elif created and run.mtime:
            elapsed = run.mtime - created
        else:
            elapsed = None
        self.tiles["elapsed"].set(human_duration(elapsed))
        budget_updates = run.updates_requested
        budget_decisions = None
        if budget_updates and m is not None and m.rows:
            per = m.column("environment_steps")[-1] / max(1, run.updates_done)
            budget_decisions = per * budget_updates
        if run.start_update and m is not None and m.rows:
            per = m.column("environment_steps")[-1] - run.start_decisions
            budget_decisions = per / max(1, run.updates_in_run) * budget_updates if budget_updates else None
            self.tiles["decisions"].set(human_count(run.decisions), (f"+{human_count(per)} in this run, of {human_count(budget_decisions)}" if budget_decisions else f"+{human_count(per)} in this run"))
            self.tiles["updates"].set(f"{run.updates_done:,}", (f"{run.updates_in_run} of {budget_updates} in this run (continuing from {run.start_update})" if budget_updates else f"continuing from {run.start_update}"))
        else:
            self.tiles["decisions"].set(human_count(run.decisions), f"of {human_count(budget_decisions)}" if budget_decisions else "")
            self.tiles["updates"].set(f"{run.updates_done:,}", f"of {budget_updates:,}" if budget_updates else "")
        speed = None
        if m is not None and m.rows:
            e2e = [v for v in self.store.derived("e2e_decisions_per_s")[-5:] if v]
            if e2e:
                speed = sum(e2e) / len(e2e)
        self.tiles["throughput"].set(f"{speed:,.0f} /s" if speed else "–", "mean of last 5 updates; reference ≈ 1,700 /s for 64 × 8" if speed else "")
        latest = run.latest
        if latest is not None:
            age = time.time() - latest.mtime if latest.mtime else None
            self.tiles["checkpoint"].set(f"update {latest.update}" if latest.update is not None else "saved",
                                         f"{human_duration(age)} ago · {human_bytes(latest.size_bytes)}" if age is not None else "")
        else:
            self.tiles["checkpoint"].set("none yet", "written after every update" if alive else "")
        if budget_updates:
            done_here = run.updates_in_run
            self.progress.setRange(0, budget_updates)
            self.progress.setValue(min(done_here, budget_updates))
            remaining = max(0, budget_updates - done_here)
            per_update = None
            if m is not None and m.rows >= 2:
                last = [a + b for a, b in zip(m.column("collection_seconds")[-10:], m.column("optimization_seconds")[-10:]) if a and b]
                per_update = sum(last) / len(last) if last else None
            eta = f" · about {human_duration(remaining * per_update)} left" if (alive and per_update) else ""
            self.progress.setFormat(f"{done_here} / {budget_updates} updates ({100 * done_here / budget_updates:.0f} %){eta}")
        else:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.progress.setFormat("budget unknown")
        self.phase.setText(self._phase_text(run, alive))

    def _phase_text(self, run: RunRecord, alive: bool) -> str:
        if run.status == "starting":
            if (run.path / "environment.json").exists():
                return "Worlds are built; the first rollout is being collected. The first update takes longer than the rest."
            if (run.path / "research_config.py").exists():
                return "Isaac Sim is up and the worlds are being built (the 64-world setup needs about one minute)…"
            return "Launching Isaac Sim…"
        if run.status == "stopping":
            return "Stop requested: the trainer is finishing the current update and will then exit."
        if run.status == "running":
            return "Running. Metrics update after every PPO update."
        if run.status == "interrupted":
            return "Stopped. Its latest checkpoint can be resumed."
        if run.status == "completed":
            return "Finished its update budget."
        return ""

    def _learning(self, run: RunRecord):
        palette = self.ctx.palette
        for spark in self.sparks.values():
            spark.view.set_palette(palette)
        if self.store is None:
            return
        sig = analysis.learning_signal(self.store.metrics)
        lost = analysis.outside_world_share(self.store.metrics)
        if lost is not None and lost >= 0.10:
            self.lost_banner.set("bad" if lost >= 0.4 else "warn", f"{100 * lost:.0f} % of the agents have left the world",
                                 "They dug through the floor and are falling; nothing ends such an episode, so they are lost to training until the 100,000-decision ceiling. "
                                 "This is a property of the experiment as defined, not a Studio fault. See the chart “Agents that left the world”.")
            self.lost_banner.show()
        else:
            self.lost_banner.hide()
        level = {"improving": "good", "worse": "warn", "no_change": "info", "too_early": "info", "none": "warn", "drained": "bad"}[sig.state]
        self.signal_banner.set(level, sig.headline, sig.detail)
        for key, spark in self.sparks.items():
            series = catalog.build_series(spark.chart, [self.store], "update")[:1]
            for s in series:
                s.smoothed = None
            spark.view.set_series(series, "Update", True)
            if series and series[0].finite_count():
                vals = [y for y in series[0].ys if y is not None]
                spark.last.setText(f"latest {vals[-1]:.3g}" + (f" · mean of last 10: {sum(vals[-10:]) / len(vals[-10:]):.3g}" if len(vals) >= 3 else ""))
            else:
                spark.last.setText("no data yet")

    def _health(self, run: RunRecord, alive: bool):
        L = self.health_labels
        sample = self.ctx.last_monitor if alive else None
        peaks = None
        if not alive or sample is None:
            sys_store = MetricsStore(run.path / "system.jsonl")
            sys_store.refresh()
            if sys_store.rows:
                live_rows = [i for i, w in enumerate(sys_store.column("workers_alive")) if w]
                if live_rows:
                    def col(key):
                        return [sys_store.column(key)[i] for i in live_rows if sys_store.column(key)[i] is not None]
                    peaks = {"uss": max(col("uss_gib") or [0]), "pss": max(col("pss_gib") or [0]), "avail": min(col("mem_available_gib") or [0]),
                             "gpu": max(col("gpu_mem_mib") or [0]), "workers": max(col("workers_alive") or [0]), "swap": max(col("swap_used_gib") or [0]),
                             "samples": len(live_rows)}
                if sample is None and alive:
                    sample = {k: sys_store.column(k)[-1] for k in sys_store.columns}
        if alive and sample is not None:
            L["memory"].setText(f"{sample.get('uss_gib', 0):.1f} GiB unique · {sample.get('pss_gib', 0):.1f} GiB proportional")
            avail = sample.get("mem_available_gib")
            L["available"].setText(f"{avail:.1f} GiB" if avail is not None else "–")
            gpu = sample.get("gpu_mem_mib")
            L["gpu"].setText(f"{gpu / 1024:.1f} GiB in use · {sample.get('gpu_util', 0)} % busy" if gpu is not None else "–")
            expected = run.workers
            wa = sample.get("workers_alive")
            L["workers"].setText(f"{wa} of {expected}" if expected else str(wa))
            L["swap"].setText(f"{sample.get('swap_used_gib', 0):.1f} GiB")
            problems = []
            if expected and wa is not None and wa < expected and run.status == "running":
                problems.append(f"only {wa} of {expected} simulation workers are alive")
            if avail is not None and avail < 3:
                problems.append(f"only {avail:.1f} GiB of memory left")
            L["recent"].setText("; ".join(problems) if problems else "none")
            if problems:
                self.health_banner.set("bad", "The run may be in trouble", "; ".join(problems) + ". If it does not recover, stop it gracefully; the last checkpoint is kept.")
                self.health_banner.show()
            else:
                self.health_banner.hide()
            return
        self.health_banner.hide()
        if peaks:
            L["memory"].setText(f"peak {peaks['uss']:.1f} GiB unique · {peaks['pss']:.1f} GiB proportional (during the run)")
            L["available"].setText(f"lowest {peaks['avail']:.1f} GiB during the run")
            L["gpu"].setText(f"peak {peaks['gpu'] / 1024:.1f} GiB in use")
            L["workers"].setText(f"{peaks['workers']} at peak" + (f" of {run.workers}" if run.workers else ""))
            L["swap"].setText(f"up to {peaks['swap']:.1f} GiB")
            L["recent"].setText("see the notice above" if run.status == "failed" else "none recorded")
        else:
            for lab in L.values():
                lab.setText("not sampled yet" if alive else "no samples were recorded for this run (the Studio samples while it watches a run)")
            L["recent"].setText("–")

    def _config(self, run: RunRecord):
        launch_rec = read_json(run.path / "studio_launch.json") or {}
        rows = [("Experiment", run.experiment), ("Run folder", str(run.path)), ("Source revision", f"{(run.revision or launch_rec.get('source_revision') or 'unknown')[:10]}"
                + (f" ({launch_rec.get('source_branch')})" if launch_rec.get("source_branch") else "")
                + (f", {launch_rec.get('dirty_files')} uncommitted file(s)" if launch_rec.get("dirty_files") else "")),
                ("Termination contract", run.termination_contract), ("World layout", f"{run.num_envs} worlds × {run.workers} workers × {run.physx_threads} PhysX threads" if run.num_envs else "–"),
                ("PPO device", (launch_rec.get("device") or run.device or "–").upper()), ("Seed", str(run.seed)),
                ("Budget", f"{run.updates_requested} updates ≈ {human_count(launch_rec.get('estimated_decisions'))} decisions" if run.updates_requested else "–"),
                ("Starting point", f"resumed from {Path(run.resume_from).parent.name} ({Path(run.resume_from).name})" if run.resume_from else "new run, untrained policy"),
                ("Legacy spawn mode", "ON (historical reproduction)" if run.legacy_spawn else "off")]
        text = "<table cellspacing=4>" + "".join(f"<tr><td><span class='muted'>{k}</span></td><td>{v}</td></tr>" for k, v in rows) + "</table>"
        if launch_rec.get("command"):
            text += f"<p class='muted'>Command: <code>{launch_rec['command']}</code></p>"
        self.config_text.setText(text)

    def _notices(self, run: RunRecord, alive: bool):
        if run.status == "failed":
            headline, details = launch.explain_failure(run, self._log_text(run))
            kept = f" Its latest checkpoint (update {run.latest.update}) is intact and can be resumed." if run.latest else " No checkpoint was saved, so it cannot be resumed."
            self.notice.set("bad", headline, "The run stopped with an error." + kept + " The technical details are below.")
            self.notice.show()
            self.failure_details.setPlainText(details)
            self.failure_details.show()
        elif run.status == "interrupted":
            self.notice.set("warn", "This run was stopped before its budget finished",
                            f"{run.updates_done} updates were completed" + (f" and the checkpoint of update {run.latest.update} is saved: you can continue from it." if run.latest else "."))
            self.notice.show()
            self.failure_details.hide()
        else:
            self.notice.hide()
            self.failure_details.hide()

    def _log_text(self, run: RunRecord) -> str:
        for candidate in (run.path / "launcher.log", self.ctx.run_root / ".studio" / f"{run.run_id}.launch.log"):
            text = launch.launch_log_tail(candidate, 120)
            if text:
                return text
        return ""
