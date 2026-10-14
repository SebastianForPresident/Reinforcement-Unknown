"""Evaluation: score a checkpoint (and the untrained baseline) on held-out terrains under identical conditions."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .. import environment as env
from .. import launch, theme
from ..runs import RunRecord, human_duration, read_json
from ..widgets import Banner, Card, InfoButton, PageHeader, make_label, open_path, page_container, scroll_page


class BarChart(QtWidgets.QWidget):
    """Grouped bars (one group per policy, one bar per mode) with 95 % interval whiskers and exact values on hover."""

    def __init__(self, palette: theme.Palette):
        super().__init__()
        self.palette_ = palette
        self.groups: list[tuple[str, list[tuple[str, float | None, list[float] | None, str]]]] = []   # (label, [(series, value, ci, tooltip)])
        self.setMinimumHeight(260)
        self.setMouseTracking(True)
        self._hover: QtCore.QPointF | None = None
        self._hits: list[tuple[QtCore.QRectF, str]] = []

    def set_data(self, groups):
        self.groups = groups
        self.update()

    def mouseMoveEvent(self, e):
        self._hover = e.position()
        self.update()

    def leaveEvent(self, e):
        self._hover = None
        self.update()

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        pal = self.palette_
        p.fillRect(self.rect(), QtGui.QColor(pal.chart_bg))
        rect = QtCore.QRectF(70, 14, self.width() - 90, self.height() - 74)
        values = [v for _, bars in self.groups for _, v, ci, _t in bars if v is not None]
        highs = [ci[1] for _, bars in self.groups for _, v, ci, _t in bars if ci]
        top = max([0.1] + highs + values)
        top = min(1.0, top * 1.15) if top <= 1.0 else top * 1.15
        font = QtGui.QFont(self.font()); font.setPointSizeF(max(8, font.pointSizeF() - 1)); p.setFont(font)
        for i in range(0, 6):
            v = top * i / 5
            y = rect.bottom() - v / top * rect.height()
            p.setPen(QtGui.QColor(pal.chart_grid)); p.drawLine(QtCore.QPointF(rect.left(), y), QtCore.QPointF(rect.right(), y))
            p.setPen(QtGui.QColor(pal.chart_axis)); p.drawText(QtCore.QRectF(0, y - 8, rect.left() - 6, 16), QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, f"{v * 100:.0f} %")
        p.setPen(QtGui.QColor(pal.chart_axis))
        p.drawRect(rect)
        p.save(); p.translate(14, rect.center().y()); p.rotate(-90)
        p.drawText(QtCore.QRectF(-rect.height() / 2, -9, rect.height(), 18), QtCore.Qt.AlignCenter, "episodes that reached the goal"); p.restore()
        self._hits = []
        if not self.groups:
            p.drawText(rect, QtCore.Qt.AlignCenter, "No evaluation selected.")
            return
        slot = rect.width() / len(self.groups)
        for g, (label, bars) in enumerate(self.groups):
            width = min(70.0, slot / (len(bars) + 1))
            left = rect.left() + g * slot + (slot - width * len(bars)) / 2
            for b, (series, value, ci, tip) in enumerate(bars):
                color = QtGui.QColor(theme.SERIES_COLORS[b % len(theme.SERIES_COLORS)])
                if value is None:
                    continue
                h = value / top * rect.height()
                bar = QtCore.QRectF(left + b * width + 3, rect.bottom() - h, width - 6, max(1.5, h))
                p.fillRect(bar, color)
                if ci:
                    cx = bar.center().x()
                    y0, y1 = rect.bottom() - ci[0] / top * rect.height(), rect.bottom() - ci[1] / top * rect.height()
                    p.setPen(QtGui.QPen(QtGui.QColor(pal.text), 1.6))
                    p.drawLine(QtCore.QPointF(cx, y0), QtCore.QPointF(cx, y1))
                    p.drawLine(QtCore.QPointF(cx - 5, y0), QtCore.QPointF(cx + 5, y0))
                    p.drawLine(QtCore.QPointF(cx - 5, y1), QtCore.QPointF(cx + 5, y1))
                p.setPen(QtGui.QColor(pal.text))
                p.drawText(QtCore.QRectF(bar.left() - 10, bar.top() - 34 if not ci else rect.bottom() - (ci[1] / top) * rect.height() - 18, bar.width() + 20, 16),
                           QtCore.Qt.AlignCenter, f"{value * 100:.1f} %")
                self._hits.append((bar.adjusted(0, -(rect.height()), 0, 0), tip))
            p.setPen(QtGui.QColor(pal.chart_axis))
            p.drawText(QtCore.QRectF(rect.left() + g * slot, rect.bottom() + 6, slot, 36), QtCore.Qt.AlignHCenter | QtCore.Qt.TextWordWrap, label)
        # legend
        names = []
        for _, bars in self.groups:
            for s, *_ in bars:
                if s not in names:
                    names.append(s)
        x = rect.left()
        for i, name in enumerate(names):
            p.fillRect(QtCore.QRectF(x, self.height() - 16, 10, 10), QtGui.QColor(theme.SERIES_COLORS[i % len(theme.SERIES_COLORS)]))
            p.setPen(QtGui.QColor(pal.text)); p.drawText(QtCore.QPointF(x + 14, self.height() - 7), name)
            x += 28 + p.fontMetrics().horizontalAdvance(name)
        if self._hover is not None:
            for r, tip in self._hits:
                if r.contains(self._hover):
                    fm = p.fontMetrics()
                    lines = tip.split("\n")
                    w = max(fm.horizontalAdvance(t) for t in lines) + 16
                    h = len(lines) * (fm.height() + 2) + 8
                    left = min(self._hover.x() + 12, self.width() - w - 4)
                    box = QtCore.QRectF(left, max(4, self._hover.y() - h - 6), w, h)
                    p.setBrush(QtGui.QColor(pal.panel)); p.setPen(QtGui.QColor(pal.border)); p.drawRoundedRect(box, 6, 6)
                    p.setPen(QtGui.QColor(pal.text))
                    for k, t in enumerate(lines):
                        p.drawText(QtCore.QPointF(box.left() + 8, box.top() + 6 + fm.ascent() + k * (fm.height() + 2)), t)
                    break


class EvaluationPage(QtWidgets.QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.process: subprocess.Popen | None = None
        self.log_path: Path | None = None
        self.out_path: Path | None = None
        self.started = 0.0
        self.total_steps = 0
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        content, self.layout_ = page_container()
        outer.addWidget(scroll_page(content))
        self.layout_.addWidget(PageHeader("Evaluation", "Does the policy actually navigate? Controlled tests on terrains it has never seen."))
        intro = Banner("info", "Training curves are not evidence of generalisation",
                       "Training statistics come from an exploring policy on the 64 training terrains. This page scores a checkpoint and the untrained baseline on held-out terrains, "
                       "with identical goals, a fixed decision horizon and no learning, so the numbers are comparable.")
        self.layout_.addWidget(intro)
        # run form
        self.form = Card("Run an evaluation", help_key="evaluation", help_callback=ctx.show_help)
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(12)
        self.run_combo = QtWidgets.QComboBox()
        self.run_combo.currentIndexChanged.connect(self._run_changed)
        self.ck_combo = QtWidgets.QComboBox()
        self.baseline = QtWidgets.QCheckBox("Include the untrained baseline (the same initial network, same seed)")
        self.baseline.setChecked(True)
        self.worlds = QtWidgets.QSpinBox(); self.worlds.setRange(4, 64); self.worlds.setValue(16)
        self.rounds = QtWidgets.QSpinBox(); self.rounds.setRange(1, 50); self.rounds.setValue(4)
        self.horizon = QtWidgets.QSpinBox(); self.horizon.setRange(100, 20000); self.horizon.setValue(1500); self.horizon.setSingleStep(100)
        self.seed = QtWidgets.QSpinBox(); self.seed.setRange(0, 2**31 - 1); self.seed.setValue(20261014)
        self.det = QtWidgets.QCheckBox("deterministic (most likely action)"); self.det.setChecked(True)
        self.sto = QtWidgets.QCheckBox("stochastic (sampled, like training)"); self.sto.setChecked(True)
        rows = [("Run", self.run_combo, None), ("Checkpoint", self.ck_combo, None), ("", self.baseline, None), ("Held-out worlds", self.worlds, None),
                ("Goal draws per world", self.rounds, None), ("Horizon (decisions per episode)", self.horizon, "truncation"), ("Evaluation seed", self.seed, None),
                ("", self.det, None), ("", self.sto, None)]
        for i, (label, widget, key) in enumerate(rows):
            grid.addWidget(make_label(label, wrap=False), i, 0)
            grid.addWidget(widget, i, 1)
            if key:
                grid.addWidget(InfoButton(key, ctx.show_help), i, 2)
        grid.setColumnStretch(1, 1)
        self.form.add(grid)
        for w in (self.worlds, self.rounds, self.horizon):
            w.valueChanged.connect(self._estimate)
        for w in (self.baseline, self.det, self.sto):
            w.toggled.connect(self._estimate)
        self.estimate = Banner("info", "", "")
        self.form.add(self.estimate)
        row = QtWidgets.QHBoxLayout()
        self.start = QtWidgets.QPushButton("Run evaluation")
        self.start.setObjectName("primary")
        self.start.clicked.connect(self._start)
        self.cancel = QtWidgets.QPushButton("Cancel")
        self.cancel.clicked.connect(self._cancel)
        self.cancel.hide()
        row.addWidget(self.start)
        row.addWidget(self.cancel)
        row.addStretch(1)
        self.form.add(row)
        self.progress = QtWidgets.QProgressBar()
        self.progress.hide()
        self.form.add(self.progress)
        self.status = make_label("", "hint")
        self.form.add(self.status)
        self.layout_.addWidget(self.form)
        # results
        self.results = Card("Results")
        top = QtWidgets.QHBoxLayout()
        self.result_combo = QtWidgets.QComboBox()
        self.result_combo.currentIndexChanged.connect(self._show_result)
        top.addWidget(make_label("Evaluation:", wrap=False))
        top.addWidget(self.result_combo, 1)
        self.open_button = QtWidgets.QPushButton("Open file")
        self.open_button.clicked.connect(lambda: self._result_path() and open_path(self._result_path()))
        top.addWidget(self.open_button)
        self.results.add(top)
        self.protocol = make_label("", "hint")
        self.results.add(self.protocol)
        self.chart = BarChart(ctx.palette)
        self.results.add(self.chart)
        self.table = QtWidgets.QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["Policy", "Mode", "Episodes", "Reached the goal", "95 % interval", "Median decisions to arrive", "Progress toward goal", "Blocks destroyed"])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(190)
        self.results.add(self.table)
        self.reading = Banner("info", "", "")
        self.results.add(self.reading)
        self.layout_.addWidget(self.results)
        self.layout_.addStretch(1)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(1000)
        ctx.runsChanged.connect(self._fill_runs)
        ctx.themeChanged.connect(lambda: (setattr(self.chart, "palette_", ctx.palette), self.chart.update()))
        self._results: list[Path] = []
        self._fill_runs()

    # ------------------------------------------------------------------------------------------ form
    def on_show(self, payload=None):
        self._fill_runs()
        if isinstance(payload, dict):
            if payload.get("run"):
                index = self.run_combo.findData(payload["run"])
                if index >= 0:
                    self.run_combo.setCurrentIndex(index)
            if payload.get("checkpoint"):
                index = self.ck_combo.findData(payload["checkpoint"])
                if index >= 0:
                    self.ck_combo.setCurrentIndex(index)
        self._estimate()
        self._fill_results()

    def _fill_runs(self):
        current = self.run_combo.currentData()
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        for run in self.ctx.runs:
            if run.checkpoints:
                self.run_combo.addItem(f"{run.run_id}  ·  {run.updates_done} updates  ·  {theme.STATUS[run.status][1]}", str(run.path))
        if self.run_combo.count() == 0:
            self.run_combo.addItem("(no run has a checkpoint yet)", None)
        index = self.run_combo.findData(current)
        self.run_combo.setCurrentIndex(max(0, index))
        self.run_combo.blockSignals(False)
        self._run_changed()

    def _run_changed(self):
        run = self.ctx.run_by_path(self.run_combo.currentData()) if self.run_combo.currentData() else None
        self.ck_combo.clear()
        if run:
            for c in run.checkpoints:
                if c.kind != "initial":
                    self.ck_combo.addItem(c.label, str(c.path))
            if any(c.kind == "initial" for c in run.checkpoints):
                for c in run.checkpoints:
                    if c.kind == "initial":
                        self.ck_combo.addItem("initial (saved untrained policy)", str(c.path))
        self._estimate()
        self._fill_results()

    def _policies(self) -> list[str]:
        out = []
        if self.baseline.isChecked():
            out.append("untrained")
        if self.ck_combo.currentData():
            out.append(f"checkpoint={self.ck_combo.currentData()}")
        return out

    def _modes(self) -> list[str]:
        return [m for m, w in (("deterministic", self.det), ("stochastic", self.sto)) if w.isChecked()]

    def _estimate(self, *_):
        policies, modes = self._policies(), self._modes()
        decisions = len(policies) * len(modes) * self.rounds.value() * self.worlds.value() * self.horizon.value()
        seconds = decisions / 900 + 70
        busy = self.ctx.active_run() is not None or self.process is not None
        problems = []
        if not policies:
            problems.append("choose a checkpoint or the baseline")
        if not modes:
            problems.append("choose at least one policy mode")
        if busy:
            problems.append("training or another evaluation is running: they compete for the CPU and would slow each other down")
        if not self.ctx.install.ok:
            problems.append("Isaac Sim was not found")
        if self.ctx.selected_experiment is None:
            problems.append("no experiment selected")
        self._problems = problems
        episodes = self.worlds.value() * self.rounds.value()
        self.estimate.set("bad" if problems else "info", "Cannot start yet" if problems else f"{episodes} episodes per policy and mode",
                          "; ".join(problems) if problems else f"About {human_duration(seconds)} (including about a minute of start-up). The uncertainty of a success rate from {episodes} episodes "
                                                               f"is roughly ±{100 * 0.98 / max(1, episodes) ** 0.5:.0f} points at 50 %: differences smaller than that are not evidence.")
        self.start.setEnabled(not problems)

    # ------------------------------------------------------------------------------------------ run
    def _start(self):
        run = self.ctx.run_by_path(self.run_combo.currentData())
        exp = self.ctx.selected_experiment
        if run is None:
            return
        cfg = Path(run.config_path) if run.config_path and Path(run.config_path).exists() else exp.path
        folder = run.path / "evaluations"
        folder.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.out_path = folder / f"eval-{stamp}.json"
        self.log_path = folder / f"eval-{stamp}.log"
        policies = self._policies()
        cmd = [str(self.ctx.install.python_sh), "-m", "casu_grid.workbench.evaluate", "--config", str(cfg), "--output", str(self.out_path),
               "--worlds", str(self.worlds.value()), "--rounds", str(self.rounds.value()), "--horizon", str(self.horizon.value()),
               "--eval-seed", str(self.seed.value()), "--init-seed", str(run.seed if run.seed is not None else 7), "--modes", ",".join(self._modes())]
        for p in policies:
            cmd += ["--policy", p]
        child_env = env.clean_child_environment(env.process_environment(self.ctx.install))
        child_env.update({"CASU_NUM_WORKERS": "4", "CASU_WORKER_PHYSX_THREADS": "2", "CASU_TRAIN_DEVICE": "cuda", "PYTHONUNBUFFERED": "1"})
        if env.inherited_overrides().get("CASU_LEGACY_MINI_SPAWN") == "1":
            self.status.setText("Legacy spawn mode is on; evaluation refused.")
            return
        handle = self.log_path.open("wb")
        self.process = subprocess.Popen(cmd, cwd=env.ROOT, env=child_env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        handle.close()
        self.started = time.time()
        self.total_steps = len(policies) * len(self._modes()) * self.rounds.value()
        self.progress.setRange(0, max(1, self.total_steps))
        self.progress.setValue(0)
        self.progress.setFormat("starting the simulator…")
        self.progress.show()
        self.cancel.show()
        self.start.setEnabled(False)
        self.status.setText(f"Writing to {self.out_path}")

    def _cancel(self):
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            self.status.setText("Cancelled.")

    def _poll(self):
        if self.process is None:
            return
        done = 0
        if self.log_path and self.log_path.exists():
            for line in self.log_path.read_text(errors="replace").splitlines():
                if line.startswith("CASU_EVAL_PROGRESS"):
                    try:
                        done = int(line.split()[1].split("/")[0])
                    except (ValueError, IndexError):
                        pass
        if done:
            self.progress.setValue(done)
            self.progress.setFormat(f"{done} / {self.total_steps} rounds · {human_duration(time.time() - self.started)}")
        if self.process.poll() is not None:
            ok = self.out_path is not None and self.out_path.exists()
            self.process = None
            self.progress.hide()
            self.cancel.hide()
            if ok:
                self.status.setText(f"Finished in {human_duration(time.time() - self.started)}.")
                self._fill_results(select=self.out_path)
            else:
                tail = launch.launch_log_tail(self.log_path, 12) if self.log_path else ""
                self.status.setText("The evaluation failed. The last lines of its log:\n" + tail)
            self._estimate()
            self.ctx.refresh_runs()

    # ------------------------------------------------------------------------------------------ results
    def _fill_results(self, select: Path | None = None):
        files = []
        for run in self.ctx.runs:
            files.extend(run.evaluations)
        files = sorted(files, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        self._results = files
        self.result_combo.blockSignals(True)
        self.result_combo.clear()
        for f in files:
            self.result_combo.addItem(f"{f.parent.parent.name}  ·  {f.stem}", str(f))
        if not files:
            self.result_combo.addItem("(no evaluation yet)", None)
        if select is not None:
            index = self.result_combo.findData(str(select))
            self.result_combo.setCurrentIndex(max(0, index))
        self.result_combo.blockSignals(False)
        self._show_result()

    def _result_path(self) -> Path | None:
        data = self.result_combo.currentData()
        return Path(data) if data else None

    def _show_result(self, *_):
        path = self._result_path()
        data = read_json(path) if path else None
        if not data:
            self.chart.set_data([])
            self.table.setRowCount(0)
            self.protocol.setText("Run an evaluation above to see the comparison here.")
            self.reading.hide()
            return
        proto = data["protocol"]
        self.protocol.setText(f"{proto['worlds']} held-out worlds × {proto['rounds']} goal draws, horizon {proto['horizon_decisions']} decisions, evaluation seed {proto['eval_seed']}, "
                              f"terrain seeds from {proto['terrain_seed_first']} (training used {proto['training_terrain_seed_first']}+), termination contract {data.get('termination_contract', 'xy1-termination-v1')}, "
                              f"source {(data.get('source_revision') or '?')[:7]}.")
        results = data["results"]
        policies = []
        for r in results:
            if r["policy"] not in policies:
                policies.append(r["policy"])
        groups = []
        for pol in policies:
            bars = []
            for r in results:
                if r["policy"] != pol:
                    continue
                s = r["summary"]
                tip = (f"{pol} · {r['mode']}\n{s['arrivals']} of {s['episodes']} episodes reached the goal\n95 % interval {100 * s['success_rate_ci95'][0]:.0f}–{100 * s['success_rate_ci95'][1]:.0f} %"
                       if s["success_rate_ci95"] else pol)
                bars.append((r["mode"], s["success_rate"], s["success_rate_ci95"], tip))
            update = next((r["checkpoint"]["update_count"] for r in results if r["policy"] == pol), None)
            groups.append((f"{pol}" + (f"\n(update {update})" if update else ""), bars))
        self.chart.set_data(groups)
        self.table.setRowCount(len(results))
        for i, r in enumerate(results):
            s = r["summary"]
            ci = s["success_rate_ci95"]
            vals = [r["policy"], r["mode"], str(s["episodes"]), f"{s['arrivals']} ({100 * s['success_rate']:.1f} %)" if s["success_rate"] is not None else "–",
                    f"{100 * ci[0]:.0f} – {100 * ci[1]:.0f} %" if ci else "–", f"{s['decisions_to_arrival_median']:.0f}" if s["decisions_to_arrival_median"] is not None else "no arrivals",
                    f"{100 * s['mean_progress']:.0f} %" if s["mean_progress"] is not None else "–", f"{s['mean_blocks_destroyed']:.1f}" if s["mean_blocks_destroyed"] is not None else "–"]
            for c, v in enumerate(vals):
                self.table.setItem(i, c, QtWidgets.QTableWidgetItem(v))
        self.reading.set(*self._reading(results))
        self.reading.show()

    @staticmethod
    def _reading(results) -> tuple[str, str, str]:
        from .. import stats
        base = {r["mode"]: r["summary"] for r in results if r["policy"] == "untrained"}
        trained = [r for r in results if r["policy"] != "untrained" and r["policy"] != "initial"]
        if not base or not trained:
            return ("info", "Compare against the untrained baseline", "Include the untrained baseline and a trained checkpoint in one evaluation to see whether anything was learned.")
        lines, verdicts = [], []
        for r in trained:
            b, t = base.get(r["mode"]), r["summary"]
            if b is None or not b.get("episodes") or not t.get("episodes"):
                continue
            d = stats.difference(t["arrivals"], t["episodes"], b["arrivals"], b["episodes"])
            if d["low"] > 0:
                verdict = "above the untrained baseline"
            elif d["high"] < 0:
                verdict = "below the untrained baseline"
            else:
                verdict = "not distinguishable from the untrained baseline"
            verdicts.append(verdict)
            lines.append(f"{r['mode']}: {100 * t['success_rate']:.0f} % vs {100 * b['success_rate']:.0f} % baseline, difference {100 * d['difference']:+.1f} points "
                         f"(95 % interval {100 * d['low']:+.1f} to {100 * d['high']:+.1f}, p = {d['p_value']:.3f}): {verdict}.")
        level = "good" if verdicts and all("above" in v for v in verdicts) else ("warn" if any("below" in v for v in verdicts) else "info")
        title = {"good": "Held-out success is above the baseline", "warn": "Held-out success is below the baseline",
                 "info": "No demonstrated difference from the untrained baseline"}[level]
        return (level, title, " ".join(lines) + " An interval that excludes zero is a necessary, not a sufficient, sign of skill: repeat with more episodes and more seeds "
                "before drawing conclusions, and remember that several comparisons were made.")
