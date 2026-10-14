"""Runs & checkpoints: browse history, inspect a run, compare runs, resume or evaluate a checkpoint."""
from __future__ import annotations

import time
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .. import analysis, launch, theme
from ..runs import CheckpointInfo, MetricsStore, RunRecord, human_bytes, human_count, human_duration
from ..widgets import Banner, Card, PageHeader, StatusBadge, make_label, open_path
from .training import parse_utc

COLUMNS = ["Status", "Run", "Created", "Progress", "Layout", "Seed", "Rev.", "Checkpoint", "Evaluated"]


class RunsPage(QtWidgets.QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self._rows: list[RunRecord] = []
        self._select_path: Path | None = None
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 14)
        outer.setSpacing(10)
        outer.addWidget(PageHeader("Runs & checkpoints", "Every run keeps its own folder; nothing here deletes anything."))
        filters = QtWidgets.QHBoxLayout()
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search runs…")
        self.search.textChanged.connect(self._rebuild)
        filters.addWidget(self.search, 1)
        self.status = QtWidgets.QComboBox()
        self.status.addItem("All statuses", "")
        for key in ("running", "completed", "interrupted", "failed"):
            self.status.addItem(theme.STATUS[key][1], key)
        self.status.currentIndexChanged.connect(self._rebuild)
        filters.addWidget(self.status)
        self.hide_smoke = QtWidgets.QCheckBox("Hide short smoke tests (under 10 updates)")
        self.hide_smoke.toggled.connect(self._rebuild)
        filters.addWidget(self.hide_smoke)
        outer.addLayout(filters)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter, 1)
        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.horizontalHeaderItem(4).setToolTip("worlds × simulation workers")
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        for col, width in enumerate((112, 215, 122, 245, 70, 48, 62, 90)):
            self.table.setColumnWidth(col, width)
        self.table.setMinimumHeight(150)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.cellDoubleClicked.connect(lambda r, c: self._open_metrics())
        splitter.addWidget(self.table)
        # detail
        detail = QtWidgets.QWidget()
        detail.setObjectName("page")
        dl = QtWidgets.QVBoxLayout(detail)
        dl.setContentsMargins(0, 8, 8, 0)
        self.detail_title = make_label("", "cardTitle", wrap=False)
        dl.addWidget(self.detail_title)
        self.detail_banner = Banner("info", "", "")
        self.detail_banner.hide()
        dl.addWidget(self.detail_banner)
        self.detail_text = make_label("", selectable=True)
        dl.addWidget(self.detail_text)
        self.ck_table = QtWidgets.QTableWidget(0, 6)
        self.ck_table.setHorizontalHeaderLabels(["Checkpoint", "Update", "Decisions", "Saved", "Size", "Can continue with the selected experiment?"])
        self.ck_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.ck_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.ck_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.ck_table.verticalHeader().setVisible(False)
        self.ck_table.horizontalHeader().setStretchLastSection(True)
        self.ck_table.setColumnWidth(0, 190)
        self.ck_table.itemSelectionChanged.connect(self._update_buttons)
        self.ck_table.setMinimumHeight(130)
        self.ck_table.setMaximumHeight(190)
        dl.addWidget(self.ck_table)
        buttons = QtWidgets.QHBoxLayout()
        self.b_metrics = self._button("Open metrics", self._open_metrics, buttons, primary=True)
        self.b_compare = self._button("Compare selected runs", self._compare, buttons)
        self.b_resume = self._button("Continue from checkpoint…", self._resume, buttons)
        self.b_eval = self._button("Evaluate checkpoint…", self._evaluate, buttons)
        self.b_folder = self._button("Open folder", lambda: self._current() and open_path(self._current().path), buttons)
        self.b_log = self._button("Open log", self._open_log, buttons)
        self.b_copy = self._button("Copy path", self._copy_path, buttons)
        buttons.addStretch(1)
        dl.addLayout(buttons)
        self.compare_box = QtWidgets.QTextBrowser()
        self.compare_box.setMinimumHeight(260)
        self.compare_box.hide()
        dl.addWidget(self.compare_box)
        dl.addStretch(1)
        detail_scroll = QtWidgets.QScrollArea()
        detail_scroll.setWidgetResizable(True)
        detail_scroll.setWidget(detail)
        splitter.addWidget(detail_scroll)
        splitter.setSizes([230, 520])
        ctx.runsChanged.connect(self._runs_changed)
        self._update_buttons()

    def _button(self, text, fn, layout, primary=False):
        b = QtWidgets.QPushButton(text)
        if primary:
            b.setObjectName("primary")
        b.clicked.connect(fn)
        layout.addWidget(b)
        return b

    # ------------------------------------------------------------------------------------------ table
    def on_show(self, payload=None):
        if isinstance(payload, (str, Path)):
            self._select_path = Path(payload)
        self.ctx.refresh_runs()
        self._rebuild()

    def _runs_changed(self):
        self._rebuild(keep=True)

    def _visible_runs(self) -> list[RunRecord]:
        text = self.search.text().strip().lower()
        out = []
        for r in self.ctx.runs:
            if self.status.currentData() and r.status != self.status.currentData():
                continue
            if self.hide_smoke.isChecked() and r.updates_done < 10 and r.status not in ("running", "starting"):
                continue
            if text and text not in f"{r.run_id} {r.experiment} {r.status} {r.revision or ''}".lower():
                continue
            out.append(r)
        return out

    def _rebuild(self, *_args, keep: bool = False):
        selected = {str(r.path) for r in self._selected_runs()} if (keep or self._select_path is None) else set()
        if self._select_path is not None:
            selected = {str(self._select_path)}
            self._select_path = None
        runs = self._visible_runs()
        sig = [(str(r.path), r.status, r.updates_done, bool(r.latest), len(r.evaluations)) for r in runs]
        if keep and sig == getattr(self, "_sig", None):
            return
        self._sig = sig
        self._rows = runs
        self.table.blockSignals(True)
        self.table.setRowCount(len(runs))
        palette = self.ctx.palette
        for i, r in enumerate(runs):
            created = parse_utc(r.created_utc)
            created_text = time.strftime("%Y-%m-%d %H:%M", time.localtime(created)) if created else "–"
            progress = (f"{r.updates_in_run}" + (f" / {r.updates_requested}" if r.updates_requested else "") + " updates"
                        + (f" (from {r.start_update})" if r.start_update else "") + f" · {human_count(r.decisions)} decisions")
            layout_text = f"{r.num_envs or '?'} × {r.workers or '?'}"
            ck = r.latest
            values = [theme.status_text(r.status), r.run_id + ("  (smoke test)" if r.is_engineering_smoke else ""), created_text, progress, layout_text,
                      str(r.seed if r.seed is not None else "–"), (r.revision or "")[:7], (f"update {ck.update}" if ck and ck.update is not None else ("saved" if ck else "none")),
                      f"{len(r.evaluations)} evaluation(s)" if r.evaluations else "–"]
            for c, v in enumerate(values):
                item = QtWidgets.QTableWidgetItem(v)
                if c == 0:
                    item.setForeground(QtGui.QColor(theme.status_color(palette, r.status)))
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.table.setItem(i, c, item)
            self.table.setRowHeight(i, 28)
        self.table.clearSelection()
        for i, r in enumerate(runs):
            if str(r.path) in selected:
                self.table.selectRow(i)
        self.table.blockSignals(False)
        if not selected and runs and not self.table.selectedItems():
            pass
        self._selection_changed()

    def _selected_runs(self) -> list[RunRecord]:
        rows = sorted({i.row() for i in self.table.selectedItems()})
        return [self._rows[r] for r in rows if r < len(self._rows)]

    def _current(self) -> RunRecord | None:
        sel = self._selected_runs()
        return sel[0] if sel else None

    # ------------------------------------------------------------------------------------------ detail
    def _selection_changed(self):
        run = self._current()
        sel = self._selected_runs()
        self.compare_box.setVisible(len(sel) >= 2)
        if run is None:
            self.detail_title.setText("Select a run to see its details")
            self.detail_text.setText("Double-click a run to open its metrics. Select two or more with Ctrl or Shift to compare them.")
            self.ck_table.setRowCount(0)
            self.detail_banner.hide()
            self._update_buttons()
            return
        self.detail_title.setText(f"{run.run_id}   ·   {theme.status_text(run.status)}")
        launch_rec = {}
        from ..runs import read_json
        launch_rec = read_json(run.path / "studio_launch.json") or {}
        rows = [("Experiment", run.experiment), ("Folder", str(run.path)),
                ("Source revision", (run.revision or "unknown")[:10] + (" (uncommitted changes)" if launch_rec.get("dirty_files") else "")),
                ("Seed / layout", f"seed {run.seed} · {run.num_envs} worlds × {run.workers} workers × {run.physx_threads} PhysX threads · PPO on {(launch_rec.get('device') or run.device or '?').upper()}"),
                ("Progress", (f"{run.updates_in_run} updates in this run (continuing from update {run.start_update}; the counter is now {run.updates_done}), " if run.start_update else f"{run.updates_done} updates, ")
                 + f"{human_count(run.decisions)} decisions in total" + (f"; {run.updates_requested} updates planned" if run.updates_requested else "")),
                ("Starting point", f"resumed from {Path(run.resume_from).parent.name}" if run.resume_from else "new run"),
                ("Artifacts", ", ".join(n for n, ok in (("metrics", run.has_metrics), ("trajectories", (run.path / 'trajectories.jsonl').exists()), ("system samples", (run.path / 'system.jsonl').exists()),
                                                         ("evaluations", bool(run.evaluations)), ("tensorboard", (run.path / 'tensorboard').exists())) if ok) or "none")]
        self.detail_text.setText("<table cellspacing=3>" + "".join(f"<tr><td><span class='muted'>{k}</span></td><td>{v}</td></tr>" for k, v in rows) + "</table>")
        if run.status == "failed":
            headline, _ = launch.explain_failure(run)
            self.detail_banner.set("bad", headline, "Open the Training page for the technical details" + (f"; the checkpoint of update {run.latest.update} survives." if run.latest else "; no checkpoint was saved."))
            self.detail_banner.show()
        else:
            self.detail_banner.hide()
        self._fill_checkpoints(run)
        if len(sel) >= 2:
            self.compare_box.setHtml(self._comparison_html(sel))
        self._update_buttons()

    def _fill_checkpoints(self, run: RunRecord):
        exp = self.ctx.selected_experiment
        cps = list(run.checkpoints)
        self.ck_table.setRowCount(len(cps))
        self._cps = cps
        palette = self.ctx.palette
        for i, c in enumerate(cps):
            compat, reason = self._compat(c, exp, run)
            saved = time.strftime("%Y-%m-%d %H:%M", time.localtime(c.mtime)) if c.mtime else "–"
            values = [c.label, str(c.update if c.update is not None else "?"), human_count(c.decisions), saved, human_bytes(c.size_bytes), ("✔ " if compat else "✖ ") + reason]
            for col, v in enumerate(values):
                item = QtWidgets.QTableWidgetItem(v)
                if col == 5:
                    item.setForeground(QtGui.QColor(palette.good if compat else palette.bad))
                self.ck_table.setItem(i, col, item)
        if cps:
            self.ck_table.selectRow(next((i for i, c in enumerate(cps) if c.kind == "latest"), 0))

    @staticmethod
    def _compat(c: CheckpointInfo, exp, run: RunRecord) -> tuple[bool, str]:
        if c.kind == "initial":
            return False, "untrained policy: use it as the evaluation baseline, or start a new run"
        if exp is None or exp.fingerprint is None:
            return False, "no experiment selected"
        if c.fingerprint is None:
            return True, "no compatibility record; the trainer will verify it on loading"
        if c.fingerprint != exp.fingerprint:
            return False, f"different learning contract than “{exp.name}”"
        if run.status in ("running", "starting", "stopping"):
            return False, "the run is still active"
        part = c.world_partition
        return True, f"matches “{exp.name}”" + (f"; needs {sum(part)} worlds on {len(part)} workers" if part else "")

    def _selected_checkpoint(self) -> CheckpointInfo | None:
        rows = sorted({i.row() for i in self.ck_table.selectedItems()})
        return self._cps[rows[0]] if rows and getattr(self, "_cps", None) and rows[0] < len(self._cps) else None

    def _update_buttons(self):
        run = self._current()
        sel = self._selected_runs()
        ck = self._selected_checkpoint()
        have = run is not None
        self.b_metrics.setEnabled(have and any(r.has_metrics for r in sel))
        self.b_compare.setEnabled(len(sel) >= 2)
        self.b_folder.setEnabled(have)
        self.b_copy.setEnabled(have)
        self.b_log.setEnabled(have)
        compat = False
        if run and ck:
            compat, reason = self._compat(ck, self.ctx.selected_experiment, run)
            self.b_resume.setToolTip(reason)
        self.b_resume.setEnabled(bool(run and ck and compat))
        self.b_eval.setEnabled(bool(run and ck))

    # ------------------------------------------------------------------------------------------ actions
    def _open_metrics(self):
        sel = [r for r in self._selected_runs() if r.has_metrics]
        if sel:
            self.ctx.navigate.emit("metrics", {"runs": [str(r.path) for r in sel]})

    def _compare(self):
        sel = self._selected_runs()
        if len(sel) >= 2:
            self.compare_box.setHtml(self._comparison_html(sel))
            self.compare_box.show()
            self._open_metrics()

    def _resume(self):
        run, ck = self._current(), self._selected_checkpoint()
        if run and ck:
            self.ctx.navigate.emit("experiments", {"resume": str(ck.path), "run": str(run.path)})

    def _evaluate(self):
        run, ck = self._current(), self._selected_checkpoint()
        if run and ck:
            self.ctx.navigate.emit("evaluation", {"run": str(run.path), "checkpoint": str(ck.path)})

    def _open_log(self):
        run = self._current()
        if run is None:
            return
        for candidate in (run.path / "launcher.log", self.ctx.run_root / ".studio" / f"{run.run_id}.launch.log"):
            if candidate.exists():
                open_path(candidate)
                return
        self.ctx.message.emit("This run has no log file.")

    def _copy_path(self):
        run = self._current()
        if run:
            QtWidgets.QApplication.clipboard().setText(str(run.path))
            self.ctx.message.emit("Path copied")

    # ------------------------------------------------------------------------------------------ comparison
    def _comparison_html(self, runs: list[RunRecord]) -> str:
        data = []
        for r in runs:
            store = MetricsStore(r.path / "metrics.jsonl")
            store.refresh()
            data.append((r, store))

        def last(store, key, n=10):
            return analysis.mean_of_last(store, key, n)

        def fmt(v, digits=3):
            return "–" if v is None else f"{v:.{digits}g}"
        metrics = [("Goals per 1,000 decisions (mean of last 10 updates)", "nav/arrivals_per_1k_decisions"), ("Progress toward goal (episodes in progress)", "nav/active_progress_mean"),
                   ("Entropy", "entropy"), ("Approximate KL", "approx_kl"), ("Clip fraction", "clip_fraction"), ("Explained variance", "explained_variance")]
        head = "<tr><th align=left></th>" + "".join(f"<th align=left>{r.run_id}</th>" for r, _ in data) + "</tr>"
        rows = [f"<tr><td><b>Status</b></td>" + "".join(f"<td>{theme.status_text(r.status)}</td>" for r, _ in data) + "</tr>",
                f"<tr><td><b>Updates / decisions</b></td>" + "".join(f"<td>{r.updates_done} / {human_count(r.decisions)}</td>" for r, _ in data) + "</tr>",
                f"<tr><td><b>Seed · layout</b></td>" + "".join(f"<td>{r.seed} · {r.num_envs}×{r.workers}</td>" for r, _ in data) + "</tr>",
                f"<tr><td><b>Revision</b></td>" + "".join(f"<td>{(r.revision or '')[:7]}</td>" for r, _ in data) + "</tr>"]
        for label, key in metrics:
            rows.append(f"<tr><td>{label}</td>" + "".join(f"<td>{fmt(last(s, key))}</td>" for _, s in data) + "</tr>")
        note = ("<p class='muted'>Comparing training curves between runs is only meaningful for runs with the same experiment, layout and budget; "
                "different seeds differ by chance alone. For a statement about skill use held-out evaluation.</p>")
        return f"<h4>Comparison of {len(runs)} runs</h4><table cellpadding=5 cellspacing=0>{head}{''.join(rows)}</table>{note}"
