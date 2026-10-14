"""Metrics: separate, interactive charts grouped by purpose; overlays for several runs; one chart can be inspected in detail."""
from __future__ import annotations

import time
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from .. import catalog, help as helptext, theme
from ..charts import ChartCard, export_csv
from ..runs import MetricsStore, RunRecord, human_count
from ..widgets import Banner, make_label, page_container, scroll_page

RANGES = [("All updates", None), ("Last 25 updates", 25), ("Last 100 updates", 100), ("Last 500 updates", 500)]


class MetricsPage(QtWidgets.QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.selected: list[Path] = []
        self.data: dict[Path, catalog.RunData] = {}
        self.category = "navigation"
        self.focus_key: str | None = None
        self._cards: dict[str, ChartCard] = {}
        self._dirty = True
        self._last_rows: dict[Path, int] = {}
        self._following = False
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(24, 18, 24, 12)
        outer.setSpacing(10)
        title_row = QtWidgets.QHBoxLayout()
        title_col = QtWidgets.QVBoxLayout()
        title_col.setSpacing(1)
        self.title = make_label("Metrics", "pageTitle", wrap=False)
        self.subtitle = make_label("", "pageSubtitle", wrap=False)
        title_col.addWidget(self.title)
        title_col.addWidget(self.subtitle)
        title_row.addLayout(title_col, 1)
        self.live = make_label("", "hint", wrap=False)
        title_row.addWidget(self.live)
        outer.addLayout(title_row)
        outer.addLayout(self._toolbar())
        self.tabs = QtWidgets.QTabBar()
        for key, label in catalog.CATEGORIES:
            self.tabs.addTab(label.replace("&", "&&"))
        self.tabs.currentChanged.connect(self._tab_changed)
        outer.addWidget(self.tabs)
        self.stack = QtWidgets.QStackedWidget()
        outer.addWidget(self.stack, 1)
        # grid page
        self.grid_content = QtWidgets.QWidget()
        self.grid_content.setObjectName("page")
        self.grid = QtWidgets.QGridLayout(self.grid_content)
        self.grid.setContentsMargins(0, 4, 8, 4)
        self.grid.setSpacing(12)
        self.empty_banner = Banner("info", "Choose a run to see its metrics", "")
        self.grid.addWidget(self.empty_banner, 0, 0, 1, 2)
        self.stack.addWidget(scroll_page(self.grid_content))
        # focus page
        self.focus_page = QtWidgets.QWidget()
        fl = QtWidgets.QHBoxLayout(self.focus_page)
        fl.setContentsMargins(0, 0, 0, 0)
        left = QtWidgets.QVBoxLayout()
        back = QtWidgets.QPushButton("◀  Back to overview")
        back.clicked.connect(self._leave_focus)
        left.addWidget(back)
        self.focus_list = QtWidgets.QListWidget()
        self.focus_list.setFixedWidth(250)
        self.focus_list.currentRowChanged.connect(self._focus_row)
        left.addWidget(self.focus_list, 1)
        fl.addLayout(left)
        right = QtWidgets.QVBoxLayout()
        self.focus_holder = QtWidgets.QVBoxLayout()
        right.addLayout(self.focus_holder, 3)
        self.stats = QtWidgets.QLabel("")
        self.stats.setWordWrap(True)
        self.stats.setTextFormat(QtCore.Qt.RichText)
        right.addWidget(self.stats)
        self.explain = QtWidgets.QLabel("")
        self.explain.setWordWrap(True)
        self.explain.setTextFormat(QtCore.Qt.RichText)
        right.addWidget(self.explain)
        fl.addLayout(right, 1)
        self.stack.addWidget(self.focus_page)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(2000)
        ctx.themeChanged.connect(self._theme)
        ctx.runsChanged.connect(self._runs_changed)

    # ------------------------------------------------------------------------------------------ toolbar
    def _toolbar(self) -> QtWidgets.QLayout:
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(10)
        self.runs_button = QtWidgets.QToolButton()
        self.runs_button.setText("Runs ▾")
        self.runs_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.runs_menu = QtWidgets.QMenu(self.runs_button)
        self.runs_button.setMenu(self.runs_menu)
        self.runs_menu.aboutToShow.connect(self._fill_runs_menu)
        row.addWidget(make_label("Runs:", wrap=False))
        row.addWidget(self.runs_button)
        row.addWidget(make_label("X axis:", wrap=False))
        self.axis = QtWidgets.QComboBox()
        for key, (label, _i) in catalog.X_AXES.items():
            self.axis.addItem(label, key)
        self.axis.currentIndexChanged.connect(self._settings_changed)
        row.addWidget(self.axis)
        self.raw = QtWidgets.QCheckBox("raw")
        self.raw.setChecked(True)
        self.smooth = QtWidgets.QCheckBox("smoothed")
        self.smooth.setChecked(True)
        self.window = QtWidgets.QSpinBox()
        self.window.setRange(1, 500)
        self.window.setValue(int(self.ctx.settings.value("smooth_window", 10)))
        self.window.setSuffix(" updates")
        self.window.setToolTip("Smoothing window (exponential moving average; missing samples stay missing)")
        for w in (self.raw, self.smooth):
            w.toggled.connect(self._settings_changed)
            row.addWidget(w)
        self.window.valueChanged.connect(self._settings_changed)
        row.addWidget(self.window)
        row.addWidget(make_label("Range:", wrap=False))
        self.range = QtWidgets.QComboBox()
        for label, n in RANGES:
            self.range.addItem(label, n)
        self.range.addItem("Zoomed (custom)", "custom")
        self.range.currentIndexChanged.connect(self._range_changed)
        row.addWidget(self.range)
        self.link = QtWidgets.QCheckBox("link zoom")
        self.link.setChecked(True)
        self.link.setToolTip("Zoom and pan all charts together")
        row.addWidget(self.link)
        reset = QtWidgets.QPushButton("Reset zoom")
        reset.clicked.connect(self._reset_all)
        row.addWidget(reset)
        export = QtWidgets.QToolButton()
        export.setText("Export ▾")
        export.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(export)
        menu.addAction("Save all charts of this tab as images…", self._export_images)
        menu.addAction("Save all data as CSV…", self._export_data)
        export.setMenu(menu)
        row.addWidget(export)
        row.addStretch(1)
        return row

    def _fill_runs_menu(self):
        self.runs_menu.clear()
        runs = self.ctx.runs
        if not runs:
            self.runs_menu.addAction("(no runs yet)").setEnabled(False)
            return
        for run in runs[:40]:
            action = self.runs_menu.addAction(f"{theme.STATUS[run.status][0]}  {run.run_id}  ·  {run.updates_done} updates")
            action.setCheckable(True)
            action.setChecked(run.path in self.selected)
            action.toggled.connect(lambda on, r=run: self._toggle_run(r, on))
        self.runs_menu.addSeparator()
        self.runs_menu.addAction("Clear selection", lambda: self.set_runs([]))

    def _toggle_run(self, run: RunRecord, on: bool):
        paths = [p for p in self.selected if p != run.path]
        if on:
            paths.append(run.path)
        if len(paths) > 6:
            self.ctx.message.emit("At most 6 runs can be overlaid.")
            paths = paths[-6:]
        self.set_runs(paths)

    # ------------------------------------------------------------------------------------------ state
    def on_show(self, payload=None):
        chart = None
        if isinstance(payload, dict):
            chart = payload.get("chart")
            paths = [Path(p) for p in payload.get("runs", [])] or ([Path(payload["run"])] if payload.get("run") else [])
            if paths:
                self.set_runs(paths)
        elif isinstance(payload, (str, Path)):
            self.set_runs([Path(payload)])
        elif not self.selected:
            active = self.ctx.active_run() or (self.ctx.runs[0] if self.ctx.runs else None)
            if active:
                self.set_runs([active.path])
        if chart and chart in catalog.CHART_BY_KEY:
            cat = catalog.CHART_BY_KEY[chart].category
            self.tabs.setCurrentIndex([k for k, _ in catalog.CATEGORIES].index(cat))
            self._enter_focus(chart)
        self.refresh(force=True)

    def set_runs(self, paths: list[Path]):
        self.selected = list(paths)
        self.data = {}
        for i, path in enumerate(self.selected):
            run = self.ctx.run_by_path(path)
            label = run.run_id if run else path.name
            rd = catalog.RunData(label, MetricsStore(path / "metrics.jsonl"), MetricsStore(path / "system.jsonl"), theme.SERIES_COLORS[i % len(theme.SERIES_COLORS)])
            self.data[path] = rd
        self._last_rows = {}
        self._dirty = True
        names = ", ".join(self.data[p].label for p in self.selected)
        self.subtitle.setText(names if names else "No run selected")
        self.runs_button.setText(f"Runs ({len(self.selected)}) ▾" if self.selected else "Runs ▾")
        self.refresh(force=True)

    def _runs_changed(self):
        pass

    def _theme(self):
        for card in self._cards.values():
            card.view.set_palette(self.ctx.palette)
        self._dirty = True
        self.refresh(force=True)

    # ------------------------------------------------------------------------------------------ settings and range
    def _settings_changed(self, *_):
        self.ctx.settings.setValue("smooth_window", self.window.value())
        self._dirty = True
        self.refresh(force=True)

    def _primary(self) -> catalog.RunData | None:
        return self.data[self.selected[0]] if self.selected else None

    def _range_for(self, n: int | None) -> tuple[float, float] | None:
        rd = self._primary()
        if n is None or rd is None or rd.metrics.rows == 0:
            return None
        xs = rd.x_values(self.axis.currentData())
        if not xs:
            return None
        lo = xs[max(0, len(xs) - n)]
        hi = xs[-1]
        pad = (hi - lo) * 0.02 or 0.5
        return lo - pad, hi + pad

    def _range_changed(self, *_):
        data = self.range.currentData()
        self._following = isinstance(data, int)
        if data == "custom":
            return
        self._apply_range(self._range_for(data))

    def _apply_range(self, rng):
        for card in self._active_cards():
            card.view.set_x_range(rng)

    def _reset_all(self):
        self.range.blockSignals(True)
        self.range.setCurrentIndex(0)
        self.range.blockSignals(False)
        self._following = False
        for card in self._active_cards():
            card.view.reset_view()

    def _view_changed(self, rng):
        sender = self.sender()
        if rng is None:
            return
        if self.link.isChecked():
            for card in self._active_cards():
                if card.view is not getattr(sender, "view", sender) and card.view is not sender:
                    card.view.set_x_range(rng)
        self.range.blockSignals(True)
        self.range.setCurrentIndex(self.range.findData("custom"))
        self.range.blockSignals(False)
        self._following = False

    # ------------------------------------------------------------------------------------------ cards
    def _category_keys(self) -> list[str]:
        cat = catalog.CATEGORIES[self.tabs.currentIndex()][0]
        return [c.spec.key for c in catalog.CHARTS if c.category == cat]

    def _card(self, key: str, large: bool = False) -> ChartCard:
        cache_key = key + ("*" if large else "")
        card = self._cards.get(cache_key)
        if card is None:
            card = ChartCard(catalog.CHART_BY_KEY[key].spec, self.ctx.palette, self.ctx.show_help, large=large)
            card.expandRequested.connect(self._enter_focus)
            card.viewChanged.connect(self._view_changed)
            self._cards[cache_key] = card
        return card

    def _active_cards(self) -> list[ChartCard]:
        if self.stack.currentIndex() == 1 and self.focus_key:
            return [self._card(self.focus_key, True)]
        return [self._card(k) for k in self._category_keys()]

    def _tab_changed(self, _i):
        self._leave_focus()
        self._layout_grid()
        self._dirty = True
        self.refresh(force=True)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._layout_grid()

    def _layout_grid(self):
        cols = 2 if self.width() > 900 else 1
        for key in {c.spec.key for c in catalog.CHARTS}:
            card = self._cards.get(key)
            if card is not None:
                self.grid.removeWidget(card)
                card.hide()
        keys = self._category_keys()
        self.empty_banner.setVisible(not self.selected)
        for i, key in enumerate(keys):
            card = self._card(key)
            self.grid.addWidget(card, 1 + i // cols, i % cols)
            card.show()
        for c in range(2):
            self.grid.setColumnStretch(c, 1 if c < cols else 0)

    # ------------------------------------------------------------------------------------------ focus
    def _enter_focus(self, key: str):
        self.focus_key = key
        self.focus_list.blockSignals(True)
        self.focus_list.clear()
        keys = self._category_keys()
        for k in keys:
            self.focus_list.addItem(catalog.CHART_BY_KEY[k].spec.title)
        self.focus_list.setCurrentRow(keys.index(key) if key in keys else 0)
        self.focus_list.blockSignals(False)
        while self.focus_holder.count():
            item = self.focus_holder.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        card = self._card(key, True)
        self.focus_holder.addWidget(card)
        card.show()
        self.stack.setCurrentIndex(1)
        self._dirty = True
        self.refresh(force=True)

    def _focus_row(self, row: int):
        keys = self._category_keys()
        if 0 <= row < len(keys):
            self._enter_focus(keys[row])

    def _leave_focus(self):
        self.focus_key = None
        self.stack.setCurrentIndex(0)
        self._layout_grid()

    # ------------------------------------------------------------------------------------------ refresh
    def refresh(self, force: bool = False):
        if not self.isVisible() and not force:
            return
        changed = force or self._dirty
        for path, rd in self.data.items():
            try:
                added = rd.refresh()
            except Exception:
                added = 0
            if added or self._last_rows.get(path) != rd.metrics.rows:
                changed = True
            self._last_rows[path] = rd.metrics.rows
        primary = self._primary()
        if primary is not None:
            rows = primary.metrics.rows
            run = self.ctx.run_by_path(self.selected[0])
            alive = run is not None and run.status in ("starting", "running", "stopping")
            skipped = primary.metrics.bad_lines
            self.live.setText((f"● live · " if alive else "") + f"{rows} updates read" + (f" · {skipped} unreadable line(s) skipped" if skipped else ""))
        else:
            self.live.setText("")
        self.empty_banner.setVisible(not self.selected)
        if not changed:
            return
        self._dirty = False
        axis = self.axis.currentData()
        xlabel, x_is_int = catalog.X_AXES[axis]
        runs = list(self.data.values())
        cards = self._active_cards()
        for card in cards:
            chart = catalog.CHART_BY_KEY[card.spec.key]
            series = catalog.build_series(chart, runs, axis) if runs else []
            if axis == "time":
                xlabel_use = "Training time (min, cumulative collection + optimisation)"
            else:
                xlabel_use = xlabel
            card.view.set_smoothing(self.window.value(), self.raw.isChecked(), self.smooth.isChecked())
            card.set_series(series, "Training time" if axis == "time" else xlabel, x_is_int)
        if self._following:
            self._apply_range(self._range_for(self.range.currentData()))
        if self.stack.currentIndex() == 1 and self.focus_key:
            self._focus_details(cards[0])

    def _focus_details(self, card: ChartCard):
        chart = catalog.CHART_BY_KEY[card.spec.key]
        stats = catalog.series_stats(card.view.series, card.view.x_range_user())
        rows = []
        for s in stats:
            if s["samples"]:
                rows.append(f"<tr><td><span style='color:{s['color']}'>■</span> {s['label']}</td><td>{s['samples']}</td><td>{s['missing']}</td>"
                            f"<td>{s['min']:.4g}</td><td>{s['mean']:.4g}</td><td>{s['max']:.4g}</td><td>{s['last']:.4g}</td></tr>")
            else:
                rows.append(f"<tr><td>{s['label']}</td><td colspan=6>no samples in the visible range</td></tr>")
        header = "<tr><th align=left>series</th><th>samples</th><th>missing</th><th>min</th><th>mean</th><th>max</th><th>latest</th></tr>"
        scope = "visible range" if card.view.x_range_user() else "whole run"
        self.stats.setText(f"<b>Numbers for the {scope}</b> (missing = updates without a sample, never counted as zero)<table cellpadding=4>{header}{''.join(rows)}</table>")
        body = helptext.body(chart.spec.help_key) if chart.spec.help_key else ""
        self.explain.setText(f"<b>How to read this chart.</b> {body or chart.spec.caption}")

    # ------------------------------------------------------------------------------------------ export
    def _export_images(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Save chart images to…")
        if not folder:
            return
        count = 0
        for card in self._active_cards():
            if card.export_png(str(Path(folder) / f"{card.spec.key}.png")):
                count += 1
        self.ctx.message.emit(f"Saved {count} chart image(s) to {folder}")

    def _export_data(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Save chart data (CSV) to…")
        if not folder:
            return
        axis = self.axis.currentData()
        count = 0
        for chart in catalog.CHARTS:
            series = catalog.build_series(chart, list(self.data.values()), axis)
            if series:
                for s in series:
                    from ..charts import smooth_values
                    s.smoothed = smooth_values(s.ys, self.window.value())
                export_csv(str(Path(folder) / f"{chart.spec.key}.csv"), series, catalog.X_AXES[axis][0])
                count += 1
        self.ctx.message.emit(f"Saved data of {count} chart(s) to {folder}")
