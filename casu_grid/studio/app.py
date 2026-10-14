"""Main window, navigation and application entry point."""
from __future__ import annotations

import sys

from PySide6 import QtCore, QtGui, QtWidgets

from . import theme
from .context import AppContext
from .pages.overview import OverviewPage

PAGES = [("overview", "Overview", "◧"), ("experiments", "Experiments", "⚗"), ("training", "Training", "▶"),
         ("metrics", "Metrics", "∿"), ("runs", "Runs & checkpoints", "▤"), ("evaluation", "Evaluation", "⚖"), ("help", "Help", "?")]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.setWindowTitle("Research Studio · XY1")
        self.resize(1360, 860)
        self.setMinimumSize(980, 640)
        geometry = ctx.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_sidebar())
        self.stack = QtWidgets.QStackedWidget()
        root.addWidget(self.stack, 1)
        self.pages: dict[str, QtWidgets.QWidget] = {}
        self._add_pages()
        ctx.navigate.connect(self.go)
        ctx.message.connect(lambda text: self.statusBar().showMessage(text, 8000))
        ctx.themeChanged.connect(self._theme_changed)
        ctx.runsChanged.connect(self._update_chip)
        ctx.sessionChanged.connect(self._update_chip)
        self.go(ctx.settings.value("page", "overview", str))
        ctx.refresh_runs()
        self._update_chip()

    def _build_sidebar(self) -> QtWidgets.QWidget:
        side = QtWidgets.QWidget()
        side.setObjectName("sidebar")
        side.setFixedWidth(210)
        layout = QtWidgets.QVBoxLayout(side)
        layout.setContentsMargins(8, 4, 8, 10)
        layout.setSpacing(2)
        brand = QtWidgets.QLabel("Research Studio")
        brand.setObjectName("brand")
        sub = QtWidgets.QLabel("XY1 point navigation")
        sub.setObjectName("brandSub")
        layout.addWidget(brand)
        layout.addWidget(sub)
        self.nav_group = QtWidgets.QButtonGroup(self)
        self.nav_buttons: dict[str, QtWidgets.QPushButton] = {}
        for key, label, glyph in PAGES:
            button = QtWidgets.QPushButton("")
            button.setCheckable(True)
            inner = QtWidgets.QHBoxLayout(button)
            inner.setContentsMargins(12, 0, 8, 0)
            inner.setSpacing(10)
            g = QtWidgets.QLabel(glyph)
            g.setFixedWidth(20)
            g.setAlignment(QtCore.Qt.AlignCenter)
            t = QtWidgets.QLabel(label)
            for lab in (g, t):
                lab.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
                lab.setObjectName("navLabel")
            inner.addWidget(g)
            inner.addWidget(t, 1)
            button.toggled.connect(lambda _on, b=button: self._style_nav(b))
            button._labels = (g, t)
            button.setMinimumHeight(38)
            button.setCursor(QtCore.Qt.PointingHandCursor)
            button.clicked.connect(lambda _c=False, k=key: self.go(k))
            self.nav_group.addButton(button)
            layout.addWidget(button)
            self.nav_buttons[key] = button
        layout.addStretch(1)
        self.chip = QtWidgets.QPushButton("")
        self.chip.setCursor(QtCore.Qt.PointingHandCursor)
        self.chip.clicked.connect(lambda: self.go("training"))
        layout.addWidget(self.chip)
        row = QtWidgets.QHBoxLayout()
        self.theme_button = QtWidgets.QPushButton("◐ Theme")
        self.theme_button.setToolTip("Switch between light, dark and the system theme")
        self.theme_button.clicked.connect(self._cycle_theme)
        row.addWidget(self.theme_button)
        layout.addLayout(row)
        return side

    def _style_nav(self, button):
        checked = button.isChecked()
        pal = self.ctx.palette
        for lab in button._labels:
            lab.setStyleSheet(f"color:{pal.accent_text if checked else pal.text}; background:transparent;" + ("font-weight:600;" if checked else ""))

    def _add_pages(self):
        from .pages.experiments import ExperimentsPage
        from .pages.training import TrainingPage
        from .pages.metrics import MetricsPage
        from .pages.runs import RunsPage
        from .pages.evaluation import EvaluationPage
        from .pages.helppage import HelpPage
        classes = {"overview": OverviewPage, "experiments": ExperimentsPage, "training": TrainingPage, "metrics": MetricsPage,
                   "runs": RunsPage, "evaluation": EvaluationPage, "help": HelpPage}
        for key, _label, _glyph in PAGES:
            page = classes[key](self.ctx)
            self.pages[key] = page
            self.stack.addWidget(page)

    def go(self, key: str, payload=None):
        if key not in self.pages:
            key = "overview"
        page = self.pages[key]
        self.stack.setCurrentWidget(page)
        self.nav_buttons[key].setChecked(True)
        self.ctx.settings.setValue("page", key)
        if hasattr(page, "on_show"):
            page.on_show(payload)

    def _cycle_theme(self):
        order = ["system", "light", "dark"]
        nxt = order[(order.index(self.ctx.theme_mode) + 1) % 3]
        self.ctx.set_theme(nxt)
        self.statusBar().showMessage(f"Theme: {nxt}", 3000)

    def _theme_changed(self):
        for b in self.nav_buttons.values():
            self._style_nav(b)
        self.theme_button.setText({"system": "◐ Theme: system", "light": "☀ Theme: light", "dark": "☾ Theme: dark"}[self.ctx.theme_mode])

    def _update_chip(self):
        run = self.ctx.active_run()
        if run is None:
            self.chip.setText("○  No run active")
            self.chip.setStyleSheet("")
            self.chip.setToolTip("Nothing is training. Start a run from Experiments.")
        else:
            color = theme.status_color(self.ctx.palette, run.status)
            self.chip.setText(f"{theme.STATUS[run.status][0]}  {theme.STATUS[run.status][1]} · update {run.updates_done}")
            self.chip.setStyleSheet(f"color:{color}; border:1px solid {color}; font-weight:700;")
            self.chip.setToolTip(f"{run.title}\nClick to open the training monitor.")

    def closeEvent(self, event):
        run = self.ctx.active_run()
        if run is not None and run.status in ("running", "starting"):
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("A run is still active")
            box.setText(f"“{run.run_id}” is still {run.status}.\n\nClosing Research Studio does not stop it: the simulator keeps running in the background. "
                        "You can reopen the Studio at any time and it will pick the run up again.")
            keep = box.addButton("Close Studio, keep training", QtWidgets.QMessageBox.AcceptRole)
            stay = box.addButton("Stay", QtWidgets.QMessageBox.RejectRole)
            box.exec()
            if box.clickedButton() is stay:
                event.ignore()
                return
        self.ctx.settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    app = QtWidgets.QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("Research Studio")
    app.setOrganizationName("ReinforcementUnknown")
    ctx = AppContext(app)
    theme.apply_application_style(app, ctx.palette)
    window = MainWindow(ctx)
    window._theme_changed()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
