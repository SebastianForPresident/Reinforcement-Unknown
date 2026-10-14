"""Open the real application in the user's desktop session (not offscreen) and save screenshots of the actual window."""
import os
import subprocess
import sys
import time
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from .. import theme
from ..app import MainWindow
from ..context import AppContext

out = Path(sys.argv[1] if len(sys.argv) > 1 else "casu_grid/results/studio/screenshots")
out.mkdir(parents=True, exist_ok=True)
app = QtWidgets.QApplication([])
ctx = AppContext(app)
theme.apply_application_style(app, ctx.palette)
window = MainWindow(ctx)
window._theme_changed()
window.resize(1360, 860)
window.show()
print("platform:", app.platformName(), "| screen:", app.primaryScreen().name(), app.primaryScreen().size().width(), "x", app.primaryScreen().size().height(), "| dpr", app.primaryScreen().devicePixelRatio())


def pump(s):
    end = time.time() + s
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)


pump(1.5)
for key, name in (("overview", "80_real_desktop_overview"), ("experiments", "81_real_desktop_experiments"), ("metrics", "82_real_desktop_metrics"), ("runs", "83_real_desktop_runs")):
    window.go(key)
    pump(1.5)
    window.grab().save(str(out / f"{name}_qt.png"))
    window.activateWindow()
    window.raise_()
    pump(0.5)
    r = subprocess.run(["spectacle", "-b", "-n", "-a", "-o", str(out / f"{name}_spectacle.png")], capture_output=True, text=True, timeout=30)
    print(name, "spectacle rc", r.returncode, (r.stderr or "")[:120].replace("\n", " "))
window.close()
