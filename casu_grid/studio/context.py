"""Application state shared by all pages: settings, the Isaac install, the run list, the supervised session and the system monitor.

Everything that touches the disk or other processes more than trivially runs on worker threads and reports back through signals, so
the interface stays responsive while a simulation saturates the CPU.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from . import environment as env
from . import launch, theme
from .experiments import ExperimentInfo, discover_experiments
from .runs import MetricsStore, RunRecord, discover_runs, last_json_line, pid_alive
from .widgets import HelpDialog


class _Signals(QtCore.QObject):
    done = QtCore.Signal(object)


class _Job(QtCore.QRunnable):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.signals = _Signals()

    def run(self):
        try:
            result = self.fn()
        except Exception as exc:        # a background failure must never take the interface down
            result = exc
        self.signals.done.emit(result)


def run_async(fn, callback, parent_pool: QtCore.QThreadPool | None = None):
    """Run ``fn`` on a worker thread and call ``callback(result)`` on the GUI thread (exceptions arrive as the result)."""
    job = _Job(fn)
    job.setAutoDelete(False)
    job.signals.done.connect(callback)
    (parent_pool or QtCore.QThreadPool.globalInstance()).start(job)
    return job


def sample_system(run_dir: Path, metrics: MetricsStore, started_at: float | None) -> dict:
    """One measurement of the supervised run: memory of its process tree, simulator workers alive, GPU and machine memory."""
    sample: dict = {"t": time.time(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    pid = 0
    try:
        pid = int((run_dir / "RUNNING").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        pass
    pss = uss = 0
    workers = 0
    procs = 0
    if pid and pid_alive(pid):
        try:
            import psutil
            root = psutil.Process(pid)
            for proc in [root, *root.children(recursive=True)]:
                try:
                    info = Path(f"/proc/{proc.pid}/smaps_rollup").read_text()
                except OSError:
                    continue
                values = {}
                for line in info.splitlines():
                    parts = line.split()
                    if len(parts) >= 2 and parts[0].rstrip(":") in ("Pss", "Private_Clean", "Private_Dirty"):
                        values[parts[0].rstrip(":")] = int(parts[1]) * 1024
                pss += values.get("Pss", 0)
                uss += values.get("Private_Clean", 0) + values.get("Private_Dirty", 0)
                procs += 1
                try:
                    if "casu_grid.isaac_shard_worker" in " ".join(proc.cmdline()):
                        workers += 1
                except Exception:
                    pass
        except Exception:
            pass
    mem = env.memory_info()
    gpu = env.gpu_info()
    metrics.refresh()
    cum = sum(v for v in metrics.column("rollout_seconds") if isinstance(v, (int, float)))
    update = metrics.column("update")[-1] if metrics.rows else None
    decisions = metrics.column("environment_steps")[-1] if metrics.rows else None
    sample.update({"pss_gib": pss / 2**30, "uss_gib": uss / 2**30, "processes": procs, "workers_alive": workers,
                   "mem_available_gib": mem["available"] / 2**30, "swap_used_gib": mem["swap_used"] / 2**30,
                   "gpu_mem_mib": gpu["used_mib"] if gpu else None, "gpu_util": gpu["util_percent"] if gpu else None,
                   "update": update, "decisions": decisions, "train_s": cum,
                   "elapsed_s": (time.time() - started_at) if started_at else None})
    return sample


class AppContext(QtCore.QObject):
    runsChanged = QtCore.Signal()
    sessionChanged = QtCore.Signal()
    navigate = QtCore.Signal(str, object)        # page key, optional payload
    themeChanged = QtCore.Signal()
    monitorSample = QtCore.Signal(object)
    message = QtCore.Signal(str)

    def __init__(self, app: QtWidgets.QApplication):
        super().__init__()
        self.app = app
        self.settings = QtCore.QSettings("ReinforcementUnknown", "ResearchStudioV2")
        self.install = env.discover_isaac(self.settings.value("isaac_sim", "", str) or None)
        default_root = env.ROOT / "experiments"
        self.run_root = Path(os.environ.get("CASU_STUDIO_RUN_ROOT") or self.settings.value("run_root", str(default_root), str)).expanduser()
        mode = self.settings.value("theme", "system", str)
        self.theme_mode = mode
        self.palette = self._resolve_palette()
        self.experiments: list[ExperimentInfo] = []
        self.runs: list[RunRecord] = []
        self.session: launch.Session | None = None
        self.selected_experiment: ExperimentInfo | None = None
        self.last_monitor: dict | None = None
        self._monitor_run: Path | None = None
        self._monitor_store: MetricsStore | None = None
        self._monitor_started: float | None = None
        self._monitor_busy = False
        self._runs_busy = False
        self.reload_experiments()
        self.runs_timer = QtCore.QTimer(self)
        self.runs_timer.timeout.connect(self.refresh_runs)
        self.runs_timer.start(3000)
        self.monitor_timer = QtCore.QTimer(self)
        self.monitor_timer.timeout.connect(self._monitor_tick)
        self.monitor_timer.start(5000)

    # ------------------------------------------------------------------------------------------ theme
    def _resolve_palette(self) -> theme.Palette:
        if self.theme_mode == "dark":
            return theme.DARK
        if self.theme_mode == "light":
            return theme.LIGHT
        return theme.DARK if theme.system_prefers_dark(self.app) else theme.LIGHT

    def set_theme(self, mode: str):
        self.theme_mode = mode
        self.settings.setValue("theme", mode)
        self.palette = self._resolve_palette()
        theme.apply_application_style(self.app, self.palette)
        self.themeChanged.emit()

    # ------------------------------------------------------------------------------------------ settings
    def set_run_root(self, path: Path):
        self.run_root = Path(path)
        self.settings.setValue("run_root", str(path))
        self.refresh_runs()

    def set_isaac(self, path: str):
        self.settings.setValue("isaac_sim", path)
        self.install = env.discover_isaac(path or None)

    # ------------------------------------------------------------------------------------------ data
    def reload_experiments(self):
        self.experiments = discover_experiments()
        if self.selected_experiment is None and self.experiments:
            last = self.settings.value("experiment", "", str)
            self.selected_experiment = next((e for e in self.experiments if str(e.path) == last), self.experiments[0])

    def select_experiment(self, exp: ExperimentInfo):
        self.selected_experiment = exp
        self.settings.setValue("experiment", str(exp.path))

    def refresh_runs(self):
        if self._runs_busy:
            return
        self._runs_busy = True
        root = self.run_root
        run_async(lambda: discover_runs(root), self._runs_loaded)

    def _runs_loaded(self, result):
        self._runs_busy = False
        if isinstance(result, Exception):
            self.message.emit(f"Could not read runs: {result}")
            return
        self.runs = result
        self.runsChanged.emit()
        # keep the session object honest: if its launcher exited, forget it (status then comes from the artifacts)
        if self.session is not None:
            launch.write_launch_record(self.session)
            if not self.session.alive():
                self.session = None
                self.sessionChanged.emit()

    def run_by_path(self, path) -> RunRecord | None:
        return next((r for r in self.runs if str(r.path) == str(path)), None)

    def active_run(self) -> RunRecord | None:
        alive = [r for r in self.runs if r.status in ("starting", "running", "stopping")]
        return alive[0] if alive else None

    def session_run(self) -> RunRecord | None:
        if self.session is None:
            return None
        return self.run_by_path(self.session.plan.run_dir)

    # ------------------------------------------------------------------------------------------ session control
    def start_training(self, plan: launch.LaunchPlan) -> launch.Session:
        session = launch.start(plan, self.install)
        self.session = session
        self.sessionChanged.emit()
        self.refresh_runs()
        return session

    def stop_training(self, run: RunRecord, force: bool = False):
        if force:
            run_async(lambda: launch.force_stop(run.path, self.session), lambda _r: self.refresh_runs())
        else:
            launch.request_stop(run.path)
        self.refresh_runs()

    # ------------------------------------------------------------------------------------------ system monitor
    def _monitor_tick(self):
        run = self.active_run()
        if run is None:
            self._monitor_run = None
            return
        if self._monitor_run != run.path:
            self._monitor_run = run.path
            self._monitor_store = MetricsStore(run.path / "metrics.jsonl")
            self._monitor_started = time.time()
        if self._monitor_busy:
            return
        self._monitor_busy = True
        path, store, started = run.path, self._monitor_store, self._monitor_started

        def work():
            sample = sample_system(path, store, started)
            try:
                with (path / "system.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(sample) + "\n")
            except OSError:
                pass
            return sample

        run_async(work, self._monitor_done)

    def _monitor_done(self, result):
        self._monitor_busy = False
        if isinstance(result, dict):
            self.last_monitor = result
            self.monitorSample.emit(result)

    # ------------------------------------------------------------------------------------------ help
    def show_help(self, key: str):
        HelpDialog(key, QtWidgets.QApplication.activeWindow()).exec()
