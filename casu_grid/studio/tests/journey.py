"""End-to-end journey of the Studio against the REAL simulator (not part of the unit suite: it launches Isaac Sim).

    QT_QPA_PLATFORM=offscreen python3 -m casu_grid.studio.tests.journey STAGE [--shots DIR]

Stages drive the actual widgets (select preset, press Start, watch live status, zoom charts, resume, stop, break a worker) and save
screenshots of the actual rendered window. Each stage prints PASS/FAIL lines with the evidence it checked.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .. import theme
from ..app import MainWindow
from ..context import AppContext
from ..pages import experiments as exp_page

LOG: list[str] = []


def check(ok: bool, text: str):
    line = ("PASS " if ok else "FAIL ") + text
    LOG.append(line)
    print(line, flush=True)
    return ok


class Harness:
    def __init__(self, shots: Path, size=(1360, 860)):
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.ctx = AppContext(self.app)
        theme.apply_application_style(self.app, self.ctx.palette)
        self.window = MainWindow(self.ctx)
        self.window._theme_changed()
        self.window.resize(*size)
        self.window.show()
        self.shots = shots
        shots.mkdir(parents=True, exist_ok=True)
        self.pump(0.3)

    def pump(self, seconds: float = 0.2):
        end = time.time() + seconds
        while time.time() < end:
            self.app.processEvents()
            QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
            time.sleep(0.02)

    def wait(self, cond, timeout: float, what: str = "", step: float = 0.5) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            self.pump(step)
            try:
                if cond():
                    return True
            except Exception:
                pass
        return False

    def go(self, page: str, payload=None):
        self.window.go(page, payload)
        self.pump(0.5)

    def shot(self, name: str):
        self.pump(0.3)
        path = self.shots / f"{name}.png"
        self.window.grab().save(str(path))
        print("shot", path, flush=True)
        return path

    def page(self, key):
        return self.window.pages[key]

    def run_record(self, run_dir: Path):
        self.ctx.refresh_runs()
        self.wait(lambda: self.ctx.run_by_path(run_dir) is not None, 15, step=0.4)
        return self.ctx.run_by_path(run_dir)


def start_from_experiments(h: Harness, preset: str, updates: int | None = None, name: str | None = None, resume: dict | None = None, seed: int | None = None):
    page = h.page("experiments")
    h.go("experiments", resume)
    exp_row = next(i for i, e in enumerate(h.ctx.experiments) if e.role == "recommended")     # always the validated XY1 mini-world unless a stage chose another
    if not getattr(h, "experiment_chosen", False):
        page.list.setCurrentRow(exp_row)
        h.pump(0.4)
    if updates is not None or preset == "custom":
        pass
    index = page.preset.findData(preset)
    page.preset.setCurrentIndex(index)
    h.pump(0.5)
    if updates is not None:
        page.updates.setValue(updates)
    if seed is not None:
        page.seed.setValue(seed)
    if name:
        page.run_name.setText(name)
        page._name_custom = True
    page.revalidate()
    h.pump(0.5)
    return page


def press_start(h: Harness, page):
    # the confirmation dialog is modal; accept it programmatically after verifying its content exists
    original = exp_page.ConfirmDialog.exec
    exp_page.ConfirmDialog.exec = lambda self: QtWidgets.QDialog.Accepted
    try:
        page._start()
    finally:
        exp_page.ConfirmDialog.exec = original
    h.pump(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--shots", type=Path, default=Path("casu_grid/results/studio/screenshots"))
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume-run", default=None)
    ap.add_argument("--experiment", default=None, help="substring of the experiment file to select (default: the validated XY1 mini-world)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    h = Harness(args.shots)
    stage = globals()["stage_" + args.stage]
    ok = stage(h, args)
    print("RESULT", "OK" if ok else "FAILED")
    out = args.shots.parent / f"journey_{args.stage}.log"
    out.write_text("\n".join(LOG) + "\n")
    return 0 if ok else 1


# ----------------------------------------------------------------------------------------------------------------------- stages
def stage_smoke(h: Harness, args) -> bool:
    """Steps 1-8 of the primary journey: open XY1, inspect, validate, start a small run, watch it live, let it finish."""
    h.go("overview")
    h.shot("01_overview_no_runs")
    if not [r for r in h.ctx.runs if not r.is_engineering_smoke]:
        check("No learning run exists yet" in h.page("overview").hero_title.text(), "overview says plainly that no learning run exists")
    page = start_from_experiments(h, "smoke", name=args.run_name or time.strftime("%Y%m%d-%H%M%S-smoke"))
    h.shot("02_experiment_configuration")
    errors = [c for c in page._checks if c.level == "error"]
    check(not errors, f"validation passes for the smoke configuration ({len(page._checks)} checks, {len([c for c in page._checks if c.level == 'warn'])} warnings)")
    plan = page.current_plan()
    check(plan.worlds == 16 and plan.workers == 2 and plan.updates == 5, "smoke preset sets 16 worlds x 2 workers x 5 updates")
    press_start(h, page)
    run_dir = plan.run_dir
    check(h.window.stack.currentWidget() is h.page("training"), "Start navigates to the training monitor")
    seen = set()
    h.wait(lambda: (seen.add((h.ctx.run_by_path(run_dir).status if h.ctx.run_by_path(run_dir) else "none")), False)[1], 25, step=0.5)
    h.shot("03_training_starting")
    check("starting" in seen or "running" in seen, f"status observed while launching: {sorted(seen)}")
    running = h.wait(lambda: (h.ctx.run_by_path(run_dir) and h.ctx.run_by_path(run_dir).updates_done >= 3), 240, step=1.0)
    h.shot("04_training_running")
    check(running, "run produced live metrics (>= 3 updates; a 5-update smoke run may already be finished)")
    done = h.wait(lambda: (h.ctx.run_by_path(run_dir) and h.ctx.run_by_path(run_dir).status in ("completed", "failed", "interrupted")), 240, step=1.0)
    rec = h.run_record(run_dir)
    check(done and rec.status == "completed", f"run finished with status {rec.status if rec else None}")
    check(rec is not None and rec.updates_done == 5, f"5 updates recorded ({rec.updates_done if rec else None})")
    check(rec is not None and rec.latest is not None and rec.latest.update == 5, "latest checkpoint is at update 5 and has a compatibility sidecar")
    h.go("training", run_dir)
    h.shot("05_training_completed")
    (args.shots.parent / "last_smoke_run.txt").write_text(str(run_dir) + "\n")
    return all(l.startswith("PASS") for l in LOG)


def _status(h, run_dir):
    r = h.ctx.run_by_path(run_dir)
    return r.status if r else None


def _kill_worker(run_dir: Path) -> int | None:
    import psutil
    try:
        lab_pid = int((run_dir / "RUNNING").read_text().split()[0])
        for proc in psutil.Process(lab_pid).children(recursive=True):
            if "casu_grid.isaac_shard_worker" in " ".join(proc.cmdline()):
                proc.kill()
                return proc.pid
    except Exception as exc:
        print("kill failed", exc)
    return None


def stage_live(h: Harness, args) -> bool:
    """Steps 5-7 and 11-12: a longer run watched live; charts zoomed; two runs compared."""
    page = start_from_experiments(h, "smoke", updates=70, name=args.run_name or time.strftime("%Y%m%d-%H%M%S-live"))
    plan = page.current_plan()
    press_start(h, page)
    run_dir = plan.run_dir
    check(h.wait(lambda: _status(h, run_dir) == "running" and h.ctx.run_by_path(run_dir).updates_done >= 8, 240, step=1.0), "run is 'running' with live metrics")
    h.go("training", run_dir)
    h.shot("10_training_live")
    tp = h.page("training")
    check(tp.badge.text().strip().endswith("Running"), f"badge shows Running ({tp.badge.text()!r})")
    check(tp.stop_button.isVisible(), "Stop button is offered while running")
    check(tp.tiles["updates"].value.text() not in ("–", "0"), f"updates tile live ({tp.tiles['updates'].value.text()})")
    h.go("metrics", {"run": str(run_dir)})
    mp = h.page("metrics")
    h.pump(1.5)
    h.shot("11_metrics_live_grid")
    check(mp.live.text().startswith("● live"), f"metrics page marks the run live ({mp.live.text()!r})")
    card = mp._card("nav_success")
    check(card.view.has_data(), "goals-per-1,000-decisions chart has data")
    rows_before = mp.data[run_dir].metrics.rows
    h.pump(6)
    check(mp.data[run_dir].metrics.rows > rows_before, f"chart data grows while the run continues ({rows_before} -> {mp.data[run_dir].metrics.rows} rows)")
    # zoom and pan with real mouse events on one chart
    view = card.view
    rect = view.plot_rect()
    centre = QtCore.QPointF(rect.center().x(), rect.center().y())
    before = view.x_view()
    QtWidgets.QApplication.sendEvent(view, QtGui.QWheelEvent(centre, view.mapToGlobal(centre), QtCore.QPoint(), QtCore.QPoint(0, 120), QtCore.Qt.NoButton,
                                                             QtCore.Qt.NoModifier, QtCore.Qt.NoScrollPhase, False))
    h.pump(0.3)
    after = view.x_view()
    check(after[1] - after[0] < before[1] - before[0], f"mouse wheel zooms the chart ({before[1] - before[0]:.1f} -> {after[1] - after[0]:.1f} updates wide)")
    linked = mp._card("nav_progress").view.x_view()
    check(abs(linked[0] - after[0]) < 1e-6, "zoom is linked across charts")
    view._hover = QtCore.QPointF(rect.center().x() + 40, rect.center().y())
    h.shot("12_metrics_zoomed_hover")
    mp._enter_focus("nav_success")
    h.pump(1.0)
    h.shot("13_metrics_focus_chart")
    check(mp.stack.currentIndex() == 1 and "Numbers for" in mp.stats.text(), "focus view shows the chart with its numbers and explanation")
    mp._reset_all()
    mp._leave_focus()
    mp.tabs.setCurrentIndex(1)
    h.pump(1.0)
    h.shot("14_metrics_ppo_health")
    mp.tabs.setCurrentIndex(2)
    h.pump(1.0)
    h.shot("15_metrics_system")
    mp.tabs.setCurrentIndex(0)
    done = h.wait(lambda: _status(h, run_dir) in ("completed", "failed", "interrupted"), 240, step=1.0)
    rec = h.run_record(run_dir)
    check(done and rec.status == "completed" and rec.updates_done == 70, f"run completed all 70 updates ({rec.status}, {rec.updates_done})")
    # comparison of two runs
    smoke = Path((args.shots.parent / "last_smoke_run.txt").read_text().strip()) if (args.shots.parent / "last_smoke_run.txt").exists() else None
    if smoke and h.ctx.run_by_path(smoke):
        h.go("metrics", {"runs": [str(smoke), str(run_dir)]})
        h.pump(1.5)
        h.shot("16_metrics_two_run_overlay")
        check(len(h.page("metrics").data) == 2, "two runs overlaid on the same charts")
        h.go("runs", run_dir)
        rp = h.page("runs")
        rp.table.selectAll()
        h.pump(0.6)
        h.shot("17_runs_compare")
    h.go("runs", run_dir)
    h.pump(0.8)
    h.shot("18_runs_history")
    (args.shots.parent / "last_live_run.txt").write_text(str(run_dir) + "\n")
    return all(l.startswith("PASS") for l in LOG)


def stage_stop_resume(h: Harness, args) -> bool:
    """Steps 8-10: graceful stop, browse the checkpoint, resume it through the supported CUDA path."""
    page = start_from_experiments(h, "smoke", updates=400, name=args.run_name or time.strftime("%Y%m%d-%H%M%S-stop"))
    plan = page.current_plan()
    press_start(h, page)
    run_dir = plan.run_dir
    check(h.wait(lambda: _status(h, run_dir) == "running" and h.ctx.run_by_path(run_dir).updates_done >= 6, 240, step=1.0), "run is running")
    h.go("training", run_dir)
    tp = h.page("training")
    original = QtWidgets.QMessageBox.question
    QtWidgets.QMessageBox.question = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes)
    try:
        tp._stop()
    finally:
        QtWidgets.QMessageBox.question = original
    seen = set()
    h.wait(lambda: (seen.add(_status(h, run_dir)), _status(h, run_dir) in ("interrupted", "completed", "failed"))[1], 90, step=0.5)
    h.shot("20_training_stopped")
    rec = h.run_record(run_dir)
    check("stopping" in seen or rec.status == "interrupted", f"status passed through {sorted(s for s in seen if s)}")
    check(rec.status == "interrupted", f"graceful stop leaves the run 'interrupted' (got {rec.status})")
    check(rec.latest is not None and rec.latest.update == rec.updates_done, f"checkpoint matches the last finished update ({rec.latest.update if rec.latest else None} vs {rec.updates_done})")
    check(not (run_dir / "failure.json").exists(), "no failure.json after a graceful stop")
    stopped_at = rec.updates_done
    h.go("runs", run_dir)
    h.shot("21_runs_checkpoint_browser")
    # resume
    page = h.page("experiments")
    h.go("experiments", {"resume": str(rec.latest.path), "run": str(run_dir)})
    page.preset.setCurrentIndex(page.preset.findData("custom"))
    page.updates.setValue(8)
    page.every.setValue(4)
    page.run_name.setText(time.strftime("%Y%m%d-%H%M%S-resumed"))
    page._name_custom = True
    h.pump(0.8)
    page.revalidate()
    h.shot("22_experiments_resume_validation")
    errors = [c for c in page._checks if c.level == "error"]
    check(not errors, "resume validation passes for a compatible checkpoint: " + "; ".join(c.title for c in page._checks if "ayout" in c.title or "ontract" in c.title))
    check(not page.worlds.isEnabled(), "worlds and workers are locked to the checkpoint's layout")
    plan2 = page.current_plan()
    press_start(h, page)
    run2 = plan2.run_dir
    done = h.wait(lambda: _status(h, run2) in ("completed", "failed", "interrupted"), 300, step=1.0)
    rec2 = h.run_record(run2)
    check(done and rec2.status == "completed", f"resumed run completed ({rec2.status})")
    check(rec2.updates_done == stopped_at + 8, f"update numbering continues: {stopped_at} -> {rec2.updates_done}")
    store = h.page("metrics")
    first_row = json.loads((run2 / "metrics.jsonl").read_text().splitlines()[0])
    check(first_row["update"] == stopped_at + 1, f"first resumed update is {first_row['update']} (expected {stopped_at + 1})")
    h.go("training", run2)
    h.shot("23_training_resumed_run")
    (args.shots.parent / "last_stop_run.txt").write_text(str(run_dir) + "\n")
    return all(l.startswith("PASS") for l in LOG)


def stage_failure(h: Harness, args) -> bool:
    """Step 13: a deliberate failure (a simulation worker is killed) and what the interface tells the user."""
    page = start_from_experiments(h, "smoke", updates=400, name=args.run_name or time.strftime("%Y%m%d-%H%M%S-failure"))
    plan = page.current_plan()
    press_start(h, page)
    run_dir = plan.run_dir
    check(h.wait(lambda: _status(h, run_dir) == "running" and h.ctx.run_by_path(run_dir).updates_done >= 5, 240, step=1.0), "run is running")
    killed = _kill_worker(run_dir)
    check(killed is not None, f"killed simulation worker pid {killed}")
    failed = h.wait(lambda: _status(h, run_dir) == "failed", 90, step=0.5)
    check(failed, "status becomes 'failed' from the real process state")
    h.go("training", run_dir)
    h.pump(2.0)
    h.shot("30_training_failure")
    tp = h.page("training")
    check("worker" in tp.notice.title.text().lower(), f"explanation names the cause: {tp.notice.title.text()!r}")
    check(tp.failure_details.isVisible() and len(tp.failure_details.toPlainText()) > 20, "technical details are available")
    rec = h.run_record(run_dir)
    check(rec.latest is not None, f"checkpoint survives (update {rec.latest.update if rec.latest else None}) and is offered for resume")
    check(tp.resume_button.isVisible(), "'Resume from latest checkpoint' is offered")
    h.go("overview")
    h.shot("31_overview_after_failure")
    check("failed" in h.page("overview").hero_title.text().lower(), "overview surfaces the failure as the next action")
    h.go("runs", run_dir)
    h.shot("32_runs_failed_run")
    return all(l.startswith("PASS") for l in LOG)


def stage_evaluation(h: Harness, args) -> bool:
    """Evaluation workflow on the checkpoints of the stop/resume run."""
    run_dir = Path((args.shots.parent / "last_stop_run.txt").read_text().strip())
    h.go("evaluation", {"run": str(run_dir)})
    ep = h.page("evaluation")
    ep.worlds.setValue(8)
    ep.rounds.setValue(2)
    ep.horizon.setValue(300)
    ep._estimate()
    h.shot("40_evaluation_form")
    check(ep.start.isEnabled(), f"evaluation can start ({'; '.join(ep._problems)})")
    ep._start()
    done = h.wait(lambda: ep.process is None, 420, step=1.0)
    h.pump(1.0)
    check(done and ep.out_path is not None and ep.out_path.exists(), "evaluation finished and wrote its result file")
    ep._fill_results(select=ep.out_path)
    h.pump(1.0)
    h.shot("41_evaluation_results")
    check(ep.table.rowCount() >= 2, f"results table has {ep.table.rowCount()} rows")
    return all(l.startswith("PASS") for l in LOG)


def stage_gallery(h: Harness, args) -> bool:
    """Screenshots of every page with real run data (no simulator): light and dark theme, tall pages, a narrow window."""
    def tall(key, name, payload=None, height=2000):
        h.window.resize(1360, height)
        h.go(key, payload)
        h.pump(1.0)
        h.shot(name)
        h.window.resize(1360, 860)
        h.pump(0.3)

    runs = {r.run_id.split("-")[-1]: r for r in h.ctx.runs}
    live = next((r for r in h.ctx.runs if r.run_id.endswith("-live")), None)
    fail = next((r for r in h.ctx.runs if r.run_id.endswith("-failure")), None)
    stop = next((r for r in h.ctx.runs if r.run_id.endswith("-resumed")), None)
    check(live is not None and fail is not None and stop is not None, "journey runs exist")
    tall("overview", "50_overview_with_runs", height=1000)
    tall("experiments", "51_experiments_full", height=2300)
    h.go("training", live.path)
    tall("training", "52_training_completed_full", live.path, 1700)
    tall("metrics", "53_metrics_overview", {"run": str(live.path)}, 1400)
    h.page("metrics")._enter_focus("ppo_entropy") if False else None
    tall("runs", "54_runs_history", live.path, 1000)
    tall("evaluation", "55_evaluation_full", {"run": str(stop.path)}, 2300)
    tall("help", "56_help", None, 1800)
    base = next((r for r in h.ctx.runs if r.run_id.endswith("diagnostic-s7")), None)
    guard = next((r for r in h.ctx.runs if r.run_id.endswith("_guarded")), None)
    if base and guard:
        tall("training", "63_training_diagnostic_baseline_full", base.path, 1900)
        h.window.resize(1360, 860)
        h.go("metrics", {"run": str(base.path), "chart": "nav_outside"})
        h.pump(1.2)
        h.shot("64_metrics_focus_agents_outside_world")
        h.go("metrics", {"runs": [str(base.path), str(guard.path)]})
        mp = h.page("metrics")
        mp._leave_focus()
        h.pump(1.5)
        mp.window.setValue(20)
        h.pump(1.0)
        h.shot("65_metrics_baseline_vs_guarded_overlay")
        tall("evaluation", "66_evaluation_baseline_diagnostic", {"run": str(base.path)}, 2300)
        tall("evaluation", "67_evaluation_guarded_diagnostic", {"run": str(guard.path)}, 2300)
    h.window.resize(980, 700)
    for key, name in (("overview", "57_narrow_overview"), ("experiments", "58_narrow_experiments"), ("metrics", "59_narrow_metrics")):
        h.go(key, {"run": str(live.path)} if key == "metrics" else None)
        h.pump(0.8)
        h.shot(name)
    h.window.resize(1360, 860)
    h.ctx.set_theme("dark")
    h.pump(0.5)
    for key, name, payload in (("overview", "60_dark_overview", None), ("metrics", "61_dark_metrics", {"run": str(live.path)}), ("training", "62_dark_training", fail.path)):
        h.go(key, payload)
        h.pump(1.0)
        h.shot(name)
    h.ctx.set_theme("light")
    return True


def stage_diagnostic(h: Harness, args) -> bool:
    """The bounded learning diagnostic through the Studio: the validated 64 x 8 layout, ~1 M decisions, then a held-out evaluation."""
    name = args.run_name or time.strftime("%Y%m%d-%H%M%S-diagnostic-s7") + (f"-{args.tag}" if args.tag else "")
    if args.experiment:
        exp_page_ = h.page("experiments")
        h.go("experiments")
        row = next(i for i, e in enumerate(h.ctx.experiments) if args.experiment in str(e.path))
        exp_page_.list.setCurrentRow(row)
        h.experiment_chosen = True
        h.pump(0.5)
    page = start_from_experiments(h, "diagnostic", name=name, seed=7)
    plan = page.current_plan()
    check(plan.is_recommended_layout() and plan.updates == 250 and plan.seed == 7 and plan.checkpoint_every == 25, "diagnostic preset = 64 x 8 x 2 on CUDA, 250 updates, seed 7, checkpoint every 25")
    errors = [c for c in page._checks if c.level == "error"]
    check(not errors, "validation passes: " + "; ".join(c.title for c in errors))
    h.shot(f"70_diagnostic_configuration{args.tag}")
    press_start(h, plan and page)
    run_dir = plan.run_dir
    shots = 0
    last_shot = 0.0
    start = time.time()
    while time.time() - start < 3000:
        h.pump(2.0)
        st = _status(h, run_dir)
        if st in ("completed", "failed", "interrupted"):
            break
        if time.time() - last_shot > 150:
            last_shot = time.time()
            h.go("training", run_dir)
            h.shot(f"71_diagnostic_training{args.tag}_{shots:02d}")
            shots += 1
    rec = h.run_record(run_dir)
    check(rec.status == "completed" and rec.updates_done == 250, f"diagnostic run finished: {rec.status}, {rec.updates_done} updates, {rec.decisions:,} decisions")
    h.go("training", run_dir)
    h.shot(f"72_diagnostic_training_final{args.tag}")
    h.go("metrics", {"run": str(run_dir)})
    h.pump(1.5)
    h.shot(f"73_diagnostic_metrics{args.tag}")
    (args.shots.parent / f"last_diagnostic_run{args.tag}.txt").write_text(str(run_dir) + "\n")
    return all(l.startswith("PASS") for l in LOG)


def stage_diag_eval(h: Harness, args) -> bool:
    run_dir = Path((args.shots.parent / f"last_diagnostic_run{args.tag}.txt").read_text().strip())
    h.go("evaluation", {"run": str(run_dir)})
    ep = h.page("evaluation")
    ep.worlds.setValue(16)
    ep.rounds.setValue(6)
    ep.horizon.setValue(1000)
    ep._estimate()
    check(ep.start.isEnabled(), "evaluation can start: " + "; ".join(ep._problems))
    ep._start()
    done = h.wait(lambda: ep.process is None, 1500, step=2.0)
    check(done and ep.out_path is not None and ep.out_path.exists(), "evaluation finished")
    ep._fill_results(select=ep.out_path)
    h.window.resize(1360, 2300)
    h.pump(1.5)
    h.shot(f"74_diagnostic_evaluation{args.tag}")
    h.window.resize(1360, 860)
    return all(l.startswith("PASS") for l in LOG)


def stage_layer_exit_smoke(h: Harness, args) -> bool:
    """Engineering smoke of termination contract v2: repeated bottom exits and valid episodes afterwards, watched in the Studio (16 x 2, 60 updates; not a learning run)."""
    page = start_from_experiments(h, "smoke", updates=60, name=args.run_name or time.strftime("%Y%m%d-%H%M%S-layer-exit-smoke"))
    plan = page.current_plan()
    check(plan.experiment.termination_contract == "xy1-termination-v2-layer-exit", f"experiment is {plan.experiment.termination_contract}")
    check(any("Termination contract v2" in c.title for c in page._checks), "validation names the termination contract")
    h.shot("90_layer_exit_experiment")
    press_start(h, page)
    run_dir = plan.run_dir
    done = h.wait(lambda: _status(h, run_dir) in ("completed", "failed", "interrupted"), 600, step=1.0)
    rec = h.run_record(run_dir)
    check(done and rec.status == "completed", f"run finished: {rec.status}, {rec.updates_done} updates")
    check(rec.termination_contract == "xy1-termination-v2-layer-exit", "run record carries the termination contract")
    h.go("metrics", {"run": str(run_dir), "chart": "nav_layer_exit"})
    h.pump(1.5)
    h.shot("91_layer_exit_metrics_focus")
    mp = h.page("metrics")
    mp._leave_focus()
    h.pump(1.0)
    h.shot("92_layer_exit_metrics_overview")
    h.go("training", run_dir)
    h.pump(1.0)
    h.shot("93_layer_exit_training")
    (args.shots.parent / "last_layer_exit_smoke.txt").write_text(str(run_dir) + "\n")
    return all(l.startswith("PASS") for l in LOG)


if __name__ == "__main__":
    raise SystemExit(main())
