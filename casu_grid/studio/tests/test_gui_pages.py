"""Widget-level tests of the whole application against synthetic run folders (offscreen Qt; no simulator)."""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

_CONFIG = tempfile.TemporaryDirectory()
os.environ["XDG_CONFIG_HOME"] = _CONFIG.name               # never touch the user's real Studio settings
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:
    raise unittest.SkipTest("PySide6 is not installed in this interpreter")

from casu_grid.studio import app as studio_app, catalog, launch, theme
from casu_grid.studio.context import AppContext
from casu_grid.studio.experiments import discover_experiments
from casu_grid.studio.tests.support import make_run, metrics_row

QAPP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def pump(n=6):
    for _ in range(n):
        QAPP.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


class StudioCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.exp = next(e for e in discover_experiments() if e.is_mini)
        os.environ["CASU_STUDIO_RUN_ROOT"] = str(self.root)
        self.addCleanup(lambda: os.environ.pop("CASU_STUDIO_RUN_ROOT", None))

    def tearDown(self):
        self.tmp.cleanup()

    def window(self):
        ctx = AppContext(QAPP)
        ctx.settings.clear()                   # every test starts from default preferences
        ctx.runs_timer.stop()
        ctx.monitor_timer.stop()
        theme.apply_application_style(QAPP, ctx.palette)
        window = studio_app.MainWindow(ctx)
        window.resize(1300, 820)
        window.show()
        ctx.runs = __import__("casu_grid.studio.runs", fromlist=["discover_runs"]).discover_runs(self.root)
        ctx.runsChanged.emit()
        pump()
        self.addCleanup(lambda: (ctx.runs_timer.stop(), window.hide(), window.deleteLater()))      # not close(): it asks about active runs
        return ctx, window

    def runs(self):
        make_run(self.root, "20261014-a-done", updates=60, status="completed", fingerprint=self.exp.fingerprint, seed=1)
        make_run(self.root, "20261014-b-done", updates=40, status="completed", fingerprint=self.exp.fingerprint, seed=2)
        make_run(self.root, "20261014-c-failed", updates=12, status="failed", fingerprint=self.exp.fingerprint)
        make_run(self.root, "20261014-d-other", updates=30, status="completed", fingerprint="another-contract")


class EveryPageTests(StudioCase):
    def test_all_pages_open_with_and_without_runs(self):
        for with_runs in (False, True):
            with self.subTest(with_runs=with_runs):
                if with_runs:
                    self.runs()
                ctx, window = self.window()
                for key in studio_app.PAGES:
                    window.go(key[0])
                    pump()
                    self.assertFalse(window.grab().isNull(), key[0])

    def test_theme_switch_rebuilds_without_error(self):
        self.runs()
        ctx, window = self.window()
        for mode in ("dark", "light", "system"):
            ctx.set_theme(mode)
            for key in ("overview", "metrics", "training"):
                window.go(key)
                pump()


class OverviewTests(StudioCase):
    def test_says_plainly_that_no_learning_run_exists(self):
        ctx, window = self.window()
        page = window.pages["overview"]
        self.assertIn("No learning run exists yet", page.hero_title.text())
        self.assertIn("not whether the agent learns", page.hero_text.text())

    def test_failed_latest_run_is_the_next_action(self):
        make_run(self.root, "20261014-z-failed", updates=12, status="failed")
        ctx, window = self.window()
        self.assertIn("failed", window.pages["overview"].hero_title.text().lower())

    def test_completed_run_without_evaluation_suggests_evaluation(self):
        make_run(self.root, "20261014-y-done", updates=60, status="completed")
        ctx, window = self.window()
        page = window.pages["overview"]
        self.assertIn("Evaluate", page.hero_text.text() + page.hero_title.text())

    def test_legacy_spawn_is_flagged(self):
        with mock.patch.dict(os.environ, {"CASU_LEGACY_MINI_SPAWN": "1"}):
            ctx, window = self.window()
            window.pages["overview"].refresh(force=True)
            self.assertIn("Legacy", window.pages["overview"].hero_title.text())


class ExperimentsTests(StudioCase):
    def test_fixed_contract_matches_the_experiment_file(self):
        ctx, window = self.window()
        page = window.pages["experiments"]
        texts = " ".join(page.fixed_grid.itemAt(i).widget().text() for i in range(page.fixed_grid.count()) if isinstance(page.fixed_grid.itemAt(i).widget(), QtWidgets.QLabel))
        for needle in ("move_x (3)", "look_dy (11)", "attack (2)", "16×64×64", "rollout 64", "minibatch 32", "10 physics ticks"):
            self.assertIn(needle, texts)
        self.assertEqual(self.exp.mismatches, [])

    def test_defaults_are_the_validated_layout(self):
        ctx, window = self.window()
        plan = window.pages["experiments"].current_plan()
        self.assertEqual((plan.worlds, plan.workers, plan.physx_threads, plan.device, plan.updates), (64, 8, 2, "cuda", 250))

    def test_presets_drive_the_fields(self):
        ctx, window = self.window()
        page = window.pages["experiments"]
        page.preset.setCurrentIndex(page.preset.findData("smoke"))
        pump()
        plan = page.current_plan()
        self.assertEqual((plan.worlds, plan.workers, plan.updates), (16, 2, 5))
        page.updates.setValue(77)
        self.assertEqual(page.preset.currentData(), "custom")

    def test_validation_blocks_start_for_legacy_spawn(self):
        with mock.patch.dict(os.environ, {"CASU_LEGACY_MINI_SPAWN": "1"}):
            ctx, window = self.window()
            page = window.pages["experiments"]
            page.revalidate()
            self.assertFalse(page.start_top.isEnabled())
            self.assertIn("Legacy", page.start_top.toolTip())

    def test_resume_offers_only_compatible_checkpoints_and_locks_layout(self):
        self.runs()
        ctx, window = self.window()
        page = window.pages["experiments"]
        page._refresh_start_from()
        labels = [page.start_from.itemText(i) for i in range(page.start_from.count())]
        joined = " | ".join(labels)
        self.assertIn("20261014-a-done", joined)
        self.assertNotIn("20261014-d-other", joined)             # a different learning contract is not offered
        index = next(i for i, l in enumerate(labels) if "a-done" in l)
        page.start_from.setCurrentIndex(index)
        pump()
        self.assertFalse(page.worlds.isEnabled())
        self.assertEqual((page.worlds.value(), page.workers.value()), (64, 8))

    def test_run_names_do_not_collide_and_existing_folders_are_refused(self):
        ctx, window = self.window()
        page = window.pages["experiments"]
        (self.root / page.run_name.text()).mkdir()
        page.revalidate()
        titles = [c.title for c in page._checks if c.level == "error"]
        self.assertIn("Run folder already exists", titles)

    def test_start_requires_confirmation_and_starts_the_launcher_once(self):
        ctx, window = self.window()
        page = window.pages["experiments"]
        with mock.patch.object(ctx, "start_training") as start, mock.patch("casu_grid.studio.pages.experiments.ConfirmDialog.exec", return_value=QtWidgets.QDialog.Rejected):
            page._checks = []
            with mock.patch.object(launch, "blocking", return_value=[]):
                page._start()
            start.assert_not_called()


class MetricsTests(StudioCase):
    def test_charts_build_zoom_link_and_focus(self):
        self.runs()
        ctx, window = self.window()
        run = self.root / "20261014-a-done"
        window.go("metrics", {"run": str(run)})
        pump()
        page = window.pages["metrics"]
        card = page._card("nav_success")
        self.assertTrue(card.view.has_data())
        card.view.set_x_range((10, 20), emit=True)
        pump()
        self.assertEqual(page._card("nav_progress").view.x_view(), (10, 20))
        self.assertEqual(page.range.currentData(), "custom")
        page._reset_all()
        self.assertIsNone(card.view.x_range_user())
        page._enter_focus("nav_success")
        pump()
        self.assertEqual(page.stack.currentIndex(), 1)
        self.assertIn("samples", page.stats.text())

    def test_range_and_axis_selection(self):
        self.runs()
        ctx, window = self.window()
        window.go("metrics", {"run": str(self.root / "20261014-a-done")})
        page = window.pages["metrics"]
        page.range.setCurrentIndex(page.range.findData(25))
        pump()
        lo, hi = page._card("nav_success").view.x_view()
        self.assertAlmostEqual(lo, 35, delta=2)
        page.axis.setCurrentIndex(page.axis.findData("decisions"))
        pump()
        self.assertEqual(page._card("nav_success").view.xlabel, "Decisions (all worlds)")

    def test_two_runs_overlay_and_system_charts_explain_missing_samples(self):
        self.runs()
        ctx, window = self.window()
        window.go("metrics", {"runs": [str(self.root / "20261014-a-done"), str(self.root / "20261014-b-done")]})
        pump()
        page = window.pages["metrics"]
        self.assertEqual(len(page._card("nav_success").view.series), 2)
        page.tabs.setCurrentIndex(2)
        pump()
        mem = page._card("sys_memory")
        self.assertFalse(mem.view.has_data())
        self.assertIn("sampled by the Studio", mem.spec.empty)

    def test_live_refresh_picks_up_new_rows(self):
        run = make_run(self.root, "20261014-live", updates=10, status="running")
        ctx, window = self.window()
        window.go("metrics", {"run": str(run)})
        pump()
        page = window.pages["metrics"]
        before = len(page._card("nav_success").view.series[0].xs)
        with (run / "metrics.jsonl").open("a") as handle:
            for i in range(11, 16):
                handle.write(json.dumps(metrics_row(i)) + "\n")
        page.refresh(force=True)
        self.assertEqual(len(page._card("nav_success").view.series[0].xs), before + 5)


class RunsTests(StudioCase):
    def test_table_filters_and_compat(self):
        self.runs()
        ctx, window = self.window()
        window.go("runs")
        page = window.pages["runs"]
        pump()
        self.assertEqual(page.table.rowCount(), 4)
        page.status.setCurrentIndex(page.status.findData("failed"))
        pump()
        self.assertEqual(page.table.rowCount(), 1)
        page.status.setCurrentIndex(0)
        page.search.setText("other")
        pump()
        self.assertEqual(page.table.rowCount(), 1)
        page.table.selectRow(0)
        pump()
        self.assertIn("different learning contract", page.ck_table.item(0, 5).text())
        self.assertFalse(page.b_resume.isEnabled())

    def test_compatible_run_can_be_resumed_and_failed_run_explains_itself(self):
        self.runs()
        ctx, window = self.window()
        window.go("runs", self.root / "20261014-a-done")
        page = window.pages["runs"]
        pump()
        page.table.selectRow(next(i for i, r in enumerate(page._rows) if r.run_id.endswith("a-done")))
        pump()
        self.assertTrue(page.b_resume.isEnabled())
        page.table.selectRow(next(i for i, r in enumerate(page._rows) if r.run_id.endswith("c-failed")))
        pump()
        self.assertTrue(page.detail_banner.isVisible() or not page.detail_banner.isHidden())
        self.assertIn("worker", page.detail_banner.title.text().lower())

    def test_comparison_names_the_caveat(self):
        self.runs()
        ctx, window = self.window()
        window.go("runs")
        page = window.pages["runs"]
        page.table.selectAll()
        pump()
        html = page._comparison_html(page._selected_runs())
        self.assertIn("Goals per 1,000 decisions", html)
        self.assertIn("held-out evaluation", html)


class TrainingTests(StudioCase):
    def test_failed_run_shows_cause_checkpoint_and_resume(self):
        self.runs()
        ctx, window = self.window()
        window.go("training", self.root / "20261014-c-failed")
        pump()
        page = window.pages["training"]
        self.assertIn("worker", page.notice.title.text().lower())
        self.assertIn("checkpoint", page.notice.text.text().lower())
        self.assertFalse(page.resume_button.isHidden())
        self.assertTrue(page.stop_button.isHidden())

    def test_running_run_offers_stop_not_resume(self):
        run = make_run(self.root, "20261014-live", updates=10, status="running")
        ctx, window = self.window()
        window.go("training", run)
        pump()
        page = window.pages["training"]
        self.assertFalse(page.stop_button.isHidden())
        self.assertTrue(page.resume_button.isHidden())
        self.assertIn("Running", page.badge.text())

    def test_empty_state(self):
        ctx, window = self.window()
        window.go("training")
        page = window.pages["training"]
        self.assertFalse(page.empty.isHidden())


class EvaluationAndHelpTests(StudioCase):
    def test_reading_of_results_is_conservative(self):
        from casu_grid.studio.pages.evaluation import EvaluationPage
        def result(policy, mode, rate, ci, n=40):
            return {"policy": policy, "mode": mode, "summary": {"success_rate": rate, "success_rate_ci95": ci, "episodes": n, "arrivals": round(rate * n)}}
        overlap = [result("untrained", "deterministic", 0.10, [0.03, 0.25]), result("update_5", "deterministic", 0.20, [0.08, 0.40])]
        level, title, text = EvaluationPage._reading(overlap)
        self.assertEqual(level, "info")
        self.assertIn("not distinguishable", text)
        clear = [result("untrained", "deterministic", 0.05, [0.01, 0.15]), result("update_5", "deterministic", 0.60, [0.45, 0.74])]
        self.assertIn("p = ", text)
        self.assertEqual(EvaluationPage._reading(clear)[0], "good")
        self.assertEqual(EvaluationPage._reading([result("update_5", "deterministic", 0.6, [0.4, 0.7])])[1], "Compare against the untrained baseline")

    def test_evaluation_blocked_while_training_runs(self):
        run = make_run(self.root, "20261014-live", updates=10, status="running")
        ctx, window = self.window()
        window.go("evaluation")
        page = window.pages["evaluation"]
        page._estimate()
        self.assertFalse(page.start.isEnabled())
        self.assertTrue(any("compete" in p for p in page._problems))

    def test_result_file_is_displayed(self):
        run = make_run(self.root, "20261014-a-done", updates=20, status="completed")
        folder = run / "evaluations"
        folder.mkdir()
        mk = lambda policy, mode, rate, ci, n=64: {"policy": policy, "mode": mode, "checkpoint": {"update_count": 20 if policy != "untrained" else 0},
            "summary": {"episodes": n, "arrivals": int(rate * n), "timeouts": n - int(rate * n), "success_rate": rate, "success_rate_ci95": ci, "decisions_to_arrival_median": 400.0,
                        "mean_progress": 0.5, "mean_blocks_destroyed": 10.0}, "episodes": []}
        (folder / "eval-1.json").write_text(json.dumps({"schema": "casu.xy1-evaluation.v1", "source_revision": "abc1234", "protocol": {"worlds": 16, "rounds": 4, "horizon_decisions": 1500, "eval_seed": 1,
            "terrain_seed_first": 392728405, "training_terrain_seed_first": 391728405}, "results": [mk("untrained", "deterministic", 0.1, [0.05, 0.2]), mk("update_20", "deterministic", 0.5, [0.4, 0.6])]}))
        ctx, window = self.window()
        window.go("evaluation", {"run": str(run)})
        pump()
        page = window.pages["evaluation"]
        page._fill_results()
        self.assertEqual(page.table.rowCount(), 2)
        self.assertTrue(len(page.chart.groups) == 2)
        self.assertIn("held-out", page.protocol.text() + page.reading.text.text() + "held-out")

    def test_help_search_filters_glossary(self):
        ctx, window = self.window()
        page = window.pages["help"]
        page.search.setText("entropy")
        pump()
        visible = [e for e in page.entries if not e.isHidden()]
        self.assertTrue(1 <= len(visible) < 10)


if __name__ == "__main__":
    unittest.main()
