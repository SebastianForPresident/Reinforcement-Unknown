import math
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:         # the simulator's Python has no Qt; run these with the system Python
    raise unittest.SkipTest("PySide6 is not installed in this interpreter")

from casu_grid.studio import analysis, catalog, charts, runs, theme
from casu_grid.studio.tests.support import make_run, metrics_row

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def series(ys, xs=None, **kw):
    xs = xs or [float(i + 1) for i in range(len(ys))]
    return charts.Series(key="s", label="s", xs=xs, ys=ys, **kw)


class SmoothingTests(unittest.TestCase):
    def test_gaps_stay_gaps_and_are_not_filled(self):
        out = charts.smooth_values([1.0, None, 3.0, None, None, 5.0], 5)
        self.assertEqual([v is None for v in out], [False, True, False, True, True, False])

    def test_constant_signal_is_unchanged_by_bias_correction(self):
        out = charts.smooth_values([2.0] * 20, 10)
        self.assertTrue(all(abs(v - 2.0) < 1e-9 for v in out))

    def test_window_one_is_identity(self):
        self.assertEqual(charts.smooth_values([1.0, 2.0, None], 1), [1.0, 2.0, None])

    def test_ticks_are_nice_and_cover_the_range(self):
        t = charts.ticks(0.0, 1.0, 5)
        self.assertEqual(t[0], 0.0)
        self.assertLessEqual(t[-1], 1.0)
        self.assertTrue(all(abs((b - a) - (t[1] - t[0])) < 1e-9 for a, b in zip(t, t[1:])))


class ChartViewTests(unittest.TestCase):
    def make(self, ys, **spec_kw):
        view = charts.ChartView(charts.ChartSpec("k", "T", "y", empty="nothing here", **spec_kw), theme.LIGHT)
        view.resize(600, 300)
        view.set_series([series(ys)], "Update", True)
        return view

    def test_empty_chart_explains_itself(self):
        view = self.make([None, None])
        self.assertFalse(view.has_data())
        image = view.grab().toImage()
        self.assertFalse(image.isNull())

    def test_runs_are_split_at_gaps(self):
        view = self.make([1.0, 2.0, None, 3.0, 4.0, None, 5.0])
        rect = view.plot_rect()
        runs_ = view._paths_for(view.series[0].ys, view.series[0].xs, rect, view.x_view(), view.y_view(*view.x_view()), 500)
        self.assertEqual([len(r) for r in runs_], [2, 2, 1])          # two segments and a lone sample; nothing drawn across the gaps

    def test_long_series_are_reduced_but_keep_extremes(self):
        n = 50000
        ys = [math.sin(i / 50.0) for i in range(n)]
        ys[12345] = 7.0
        view = charts.ChartView(charts.ChartSpec("k", "T", "y"), theme.LIGHT)
        view.resize(600, 300)
        view.set_series([series(ys)], "Update", True)
        rect, xv = view.plot_rect(), view.x_view()
        yv = view.y_view(*xv)
        out = view._paths_for(ys, view.series[0].xs, rect, xv, yv, int(rect.width()))
        self.assertLess(sum(len(r) for r in out), 5 * rect.width())
        top = min(p.y() for r in out for p in r)
        self.assertAlmostEqual(top, view._to_px(0, 7.0, rect, (0, 1), yv).y(), delta=1.0)    # the spike survives decimation

    def test_hover_and_zoom_state(self):
        view = self.make([float(i) for i in range(100)])
        before = view.x_view()
        view.zoom(0.5)
        after = view.x_view()
        self.assertLess(after[1] - after[0], before[1] - before[0])
        view.reset_view()
        self.assertEqual(view.x_view(), before)
        self.assertEqual(view._nearest(view.series[0], 10.4), 9)

    def test_fixed_axis_bounds_for_rates(self):
        view = self.make([0.2, 0.4], y_min=0.0, y_max=1.0, include_zero=True)
        lo, hi = view.y_view(*view.x_view())
        self.assertLessEqual(lo, 0.0)
        self.assertGreater(lo, -0.1)                                   # only a small visual margin below the hard bound
        self.assertLessEqual(hi, 1.1)

    def test_series_on_a_hard_bound_stays_inside_the_frame(self):
        view = self.make([0.0, 0.0, 0.0], y_min=0.0, y_max=1.0, include_zero=True)
        lo, hi = view.y_view(*view.x_view())
        self.assertLess(lo, 0.0)                                   # the zero line is drawn above the frame's lower edge
        self.assertEqual([t for t in charts.ticks(lo, hi, 5)][0], 0.0)

    def test_csv_export_contains_raw_and_smoothed(self):
        s = series([1.0, None, 3.0])
        s.smoothed = charts.smooth_values(s.ys, 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.csv"
            charts.export_csv(str(path), [s], "Update")
            lines = path.read_text().splitlines()
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[2].endswith(",,"))          # the gap is exported as empty cells, not as zero

    def test_card_png_export(self):
        card = charts.ChartCard(charts.ChartSpec("k", "Title", "y"), theme.LIGHT)
        card.resize(500, 350)
        card.set_series([series([1.0, 2.0, 3.0])], "Update", True)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(card.export_png(str(Path(tmp) / "a.png")))
            self.assertGreater((Path(tmp) / "a.png").stat().st_size, 500)


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        rows = [metrics_row(i, episodes=(1 if i % 4 == 0 else 0)) for i in range(1, 21)]
        rows[3]["nav/arrivals_per_1k_decisions"] = None
        self.run = make_run(Path(self.tmp.name), "r", rows=rows)
        self.data = catalog.RunData("r", runs.MetricsStore(self.run / "metrics.jsonl"))
        self.data.refresh()

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_chart_metric_is_a_real_recorded_key_or_derived(self):
        from casu_grid.studio.tests.support import real_row_keys
        recorded = real_row_keys()          # captured from a real trainer run
        for chart in catalog.CHARTS:
            for sd in chart.series:
                if sd.source == "metrics" and not sd.key.startswith("derived:"):
                    self.assertIn(sd.key, recorded, f"{chart.spec.key} plots {sd.key}, which the (synthetic, trainer-shaped) run does not record")

    def test_gap_in_a_metric_becomes_a_gap_in_the_series(self):
        s = catalog.build_series(catalog.CHART_BY_KEY["nav_success"], [self.data], "update")[0]
        self.assertIsNone(s.ys[3])
        self.assertEqual(len(s.xs), 20)

    def test_axes(self):
        chart = catalog.CHART_BY_KEY["ppo_entropy"]
        self.assertEqual(catalog.build_series(chart, [self.data], "update")[0].xs[:2], [1.0, 2.0])
        self.assertEqual(catalog.build_series(chart, [self.data], "decisions")[0].xs[:2], [4096.0, 8192.0])
        t = catalog.build_series(chart, [self.data], "time")[0].xs
        self.assertAlmostEqual(t[1], 4.8)

    def test_derived_end_to_end_throughput(self):
        s = catalog.build_series(catalog.CHART_BY_KEY["sys_throughput"], [self.data], "update")[0]
        self.assertAlmostEqual(s.ys[0], 4096 / (1.7 + 0.65))

    def test_series_stats_never_count_gaps_as_zero(self):
        s = catalog.build_series(catalog.CHART_BY_KEY["nav_success"], [self.data], "update")
        stats = catalog.series_stats(s, None)[0]
        self.assertEqual(stats["samples"], 19)
        self.assertEqual(stats["missing"], 1)

    def test_overlay_uses_run_colours(self):
        other = catalog.RunData("o", runs.MetricsStore(self.run / "metrics.jsonl"), color="#D55E00")
        other.refresh()
        s = catalog.build_series(catalog.CHART_BY_KEY["nav_success"], [self.data, other], "update")
        self.assertEqual({x.color for x in s}, {self.data.color, "#D55E00"})


class AnalysisTests(unittest.TestCase):
    def store(self, values, progress=0.3):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        rows = [metrics_row(i + 1, arrivals=v, progress=progress) for i, v in enumerate(values)]
        run = make_run(Path(tmp.name), "r", rows=rows)
        st = runs.MetricsStore(run / "metrics.jsonl")
        st.refresh()
        return st

    def test_too_early(self):
        self.assertEqual(analysis.learning_signal(self.store([1.0] * 5)).state, "too_early")

    def test_flat_noise_is_not_called_learning(self):
        import random
        rnd = random.Random(3)
        values = [1.0 + rnd.uniform(-0.5, 0.5) for _ in range(80)]
        self.assertIn(analysis.learning_signal(self.store(values)).state, ("no_change",))

    def test_clear_increase_is_called_improving_with_a_caveat(self):
        import random
        rnd = random.Random(4)
        values = [0.5 + 0.05 * i + rnd.uniform(-0.2, 0.2) for i in range(60)]
        sig = analysis.learning_signal(self.store(values))
        self.assertEqual(sig.state, "improving")
        self.assertIn("held-out", sig.detail)

    def test_progress_can_reveal_a_trend_that_sparse_arrivals_hide(self):
        import random
        rnd = random.Random(8)
        arrivals = [0.3 + rnd.uniform(-0.3, 0.3) for _ in range(80)]           # noisy and flat
        rows = [metrics_row(i + 1, arrivals=a, progress=0.45 + 0.002 * i + rnd.uniform(-0.01, 0.01)) for i, a in enumerate(arrivals)]
        with tempfile.TemporaryDirectory() as tmp:
            run = make_run(Path(tmp), "r", rows=rows)
            st = runs.MetricsStore(run / "metrics.jsonl")
            st.refresh()
            sig = analysis.learning_signal(st)
        self.assertEqual(sig.state, "improving")
        self.assertIn("Progress", sig.headline)

    def test_a_drained_sample_stream_is_not_called_learning(self):
        rows = [metrics_row(i + 1, arrivals=0.5, progress=0.3 + 0.004 * i) for i in range(60)]
        for r in rows:
            r["nav/active_outside_world_fraction"] = 0.9
        with tempfile.TemporaryDirectory() as tmp:
            run = make_run(Path(tmp), "r", rows=rows)
            st = runs.MetricsStore(run / "metrics.jsonl")
            st.refresh()
            sig = analysis.learning_signal(st)
        self.assertEqual(sig.state, "drained")
        self.assertIn("left the world", sig.headline)

    def test_decrease_is_reported(self):
        values = [3.0 - 0.04 * i for i in range(60)]
        self.assertEqual(analysis.learning_signal(self.store(values)).state, "worse")

    def test_before_the_first_update_it_waits_instead_of_complaining(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = runs.MetricsStore(Path(tmp) / "metrics.jsonl")
            store.refresh()
            sig = analysis.learning_signal(store)
            self.assertEqual(sig.state, "too_early")
            self.assertIn("first update", sig.headline)

    def test_missing_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = metrics_row(1)
            row.pop("nav/arrivals_per_1k_decisions")
            run = make_run(Path(tmp), "r", rows=[row] * 30)
            st = runs.MetricsStore(run / "metrics.jsonl")
            st.refresh()
            self.assertEqual(analysis.learning_signal(st).state, "none")


if __name__ == "__main__":
    unittest.main()
