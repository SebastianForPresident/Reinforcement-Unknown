import json
import os
import tempfile
import unittest
from pathlib import Path

from casu_grid.studio import runs
from casu_grid.studio.tests.support import make_run, metrics_row


class MetricsStoreTests(unittest.TestCase):
    def test_incremental_reads_only_new_rows_and_keeps_partial_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.jsonl"
            store = runs.MetricsStore(path)
            self.assertEqual(store.refresh(), 0)                       # file does not exist yet
            path.write_text(json.dumps(metrics_row(1)) + "\n" + json.dumps(metrics_row(2))[:40])
            self.assertEqual(store.refresh(), 1)                       # the incomplete second line must wait
            with path.open("a") as handle:
                handle.write(json.dumps(metrics_row(2))[40:] + "\n" + json.dumps(metrics_row(3)) + "\n")
            self.assertEqual(store.refresh(), 2)
            self.assertEqual(store.column("update"), [1, 2, 3])
            self.assertEqual(store.refresh(), 0)

    def test_malformed_lines_are_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.jsonl"
            path.write_text(json.dumps(metrics_row(1)) + "\nnot json at all\n[1, 2]\n" + json.dumps(metrics_row(2)) + "\n")
            store = runs.MetricsStore(path)
            self.assertEqual(store.refresh(), 2)
            self.assertEqual(store.bad_lines, 2)

    def test_missing_values_stay_missing_and_new_columns_back_fill(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.jsonl"
            first = metrics_row(1)
            first.pop("clip_fraction")
            path.write_text(json.dumps(first) + "\n" + json.dumps(metrics_row(2)) + "\n")
            store = runs.MetricsStore(path)
            store.refresh()
            self.assertEqual(store.column("clip_fraction"), [None, 0.05])         # never a fabricated zero
            self.assertEqual(store.column("nav/arrival_rate"), [None, None])
            self.assertFalse(store.has("nav/arrival_rate"))
            self.assertEqual(store.column("not_a_metric"), [None, None])

    def test_truncated_or_replaced_file_restarts_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.jsonl"
            path.write_text("".join(json.dumps(metrics_row(i)) + "\n" for i in range(1, 6)))
            store = runs.MetricsStore(path)
            store.refresh()
            path.write_text(json.dumps(metrics_row(1)) + "\n")
            store.refresh()
            self.assertEqual(store.rows, 1)

    def test_last_json_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.jsonl"
            path.write_text(json.dumps({"a": 1}) + "\n" + json.dumps({"a": 2}) + "\n{broken")
            self.assertEqual(runs.last_json_line(path), {"a": 2})
            self.assertIsNone(runs.last_json_line(Path(tmp) / "none"))


class StatusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def status(self, name, **kw):
        make_run(self.root, name, **kw)
        return runs.load_run(self.root / name).status

    def test_status_comes_from_process_and_artifacts(self):
        self.assertEqual(self.status("done", status="completed"), "completed")
        self.assertEqual(self.status("bad", status="failed"), "failed")
        self.assertEqual(self.status("stopped", status="interrupted"), "interrupted")
        self.assertEqual(self.status("live", status="running"), "running")
        self.assertEqual(self.status("starting", status="running", rows=[]), "starting")

    def test_dead_owner_without_outcome_is_not_reported_as_running(self):
        make_run(self.root, "ghost", status="running", pid=2 ** 22 + 12345)
        self.assertEqual(runs.load_run(self.root / "ghost").status, "interrupted")      # has metrics
        make_run(self.root, "ghost2", status="running", pid=2 ** 22 + 12346, rows=[])
        self.assertEqual(runs.load_run(self.root / "ghost2").status, "failed")           # never produced anything

    def test_stop_request_shows_stopping_only_while_alive(self):
        run = make_run(self.root, "stopping", status="running")
        (run / "STOP").write_text("x")
        self.assertEqual(runs.load_run(run).status, "stopping")

    def test_foreign_directories_are_ignored(self):
        (self.root / "random").mkdir()
        (self.root / "random" / "config.json").write_text("{not json")
        make_run(self.root, "real", status="completed")
        found = runs.discover_runs(self.root)
        self.assertEqual([r.run_id for r in found], ["real"])

    def test_checkpoint_sidecar(self):
        make_run(self.root, "r", status="completed", fingerprint="abc", workers=8)
        run = runs.load_run(self.root / "r")
        self.assertEqual(run.latest.update, 30)
        self.assertEqual(run.latest.fingerprint, "abc")
        self.assertEqual(run.latest.world_partition, [8] * 8)
        self.assertEqual(run.latest.termination_contract, "xy1-termination-v2-layer-exit")
        self.assertEqual(run.termination_contract, "xy1-termination-v2-layer-exit")

    def test_runs_and_checkpoints_without_a_contract_id_are_the_reference_xy1(self):
        make_run(self.root, "old", status="completed", termination=None)
        run = runs.load_run(self.root / "old")
        self.assertEqual((run.termination_contract, run.latest.termination_contract), ("xy1-termination-v1", "xy1-termination-v1"))


class FormatTests(unittest.TestCase):
    def test_human_formats(self):
        self.assertEqual(runs.human_duration(75), "75 s")
        self.assertEqual(runs.human_duration(None), "–")
        self.assertEqual(runs.human_count(1_024_000), "1.02 M")
        self.assertEqual(runs.human_bytes(3 * 2 ** 30), "3.0 GiB")


if __name__ == "__main__":
    unittest.main()


class ResumedRunTests(unittest.TestCase):
    def test_resumed_run_knows_where_it_started(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = make_run(root, "parent", updates=9)
            child = make_run(root, "child", rows=[metrics_row(i) for i in range(10, 18)])
            manifest = json.loads((child / "config.json").read_text())
            manifest["resume_from"] = str(parent / "latest.pt")
            (child / "config.json").write_text(json.dumps(manifest))
            run = runs.load_run(child)
            self.assertEqual((run.start_update, run.updates_done, run.updates_in_run), (9, 17, 8))
