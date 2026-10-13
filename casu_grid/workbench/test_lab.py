from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path

from . import lab


class LabUsabilityTests(unittest.TestCase):
    def test_scaffold_validates_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "my_exp.py"
            created = lab.scaffold("My question", path, hypothesis="test a change")
            self.assertEqual(created, path.resolve())
            self.assertIn("Hypothesis: test a change", path.read_text())
            summary = lab.inspect_config(path)
            self.assertEqual(summary["physics_ticks_per_decision"], 10)
            self.assertTrue(summary["callbacks_defined"])
            config, _ = lab._load_config(path)
            self.assertEqual(config.EXPERIMENT["hypothesis"], "test a change")
            self.assertTrue(config.EXPERIMENT["id"])
            self.assertEqual(config.EXPERIMENT["config_version"], 1)
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                lab.scaffold("overwrite", path)

    def test_legacy_engineering_template_source_is_not_rewritten_by_scaffolding_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "copy.py"
            lab.scaffold("copy", path)
            config, _ = lab._load_config(path)
            self.assertEqual(config.EXPERIMENT["name"], "copy")
            self.assertIn("temporary engineering-only", Path(lab.TEMPLATE).read_text())

    def test_invalid_config_names_missing_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.py"
            path.write_text("ACTION = None\n")
            with self.assertRaisesRegex(ValueError, "missing required"):
                lab.inspect_config(path)

    def test_missing_reward_callback_is_actionable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "no_reward_callback.py"
            source = Path(lab.TEMPLATE).read_text().replace("\ndef callbacks():", "\n# callbacks intentionally omitted\ndef callbacks_disabled():")
            # Remove callback implementation while retaining the real contracts.
            source = source[:source.index("# callbacks intentionally omitted")]
            path.write_text(source)
            with self.assertRaisesRegex(ValueError, "zero reward"):
                lab.inspect_config(path)

    def _run(self, root: Path, name: str, rows, *, complete=False, running=False):
        path = root / name; path.mkdir()
        manifest = {"schema": "casu.training-run.v1", "git_revision": "abc",
                    "config_sha256": name, "contracts": {"rewards": {"terms": []}},
                    "environment": {"num_envs": 2, "terrain_mode": "shared"}}
        (path / "config.json").write_text(json.dumps(manifest))
        (path / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows) + '{bad')
        if complete: (path / "COMPLETE").write_text("done")
        if running: (path / "RUNNING").write_text("999999999")
        if rows: (path / "latest.pt").write_bytes(b"checkpoint marker")
        return path

    def test_discovery_partial_metrics_status_and_artifact_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._run(root, "done", [{"update": 2, "environment_steps": 64,
                "mean_completed_return": 1.2, "policy_decisions_per_second": 10}], complete=True)
            self._run(root, "stale", [{"update": 1}], running=True)
            rows = lab.discover_runs(root)
            by_name = {r["run_id"]: r for r in rows}
            self.assertEqual(by_name["done"]["status"], "completed")
            self.assertEqual(by_name["stale"]["status"], "interrupted")
            self.assertEqual(by_name["done"]["environment_steps"], 64)
            self.assertIn("incomplete/corrupt", by_name["done"]["metrics_warning"])
            self.assertTrue(by_name["done"]["latest_checkpoint"].endswith("latest.pt"))

    def test_comparison_reports_contract_differences_and_step_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self._run(root, "a", [{"update": 1, "environment_steps": 20, "entropy": 0.7},
                                       {"update": 2, "environment_steps": 40, "entropy": 0.6}])
            b = self._run(root, "b", [{"update": 1, "environment_steps": 20, "entropy": 0.5},
                                       {"update": 2, "environment_steps": 40, "entropy": 0.2}])
            manifest = json.loads((b / "config.json").read_text())
            manifest["environment"]["num_envs"] = 4
            (b / "config.json").write_text(json.dumps(manifest))
            result = lab.compare_runs([a, b])
            self.assertEqual(result["configuration_differences"]["environment.num_envs"], [2, 4])
            self.assertEqual(result["metrics_by_environment_decisions"]["a"][0]["environment_steps"], 20)
            self.assertTrue(result["reward_contracts_match"])
            self.assertIn("entropy", result["aligned_metrics_at_shared_environment_decision_budgets"])

    def test_cli_safe_errors_and_machine_readable_run_listing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._run(root, "one", [])
            # parser dispatch writes JSON and returns cleanly
            self.assertEqual(lab.main(["--experiments", str(root), "runs", "--json"]), 0)
            self.assertEqual(lab.main(["--experiments", str(root), "status", "missing"]), 2)

    def test_launch_preview_works_without_isaac_and_uses_shell_quoted_paths(self):
        with tempfile.TemporaryDirectory(prefix="path with spaces ") as tmp:
            config = Path(tmp) / "template copy.py"
            config.write_bytes(Path(lab.TEMPLATE).read_bytes())
            run_dir = Path(tmp) / "output run"
            self.assertEqual(lab.main(["launch", str(config), "--preview", "--engineering-smoke",
                                       "--run-dir", str(run_dir), "--updates", "2"]), 0)
            self.assertFalse(run_dir.exists())

    def test_historical_files_are_not_touched(self):
        # Scaffold operates only on its explicit destination; this fixture models
        # preservation of legacy artifacts without accessing/deleting repository data.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); historical = root / "PPO_Python" / "old.json"
            historical.parent.mkdir(); historical.write_text("legacy")
            lab.scaffold("new", root / "new.py")
            self.assertEqual(historical.read_text(), "legacy")

    def test_comparison_requires_two_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self._run(Path(tmp), "one", [])
            with self.assertRaisesRegex(ValueError, "at least two"):
                lab.compare_runs([run])

    def test_resume_uses_exact_snapshot_when_source_moved_or_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self._run(Path(tmp), "run", [])
            source = run / "missing.py"
            source.write_text("changed")
            snapshot = run / "research_config.py"
            snapshot.write_text("exact config")
            manifest = json.loads((run / "config.json").read_text())
            manifest["config_path"] = str(source)
            manifest["config_sha256"] = __import__("hashlib").sha256(snapshot.read_bytes()).hexdigest()
            (run / "config.json").write_text(json.dumps(manifest))
            self.assertEqual(lab._resolve_config_for_run(run, None), snapshot.resolve())

    def test_run_path_can_be_supplied_relative_to_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(tmp)
                root = Path("runs"); root.mkdir()
                run = self._run(root, "sample", [])
                self.assertEqual(lab._find_run("runs/sample", Path("other-root").resolve()), run.resolve())
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
