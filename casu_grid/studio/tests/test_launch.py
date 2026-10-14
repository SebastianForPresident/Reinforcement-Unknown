import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from casu_grid.studio import environment as env
from casu_grid.studio import launch
from casu_grid.studio.experiments import ExperimentInfo, discover_experiments
from casu_grid.studio.runs import CheckpointInfo


def mini() -> ExperimentInfo:
    return next(e for e in discover_experiments() if e.is_mini)


def plan(root: Path, **kw) -> launch.LaunchPlan:
    return launch.LaunchPlan(experiment=mini(), run_root=root, run_name="t-run", **kw)


_TMP = tempfile.TemporaryDirectory()
_FILE = Path(_TMP.name) / "latest.pt"
_FILE.write_bytes(b"x")


def checkpoint(fingerprint, partition):
    return CheckpointInfo(_FILE, "latest", 5, 5000, None, 10, fingerprint, partition, 0.0)


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.install = env.discover_isaac()

    def tearDown(self):
        self.tmp.cleanup()

    def levels(self, p, **patches):
        return {c.title: c.level for c in launch.validate(p, self.install)}

    def test_default_layout_is_the_validated_xy1_layout(self):
        p = plan(self.root)
        self.assertEqual((p.worlds, p.workers, p.physx_threads, p.device), (64, 8, 2, "cuda"))
        self.assertTrue(p.is_recommended_layout())
        self.assertEqual(p.decisions_per_update, 4096)

    def test_command_pins_the_scientific_settings(self):
        cmd = launch.build_command(plan(self.root), self.install)
        text = " ".join(cmd)
        for needle in ("launch", "--terrain-mode independent", "--num-envs 64", "--workers 8", "--physx-threads 2", "--train-device cuda", "--save-initial-checkpoint"):
            self.assertIn(needle, text)
        self.assertNotIn("--resume", text)

    def test_resume_command_does_not_save_an_initial_checkpoint(self):
        p = plan(self.root, resume=checkpoint("x", None))
        self.assertNotIn("--save-initial-checkpoint", launch.build_command(p, self.install))
        self.assertIn("--resume", launch.build_command(p, self.install))

    def test_legacy_spawn_blocks_launch(self):
        with mock.patch.dict(os.environ, {"CASU_LEGACY_MINI_SPAWN": "1"}):
            checks = launch.validate(plan(self.root), self.install)
        errors = [c.title for c in launch.blocking(checks)]
        self.assertTrue(any("Legacy spawn" in t for t in errors), errors)

    def test_execution_overrides_are_removed_from_the_child_environment(self):
        cleaned = env.clean_child_environment({"CASU_NUM_WORKERS": "3", "CASU_PPO_EXECUTION": "reference", "PATH": "/bin", "CASU_LEGACY_MINI_SPAWN": "1"})
        self.assertEqual(set(cleaned), {"PATH", "CASU_LEGACY_MINI_SPAWN"})          # the legacy flag is never silently removed: validation refuses it

    def test_invalid_layouts_are_errors_not_surprises(self):
        self.assertIn("At least 2 PhysX threads per worker", self.levels(plan(self.root, physx_threads=1)))
        self.assertEqual(self.levels(plan(self.root, physx_threads=1))["At least 2 PhysX threads per worker"], launch.ERROR)
        self.assertEqual(self.levels(plan(self.root, worlds=4, workers=8))["More workers than worlds"], launch.ERROR)
        self.assertEqual(self.levels(plan(self.root, workers=16))["More simulation threads than CPU threads"], launch.WARN)

    def test_existing_run_folder_is_never_overwritten(self):
        (self.root / "t-run").mkdir()
        self.assertEqual(self.levels(plan(self.root))["Run folder already exists"], launch.ERROR)

    def test_cpu_request_without_gpu_does_not_fall_back_silently(self):
        with mock.patch.object(env, "gpu_info", return_value=None):
            levels = self.levels(plan(self.root, device="cuda"))
        self.assertEqual(levels["No NVIDIA GPU visible"], launch.ERROR)

    def test_resume_checks(self):
        exp = mini()
        good = launch.resume_checks(plan(self.root, resume=checkpoint(exp.fingerprint, [8] * 8)))
        self.assertFalse(launch.blocking(good))
        wrong_contract = launch.resume_checks(plan(self.root, resume=checkpoint("other", [8] * 8)))
        self.assertTrue(any("contract" in c.title.lower() for c in launch.blocking(wrong_contract)))
        wrong_layout = launch.resume_checks(plan(self.root, workers=4, resume=checkpoint(exp.fingerprint, [8] * 8)))
        self.assertTrue(any("layout" in c.title.lower() for c in launch.blocking(wrong_layout)))
        unknown = launch.resume_checks(plan(self.root, resume=checkpoint(None, None)))
        self.assertFalse(launch.blocking(unknown))
        self.assertTrue(any(c.level == launch.WARN for c in unknown))

    def test_shard_sizes_match_the_trainers_partition(self):
        try:
            from casu_grid.isaac_sharded_env import shard_sizes
        except ModuleNotFoundError as exc:
            self.skipTest(f"needs the simulator's Python ({exc})")
        for worlds, workers in ((64, 8), (16, 2), (10, 4), (7, 3)):
            self.assertEqual(launch.shard_sizes(worlds, workers), list(shard_sizes(worlds, workers)))

    def test_budget_arithmetic(self):
        p = plan(self.root, updates=250)
        self.assertEqual(p.total_decisions, 1_024_000)
        self.assertAlmostEqual(p.estimated_seconds(1700.0), 602.35, places=1)

    def test_failure_explanations_are_plain_language(self):
        class R:
            failure = {"log_tail": "casu_grid.isaac_sharded_env.WorkerFailure: worker 3 closed its pipe"}
        self.assertIn("worker", launch.explain_failure(R())[0].lower())
        R.failure = {"log_tail": "FloatingPointError: non-finite PPO loss or gradient norm"}
        self.assertIn("diverged", launch.explain_failure(R())[0].lower())
        R.failure = {"log_tail": "ValueError: PPO checkpoint learning contract is incompatible with this run"}
        self.assertIn("checkpoint", launch.explain_failure(R())[0].lower())
        self.assertIn("without leaving", launch.explain_failure(None)[0])


if __name__ == "__main__":
    unittest.main()
