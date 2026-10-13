import os
import unittest
from unittest import mock

from casu_grid.isaac_navigation_env import IsaacLayer1EnvConfig
from casu_grid.isaac_sharded_env import ShardedBatchedEnv, WorkerFailure, resolve_worker_count, shard_sizes


class ShardingPlanTests(unittest.TestCase):
    def test_worker_failure_is_raised_and_terminates_owned_siblings(self):
        import tempfile
        from pathlib import Path

        class FakeProcess:
            def __init__(self): self.terminated = False
            def poll(self): return -15 if self.terminated else None
            def terminate(self): self.terminated = True
            def wait(self, timeout=None): return -15

        first, failed = FakeProcess(), FakeProcess()
        with tempfile.TemporaryDirectory() as tmp:
            env = object.__new__(ShardedBatchedEnv)
            env._procs = [first, failed]
            env.log_dir = Path(tmp)
            env.closed = False
            (env.log_dir / "worker_1.log").write_text("worker traceback evidence")
            with self.assertRaisesRegex(WorkerFailure, "worker 1 reported an error") as caught:
                env._fail("worker 1 reported an error", 1)
            self.assertIn("worker traceback evidence", str(caught.exception))
            self.assertTrue(first.terminated)
            self.assertTrue(failed.terminated)
            self.assertTrue(env.closed)

    def test_shards_are_contiguous_balanced_and_complete(self):
        for total in range(1, 40):
            for workers in range(1, 9):
                sizes = shard_sizes(total, workers)
                self.assertEqual(sum(sizes), total)
                self.assertLessEqual(max(sizes) - min(sizes), 1)
                self.assertLessEqual(len(sizes), min(total, workers))

    def test_explicit_worker_count_is_capped_by_world_count(self):
        self.assertEqual(resolve_worker_count(IsaacLayer1EnvConfig(num_envs=3, num_workers=8)), 3)
        self.assertEqual(resolve_worker_count(IsaacLayer1EnvConfig(num_envs=64, num_workers=4)), 4)

    def test_auto_uses_at_least_four_worlds_per_worker_and_at_most_four_workers(self):
        with mock.patch("os.cpu_count", return_value=16):
            auto = lambda n: resolve_worker_count(IsaacLayer1EnvConfig(num_envs=n, num_workers=0))
            self.assertEqual([auto(n) for n in (1, 3, 4, 8, 16, 64, 512)], [1, 1, 1, 2, 4, 4, 4])
        with mock.patch("os.cpu_count", return_value=4):
            self.assertEqual(resolve_worker_count(IsaacLayer1EnvConfig(num_envs=64, num_workers=0)), 1)

    def test_environment_variable_selects_workers(self):
        for value, expected in (("1", 1), ("4", 4), ("auto", 0)):
            with mock.patch.dict(os.environ, {"CASU_NUM_WORKERS": value}):
                self.assertEqual(IsaacLayer1EnvConfig().num_workers, expected)

    def test_validation_rejects_unusable_worker_settings(self):
        with self.assertRaises(ValueError):
            IsaacLayer1EnvConfig(worker_physx_threads=1).validate()    # one PhysX thread hangs the scene
        with self.assertRaises(ValueError):
            IsaacLayer1EnvConfig(num_workers=-1).validate()


if __name__ == "__main__":
    unittest.main()
