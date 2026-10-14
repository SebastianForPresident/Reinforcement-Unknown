"""Run unittest modules that need USD (pxr) inside a headless Kit process.

    python casu_grid/perf/run_kit_tests.py casu_grid.tests.test_incremental_terrain [more modules]
"""
import sys
import unittest
from pathlib import Path

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
suite = unittest.defaultTestLoader.loadTestsFromNames([m for m in sys.argv[1:] if not m.startswith("-")])
result = unittest.TextTestRunner(verbosity=2).run(suite)
print("KIT_TESTS", "OK" if result.wasSuccessful() else "FAILED", result.testsRun, len(result.failures), len(result.errors), flush=True)
sys.stdout.flush()
app.close()
