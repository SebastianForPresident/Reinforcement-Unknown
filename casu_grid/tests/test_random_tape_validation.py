import gzip
import json
import unittest
from pathlib import Path

from casu_grid.source_traces import load_source_trace
from casu_grid.validation.analyze_random_tapes import poisson_ci
from casu_grid.validation.random_tapes import tape

ROOT = Path(__file__).resolve().parents[1] / "validation" / "casu_random_tapes"
SEEDS = range(2001, 2011)


class RandomTapeTests(unittest.TestCase):
    def test_tape_generator_is_reproducible_and_neutral_on_unused_heads(self):
        a, b = tape(2001, 200), tape(2001, 200)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 200)
        self.assertTrue(all(row[2:3] == [1] and row[4:] == [4, 5, 0] for row in a))
        self.assertNotEqual(tape(2001, 20), tape(2002, 20))

    def test_recorded_tapes_match_generator_and_checksums(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                trace = load_source_trace(f"rand_{seed}", ROOT)          # verifies SHA-256 and row counts
                self.assertEqual([list(a) for a in trace.actions], tape(seed, 200))
                self.assertEqual(len(trace.pre_fixed), 2000)
                self.assertEqual(len(trace.post_physics), 2000)

    def test_outcomes_have_one_row_per_decision(self):
        for seed in SEEDS:
            outcomes = json.loads((ROOT / f"rand_{seed}" / "outcomes.json").read_text())
            self.assertEqual(len(outcomes["steps"]), 200)
            self.assertEqual(max(s["outcome"]["AttackSwingsExecuted"] for s in outcomes["steps"]), 0)

    def test_casu_repeats_agree(self):
        def rb(name):
            with gzip.open(ROOT / name / "post_physics.jsonl.gz", "rt") as stream:
                stream.readline()
                return [json.loads(line)["extra"]["rb_position"] for line in stream]
        for seed in (2001, 2002):
            a, b = rb(f"rand_{seed}"), rb(f"rand_{seed}_aa")
            self.assertLess(max(max(abs(p[0] - q[0]), abs(p[1] - q[1])) for p, q in zip(a, b)), 1e-4)

    def test_poisson_interval(self):
        lower, upper = poisson_ci(0, 2000)
        self.assertEqual(lower, 0.0)
        self.assertAlmostEqual(upper * 2000, 3.689, places=2)           # -ln(0.025)
        lower, upper = poisson_ci(10, 1000)
        self.assertAlmostEqual(lower * 1000, 4.795, places=2)
        self.assertAlmostEqual(upper * 1000, 18.39, places=1)


if __name__ == "__main__":
    unittest.main()
