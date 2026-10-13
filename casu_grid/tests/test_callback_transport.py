import importlib.util
import tempfile
import unittest
from pathlib import Path

from casu_grid import callback_transport as T

SCALE = 3


def top_level(x):
    return x * SCALE


def make_closure(offset, helper=top_level):
    def inner(x):
        return helper(x) + offset
    return inner


def make_wrapper(callback):
    def wrapped(x):
        try:
            return callback(x)
        except Exception as exc:
            raise ValueError(f"wrapped failure: {exc}") from exc
    return wrapped


class CallbackTransportTests(unittest.TestCase):
    def roundtrip(self, obj):
        return T.loads(T.dumps(obj))

    def test_importable_function_travels_by_reference(self):
        self.assertIs(self.roundtrip(top_level), top_level)

    def test_closure_and_wrapper_chain_are_sent_by_value(self):
        fn = make_wrapper(make_closure(10))
        copy = self.roundtrip(fn)
        self.assertIsNot(copy, fn)
        self.assertEqual(copy(4), fn(4))
        self.assertEqual(copy(4), 4 * SCALE + 10)
        with self.assertRaisesRegex(ValueError, "wrapped failure"):
            self.roundtrip(make_wrapper(lambda v: 1 / 0))(1)

    def test_lambda_and_tuple_of_callbacks(self):
        bundle = {"a": (lambda v: v + 1, make_closure(2)), "n": 5}
        copy = self.roundtrip(bundle)
        self.assertEqual(copy["a"][0](1), 2)
        self.assertEqual(copy["a"][1](1), 5)
        self.assertEqual(copy["n"], 5)

    def test_function_of_a_module_loaded_from_a_path(self):
        source = ("import math\nLIMIT = 7\ndef helper(v):\n    return min(v, LIMIT)\n"
                  "def callbacks():\n    scale = math.sqrt(4)\n    def encode(v):\n        return helper(v) * scale\n"
                  "    return encode\n")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "research_config.py"
            path.write_text(source)
            spec = importlib.util.spec_from_file_location("research_config", path)
            module = importlib.util.module_from_spec(spec)      # deliberately not in sys.modules
            spec.loader.exec_module(module)
            encode = module.callbacks()
            copy = self.roundtrip(encode)
            self.assertEqual(copy(100), encode(100))
            self.assertEqual(copy(3), 6.0)

    def test_unpicklable_capture_gives_a_clear_error(self):
        def make():
            handle = open(__file__)
            def bad(v):
                return handle.name
            return bad
        with self.assertRaisesRegex(ValueError, "num_workers=1"):
            T.dumps(make())


if __name__ == "__main__":
    unittest.main()
