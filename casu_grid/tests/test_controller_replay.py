import unittest

from casu_grid.validation.controller_replay import validate_fixed_update, validate_update


class ControllerReplayTests(unittest.TestCase):
    """The ported Body logic reproduces recorded CasU ticks when seeded from them."""

    def check_fixed(self, name):
        report = validate_fixed_update(name)
        self.assertGreaterEqual(report["rows"], 690)
        err = report["max_abs_error"]
        self.assertLess(err["vx"], 1e-4)
        self.assertLess(err["vy"], 1e-4)
        self.assertLess(err["x"], 1e-4)
        self.assertLess(err["y"], 1e-4)
        self.assertEqual(err["gravity_scale"], 0.0)
        self.assertEqual(err["ended_jump"], 0.0)
        self.assertLess(err["max_speed"], 1e-4)

    def check_update(self, name, max_crouch_rows):
        report = validate_update(name)
        bad = report["mismatching_rows"]
        # discrete controller state is exact
        for key in ("grounded", "sliding_left", "sliding_right", "ended_jump", "wants_jump_input",
                    "move_x", "vy_after_jump", "ground_normal_x", "first_wall_jump",
                    "jump_cooldown", "time_since_grounded", "time_since_slid_left",
                    "time_since_slid_right", "time_slid_for"):
            self.assertEqual(bad.get(key, 0), 0, key)
        # crouch squash depends on Box2D contact-begin events; allow a documented few rows
        self.assertLessEqual(bad.get("crouch_amount", 0), max_crouch_rows)

    def test_flat_master_fixed_update(self):
        self.check_fixed("flat_master")

    def test_course_fixed_update(self):
        self.check_fixed("course_v1")

    def test_flat_master_update_and_actions(self):
        self.check_update("flat_master", max_crouch_rows=2)

    def test_course_update_and_actions(self):
        self.check_update("course_v1", max_crouch_rows=6)


if __name__ == "__main__":
    unittest.main()
