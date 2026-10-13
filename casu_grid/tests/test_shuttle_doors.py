import unittest

from casu_grid.shuttle_doors import ShuttleDoors


class ShuttleDoorTests(unittest.TestCase):
    def make(self):
        return ShuttleDoors(trigger_boxes=((3.5, 506.5, 4.5, 507.5), (-4.5, 506.5, -3.5, 507.5)),
                            base_centers=((5.5, 508.0), (-5.5, 508.0)))

    def test_doors_stay_closed_until_the_player_reaches_a_trigger(self):
        doors = self.make()
        for _ in range(500):
            doors.tick(0.02, (-0.8, 505.1, 0.8, 510.1))
        self.assertFalse(doors.activated)
        self.assertEqual(doors.offset(), 0.0)

    def test_timeline_after_activation(self):
        doors = self.make()
        body = (3.4, 505.1, 5.0, 510.1)   # overlaps the right trigger
        moved_at = None
        for tick in range(1, 600):
            if doors.tick(0.02, body) and moved_at is None:
                moved_at = tick
        self.assertTrue(doors.activated)
        self.assertAlmostEqual(moved_at * 0.02, 2.0, delta=0.05)    # starts after 2 s
        self.assertAlmostEqual(doors.offset(), 6.0)                 # fully raised
        self.assertTrue(doors.destroyed)

    def test_partial_lift_is_linear_in_time(self):
        doors = self.make()
        body = (3.4, 505.1, 5.0, 510.1)
        for _ in range(int(round((2.0 + 0.5 / 0.75) / 0.02))):  # half way through the lift
            doors.tick(0.02, body)
        self.assertAlmostEqual(doors.offset(), 3.0, delta=0.15)


if __name__ == "__main__":
    unittest.main()
