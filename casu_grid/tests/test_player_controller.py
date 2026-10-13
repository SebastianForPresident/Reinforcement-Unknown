import unittest

from casu_grid.player_controller import (
    StandingAction,
    StandingPhysicsConfig,
    StandingPlayerController,
)


class StandingControllerTests(unittest.TestCase):
    def setUp(self):
        self.controller = StandingPlayerController()

    def test_source_baseline_and_force_are_exposed(self):
        cfg = self.controller.config
        self.assertEqual((cfg.mass_kg, cfg.collider_width, cfg.collider_height),
                         (49.28, 1.6, 5.0))
        self.assertEqual((cfg.move_force, cfg.max_speed, cfg.jump_speed),
                         (4000.0, 25.0, 26.0))
        self.controller.set_action(StandingAction(move_x=1))
        command = self.controller.pre_physics_tick(velocity_x=0, velocity_y=0,
                                                    grounded=True)
        self.assertEqual(command.horizontal_force, 4000.0)
        self.assertEqual(command.gravity_scale, 1.0)

    def test_grounded_speed_cap_and_neutral_slowdown(self):
        self.controller.set_action(StandingAction(move_x=1))
        cap = self.controller.pre_physics_tick(velocity_x=30, velocity_y=0,
                                                grounded=True)
        self.assertEqual(cap.horizontal_velocity_override, 25.0)
        self.controller.set_action(StandingAction())
        brake = self.controller.pre_physics_tick(velocity_x=10, velocity_y=0,
                                                  grounded=True)
        self.assertAlmostEqual(brake.horizontal_velocity_override, 8.0)

    def test_slippery_ground_reduces_force_and_disables_slowdown(self):
        self.controller.set_action(StandingAction(move_x=1))
        command = self.controller.pre_physics_tick(velocity_x=-10, velocity_y=0,
                                                    grounded=True, slippery=True)
        self.assertAlmostEqual(command.horizontal_force, 800.0)
        self.assertIsNone(command.horizontal_velocity_override)

    def test_jump_press_release_gravity_and_coyote_window(self):
        self.controller.set_action(StandingAction(jump=True))
        jump = self.controller.pre_physics_tick(velocity_x=0, velocity_y=0,
                                                 grounded=True)
        self.assertTrue(jump.jump_started)
        self.assertEqual(jump.vertical_velocity_override, 26.0)
        self.controller.set_action(StandingAction(jump=False))
        released = self.controller.pre_physics_tick(velocity_x=0, velocity_y=10,
                                                     grounded=False)
        self.assertEqual(released.gravity_scale, 2.5)
        coyote = StandingPlayerController()
        coyote.state.time_since_grounded = 0.10
        coyote.set_action(StandingAction(jump=True))
        self.assertTrue(coyote.pre_physics_tick(velocity_x=0, velocity_y=-1,
                                                grounded=False).jump_started)

    def test_crouch_collision_matches_half_height_target_and_offset(self):
        self.controller.set_action(StandingAction(crouch=True))
        command = self.controller.pre_physics_tick(velocity_x=0, velocity_y=0,
                                                    grounded=True, dt=1 / 6)
        self.assertEqual(command.collider_height, 2.5)
        self.assertEqual(command.collider_vertical_offset, -1.25)

    def test_crouch_smoothly_interpolates_and_can_recover(self):
        self.controller.set_action(StandingAction(crouch=True))
        half = self.controller.pre_physics_tick(velocity_x=0, velocity_y=0,
                                                grounded=True, dt=0.02)
        self.assertAlmostEqual(half.collider_height, 5.0 * (1 - 0.5 * 0.12))
        self.controller.set_action(StandingAction(crouch=False))
        recovered = self.controller.pre_physics_tick(velocity_x=0, velocity_y=0,
                                                      grounded=True, dt=0.02)
        self.assertGreater(recovered.collider_height, half.collider_height)
        self.assertEqual(recovered.collider_vertical_offset,
                         -5.0 * 0.25 * (0.12 * (1 - 0.12)))

    def test_invalid_actions_and_physical_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            StandingAction(move_x=2)
        with self.assertRaises(ValueError):
            StandingPhysicsConfig(mass_kg=0)
        with self.assertRaises(ValueError):
            self.controller.pre_physics_tick(velocity_x=0, velocity_y=0,
                                             grounded=False, dt=0)


if __name__ == "__main__":
    unittest.main()
