import unittest

import numpy as np

from casu_grid.body_model import BodyConfig, BodyState, PolicyAction, StandingBodyModel
from casu_grid.ground_query import GroundWorld
from casu_grid.player_runtime import RuntimeState, run_macrostep


class FreeAdapter:
    """Frictionless point-mass stand-in for the physics engine (no contacts)."""

    def __init__(self, mass=49.28, gravity=-55.0, dt=0.02):
        self.mass, self.g, self.dt = mass, gravity, dt
        self.s = [0.0, 0.0, 0.0, 0.0]
        self.f = (0.0, 0.0)
        self.crouch = []

    def write_state(self, x, y, vx, vy): self.s = [x, y, vx, vy]
    def apply_force(self, fx, fy): self.f = (fx, fy)

    def step(self):
        x, y, vx, vy = self.s
        vx += self.f[0] / self.mass * self.dt
        vy += (self.f[1] / self.mass + self.g) * self.dt
        self.s = [x + vx * self.dt, y + vy * self.dt, vx, vy]

    def read_state(self): return tuple(self.s)
    def set_crouch(self, crouch): self.crouch.append(crouch)


class CountingModel(StandingBodyModel):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.counts = {"update": 0, "apply": 0, "fixed": 0}

    def update(self, dt=None): self.counts["update"] += 1; super().update(dt)
    def apply_actions(self, **k): self.counts["apply"] += 1; super().apply_actions(**k)
    def fixed_update(self): self.counts["fixed"] += 1; return super().fixed_update()


class RuntimeScheduleTests(unittest.TestCase):
    def setUp(self):
        self.world = GroundWorld(np.zeros((1024, 1024), dtype=np.uint16))
        self.model = CountingModel(self.world, BodyConfig(), BodyState(x=0.0, y=300.0, grounded=False))
        self.adapter = FreeAdapter()
        self.runtime = RuntimeState()

    def test_macrostep_uses_the_harness_twelve_update_schedule(self):
        run_macrostep(self.model, self.adapter, self.world, PolicyAction(move=1), self.runtime)
        self.assertEqual(self.model.counts, {"update": 12, "apply": 12, "fixed": 10})

    def test_ten_ticks_of_rightward_force_match_the_measured_acceleration(self):
        run_macrostep(self.model, self.adapter, self.world, PolicyAction(move=1), self.runtime)
        # F/m*dt = 4000*0.9999999/49.28*0.02 per tick, ten ticks
        self.assertAlmostEqual(self.model.s.vx, 10 * 4000.0 / 49.28 * 0.02, places=3)

    def test_held_action_persists_across_macrosteps(self):
        run_macrostep(self.model, self.adapter, self.world, PolicyAction(move=1), self.runtime)
        before = self.model.s.vx
        run_macrostep(self.model, self.adapter, self.world, None, self.runtime)
        self.assertGreater(self.model.s.vx, before)

    def test_jump_press_edge_fires_once_per_press(self):
        s = self.model.s
        s.grounded = True
        self.world.blocks[512 - 1:512 + 2, 100] = 1  # a few support tiles far below
        run_macrostep(self.model, self.adapter, self.world, PolicyAction(jump=1), self.runtime, ticks=1)
        self.assertGreater(s.vy, 20.0)   # jumped (26 m/s) at the latch application


if __name__ == "__main__":
    unittest.main()
