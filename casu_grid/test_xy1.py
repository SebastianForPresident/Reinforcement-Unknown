"""Focused non-Kit tests for XY1 geometry, action, reward and observation contracts."""
import unittest
import numpy as np
from casu_grid.xy1 import SPATIAL_SHAPE, BestDistanceReward, within_arrival_tiles
from casu_grid.xy1_callbacks import action_decoder
from casu_grid.player_controller import StandingAction

class XY1ContractTests(unittest.TestCase):
    def test_relative_goal_sign_and_window_dimensions(self):
        self.assertEqual(SPATIAL_SHAPE,(16,85,49))
        # Positive dx is right; positive dy is up in world coordinates.
        player=(8,20);goal=(13,17)
        self.assertEqual((goal[0]-player[0],goal[1]-player[1]),(5,-3))

    def test_observation_channel_markers_and_goal_offsets(self):
        from casu_grid.xy1 import encode_xy1_observation
        class Grid:
            width=100;height=100
            blocks=np.zeros((100,100),dtype=np.uint16);fluids=np.zeros((100,100),dtype=np.uint8)
            @staticmethod
            def world_to_cell(x,y,w,h):return (50,50)
        class S: current_action=type('A',(),{'look_dx':2,'look_dy':-1})();x=0.;y=0.
        class Slot:
            grid=Grid();model=type('M',(),{'s':S()})();goal_cell=(55,47)
        class Env:slot=Slot();grid=Slot.grid;model=Slot.model
        obs=encode_xy1_observation(Env())
        self.assertEqual(obs['spatial'].shape,SPATIAL_SHAPE)
        self.assertEqual(np.argwhere(obs['spatial'][14]).tolist(),[[42,24]])
        self.assertEqual(np.argwhere(obs['spatial'][15]).tolist(),[[44,23]])
        self.assertEqual(obs['goal_relative'].tolist(),[5,-3])

    def test_uniform_goal_sampler_covers_full_window_without_rejection(self):
        rng=np.random.default_rng(12)
        dx=rng.integers(-42,43,10000);dy=rng.integers(-24,25,10000)
        self.assertEqual((dx.min(),dx.max(),dy.min(),dy.max()),(-42,42,-24,24))
        self.assertLess(abs(float((dx==0).mean())-1/85),.02)

    def test_spawn_clearance_uses_complete_body_box(self):
        class World:
            def __init__(self):self.calls=[]
            def overlap_box(self,center,size):self.calls.append((center,size));return False
        class E:pass
        from casu_grid.isaac_navigation_env import IsaacLayer1NavigationEnv
        # Invoke unbound helper with minimal shape objects.
        e=E();e.controller_config=type('C',(),{'orig_col_size':(1.6,5.0)})();e.cfg=type('Cfg',(),{'spawn_clearance_attempts':2})();e.grid=type('G',(),{'width':20,'height':20})();e.slot=type('S',(),{'ground_world':World()})();
        spawn=IsaacLayer1NavigationEnv._sample_clear_spawn(e,np.random.default_rng(1))
        self.assertEqual(len(e.slot.ground_world.calls),1)
        self.assertEqual(e.slot.ground_world.calls[0][1],(1.58,4.98))
        self.assertEqual(len(spawn),2)

    def test_action_decoder_maps_all_seven_branches(self):
        action=action_decoder({'move_x':np.array([0]),'jump':np.array([1]),'move_y':np.array([2]),
            'crouch':np.array([1]),'look_dx':np.array([8]),'look_dy':np.array([0]),'attack':np.array([1])})
        self.assertEqual(action,StandingAction(-1,True,True,1,True,4,-5))

    def test_best_distance_reward_has_no_backtracking_reward_and_resets(self):
        r=BestDistanceReward(10)
        value,terms=r((9,0),(0,0));self.assertAlmostEqual(value,.1);self.assertAlmostEqual(terms['distance_improvement'],.1)
        self.assertEqual(r((10,0),(0,0))[0],0)
        value,terms=r((0,0),(0,0),arrived=True);self.assertAlmostEqual(value,10.9)
        self.assertEqual(terms['arrival_bonus'],10)
        r.reset((5,0),(0,0));self.assertEqual(r.best_distance,5)

    def test_arrival_is_nine_tiles(self):
        self.assertTrue(within_arrival_tiles((11,9),(10,10)))
        self.assertFalse(within_arrival_tiles((12,10),(10,10)))

    def test_terrain_damage_uses_existing_health_and_destroy_transition(self):
        from casu_grid.state import CasUGridState
        grid=CasUGridState(width=8,height=8)
        grid.set_block(3,4,2) # source health = 25
        self.assertFalse(grid.damage_block(3,4,12.5,ignore_loot=True))
        self.assertEqual(grid.get_block(3,4),2)
        self.assertTrue(grid.damage_block(3,4,12.5,ignore_loot=True))
        self.assertEqual(grid.get_block(3,4),0)

if __name__=='__main__':unittest.main()
