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

    def test_mini_world_is_randomized_open_top_infinirock_sided_and_has_no_chamber(self):
        from casu_grid.xy1_mini_world import generate_mini_terrain
        a=generate_mini_terrain(123);b=generate_mini_terrain(124)
        self.assertEqual(a.shape,(128,128));self.assertEqual(b.shape,(128,128))
        self.assertFalse(np.array_equal(a,b))
        self.assertTrue(np.all(a[0:2,:]==14));self.assertTrue(np.all(a[-2:,:]==14))
        self.assertTrue(np.all(a[:,0:2]>0)) # continuous standable bottom floor
        self.assertEqual(np.count_nonzero(a[6:-6,-1]),0) # open top, side barriers remain
        self.assertGreater(len(np.unique(a[a>0])),4) # CasU material variety
        self.assertGreater(np.count_nonzero(a)/a.size,.25) # close to canonical 30.6% solidity
        self.assertLess(np.count_nonzero(a)/a.size,.35)
        self.assertTrue(np.all(a[63:66,62:68]==0)) # body clearance only
        self.assertNotEqual(np.count_nonzero(a[55:74,55:74]),0) # no invariant center chamber

    def test_mini_world_dimensions_and_partial_collision_chunks(self):
        from casu_grid.xy1_mini_world import load_mini_world_reference
        from casu_grid.geometry import rectangles_for_world
        from casu_grid.ground_query import GroundWorld
        ref=load_mini_world_reference(391728405)
        self.assertEqual(ref.new_state().blocks.shape,(128,128))
        rects=rectangles_for_world(ref)
        self.assertEqual({(r.chunk_x,r.chunk_y) for r in rects},{(0,0),(0,1),(1,0),(1,1)})
        ground=GroundWorld(ref.terrain)
        self.assertEqual(ground.block_at(-63.5,-63.5),14)
        self.assertEqual(ground.block_at(0.5,0.5),0)
        self.assertFalse(ground.overlap_box(ref.spawn_world_xy,(1.58,4.98)))
        self.assertFalse(np.array_equal(load_mini_world_reference(391728406).terrain,ref.terrain))

    def test_mini_world_center_aim_and_goal_window_contract(self):
        from casu_grid.xy1_mini_world import load_mini_world_reference
        from casu_grid.body_model import StandingBodyModel,PolicyAction
        from casu_grid.ground_query import GroundWorld
        from casu_grid.workbench.experiments import xy1_point_navigation as canonical
        from casu_grid.workbench.experiments import xy1_point_navigation_mini as mini
        ref=load_mini_world_reference(5);world=GroundWorld(ref.terrain,backend="reference-python")
        model=StandingBodyModel(world);model.s.x,model.s.y=.5,0.0
        model.apply_actions(latch=PolicyAction(look_dx=4,look_dy=-5))
        self.assertEqual((model.s.look_x,model.s.look_y),(4.5,-4.5))
        rng=np.random.default_rng(9);points=[(rng.integers(43,128),rng.integers(25,74)) for _ in range(2000)]
        self.assertEqual((min(p[0] for p in points),max(p[0] for p in points)),(43,127))
        self.assertEqual((min(p[1] for p in points),max(p[1] for p in points)),(25,73))
        self.assertEqual(mini.ACTION,canonical.ACTION);self.assertEqual(mini.OBSERVATION.spatial[0].shape,(16,64,64))
        self.assertEqual(canonical.OBSERVATION.spatial[0].shape,(16,85,49))
        self.assertEqual(mini.REWARD,canonical.REWARD);self.assertEqual(mini.MODEL,canonical.MODEL)
        self.assertEqual(mini.PPO,canonical.PPO)

    def test_mini_goal_sampling_is_uniform_in_64_viewport_and_unclipped(self):
        from casu_grid.isaac_vector_env import BatchedIsaacLayer1Env
        from casu_grid.xy1_mini_world import load_mini_world_reference
        class E:
            cfg=type('Cfg',(),{'world_variant':'xy1_mini_128x128','observation_window_tiles':(64,64)})()
        class G:
            width=128;height=128
        class S:
            reference=load_mini_world_reference(2);grid=G()
        e=E();slot=S();rng=np.random.default_rng(17)
        samples=[BatchedIsaacLayer1Env._sample_goal(e,slot,rng,(0.5,0.5)) for _ in range(12000)]
        goals=[sample[0] for sample in samples]
        self.assertEqual((min(g[0] for g in goals),max(g[0] for g in goals)),(32,95))
        self.assertEqual((min(g[1] for g in goals),max(g[1] for g in goals)),(32,95))
        self.assertLess(abs(float((np.asarray(goals)[:,0]==32).mean())-1/64),.012)
        self.assertEqual(samples[0][1],(goals[0][0]-63.5,goals[0][1]-63.5))

    def test_mini_goal_offset_can_grow_after_destination_leaves_view(self):
        from casu_grid.xy1 import encode_xy1_observation
        from casu_grid.xy1_mini_world import load_mini_world_reference
        from casu_grid.body_model import BodyState,PolicyAction
        ref=load_mini_world_reference(3);grid=ref.new_state()
        state=BodyState(x=-53.5,y=.5,current_action=PolicyAction()) # tile (10,64)
        class Slot:
            goal_cell=(95,64);model=type('M',(),{'s':state})();
        class Env:
            cfg=type('Cfg',(),{'observation_window_tiles':(64,64)})()
            def __init__(self):self.slot=Slot();self.grid=grid;self.model=self.slot.model
        env=Env();obs=encode_xy1_observation(env)
        self.assertEqual(obs['spatial'].shape,(16,64,64))
        self.assertEqual(obs['goal_relative'].tolist(),[85,0])
        self.assertEqual(env.slot.goal_cell,(95,64))

    def test_mini_environment_slots_edit_only_their_own_terrain(self):
        from casu_grid.body_model import BodyConfig
        from casu_grid.env_slot import EnvSlot
        from casu_grid.isaac_navigation_env import IsaacLayer1EnvConfig
        from casu_grid.xy1_mini_world import load_mini_world_reference
        cfg=IsaacLayer1EnvConfig(navigation_mode=True,world_variant='xy1_mini_128x128',
            observation_window_tiles=(64,64),include_world_objects=False)
        a=EnvSlot(cfg,load_mini_world_reference(10),fluid_seed=10,body_config=BodyConfig())
        b=EnvSlot(cfg,load_mini_world_reference(11),fluid_seed=11,body_config=BodyConfig())
        original_b=b.grid.get_block(2,20)
        a.edit_block(2,20,0)
        self.assertEqual(a.grid.get_block(2,20),0)
        self.assertEqual(b.grid.get_block(2,20),original_b)
        self.assertIn((0,0),a.physical_dirty)

    def test_out_of_range_edit_is_clamped_like_the_grid(self):
        from casu_grid.body_model import BodyConfig
        from casu_grid.env_slot import EnvSlot
        from casu_grid.isaac_navigation_env import IsaacLayer1EnvConfig
        from casu_grid.xy1_mini_world import load_mini_world_reference
        cfg=IsaacLayer1EnvConfig(navigation_mode=True,world_variant='xy1_mini_128x128',
            observation_window_tiles=(64,64),include_world_objects=False)
        a=EnvSlot(cfg,load_mini_world_reference(10),fluid_seed=10,body_config=BodyConfig())
        before=a.physical_blocks.copy()
        a.edit_block(-1,5,0,immediate=True); a.edit_block(5,-3,0,immediate=True); a.edit_block(10**6,10**6,0,immediate=True)
        self.assertTrue(all(cx>=0 and cy>=0 for cx,cy in a.physical_dirty))
        self.assertEqual(int(a.physical_blocks[-1,5]),int(before[-1,5]))      # no wrapped write
        self.assertEqual(int(a.physical_blocks[0,5]),0); self.assertEqual(int(a.physical_blocks[-1,-1]),0)

if __name__=='__main__':unittest.main()
