"""Build bounded per-tile collision fixtures and A/B query them in Isaac."""
import argparse
import json
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--source-profile", type=Path, required=True)
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
launcher = AppLauncher(args)
app = launcher.app

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaaclab.sim as sim_utils
import isaacsim.core.utils.stage as stage_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from casu_grid.collision import fixture_set, fixture_specs

def main():
    print("ISAAC_COLLISION_FIXTURE_START", flush=True)
    source = json.loads(args.source_profile.read_text())
    source_types = {row["id"]: row["sample"]
                    for row in source["tile_types"]["block_types"]}
    fixtures = {fixture.name: fixture for fixture in fixture_set()}
    fixture_cells = {
        1: fixtures["isolated_rock"].cells[0],
        16: fixtures["adjacent_soil"].cells[0],
        7: fixtures["glass_tile"].cells[0],
    }
    measured_ids = (1, 16, 7)  # Rock, Soil and Glass source asset types.
    for block_id in measured_ids:
        if block_id not in source_types or block_id not in fixture_cells:
            raise RuntimeError(f"source profile or fixture lacks block ID {block_id}")
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device="cpu",
                                      gravity=(0.0, 0.0, -55.0))
    sim = sim_utils.SimulationContext(sim_cfg)
    print("ISAAC_COLLISION_SIM_READY", flush=True)
    try:
        cell_specs = [spec for fixture in fixture_set()
                      for spec in fixture_specs(fixture)]
        for spec in cell_specs:
            cfg = sim_utils.CuboidCfg(
                size=tuple(spec["size_xyz"]),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True))
            cfg.func(spec["prim_path"], cfg,
                     translation=tuple(spec["isaac_xyz"]))
        probes = {}
        for block_id, cell in fixture_cells.items():
            center_x = cell.x - 1024 / 2 + 0.5
            center_z = cell.y - 1024 / 2 + 0.5
            body_cfg = sim_utils.CuboidCfg(
                size=(0.25, 0.2, 0.25),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=False, disable_gravity=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True))
            probe_cfg = RigidObjectCfg(
                prim_path=f"/World/PassiveProbes/Probe_{block_id}",
                spawn=body_cfg,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(center_x, 0.0, center_z + 2.0)))
            probes[block_id] = RigidObject(cfg=probe_cfg)
        print("ISAAC_COLLISION_GEOMETRY_BUILT", len(cell_specs), flush=True)
        stage_utils.update_stage()
        sim.reset()
        print("ISAAC_COLLISION_SIM_RESET", sim.is_playing(), flush=True)
        sim.play()
        print("ISAAC_COLLISION_SIM_PLAY", sim.is_playing(), flush=True)
        comparisons = []
        traces = {block_id: [] for block_id in measured_ids}
        for tick in range(60):
            sim.step(render=False)
            for block_id, probe in probes.items():
                probe.update(sim.get_physics_dt())
                if tick < 12 or tick % 4 == 3:
                    traces[block_id].append({
                        "tick": tick + 1,
                        "root_xyz": probe.data.root_pos_w[0].tolist(),
                    })
        for block_id in measured_ids:
            sample = source_types[block_id]
            cell = fixture_cells[block_id]
            center_x = cell.x - 1024 / 2 + 0.5
            center_z = cell.y - 1024 / 2 + 0.5
            expected = sample.get("layer6_overlap_pattern_0p04_cell_spacing_from_minus0p52")
            if expected is None:
                expected = sample["layer6_overlap_pattern_0p04_cell_spacing"]
            probe = probes[block_id]
            rest = probe.data.root_pos_w[0].tolist()
            comparisons.append({
                "block_id": block_id, "tile_name": sample["name"],
                "tile_collider_type": sample["asset_collider_type"],
                "grid_xy": [cell.x, cell.y], "source_world_center_xy": [center_x, center_z],
                "isaac_plane": "X/Z with Y extrusion", "samples": len(expected) ** 2,
                "source_center_sample_is_solid": expected[len(expected)//2][len(expected)//2] == "#",
                "probe_size_xyz": [0.25, 0.2, 0.25], "probe_mass_kg": 1.0,
                "probe_rest_root_xyz_after_60_ticks": rest,
                "probe_trace": traces[block_id],
            })
            print("ISAAC_COLLISION_PROBE_DONE", block_id, rest, flush=True)
        result = {
            "schema": "casu.isaac-collision-fixtures.v1",
            "fixture_count": len(fixture_set()), "static_cell_collider_count": len(cell_specs),
            "representation": "one static unit cuboid per non-air tile; separate chunk prim parents",
            "time_step_s": sim_cfg.dt, "gravity_xyz": [0.0, 0.0, -55.0],
            "material": "unassigned/default on both source collider and Isaac fixtures",
            "source_profile": str(args.source_profile),
            "source_to_isaac_probe_comparisons": comparisons,
            "geometry_result": "fixtures built; dynamic probes completed",
        }
        output = args.output or Path(__file__).resolve().parent / "results" / "mission11_isaac_collision_trace.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print("ISAAC_COLLISION_FIXTURE_OK", output, len(cell_specs), flush=True)
    finally:
        sim.clear_all_callbacks()
        sim.clear_instance()

try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    app.close()
