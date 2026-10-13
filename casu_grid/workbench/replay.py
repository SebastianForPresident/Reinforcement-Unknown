"""Replay a recorded training run's trajectories and check them against the log.

    "$ISAAC_SIM/python.sh" -m casu_grid.workbench.replay \
        --config casu_grid/workbench/engineering_smoke.py --run-dir experiments/<run>

Reads ``trajectories.jsonl`` (written by the trainer), rebuilds the same canonical
environment from ``config.json``, feeds the recorded per-step actions back in, mirrors
the trainer's reset rules, and writes under ``<run-dir>/replay/``:

* ``replay.csv``   one row per (step, world): pose, velocity, reward, termination reason;
* ``paths.png``    x/y path of every world (episode boundaries marked);
* ``summary.json`` determinism check (max deviation of replayed vs. logged pose/velocity).
"""
from __future__ import annotations
import argparse, csv, importlib.util, json, sys
import hashlib
import os
from pathlib import Path

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--config', required=True, help='the Python config file the run was trained with')
p.add_argument('--run-dir', type=Path, required=True)
p.add_argument('--max-steps', type=int, default=None, help='replay only the first N logged steps')
p.add_argument('--tolerance', type=float, default=1e-4, help='allowed |replayed - logged| for the determinism check')
AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); args.headless = True
launcher = AppLauncher(args)
app = launcher.app
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from casu_grid.isaac_navigation_env import IsaacLayer1EnvConfig, IsaacLayer1NavigationEnv


def main():
    spec = importlib.util.spec_from_file_location('research_config', args.config)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    run = json.loads((args.run_dir / 'config.json').read_text())
    component_snapshot=args.run_dir/'research_component.json'
    if component_snapshot.is_file():
        module.REWARD_COMPONENT_REF=json.loads(component_snapshot.read_text())
    current_config_sha256 = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    environment = run['environment']
    count = int(environment['num_envs'])
    recorded = {}
    environment_path = args.run_dir / 'environment.json'
    if environment_path.is_file():
        recorded = json.loads(environment_path.read_text()).get('execution', {})
    recorded_layout = recorded.get('worlds_per_worker')
    requested_workers = os.environ.get('CASU_NUM_WORKERS')
    if requested_workers is not None:
        normalized = requested_workers.strip().lower()
        if normalized != 'auto' and normalized.isdigit():
            requested_count = max(1, min(int(normalized), count))
            if recorded_layout and requested_count != len(recorded_layout):
                raise ValueError(f"Replay worker-layout mismatch: run used {len(recorded_layout)} workers with worlds_per_worker={recorded_layout}; CASU_NUM_WORKERS requests {requested_count}. Set CASU_NUM_WORKERS to the recorded worker count for layout-matched replay.")
        elif normalized == 'auto' and recorded_layout:
            from casu_grid.isaac_sharded_env import resolve_worker_count, shard_sizes
            replay_config = IsaacLayer1EnvConfig(num_envs=count, num_workers=0)
            replay_layout = shard_sizes(count, resolve_worker_count(replay_config))
            if replay_layout != recorded_layout:
                raise ValueError(f"Replay worker-layout mismatch: run used worlds_per_worker={recorded_layout}; CASU_NUM_WORKERS=auto resolves to {replay_layout}. Set CASU_NUM_WORKERS to the recorded worker count for layout-matched replay.")
    elif recorded_layout and len(recorded_layout) > 1:
            os.environ['CASU_NUM_WORKERS'] = str(len(recorded_layout))
    kwargs = {'episode_length_steps': environment['episode_steps']} if environment.get('episode_steps') else {}
    cfg = IsaacLayer1EnvConfig(seed=391728405, num_envs=count, spawn_mode='canonical',
                               geometry_scope=environment['geometry_scope'], **kwargs)
    callbacks = module.callbacks() if hasattr(module, 'callbacks') else (None, None, None)
    from casu_grid.workbench.components import resolve_reward_reference
    callbacks = resolve_reward_reference(module, Path(args.config), callbacks)
    if count == 1:
        env = IsaacLayer1NavigationEnv(cfg, action_decoder=callbacks[0], observation_encoder=callbacks[1],
                                       reward_function=callbacks[2])
    else:
        from casu_grid.isaac_vector_env import BatchedIsaacLayer1Env
        env = BatchedIsaacLayer1Env(cfg, action_decoder=callbacks[0], observation_encoder=callbacks[1],
                                    reward_function=callbacks[2], terrain_mode=environment['terrain_mode'])
    out_dir = args.run_dir / 'replay'
    out_dir.mkdir(exist_ok=True)
    worst = 0.0
    worst_detail = None
    first_divergence = None
    steps = episodes_ended = 0
    paths: list[list[list[tuple[float, float]]]] = [[[]] for _ in range(count)]
    try:
        with (args.run_dir / 'trajectories.jsonl').open() as log, \
                (out_dir / 'replay.csv').open('w', newline='') as sink:
            writer = csv.writer(sink)
            writer.writerow(['global_step', 'world', 'actions_json', 'x', 'y', 'vx', 'vy',
                             'replayed_reward', 'logged_reward', 'terminated', 'truncated',
                             'termination_reason'])
            for line in log:
                row = json.loads(line)
                if row['schema'] == 'casu.trajectory-reset.v1':
                    env.reset()
                    paths = [[[]] for _ in range(count)]
                    continue
                if args.max_steps is not None and steps >= args.max_steps:
                    break
                action = {name: np.asarray(values) for name, values in row['actions'].items()}
                _, reward, terminated, truncated, info = env.step(action)
                players = info['player'] if count > 1 else [info['player']]
                logged = row['info']['player'] if count > 1 else [row['info']['player']]
                reasons = info.get('termination_reason')
                reasons = list(reasons) if isinstance(reasons, (list, tuple, np.ndarray)) else [reasons] * count
                reward = np.asarray(reward).reshape(count)
                done = np.logical_or(np.asarray(terminated).reshape(count), np.asarray(truncated).reshape(count))
                for world in range(count):
                    now, then = players[world], logged[world]
                    errors = {key: abs(float(now[key]) - float(then[key]))
                              for key in ('x', 'y', 'vx', 'vy')}
                    deviation = max(errors.values())
                    if deviation > worst:
                        worst = deviation
                        worst_detail = {"global_step": row["global_step"], "world": world,
                                        "field_errors": errors, "max_abs_error": deviation}
                    if first_divergence is None and deviation > args.tolerance:
                        first_divergence = {"global_step": row["global_step"], "world": world,
                                            "field_errors": errors, "max_abs_error": deviation,
                                            "replayed": {key: float(now[key]) for key in errors},
                                            "recorded": {key: float(then[key]) for key in errors}}
                    action_row = {key: np.asarray(value)[world].tolist() for key, value in action.items()}
                    logged_reward = row.get('reward', [None] * count)
                    logged_reward = logged_reward[world] if isinstance(logged_reward, list) else logged_reward
                    writer.writerow([row['global_step'], world, json.dumps(action_row, sort_keys=True),
                                     now['x'], now['y'], now['vx'], now['vy'],
                                     float(reward[world]), logged_reward, bool(np.asarray(terminated).reshape(count)[world]),
                                     bool(np.asarray(truncated).reshape(count)[world]), reasons[world] or ''])
                    paths[world][-1].append((float(now['x']), float(now['y'])))
                    if done[world]:
                        paths[world].append([])
                        episodes_ended += 1
                if done.any():
                    if hasattr(env, 'reset_done'):
                        env.reset_done(done)
                    else:
                        env.reset()
                steps += 1
        replay_performance = env.performance_info() if callable(getattr(env, 'performance_info', None)) else None
        replay_layout = replay_performance.get('worlds_per_worker') if replay_performance else [count]
        layout_matches = not recorded_layout or replay_layout == recorded_layout
        summary = {'schema': 'casu.replay-summary.v1', 'replayed_steps': steps, 'num_envs': count,
                   'recorded_worlds_per_worker': recorded_layout, 'replay_worlds_per_worker': replay_layout,
                   'layout_matches_recorded': layout_matches,
                   'episodes_ended': episodes_ended, 'max_abs_pose_velocity_deviation': worst,
                   'worst_deviation': worst_detail, 'first_divergence': first_divergence,
                   'run_source_revision': run.get('git_revision'),
                   'run_config_sha256': run.get('config_sha256'),
                   'replay_config_sha256': current_config_sha256,
                   'config_matches_recorded': not run.get('config_sha256') or current_config_sha256 == run.get('config_sha256'),
                   'tolerance': args.tolerance,
                   'deterministic': bool(worst <= args.tolerance and layout_matches)}
        (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            for world, episodes in enumerate(paths):
                for points in episodes:
                    if len(points) > 1:
                        xs, ys = zip(*points)
                        ax.plot(xs, ys, marker='.', linewidth=1, label=f'world {world}' if points is episodes[0] else None)
            ax.set_xlabel('x (world units)'); ax.set_ylabel('y (world units)'); ax.set_aspect('equal', 'datalim')
            ax.legend(loc='best', fontsize=7); ax.grid(alpha=0.3)
            fig.savefig(out_dir / 'paths.png', dpi=120, bbox_inches='tight')
        except Exception as error:   # plotting is a convenience; the CSV is the record
            print('REPLAY_PLOT_SKIPPED', error, flush=True)
        print('REPLAY_SUMMARY', json.dumps(summary), flush=True)
    finally:
        env.close()


try:
    main()
finally:
    app.close()
