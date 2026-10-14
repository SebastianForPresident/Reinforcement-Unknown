"""Headless research-config training entry point for canonical Layer 1."""
from __future__ import annotations
import argparse, importlib.util, json, subprocess, time, traceback
import os
import hashlib
from pathlib import Path
import sys
import numpy as np
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser()
p.add_argument('--config', required=True, help='Python file defining ACTION, OBSERVATION, MODEL, PPO and callbacks()')
p.add_argument('--run-dir', type=Path, default=None)
p.add_argument('--updates', type=int, default=100)
p.add_argument('--num-envs', type=int, default=1)
p.add_argument('--seed', type=int, default=391728405)
p.add_argument('--geometry-scope', choices=('full','spawn_chunk'), default='full')
p.add_argument('--terrain-mode', choices=('shared','independent'), default='shared',
               help='multi-env only: shared static terrain (fast) or per-world terrain copies (allows terrain edits)')
p.add_argument('--episode-steps', type=int, default=None,
               help='truncate episodes after this many policy decisions (default: environment default, 500)')
p.add_argument('--resume', type=Path)
p.add_argument('--checkpoint-every', type=int, default=0,
               help='also keep a numbered copy checkpoints/update_NNNNNN.pt every N updates (latest.pt is always written after every update)')
p.add_argument('--save-initial-checkpoint', action='store_true',
               help='save the untrained policy as checkpoints/initial.pt before the first update (fresh runs only)')
p.add_argument('--train-device', default=os.environ.get('CASU_TRAIN_DEVICE','auto'),
               help="device of the PPO model, rollout buffers and update: 'cpu', 'cuda' or 'auto' (CUDA when available); env CASU_TRAIN_DEVICE")
AppLauncher.add_app_launcher_args(p); args=p.parse_args(); args.headless=True
launcher=AppLauncher(args); app=launcher.app
print(f"CASU_TRAIN_ENTRY config={args.config} run_dir={args.run_dir} updates={args.updates}",flush=True)
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from casu_grid.isaac_navigation_env import IsaacLayer1EnvConfig, IsaacLayer1NavigationEnv
from casu_grid.workbench.ppo import PPOTrainer, write_metrics
from casu_grid.workbench.contracts import validate_contracts
from casu_grid.workbench.runtime_validation import checked_callbacks
from gymnasium import spaces

active_run_dir: Path | None = None

def main():
    global active_run_dir
    print("CASU_TRAIN_MAIN",flush=True)
    spec=importlib.util.spec_from_file_location('research_config',args.config)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    print("CASU_TRAIN_CONFIG_LOADED",flush=True)
    config=module
    if args.resume:
        component_manifest=args.resume.expanduser().resolve().parent/'research_component.json'
        if component_manifest.is_file():
            component_reference=json.loads(component_manifest.read_text())
            if component_reference.get('snapshot_source'):
                component_reference['snapshot_source']=str(args.resume.expanduser().resolve().parent/component_reference['snapshot_source'])
            config.REWARD_COMPONENT_REF=component_reference
    validate_contracts(config.ACTION,config.OBSERVATION,config.REWARD,config.MODEL,config.PPO)
    config_source=Path(args.config).resolve().read_bytes()
    print("CASU_TRAIN_CONTRACTS_VALID",flush=True)
    if args.num_envs <= 0:
        raise ValueError('--num-envs must be positive')
    if args.updates <= 0:
        raise ValueError('--updates must be positive')
    if args.geometry_scope == 'spawn_chunk':
        raise ValueError('spawn_chunk is a diagnostic-only geometry scope; PPO training requires full canonical terrain')
    if os.environ.get('CASU_LEGACY_MINI_SPAWN','0')=='1' and os.environ.get('CASU_REPRODUCE_LEGACY','0')!='1':
        raise ValueError('CASU_LEGACY_MINI_SPAWN=1 restores the pre-Optimization-IV mis-placed mini-world workload and is only for explicit '
                         'historical reproduction; set CASU_REPRODUCE_LEGACY=1 as well to acknowledge that, or unset it for real training')
    run_dir=args.run_dir or Path('experiments')/time.strftime('%Y%m%d-%H%M%S')
    active_run_dir = run_dir
    # The researcher CLI reserves the directory and writes a RUNNING marker;
    # direct legacy invocations retain the original create-if-absent behavior.
    run_dir.mkdir(parents=True,exist_ok=os.environ.get('CASU_WORKBENCH_LAUNCHER_PID') is not None)
    print(f"CASU_TRAIN_RUN_DIR {run_dir}",flush=True)
    source=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip() or None
    working_tree=subprocess.run(['git','status','--porcelain'],capture_output=True,text=True).stdout.splitlines()
    (run_dir/'config.json').write_text(json.dumps({
        'schema':'casu.training-run.v1','run_id':run_dir.name,
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'seed':args.seed,'git_revision':source,
        'working_tree_status':working_tree,
        'config_path':str(Path(args.config).resolve()),
        'experiment':getattr(config,'EXPERIMENT',{'name':Path(args.config).stem,
            'description':'','hypothesis':'','notes':'','parent':None}),
        'config_sha256':hashlib.sha256(config_source).hexdigest(),
        'config_snapshot_path':str((run_dir/'research_config.py').resolve()),
        'contracts':validate_contracts(config.ACTION,config.OBSERVATION,config.REWARD,config.MODEL,config.PPO),
        'environment':{'canonical_seed':391728405,'num_envs':args.num_envs,'geometry_scope':args.geometry_scope,
                       'terrain_mode':args.terrain_mode if args.num_envs>1 else 'single',
                       'episode_steps':args.episode_steps or getattr(config,'ENVIRONMENT',{}).get('episode_length_steps'),
                       'navigation_mode':bool(getattr(config,'ENVIRONMENT',{}).get('navigation_mode',False)),
                       'world_variant':getattr(config,'ENVIRONMENT',{}).get('world_variant','canonical'),
                       'player_collider':getattr(config,'ENVIRONMENT',{}).get('player_collider','compound'),
                       'physics_dt':0.02,'action_decimation':10},
        'resume_from':os.environ.get('CASU_WORKBENCH_RESUME_FROM') or None,
        'run_kind':os.environ.get('CASU_WORKBENCH_RUN_KIND') or 'direct_training',
        'config_source_sha256':hashlib.sha256(config_source).hexdigest()},indent=2)+'\n')
    (run_dir/'research_config.py').write_bytes(config_source)
    reward_reference=getattr(config,'REWARD_COMPONENT_REF',None)
    if reward_reference:
        from casu_grid.workbench.components import snapshot_reward_reference
        reward_reference=snapshot_reward_reference(reward_reference,Path(args.config),run_dir)
        (run_dir/'research_component.json').write_text(json.dumps(reward_reference,indent=2)+'\n')
        runtime_reference=dict(reward_reference)
        runtime_reference['snapshot_source']=str(run_dir/reward_reference['snapshot_source'])
        config.REWARD_COMPONENT_REF=runtime_reference
    callbacks=config.callbacks() if hasattr(config,'callbacks') else (None,None,None)
    from casu_grid.workbench.components import resolve_reward_reference
    callbacks=resolve_reward_reference(config,Path(args.config),callbacks)
    callbacks=checked_callbacks(callbacks,config.OBSERVATION,config.REWARD)
    env=trainer=None
    env_init_started=time.perf_counter()
    try:
        env_config=IsaacLayer1EnvConfig(
            seed=391728405,num_envs=args.num_envs,spawn_mode='canonical',geometry_scope=args.geometry_scope,
            navigation_mode=bool(getattr(config,'ENVIRONMENT',{}).get('navigation_mode',False)),
            world_variant=getattr(config,'ENVIRONMENT',{}).get('world_variant','canonical'),
            observation_window_tiles=tuple(getattr(config,'ENVIRONMENT',{}).get('observation_window_tiles',(85,49))),
            player_collider=getattr(config,'ENVIRONMENT',{}).get('player_collider','compound'),
            **({'episode_length_steps':args.episode_steps} if args.episode_steps else
               ({'episode_length_steps':int(config.ENVIRONMENT['episode_length_steps'])} if getattr(config,'ENVIRONMENT',{}).get('episode_length_steps') else {})))
        if args.num_envs == 1:
            from casu_grid.isaac_vector_env import BatchedIsaacLayer1Env
            env=BatchedIsaacLayer1Env(env_config,
                action_decoder=callbacks[0],observation_encoder=callbacks[1],reward_function=callbacks[2],
                terrain_mode=args.terrain_mode)
        else:
            from casu_grid.isaac_vector_env import BatchedIsaacLayer1Env
            env=BatchedIsaacLayer1Env(env_config,
                action_decoder=callbacks[0],observation_encoder=callbacks[1],reward_function=callbacks[2],
                terrain_mode=args.terrain_mode)
        env.action_space=spaces.Dict({
            head.name: (spaces.Discrete(head.size) if head.kind=='categorical'
                        else spaces.Box(head.low,head.high,shape=(head.size,),dtype=np.float32))
            for head in config.ACTION.heads})
        env.observation_space=spaces.Dict({
            field.name: spaces.Box(-np.inf,np.inf,shape=field.shape,dtype=np.float32)
            for field in config.OBSERVATION.spatial + config.OBSERVATION.scalar})
        if any(head.hold_physics_ticks != 10 for head in config.ACTION.heads):
            raise ValueError('this environment pins every action head to CasU’s ten-physics-tick cadence')
        import torch
        train_device='cuda' if (args.train_device=='auto' and torch.cuda.is_available()) else ('cpu' if args.train_device=='auto' else args.train_device)
        trainer=PPOTrainer(env,config.ACTION,config.OBSERVATION,config.MODEL,config.PPO,
                           device=train_device,seed=args.seed,log_dir=run_dir/'tensorboard')
        if args.resume: trainer.load(args.resume)
        performance = env.performance_info() if callable(getattr(env, 'performance_info', None)) else {
            'backend': {'requested': os.environ.get('CASU_BACKEND', 'auto'), 'active': 'unknown'},
            'num_workers': 1, 'worlds': args.num_envs, 'worlds_per_worker': [args.num_envs]}
        world_variant=getattr(config,'ENVIRONMENT',{}).get('world_variant','canonical')
        environment_manifest={
            'performance': performance,
            'execution': {'requested_backend': os.environ.get('CASU_BACKEND', 'auto'),
                          'effective_backend': performance.get('backend'),
                          'requested_workers': os.environ.get('CASU_NUM_WORKERS', '1'),
                          'effective_workers': performance.get('num_workers'),
                          'num_envs': args.num_envs,
                          'worlds_per_worker': performance.get('worlds_per_worker', [args.num_envs])},
            'world_variant':world_variant,
            'world_fixture_id':env.reference.manifest['fixture_id'],
            'world_content_id':env.reference.manifest['content_id'],
            'env_startup_seconds':time.perf_counter()-env_init_started,
            'terrain_box_colliders_total':sum(int(g.get('box_collider_count',0)) for g in env.geometry_by_env),
            'terrain_occupied_cells_per_world':[int(g.get('occupied_cells',0)) for g in env.geometry_by_env],
            'geometry':env.geometry_by_env if hasattr(env,'geometry_by_env') else env.geometry_info,
            'geometry_build_seconds':env.geometry_build_seconds,
            'physics_dt':env.cfg.physics_dt,
            'action_decimation':env.cfg.action_decimation,
            'fluid_rng_seed':env.cfg.fluid_rng_seed,
            'liquid_affect_phase':env.cfg.liquid_affect_phase,
        }
        if world_variant == 'canonical':
            # Keep the established metadata keys for downstream canonical-run tools.
            environment_manifest['canonical_fixture_id']=env.reference.manifest['fixture_id']
            environment_manifest['canonical_content_id']=env.reference.manifest['content_id']
        (run_dir/'environment.json').write_text(json.dumps(environment_manifest,indent=2)+'\n')
        checkpoint_metadata={'source_revision':source,'run_dir':str(run_dir)}
        if args.save_initial_checkpoint and not args.resume:
            trainer.save(run_dir/'checkpoints'/'initial.pt',metadata=dict(checkpoint_metadata,kind='initial_untrained_policy'))
        stop_requested=False
        for _ in range(args.updates):
            if (run_dir/'STOP').exists():      # graceful stop requested by the Studio: finish at an update boundary
                stop_requested=True
                break
            metrics=trainer.train_update()
            write_metrics(run_dir/'metrics.jsonl',metrics)
            trainer.save(run_dir/'latest.pt',metadata=checkpoint_metadata)
            if args.checkpoint_every and trainer.update_count % args.checkpoint_every == 0:
                trainer.save(run_dir/'checkpoints'/f'update_{trainer.update_count:06d}.pt',metadata=dict(checkpoint_metadata,kind='scheduled'))
            print(json.dumps(metrics,sort_keys=True),flush=True)
        if stop_requested:
            (run_dir/'INTERRUPTED').write_text(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+' graceful stop requested\n')
            (run_dir/'COMPLETE').unlink(missing_ok=True)
            print('CASU_TRAIN_STOPPED_BY_REQUEST update=%d'%trainer.update_count,flush=True)
        else:
            (run_dir/'COMPLETE').write_text(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+'\n')
            (run_dir/'failure.json').unlink(missing_ok=True)
            (run_dir/'INTERRUPTED').unlink(missing_ok=True)
    finally:
        pending_exception = sys.exc_info()[0] is not None
        for close in ((trainer.close if trainer is not None else None),
                      (env.close if env is not None else None)):
            if close is None:
                continue
            try:
                close()
            except BaseException:
                if not pending_exception:
                    raise
                traceback.print_exc()
        if trainer is not None:
            trainer.env = None

try:
    main()
except KeyboardInterrupt:
    run_dir = active_run_dir
    if run_dir is not None:
        try:
            (run_dir/'INTERRUPTED').write_text(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+'\n')
        except OSError:
            pass
    traceback.print_exc()
    raise SystemExit(130)
except BaseException as exc:
    traceback.print_exc()
    run_dir = active_run_dir
    if run_dir is not None:
        try:
            (run_dir/'failure.json').write_text(json.dumps({
                'status':'failed','error_type':type(exc).__name__,'message':str(exc),
                'finished_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                'traceback':traceback.format_exc()},indent=2)+'\n')
            (run_dir/'COMPLETE').unlink(missing_ok=True)
        except OSError:
            traceback.print_exc()
    raise SystemExit(1)
finally:
    try:
        app.close()
    except BaseException:
        traceback.print_exc()
