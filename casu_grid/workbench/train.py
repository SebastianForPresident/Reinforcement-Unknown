"""Headless research-config training entry point for canonical Layer 1."""
from __future__ import annotations
import argparse, importlib.util, json, subprocess, time, traceback
import hashlib
from dataclasses import replace
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
p.add_argument('--resume', type=Path)
AppLauncher.add_app_launcher_args(p); args=p.parse_args(); args.headless=True
launcher=AppLauncher(args); app=launcher.app
print(f"CASU_TRAIN_ENTRY config={args.config} run_dir={args.run_dir} updates={args.updates}",flush=True)
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from casu_grid.isaac_navigation_env import IsaacLayer1EnvConfig, IsaacLayer1NavigationEnv
from casu_grid.workbench.ppo import PPOTrainer, write_metrics
from casu_grid.workbench.contracts import validate_contracts
from gymnasium import spaces

def main():
    print("CASU_TRAIN_MAIN",flush=True)
    spec=importlib.util.spec_from_file_location('research_config',args.config)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    print("CASU_TRAIN_CONFIG_LOADED",flush=True)
    config=module
    validate_contracts(config.ACTION,config.OBSERVATION,config.REWARD,config.MODEL,config.PPO)
    print("CASU_TRAIN_CONTRACTS_VALID",flush=True)
    if args.num_envs <= 0:
        raise ValueError('--num-envs must be positive')
    if args.updates <= 0:
        raise ValueError('--updates must be positive')
    if args.geometry_scope == 'spawn_chunk':
        raise ValueError('spawn_chunk is a diagnostic-only geometry scope; PPO training requires full canonical terrain')
    run_dir=args.run_dir or Path('experiments')/time.strftime('%Y%m%d-%H%M%S')
    run_dir.mkdir(parents=True,exist_ok=False)
    print(f"CASU_TRAIN_RUN_DIR {run_dir}",flush=True)
    source=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip() or None
    working_tree=subprocess.run(['git','status','--porcelain'],capture_output=True,text=True).stdout.splitlines()
    (run_dir/'config.json').write_text(json.dumps({
        'schema':'casu.training-run.v1','seed':args.seed,'git_revision':source,
        'working_tree_status':working_tree,
        'config_path':str(Path(args.config).resolve()),
        'config_sha256':hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        'contracts':validate_contracts(config.ACTION,config.OBSERVATION,config.REWARD,config.MODEL,config.PPO),
        'environment':{'canonical_seed':391728405,'num_envs':args.num_envs,'geometry_scope':args.geometry_scope,
                       'physics_dt':0.02,'action_decimation':10}},indent=2)+'\n')
    callbacks=config.callbacks() if hasattr(config,'callbacks') else (None,None,None)
    env=trainer=None
    try:
        env_config=IsaacLayer1EnvConfig(
            seed=391728405,num_envs=args.num_envs,spawn_mode='canonical',geometry_scope=args.geometry_scope)
        if args.num_envs == 1:
            env=IsaacLayer1NavigationEnv(env_config,
                action_decoder=callbacks[0],observation_encoder=callbacks[1],reward_function=callbacks[2])
        else:
            from casu_grid.isaac_vector_env import BatchedIsaacLayer1Env
            env=BatchedIsaacLayer1Env(env_config,
                action_decoder=callbacks[0],observation_encoder=callbacks[1],reward_function=callbacks[2])
        env.action_space=spaces.Dict({
            head.name: (spaces.Discrete(head.size) if head.kind=='categorical'
                        else spaces.Box(head.low,head.high,shape=(head.size,),dtype=np.float32))
            for head in config.ACTION.heads})
        env.observation_space=spaces.Dict({
            field.name: spaces.Box(-np.inf,np.inf,shape=field.shape,dtype=np.float32)
            for field in config.OBSERVATION.spatial + config.OBSERVATION.scalar})
        if any(head.hold_physics_ticks != 10 for head in config.ACTION.heads):
            raise ValueError('this environment pins every action head to CasU’s ten-physics-tick cadence')
        trainer=PPOTrainer(env,config.ACTION,config.OBSERVATION,config.MODEL,config.PPO,
                           device='cpu',seed=args.seed,log_dir=run_dir/'tensorboard')
        if args.resume: trainer.load(args.resume)
        environment_manifest={
            'canonical_fixture_id':env.reference.manifest['fixture_id'],
            'canonical_content_id':env.reference.manifest['content_id'],
            'geometry':env.geometry_by_env if hasattr(env,'geometry_by_env') else env.geometry_info,
            'geometry_build_seconds':env.geometry_build_seconds,
            'physics_dt':env.cfg.physics_dt,
            'action_decimation':env.cfg.action_decimation,
            'fluid_rng_seed':env.cfg.fluid_rng_seed,
            'liquid_affect_phase':env.cfg.liquid_affect_phase,
        }
        (run_dir/'environment.json').write_text(json.dumps(environment_manifest,indent=2)+'\n')
        for _ in range(args.updates):
            metrics=trainer.train_update()
            write_metrics(run_dir/'metrics.jsonl',metrics)
            trainer.save(run_dir/'latest.pt',metadata={'source_revision':source,'run_dir':str(run_dir)})
            print(json.dumps(metrics,sort_keys=True),flush=True)
    finally:
        if trainer is not None: trainer.close(); trainer.env=None
        if env is not None: env.close()
        app.close()

try:
    main()
except BaseException:
    traceback.print_exc()
    try:
        app.close()
    except BaseException:
        pass
    raise SystemExit(1)
