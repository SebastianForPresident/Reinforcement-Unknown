"""Bounded one-world Isaac observation and reward callback probe."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config",required=True,type=Path)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();args.headless=True
app=AppLauncher(args).app
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def _describe(observation):
    import numpy as np
    if not isinstance(observation,dict):raise ValueError(f"observation must be a mapping, got {type(observation).__name__}")
    result={}
    for key,value in observation.items():
        array=np.asarray(value)
        numeric=np.asarray(value,dtype=np.float64)
        result[key]={"shape":list(array.shape),"dtype":str(array.dtype),"finite":bool(np.isfinite(numeric).all()),
                     "min":float(numeric.min()) if numeric.size else None,"max":float(numeric.max()) if numeric.size else None,
                     "mean":float(numeric.mean()) if numeric.size else None}
        if not result[key]["finite"]:raise ValueError(f"observation field {key} contains non-finite values")
    return result


def main():
    import numpy as np
    spec=importlib.util.spec_from_file_location("research_config",args.config.expanduser().resolve())
    if spec is None or spec.loader is None:raise ValueError(f"cannot load config {args.config}")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from casu_grid.workbench.contracts import validate_contracts
    from casu_grid.workbench.components import resolve_reward_reference
    from casu_grid.workbench.runtime_validation import checked_callbacks
    from casu_grid.isaac_navigation_env import IsaacLayer1EnvConfig,IsaacLayer1NavigationEnv
    validate_contracts(module.ACTION,module.OBSERVATION,module.REWARD,module.MODEL,module.PPO)
    callbacks=module.callbacks() if hasattr(module,"callbacks") else (None,None,None)
    if getattr(module,"REWARD_COMPONENT_REF",None):
        component_manifest=args.config.expanduser().resolve().parent/"research_component.json"
        if component_manifest.is_file():module.REWARD_COMPONENT_REF=json.loads(component_manifest.read_text())
    callbacks=resolve_reward_reference(module,args.config,callbacks)
    callbacks=checked_callbacks(callbacks,module.OBSERVATION,module.REWARD)
    env=IsaacLayer1NavigationEnv(IsaacLayer1EnvConfig(seed=391728405,num_envs=1,spawn_mode="canonical",geometry_scope="full"),
                                action_decoder=callbacks[0],observation_encoder=callbacks[1],reward_function=callbacks[2])
    try:
        observation,_=env.reset(seed=391728405)
        reset_before=_describe(observation)
        actions={}
        for head in module.ACTION.heads:
            actions[head.name]=np.asarray([0],dtype=np.int64) if head.kind=="categorical" else np.zeros((1,head.size),dtype=np.float32)
        stepped,reward,terminated,truncated,info=env.step(actions)
        step_description=_describe(stepped)
        observation_after,_=env.reset(seed=391728405)
        reset_after=_describe(observation_after)
        declared={field.name:list(field.shape) for field in module.OBSERVATION.spatial+module.OBSERVATION.scalar}
        expected={key:[1]+shape for key,shape in declared.items()}
        reports=(reset_before,step_description,reset_after)
        for label,report in zip(("reset_before","step","reset_after"),reports):
            shapes={key:value["shape"] for key,value in report.items()}
            if shapes!=expected:raise ValueError(f"{label} observation shapes {shapes} differ from declared {expected}")
        if reset_before!=reset_after:
            raise ValueError("reset observation summaries changed across canonical resets")
        output={"probe":"one canonical reset, one policy decision, one reset","status":"passed",
                "config":str(args.config.expanduser().resolve()),"seed":391728405,
                "physics_dt_seconds":env.cfg.physics_dt,"physics_ticks_per_decision":env.cfg.action_decimation,
                "declared_field_shapes":declared,"expected_runtime_batch_shapes":expected,
                "observations":{"reset_before":reset_before,"step":step_description,"reset_after":reset_after},
                "reward":{"scalar":np.asarray(reward).astype(float).reshape(-1).tolist(),"terms":info.get("reward_terms",{}),
                          "terminated":np.asarray(terminated).astype(bool).reshape(-1).tolist(),"truncated":np.asarray(truncated).astype(bool).reshape(-1).tolist()},
                "callback_source":"selected config/component callback; bounded probe, not a task-quality evaluation"}
        print(json.dumps(output,indent=2,default=str),flush=True)
    finally:env.close()


try:main()
finally:app.close()
