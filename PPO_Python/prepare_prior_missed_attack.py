"""Prepare clean PPO_6 with a temporary attack prior and miss penalty."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import CasualtiesEnv
import Train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError(args.destination)
    if Train.CheckpointStep(args.source) != Train.CB1_INTERVENTION_SOURCE_STEP:
        raise ValueError("Expected the accepted 51,843,364-step PPO_5 source")

    args.destination.mkdir(parents=True)
    target = args.destination / args.source.name
    shutil.copy2(args.source, target)
    environment = {
        "PPO_NATURAL_STAMINA": "0",
        "PPO_ATTACK_LOGIT_PRIOR": "1",
        "PPO_MISSED_ATTACK_PENALTY": "1",
        "PPO_ATTACK_BUDGET": "0",
        "PPO_ATTACK_PERMISSION": "0",
        "PPO_ATTACK_PENALTY": "0",
        "PPO_ATTACK_TERMINAL": "0",
        "PPO_ATTACK_PHASES": "0",
        "PPO_ATTACK_CURRICULUM": "0",
        "PPO_ATTACK_EXPLORATION": "0",
    }
    os.environ.update(environment)
    env = CasualtiesEnv.Env()
    Train.ValidateOrWriteProtocolManifest(args.destination, env, resuming=False)
    Train.ValidateOrWriteInterventionManifest(args.destination, target)
    Train.ValidateOrWriteProtocolManifest(args.destination, env, resuming=True)
    env.close()

    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    report = {
        "source": str(args.source.resolve()),
        "source_sha256": source_hash,
        "destination": str(target.resolve()),
        "destination_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "checkpoint_copied_without_model_surgery": True,
        "observation_shape_unchanged": True,
        "environment": environment,
    }
    (args.destination / "prior_missed_attack_preparation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
