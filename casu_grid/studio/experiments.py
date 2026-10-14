"""Experiment discovery, contract summaries and the learning-contract fingerprint (no Qt, no torch)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..workbench import lab
from ..workbench.contracts import TERMINATION_V1, TERMINATION_V1_OUT_OF_WORLD, TERMINATION_V2_LAYER_EXIT, termination_contract_id
from .environment import ROOT

EXPERIMENT_DIR = ROOT / "casu_grid" / "workbench" / "experiments"

# The scientific contract of XY1 as the Studio presents it. Every number is checked against the loaded configuration (see
# ``contract_mismatches``) so the screen can never claim something the experiment file does not say.
XY1_EXPECTED = {
    "action_heads": [3, 2, 3, 2, 9, 11, 2],
    "action_names": ["move_x", "jump", "move_y", "crouch", "look_dx", "look_dy", "attack"],
    "physics_ticks_per_decision": 10,
    "spatial_shape": (16, 64, 64),
    "scalar_shape": (2,),
    "world": "128 × 128 tiles",
    "horizon": 100_000,
    "rollout_steps": 64, "minibatch_size": 32, "epochs": 2,
}

# the validated execution layout (Optimization IV): not part of the science, but the one that was measured and tested
RECOMMENDED = {"worlds": 64, "workers": 8, "physx_threads": 2, "device": "cuda", "throughput": 1700.0}


@dataclass
class ExperimentInfo:
    path: Path
    name: str
    description: str
    hypothesis: str
    notes: str
    sha256: str
    is_xy1: bool
    is_mini: bool
    summary: dict = field(default_factory=dict)         # lab.inspect_config result (None when the file is not valid)
    error: str | None = None
    fingerprint: str | None = None
    mismatches: list[str] = field(default_factory=list)
    variant_note: str | None = None        # set for exploratory variants whose rules differ from the validated XY1 contract
    termination_contract: str = TERMINATION_V1
    role: str = "other"                     # recommended (termination v2) | reference (v1, reproducible) | superseded | other


def contract_fingerprint(config) -> str:
    """Same digest ``PPOTrainer`` stores in every checkpoint (actions, observations, model, PPO)."""
    payload = json.dumps({"actions": asdict(config.ACTION), "observations": asdict(config.OBSERVATION),
                          "model": asdict(config.MODEL), "ppo": asdict(config.PPO)},
                         sort_keys=True, separators=(",", ":"), default=list).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def contract_mismatches(summary: dict) -> list[str]:
    """Differences between a loaded XY1 config and the contract the Studio describes. Empty for the intended XY1 setup."""
    issues = []
    sizes = [h["size"] for h in summary["actions"]]
    if sizes != XY1_EXPECTED["action_heads"]:
        issues.append(f"action heads are {sizes}, expected {XY1_EXPECTED['action_heads']}")
    if summary["physics_ticks_per_decision"] != XY1_EXPECTED["physics_ticks_per_decision"]:
        issues.append(f"{summary['physics_ticks_per_decision']} physics ticks per decision, expected 10")
    spatial = [tuple(f["shape"]) for f in summary["observations"] if len(f["shape"]) == 3]
    if XY1_EXPECTED["spatial_shape"] not in spatial:
        issues.append(f"spatial observation is {spatial}, expected {XY1_EXPECTED['spatial_shape']}")
    ppo = summary["ppo"]
    for key in ("rollout_steps", "minibatch_size", "epochs"):
        if ppo[key] != XY1_EXPECTED[key]:
            issues.append(f"PPO {key} is {ppo[key]}, expected {XY1_EXPECTED[key]}")
    return issues


def load_experiment(path: Path) -> ExperimentInfo:
    path = Path(path)
    try:
        summary = lab.inspect_config(path)
        cfg, _ = lab._load_config(path)
        experiment = getattr(cfg, "EXPERIMENT", {}) or {}
        environment = getattr(cfg, "ENVIRONMENT", {}) or {}
        is_xy1 = bool(environment.get("navigation_mode"))
        info = ExperimentInfo(path=path, name=experiment.get("name") or path.stem, description=experiment.get("description", ""),
                              hypothesis=experiment.get("hypothesis", ""), notes=experiment.get("notes", ""), sha256=summary["sha256"],
                              is_xy1=is_xy1, is_mini=environment.get("world_variant") == "xy1_mini_128x128", summary=summary,
                              fingerprint=contract_fingerprint(cfg))
        if is_xy1:
            info.mismatches = contract_mismatches(summary)
            info.termination_contract = termination_contract_id(environment)
            info.role = {TERMINATION_V2_LAYER_EXIT: "recommended", TERMINATION_V1: "reference", TERMINATION_V1_OUT_OF_WORLD: "superseded"}.get(info.termination_contract, "other")
            if info.role == "superseded":
                info.variant_note = "SUPERSEDED exploratory variant: episodes are truncated when the agent leaves the world"
                info.mismatches.append(info.variant_note)
        return info
    except Exception as exc:             # a broken experiment file is shown with its error, it does not break the Studio
        return ExperimentInfo(path=path, name=path.stem, description="", hypothesis="", notes="", sha256="", is_xy1=False, is_mini=False,
                              error=f"{type(exc).__name__}: {exc}")


def discover_experiments(directory: Path = EXPERIMENT_DIR) -> list[ExperimentInfo]:
    found = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.py")):
            if not path.name.startswith("_"):
                found.append(load_experiment(path))
    # the intended XY1 experiment first
    order = {"recommended": 0, "reference": 1, "other": 2, "superseded": 3}
    found.sort(key=lambda e: (not e.is_mini, order.get(e.role, 2), e.name))
    return found
