"""Researcher-facing command line for the CasU Isaac workbench.

File discovery and analysis are pure Python; commands that initialize Isaac are
explicitly delegated to the existing trainer or deterministic replay module.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
from pathlib import Path
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from .contracts import validate_contracts

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENTS = ROOT / "experiments"
TEMPLATE = Path(__file__).with_name("engineering_smoke.py")


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc


def _load_config(path: Path):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Configuration not found: {path}")
    spec = importlib.util.spec_from_file_location("casu_research_config", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load Python configuration: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        required = ("ACTION", "OBSERVATION", "REWARD", "MODEL", "PPO")
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise ValueError(f"{path}: missing required configuration members: {', '.join(missing)}")
        contracts = validate_contracts(*(getattr(module, name) for name in required))
        if not callable(getattr(module, "callbacks", None)):
            if contracts["rewards"]["terms"]:
                raise ValueError("callbacks() is required because reward terms are declared; otherwise training would receive a zero reward")
            if contracts["observations"]["spatial"] or [
                    (field["name"], field["shape"]) for field in contracts["observations"]["scalar"]] != [("player_state", [8])]:
                raise ValueError("callbacks() is required for the configured observation fields; the built-in adapter only supplies player_state with shape (8,)")
            if any(head["name"] not in {"move_x", "move_y", "jump", "crouch"}
                   for head in contracts["actions"]["heads"]):
                raise ValueError("callbacks() is required to decode the configured action head names")
            native_sizes = {"move_x": 3, "move_y": 3, "jump": 2, "crouch": 2}
            if any(head["kind"] != "categorical" or head["size"] != native_sizes[head["name"]]
                   for head in contracts["actions"]["heads"]):
                raise ValueError("callbacks() is required for action types/sizes beyond the built-in move_x/move_y/jump/crouch adapter")
    except Exception as exc:
        raise ValueError(f"Invalid configuration {path}: {exc}") from exc
    return module, contracts


def scaffold(name: str, destination: Path, *, hypothesis: str = "", notes: str = "", parent: str | None = None) -> Path:
    dest = destination.expanduser().resolve()
    if dest.exists():
        raise ValueError(f"Refusing to overwrite existing experiment/configuration: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    text = TEMPLATE.read_text(encoding="utf-8")
    header_lines = [f"# {line}" for line in [f"Research experiment: {name}",
        "Identity metadata; values below are engineering examples requiring review.",
        "Learning contracts are research choices owned by the researcher.",
        f"Hypothesis: {hypothesis or '(add when ready)'}",
        f"Notes: {notes or '(add when ready)'}",
        f"Created UTC: {datetime.now(timezone.utc).isoformat()}"]]
    header = "\n".join(header_lines) + "\n\n"
    identity = "EXPERIMENT = " + repr({"id": uuid.uuid4().hex, "name": name,
        "description": notes, "hypothesis": hypothesis, "notes": notes, "parent": parent,
        "created_utc": datetime.now(timezone.utc).isoformat(), "config_version": 1}) + "\n\n"
    if re.search(r"(?m)^EXPERIMENT = ", text):
        text = re.sub(r"(?ms)^EXPERIMENT = \{.*?^\}\n\n", identity, text, count=1)
    else:
        text = identity + text
    dest.write_text(header + text, encoding="utf-8")
    return dest


def inspect_config(path: Path) -> dict[str, Any]:
    module, contracts = _load_config(path)
    source = path.expanduser().resolve().read_bytes()
    # The lightweight model constructor only needs torch and does not initialize Isaac.
    model = contracts["model"]
    params = None
    parameter_note = None
    try:
        from .model import ConfigurableActorCritic, count_parameters
        params = count_parameters(ConfigurableActorCritic(module.ACTION, module.OBSERVATION, module.MODEL))
    except Exception:
        parameter_note = "Unavailable in this Python; use the Isaac bundled Python to compute model parameters."
    actions = contracts["actions"]
    observations = contracts["observations"]
    rewards = contracts["rewards"]
    ppo = contracts["ppo"]
    return {
        "config": str(path.expanduser().resolve()),
        "sha256": hashlib.sha256(source).hexdigest(),
        "actions": [{"name": h["name"], "kind": h["kind"], "size": h["size"],
                     "bounds": [h["low"], h["high"]] if h["kind"] == "continuous" else None,
                     "mapping": actions["mapping"].get(h["name"]),
                     "hold_physics_ticks": h["hold_physics_ticks"]} for h in actions["heads"]],
        "physics_ticks_per_decision": 10,
        "observations": [{"name": f["name"], "shape": f["shape"], "source": f["source"],
                           "normalization": f["normalization"], "mask": f["mask"]}
                          for f in observations["spatial"] + observations["scalar"]],
        "coordinate_convention": observations["coordinate_convention"],
        "reward_terms": rewards["terms"],
        "model": model, "parameter_count": params, "parameter_count_note": parameter_note, "ppo": ppo,
        "checkpoint_compatibility": "Action, observation, model, and PPO contracts are fingerprinted.",
        "environment": {"canonical_seed": 391728405, "physics_dt_seconds": 0.02,
                        "action_decimation": 10, "training_geometry": "full only",
                        "terrain_modes": ["shared", "independent"]},
        "callbacks_defined": callable(getattr(module, "callbacks", None)),
        "mode_notice": "Contract preflight only; it does not initialize Isaac Sim or run callback probes.",
    }


def _read_metrics(path: Path, *, max_rows: int | None = None) -> tuple[list[dict], str | None]:
    rows, error = [], None
    if not path.exists():
        return rows, "metrics.jsonl is missing"
    try:
        # Incremental line reading naturally tolerates a final incomplete append.
        with path.open("rb") as stream:
            if max_rows is not None:
                stream.seek(0, os.SEEK_END)
                end = stream.tell()
                stream.seek(max(0, end - 1_048_576))
                if stream.tell() > 0:
                    stream.readline()  # discard a possibly partial first record
            lines = stream
            for line_no, raw_line in enumerate(lines, 1):
                line = raw_line.decode("utf-8", errors="replace")
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        rows.append(row)
                        if max_rows is not None and len(rows) > max_rows: rows.pop(0)
                except json.JSONDecodeError:
                    error = f"incomplete/corrupt metrics row in inspected window at record {line_no}; ignored bad row"
    except OSError as exc:
        error = str(exc)
    return rows, error


def _status(run: Path, manifest: dict, rows: list[dict]) -> str:
    status = manifest.get("status")
    if status in {"configured", "running", "completed", "interrupted", "failed"}:
        return str(status)
    if (run / "failure.json").exists(): return "failed"
    if (run / "INTERRUPTED").exists(): return "interrupted"
    if (run / "RUNNING").exists(): return "running" if _pid_alive(run / "RUNNING") else "interrupted"
    if (run / "COMPLETE").exists(): return "completed"
    if rows:
        return "interrupted"  # metrics without explicit completion evidence are not success
    if (run / "config.json").exists(): return "unknown/stale"
    return "unknown"


def _pid_alive(marker: Path) -> bool:
    try:
        pid = int(marker.read_text().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def discover_runs(root: Path = DEFAULT_EXPERIMENTS) -> list[dict[str, Any]]:
    root = root.expanduser().resolve()
    if not root.exists(): return []
    runs = []
    for run in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name):
        cfg_path = run / "config.json"
        if not cfg_path.exists(): continue
        try: manifest = _json(cfg_path)
        except ValueError as exc: manifest = {"manifest_error": str(exc)}
        rows, metrics_error = _read_metrics(run / "metrics.jsonl", max_rows=2)
        last = rows[-1] if rows else {}
        env = manifest.get("environment", {})
        latest = run / "latest.pt"
        runs.append({
            "run_id": run.name, "experiment_name": (manifest.get("experiment") or {}).get("name"), "path": str(run), "created": manifest.get("created_utc") or
                datetime.fromtimestamp(run.stat().st_mtime, timezone.utc).isoformat(),
            "status": _status(run, manifest, rows), "revision": manifest.get("git_revision"),
            "config_sha256": manifest.get("config_sha256"), "num_envs": env.get("num_envs"),
            "resume_from": manifest.get("resume_from"),
            "terrain_mode": env.get("terrain_mode", env.get("geometry_scope")),
            "environment_steps": last.get("environment_steps"), "update": last.get("update"),
            "latest_checkpoint": str(latest) if latest.is_file() else None,
            "mean_completed_return": last.get("mean_completed_return"),
            "completed_episodes": last.get("completed_episodes"),
            "policy_decisions_per_second": last.get("policy_decisions_per_second"),
            "physics_ticks_per_second": last.get("physics_ticks_per_second"),
            "metrics_rows": len(rows), "metrics_warning": metrics_error,
            "manifest_error": manifest.get("manifest_error"),
            "trajectory_available": (run / "trajectories.jsonl").is_file(),
        })
    return runs


def _count_trajectory_steps(path: Path) -> int | None:
    if not path.exists(): return None
    count = 0
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    if json.loads(line).get("schema") == "casu.trajectory-step.v1": count += 1
                except json.JSONDecodeError: continue
    except OSError: return None
    return count


def _find_run(run_id_or_path: str, root: Path) -> Path:
    candidate = Path(run_id_or_path).expanduser()
    if not candidate.is_absolute():
        rooted = root / candidate
        candidate = rooted if rooted.is_dir() else Path.cwd() / candidate
    candidate = candidate.resolve()
    if not candidate.is_dir(): raise ValueError(f"Run directory not found: {candidate}")
    if not (candidate / "config.json").is_file(): raise ValueError(f"No config.json in run directory: {candidate}")
    return candidate


def _isaac_python_command(*, preview: bool = False) -> str:
    if importlib.util.find_spec("isaaclab") is not None:
        return sys.executable
    sim_root = os.environ.get("ISAAC_SIM")
    if sim_root:
        return str(Path(sim_root).expanduser() / "python.sh")
    if preview:
        return "$ISAAC_SIM/python.sh"
    raise ValueError("This operation requires Isaac Lab Python. Set ISAAC_SIM to the Isaac Sim directory and use \"$ISAAC_SIM/python.sh\" (see ISAAC_LAB_SETUP.md).")


def _display_command(command: list[str]) -> str:
    display = shlex.join(map(str, command))
    return display.replace(shlex.quote("$ISAAC_SIM/python.sh"), "$ISAAC_SIM/python.sh")


def _flatten(value: Any, prefix="") -> dict[str, Any]:
    out = {}
    if isinstance(value, dict):
        for key, item in value.items(): out.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for i, item in enumerate(value): out.update(_flatten(item, f"{prefix}[{i}]"))
    else: out[prefix] = value
    return out


def compare_runs(paths: list[Path]) -> dict[str, Any]:
    if len(paths) < 2: raise ValueError("Compare requires at least two run IDs or paths")
    runs, manifests, metric_rows = [], [], []
    for path in paths:
        p = path.resolve(); manifest = _json(p / "config.json"); rows, warning = _read_metrics(p / "metrics.jsonl")
        runs.append({"run_id": p.name, "path": str(p), "status": _status(p, manifest, rows),
                     "metrics_warning": warning})
        manifests.append(manifest); metric_rows.append(rows)
    flat = [_flatten(m.get("contracts", {})) | _flatten(m.get("environment", {}), "environment") for m in manifests]
    keys = sorted(set().union(*(x.keys() for x in flat)))
    differences = {key: [x.get(key, "<missing>") for x in flat]
                   for key in keys if len({json.dumps(x.get(key, "<missing>"), sort_keys=True) for x in flat}) > 1}
    fingerprints = [m.get("config_sha256") for m in manifests]
    reward_match = len({json.dumps(m.get("contracts", {}).get("rewards", {}), sort_keys=True) for m in manifests}) == 1
    learning_match = len({json.dumps(m.get("contracts", {}), sort_keys=True) for m in manifests}) == 1
    source_match = len({m.get("config_sha256") for m in manifests}) == 1
    metric_names = sorted(set().union(*(set(row) for series in metric_rows for row in series))) if any(metric_rows) else []
    aligned = {}
    for key in metric_names:
        if not all(isinstance(r.get(key), (int, float)) and math.isfinite(r[key]) for rs in metric_rows for r in rs):
            continue
        aligned[key] = [{"environment_steps": row.get("environment_steps"), "update": row.get("update"), "value": row[key]}
                        for rs in metric_rows for row in rs]
    aligned_metrics = {}
    x_values = []
    budgets = []
    for rows in metric_rows:
        points = sorted({int(row["environment_steps"]) for row in rows
                         if isinstance(row.get("environment_steps"), (int, float))})
        if points: x_values.append(points)
    if len(x_values) == len(metric_rows) and x_values:
        low, high = max(points[0] for points in x_values), min(points[-1] for points in x_values)
        budgets = sorted({x for points in x_values for x in points if low <= x <= high}) if low <= high else []
        for key in metric_names:
            curves = {}
            for run, rows in zip(runs, metric_rows):
                points = [(float(row["environment_steps"]), float(row[key])) for row in rows
                          if isinstance(row.get("environment_steps"), (int, float))
                          and isinstance(row.get(key), (int, float)) and math.isfinite(row[key])]
                if not points or not budgets: continue
                curves[run["run_id"]] = [
                    {"environment_steps": budget,
                     "value": next((v for x, v in points if x == budget),
                                   next((v0 + (v1-v0)*(budget-x0)/(x1-x0)
                                         for (x0,v0),(x1,v1) in zip(points, points[1:])
                                         if x0 < budget < x1), None))}
                    for budget in budgets]
            if curves: aligned_metrics[key] = curves
    if not reward_match:
        compare_warning = "Reward contracts differ; training rewards and completion behavior may not be directly comparable."
    elif not source_match:
        compare_warning = "Config source hashes differ; callback code or metadata may change the effective objective despite matching declared reward contracts."
    elif not learning_match:
        compare_warning = "Learning contracts differ; compare outcomes with the action/observation/model/PPO differences in view."
    else:
        compare_warning = None
    return {"schema": "casu.run-comparison.v1", "runs": runs,
            "config_fingerprints": fingerprints, "configuration_differences": differences,
            "reward_contracts_match": reward_match, "learning_contracts_match": learning_match,
            "config_sources_match": source_match,
            "comparability_warning": compare_warning,
            "metrics_by_environment_decisions": {run["run_id"]: rows for run, rows in zip(runs, metric_rows)},
            "aligned_metrics_at_shared_environment_decision_budgets": aligned_metrics,
            "shared_decision_budgets": budgets if x_values else [],
            "available_numeric_metrics": sorted(aligned),
            "checkpoints": [str((Path(r["path"]) / "latest.pt")) if (Path(r["path"]) / "latest.pt").exists() else None for r in runs]}


def _summary(run: Path) -> dict:
    manifest = _json(run / "config.json"); rows, warning = _read_metrics(run / "metrics.jsonl", max_rows=100)
    if not rows: raise ValueError(f"No complete metrics rows in {run}")
    last = rows[-1]
    keys = ("update", "environment_steps", "environment_count", "completed_episodes", "mean_completed_return",
            "mean_completed_episode_length", "policy_decisions_per_second", "physics_ticks_per_second",
            "policy_loss", "value_loss", "entropy", "approx_kl", "optimization_seconds", "rollout_seconds")
    checkpoint = run / "latest.pt"
    try:
        created = datetime.fromisoformat(str(manifest.get("created_utc", "")).replace("Z", "+00:00"))
        elapsed_seconds = max(0.0, (datetime.now(timezone.utc) - created).total_seconds())
    except (TypeError, ValueError):
        elapsed_seconds = None
    checkpoint_age_seconds = max(0.0, time.time() - checkpoint.stat().st_mtime) if checkpoint.is_file() else None
    return {"run_id": run.name, "path": str(run), "status": _status(run, manifest, rows),
            "revision": manifest.get("git_revision"), "config_sha256": manifest.get("config_sha256"),
            "resume_from": manifest.get("resume_from"),
            "elapsed_seconds_since_creation": elapsed_seconds,
            "checkpoint_age_seconds": checkpoint_age_seconds,
            "failure": _json(run / "failure.json") if (run / "failure.json").is_file() else None,
            "latest": {k: last.get(k) for k in keys},
            "reward_terms": {k.removeprefix("reward_term/"): v for k, v in last.items() if k.startswith("reward_term/")},
            "action_frequencies": {k.removeprefix("action/"): v for k, v in last.items() if k.startswith("action/")},
            "termination_counts": {k.removeprefix("termination/"): v for k, v in last.items() if k.startswith("termination/")},
            "metrics_rows_in_recent_window": len(rows), "warning": warning,
            "trajectory_available": (run / "trajectories.jsonl").is_file(),
            "tensorboard": str(run / "tensorboard") if (run / "tensorboard").exists() else None,
            "checkpoint": str(run / "latest.pt") if (run / "latest.pt").is_file() else None}


def _print_table(rows: list[dict], columns: list[tuple[str, str]]) -> None:
    if not rows: print("No runs found."); return
    widths = [max(len(title), *(len(str(r.get(key) if r.get(key) is not None else "unknown")) for r in rows)) for key, title in columns]
    print("  ".join(title.ljust(width) for (_, title), width in zip(columns, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(row.get(key) if row.get(key) is not None else "unknown").ljust(width)
                        for (key, _), width in zip(columns, widths)))


def _resolve_config_for_run(run: Path, supplied: str | None) -> Path:
    manifest = _json(run / "config.json")
    if supplied: return Path(supplied).expanduser().resolve()
    raw = manifest.get("config_path")
    if not raw: raise ValueError("Run does not record config_path; pass --config explicitly")
    candidate = Path(raw)
    expected_hash = manifest.get("config_sha256")
    if candidate.is_file() and (not expected_hash or hashlib.sha256(candidate.read_bytes()).hexdigest() == expected_hash):
        return candidate.resolve()
    snapshot = run / "research_config.py"
    if snapshot.is_file() and (not expected_hash or hashlib.sha256(snapshot.read_bytes()).hexdigest() == expected_hash):
        return snapshot.resolve()
    raise ValueError(f"Recorded config is unavailable or has changed ({candidate}); pass --config pointing to the exact preserved copy")


def _metric_plot(run: Path, output: Path) -> Path:
    rows, _ = _read_metrics(run / "metrics.jsonl")
    if not rows: raise ValueError(f"No complete metrics to plot in {run}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise ValueError(f"Plotting requires matplotlib in this Python environment: {exc}") from exc
    keys = [k for k in ("mean_completed_return", "mean_completed_episode_length", "policy_loss", "value_loss",
                         "completed_episodes", "entropy", "approx_kl", "policy_decisions_per_second", "physics_ticks_per_second")
            if any(isinstance(r.get(k), (int, float)) for r in rows)]
    reward_terms = sorted({k for r in rows for k in r if k.startswith("reward_term/")})
    action_metrics = sorted({k for r in rows for k in r if k.startswith("action/")})
    termination_metrics = sorted({k for r in rows for k in r if k.startswith("termination/")})
    keys += reward_terms + action_metrics + termination_metrics
    if not keys: raise ValueError("No supported numeric training metrics are available")
    cols = 2; nrows = math.ceil(len(keys) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(12, 3.2 * nrows), squeeze=False)
    x = [r.get("environment_steps", r.get("update", i)) for i, r in enumerate(rows)]
    for i, key in enumerate(keys):
        ax = axes.flat[i]
        vals = [r.get(key) for r in rows]
        ax.plot(x, vals, marker="." if len(rows) < 30 else None)
        ax.set_title(key.replace("_", " ")); ax.set_xlabel("Environment decisions (steps across all worlds)")
        ax.grid(alpha=.25)
    for ax in list(axes.flat)[len(keys):]: ax.remove()
    fig.suptitle(f"Training metrics — {run.name}")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True); fig.savefig(output, dpi=140, bbox_inches="tight"); plt.close(fig)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="casu-lab", description="Create, validate, run, monitor, compare, and replay CasU Isaac research experiments.")
    parser.add_argument("--experiments", type=Path, default=DEFAULT_EXPERIMENTS, help="run directory (default: ./experiments)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("about", help="summarize the active laboratory and fidelity boundaries")
    create = sub.add_parser("new", help="scaffold an editable Python experiment config (engineering example values)")
    create.add_argument("name"); create.add_argument("--output", type=Path, required=True); create.add_argument("--hypothesis", default=""); create.add_argument("--notes", default=""); create.add_argument("--parent", default=None, help="optional parent experiment ID")
    validate = sub.add_parser("validate", help="preflight contract shapes and settings without starting Isaac")
    validate.add_argument("config", type=Path); validate.add_argument("--json", action="store_true")
    launch = sub.add_parser("launch", help="preview or launch the existing Isaac PPO trainer")
    launch.add_argument("config", type=Path); launch.add_argument("--run-dir", type=Path); launch.add_argument("--updates", type=int, default=100)
    launch.add_argument("--num-envs", type=int, default=1); launch.add_argument("--episode-steps", type=int); launch.add_argument("--terrain-mode", choices=("shared", "independent"), default="shared")
    launch.add_argument("--seed", type=int, default=391728405, help="PPO policy/sampling RNG seed; the world fixture remains canonical seed 391728405"); launch.add_argument("--resume", type=Path); launch.add_argument("--preview", action="store_true"); launch.add_argument("--engineering-smoke", action="store_true", help="explicitly label the engineering-only template")
    runs = sub.add_parser("runs", help="discover runs and classify status from artifacts")
    runs.add_argument("--json", action="store_true")
    configs = sub.add_parser("configs", help="list configured Python experiments that have not yet become runs")
    configs.add_argument("--directory", type=Path, default=Path("casu_grid/workbench/experiments"))
    configs.add_argument("--json", action="store_true")
    status = sub.add_parser("status", help="show metrics and artifacts for one run"); status.add_argument("run")
    compare = sub.add_parser("compare", help="compare configuration and metrics aligned by environment decisions")
    compare.add_argument("runs", nargs="+"); compare.add_argument("--output", type=Path); compare.add_argument("--json", action="store_true")
    plot = sub.add_parser("plot", help="generate labeled training-metric plots")
    plot.add_argument("run"); plot.add_argument("--output", type=Path)
    replay = sub.add_parser("replay", help="replay a run's stored actions and export CSV/plots")
    replay.add_argument("run"); replay.add_argument("--config"); replay.add_argument("--max-steps", type=int); replay.add_argument("--tolerance", type=float, default=1e-4); replay.add_argument("--preview", action="store_true")
    resume = sub.add_parser("resume", help="preview or continue a compatible checkpoint in a new run folder")
    resume.add_argument("run"); resume.add_argument("--updates", type=int, default=100); resume.add_argument("--output", type=Path); resume.add_argument("--preview", action="store_true")
    tb = sub.add_parser("tensorboard", help="open TensorBoard for a run (requires tensorboard in the selected Python)")
    tb.add_argument("run"); tb.add_argument("--port", type=int, default=6006); tb.add_argument("--open", action="store_true")
    guide = sub.add_parser("guide", help="show first-session commands and links")
    watch = sub.add_parser("watch", help="poll a run summary without streaming raw trainer logs")
    watch.add_argument("run"); watch.add_argument("--interval", type=float, default=15.0)
    watch.add_argument("--count", type=int, default=0, help="number of changed summaries; 0 waits until Ctrl-C")
    sub.add_parser("menu", help="open a guided interactive terminal menu")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(); args = parser.parse_args(argv); root = args.experiments.expanduser().resolve()
    try:
        if args.command == "about":
            print("CasU Layer 1 Isaac research workbench\nCurrent interface: casu_grid.workbench.lab\nCanonical seed: 391728405 | physics: 50 ticks/s | policy cadence: 10 physics ticks/decision\nFidelity: useful source-validated surrogate, not bit-exact CasU; see casu_grid/ACCEPTANCE_REPORT.md\nHistorical Unity research remains in PPO_Python/; new Isaac experiments belong under experiments/.")
        elif args.command == "guide":
            print("1) Check setup: read ISAAC_LAB_SETUP.md\n2) Create: python -m casu_grid.workbench.lab new 'My question' --output casu_grid/workbench/experiments/my_question.py\n3) Edit the explicitly labeled research contracts/callbacks, then validate:\n   python -m casu_grid.workbench.lab validate casu_grid/workbench/experiments/my_question.py\n4) Preview, then launch with Isaac Python:\n   $ISAAC_SIM/python.sh -m casu_grid.workbench.lab launch casu_grid/workbench/experiments/my_question.py --updates 3 --run-dir experiments/my-question --preview\n   $ISAAC_SIM/python.sh -m casu_grid.workbench.lab launch casu_grid/workbench/experiments/my_question.py --updates 3 --run-dir experiments/my-question\n5) Inspect: runs, status RUN, watch RUN, plot RUN, compare RUN_A RUN_B, replay RUN.\nSee casu_grid/workbench/README.md and USABILITY_ACCEPTANCE_REPORT.md.")
        elif args.command == "new":
            path = scaffold(args.name, args.output, hypothesis=args.hypothesis, notes=args.notes, parent=args.parent); print(f"Created {path}\nTemplate values are engineering examples; review every learning contract before a research run.")
        elif args.command == "validate":
            result = inspect_config(args.config)
            if args.json: print(json.dumps(result, indent=2, default=str))
            else:
                print(f"Valid Python contract config: {result['config']}\nSHA-256: {result['sha256']}\nMode: {result['mode_notice']}\n\nActions (10 physics ticks/decision):")
                for h in result["actions"]: print(f"  {h['name']}: {h['kind']} size={h['size']} bounds={h['bounds']} mapping={h['mapping']} hold={h['hold_physics_ticks']}")
                print("Observations:")
                for f in result["observations"]: print(f"  {f['name']}: {f['shape']} from {f['source']} normalization={f['normalization']} mask={f['mask']}")
                print("Reward terms:")
                for t in result["reward_terms"]: print(f"  {t['name']}: source={t['source']} weight={t['weight']} terminal={t['terminal']}")
                print(f"Model: {result['model']}\nParameter count: {result['parameter_count'] if result['parameter_count'] is not None else result['parameter_count_note']}\nPPO: {result['ppo']}\nCallbacks defined: {result['callbacks_defined']}\nEnvironment: {result['environment']}")
                print("NOTE: this validates declared contracts, not runtime callback outputs. Isaac initialization is separate.")
        elif args.command == "launch":
            cfg, _ = _load_config(args.config)
            python_cmd = _isaac_python_command(preview=args.preview)
            if args.resume and importlib.util.find_spec("isaaclab") is None:
                raise ValueError("Checkpoint fingerprint inspection requires the Isaac bundled Python (torch). Use the configured python.sh interpreter.")
            if args.updates <= 0 or args.num_envs <= 0 or (args.episode_steps is not None and args.episode_steps <= 0): raise ValueError("updates, num-envs, and episode-steps must be positive")
            if any(h.hold_physics_ticks != 10 for h in cfg.ACTION.heads): raise ValueError("Current environment requires exactly 10 physics ticks per action decision")
            if args.resume and not args.resume.is_file(): raise ValueError(f"Resume checkpoint not found: {args.resume}")
            if args.resume:
                # Validate signature without creating Isaac: reconstruct fingerprint from contracts.
                import torch
                try:
                    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
                except Exception as exc:
                    raise ValueError(f"Could not read resume checkpoint {args.resume}: {exc}") from exc
                expected_config = cfg
                from dataclasses import asdict
                payload = json.dumps({"actions": asdict(cfg.ACTION), "observations": asdict(cfg.OBSERVATION), "model": asdict(cfg.MODEL), "ppo": asdict(cfg.PPO)}, sort_keys=True, separators=(",", ":"), default=list).encode()
                expected = hashlib.sha256(payload).hexdigest()
                if checkpoint.get("schema") != "casu.ppo-checkpoint.v1" or checkpoint.get("contract_fingerprint") != expected:
                    raise ValueError("Resume checkpoint is incompatible with this config (action/observation/model/PPO contract fingerprint differs)")
            run_dir = (args.run_dir or (root / time.strftime("%Y%m%d-%H%M%S"))).expanduser().resolve()
            if run_dir.exists(): raise ValueError(f"Run directory already exists; choose a new path: {run_dir}")
            cmd = [python_cmd, "-m", "casu_grid.workbench.train", "--config", str(args.config.expanduser().resolve()), "--run-dir", str(run_dir), "--updates", str(args.updates), "--num-envs", str(args.num_envs), "--seed", str(args.seed), "--geometry-scope", "full", "--terrain-mode", args.terrain_mode]
            if args.episode_steps: cmd += ["--episode-steps", str(args.episode_steps)]
            if args.resume: cmd += ["--resume", str(args.resume.resolve())]
            if args.engineering_smoke: print("MODE: explicitly labeled ENGINEERING SMOKE; this is not an approved Timmy research experiment.")
            else: print("MODE: researcher-configured PPO training. Verify your scientific contracts before launch.")
            print("Effective launch: " + _display_command(cmd))
            print(f"Run output: {run_dir}")
            if args.preview: return 0
            if not args.engineering_smoke and Path(args.config).resolve() == TEMPLATE.resolve(): raise ValueError("Refusing to launch engineering_smoke.py as a research run without --engineering-smoke")
            run_dir.parent.mkdir(parents=True, exist_ok=True)
            run_dir.mkdir(parents=True)
            marker = run_dir / "RUNNING"
            marker.write_text(f"{os.getpid()}\n", encoding="utf-8")
            try:
                child_env = os.environ.copy()
                child_env["CASU_WORKBENCH_LAUNCHER_PID"] = str(os.getpid())
                child_env["CASU_WORKBENCH_RESUME_FROM"] = str(args.resume.resolve()) if args.resume else ""
                child_env["CASU_WORKBENCH_RUN_KIND"] = "engineering_smoke" if args.engineering_smoke else "research_training"
                result = subprocess.run(cmd, cwd=ROOT, env=child_env)
                if result.returncode == 0:
                    (run_dir / "COMPLETE").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
                else:
                    (run_dir / "failure.json").write_text(json.dumps({"exit_code": result.returncode,
                        "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
                return int(result.returncode)
            except KeyboardInterrupt:
                (run_dir / "INTERRUPTED").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
                print(f"Interrupted. Checkpoint, if present: {run_dir / 'latest.pt'}")
                return 130
            finally:
                marker.unlink(missing_ok=True)
        elif args.command == "runs":
            found = discover_runs(root)
            if args.json: print(json.dumps(found, indent=2))
            else: _print_table(found, [("run_id", "Run"), ("status", "Status"), ("update", "Update"), ("environment_steps", "Env decisions"), ("mean_completed_return", "Mean return"), ("latest_checkpoint", "Checkpoint")])
        elif args.command == "configs":
            directory = args.directory.expanduser().resolve()
            found = []
            if directory.exists():
                for config_path in sorted(directory.glob("*.py")):
                    if config_path.name.startswith("_"): continue
                    found.append({"status": "configured", "name": config_path.stem,
                                  "path": str(config_path),
                                  "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()})
            if args.json: print(json.dumps(found, indent=2))
            else: _print_table(found, [("name", "Experiment"), ("status", "Status"), ("path", "Config path"), ("sha256", "SHA-256")])
        elif args.command == "status":
            run = _find_run(args.run, root)
            try: report = _summary(run)
            except ValueError as exc:
                manifest = _json(run / "config.json")
                report = {"run_id": run.name, "path": str(run), "status": _status(run, manifest, []),
                          "revision": manifest.get("git_revision"), "config_sha256": manifest.get("config_sha256"),
                          "latest_metrics": None, "checkpoint": str(run / "latest.pt") if (run / "latest.pt").is_file() else None,
                          "warning": str(exc)}
            print(json.dumps(report, indent=2))
            if report["warning"]: print(f"WARNING: {report['warning']}", file=sys.stderr)
        elif args.command == "watch":
            if args.interval < 1 or args.count < 0: raise ValueError("watch interval must be >=1 second and count must be >=0")
            run = _find_run(args.run, root); previous = None; emitted = 0
            print(f"Watching {run}; refresh {args.interval:g}s; Ctrl-C to stop")
            try:
                while True:
                    try: summary = _summary(run)
                    except ValueError:
                        manifest = _json(run / "config.json")
                        summary = {"run_id": run.name, "status": _status(run, manifest, []),
                                   "warning": "No complete metrics rows yet", "latest": None}
                    signature = json.dumps(summary, sort_keys=True)
                    if signature != previous:
                        print(json.dumps(summary, indent=2)); previous = signature; emitted += 1
                        if args.count and emitted >= args.count: break
                    if summary.get("status") in {"completed", "failed", "interrupted"}: break
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                print("Stopped watching; training process was not signalled.")
        elif args.command == "compare":
            paths = [_find_run(name, root) for name in args.runs]; result = compare_runs(paths)
            if args.output:
                output = args.output.expanduser().resolve()
                if output.exists(): raise ValueError(f"Refusing to overwrite comparison output: {output}")
                output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2) + "\n")
                print(f"Machine-readable comparison: {output}")
            if args.json: print(json.dumps(result, indent=2))
            else:
                print("Compared: " + ", ".join(f"{r['run_id']} [{r['status']}] {r['path']}" for r in result["runs"]))
                print(f"Reward contracts match: {result['reward_contracts_match']}")
                if result["comparability_warning"]: print("WARNING: " + result["comparability_warning"])
                print("Configuration differences:")
                if not result["configuration_differences"]: print("  (none)")
                for key, values in result["configuration_differences"].items(): print(f"  {key}: " + " | ".join(map(str, values)))
                print(f"Shared environment-decision budgets: {result['shared_decision_budgets']}")
                print("Metric curves use recorded steps across all worlds; use --json for aligned values and per-run records.")
        elif args.command == "plot":
            run = _find_run(args.run, root); output = args.output.expanduser().resolve() if args.output else run / "analysis" / "training_metrics.png"
            print(f"Wrote { _metric_plot(run, output) }")
        elif args.command == "replay":
            run = _find_run(args.run, root); config = _resolve_config_for_run(run, args.config)
            if importlib.util.find_spec("isaaclab") is None:
                raise ValueError("Deterministic replay requires Isaac Lab Python. Use the configured Isaac Sim python.sh interpreter (see ISAAC_LAB_SETUP.md).")
            trajectory = run / "trajectories.jsonl"
            if not trajectory.is_file() or _count_trajectory_steps(trajectory) == 0: raise ValueError(f"No recorded trajectory steps in {trajectory}")
            _, _ = _load_config(config)
            replay_python = _isaac_python_command(preview=args.preview)
            cmd = [replay_python, "-m", "casu_grid.workbench.replay", "--config", str(config), "--run-dir", str(run), "--tolerance", str(args.tolerance)]
            if args.max_steps is not None: cmd += ["--max-steps", str(args.max_steps)]
            print("Deterministic simulator replay (not rendered gameplay): " + _display_command(cmd))
            print(f"Expected outputs: {run / 'replay' / 'replay.csv'}, {run / 'replay' / 'paths.png'}, {run / 'replay' / 'summary.json'}")
            if args.preview: return 0
            _isaac_python_command()
            result = subprocess.run(cmd, cwd=ROOT)
            summary_path = run / "replay" / "summary.json"
            if result.returncode == 0 and summary_path.is_file():
                summary = _json(summary_path)
                if not summary.get("deterministic", False):
                    print(f"REPLAY DIVERGED: first divergence={summary.get('first_divergence')}; full report={summary_path}", file=sys.stderr)
                    return 1
                if not summary.get("config_matches_recorded", True):
                    print(f"WARNING: replay config hash differs from recorded run; recorded={summary.get('run_config_sha256')} replay={summary.get('replay_config_sha256')}", file=sys.stderr)
            return int(result.returncode)
        elif args.command == "resume":
            source = _find_run(args.run, root); checkpoint = source / "latest.pt"
            if not checkpoint.is_file(): raise ValueError(f"No latest.pt checkpoint in {source}")
            config = _resolve_config_for_run(source, None); _load_config(config)
            output = args.output.expanduser().resolve() if args.output else root / f"{source.name}-resume-{time.strftime('%Y%m%d-%H%M%S')}"
            if output.exists(): raise ValueError(f"Resume output already exists: {output}")
            python_cmd = _isaac_python_command(preview=args.preview)
            cmd = [python_cmd, "-m", "casu_grid.workbench.lab", "--experiments", str(root), "launch", str(config), "--run-dir", str(output), "--updates", str(args.updates), "--resume", str(checkpoint)]
            source_manifest = _json(source / "config.json")
            experiment_meta = source_manifest.get("experiment") or {}
            is_engineering = (source_manifest.get("run_kind") == "engineering_smoke"
                              or "engineering-only" in json.dumps(experiment_meta).lower()
                              or config.resolve() == TEMPLATE.resolve())
            if is_engineering:
                cmd.append("--engineering-smoke")
            print(f"Resume from exact checkpoint: {checkpoint}\nNew output folder: {output}\nCommand: " + _display_command(cmd))
            if args.preview: return 0
            if importlib.util.find_spec("isaaclab") is None:
                _isaac_python_command()
            return int(subprocess.run(cmd, cwd=ROOT).returncode)
        elif args.command == "tensorboard":
            run = _find_run(args.run, root); logdir = run / "tensorboard"
            if not logdir.is_dir(): raise ValueError(f"TensorBoard event directory missing: {logdir}")
            if importlib.util.find_spec("tensorboard") is None:
                raise ValueError("TensorBoard is not installed in this Python; use the Isaac bundled runtime or install the existing workbench requirements.")
            command = [sys.executable, "-m", "tensorboard.main", "--logdir", str(logdir), "--host", "127.0.0.1", "--port", str(args.port)]
            url = f"http://127.0.0.1:{args.port}"
            print(f"Local-only TensorBoard: {url} (Ctrl-C to stop)")
            if args.open:
                import webbrowser
                webbrowser.open(url)
            return int(subprocess.run(command, cwd=ROOT).returncode)
        elif args.command == "menu":
            print("CasU Isaac research workbench — guided command menu")
            print("Operations: about | guide | new | validate | launch | configs | runs | status | watch | compare | plot | replay | resume | quit")
            while True:
                try: line = input("lab> ").strip()
                except (EOFError, KeyboardInterrupt): print(); break
                if line in {"quit", "exit", "q"}: break
                if not line: continue
                try:
                    code = main(["--experiments", str(root), *shlex.split(line)])
                    if code: print(f"Command returned {code}")
                except SystemExit as exc:
                    if exc.code not in (0, None): print(f"Command returned {exc.code}")
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"casu-lab: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
