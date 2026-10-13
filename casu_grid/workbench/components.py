"""Explicit, subprocess-isolated researcher component registry for the workbench.

Component modules are trusted Python. Discovery reads a JSON sidecar only;
validation imports the selected source in a short-lived worker process.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

CONTRACT_VERSION = 1


def _metadata_path(source: Path) -> Path:
    return source.with_suffix(source.suffix + ".component.json")


def discover_components(directory: Path) -> list[dict[str, Any]]:
    root = directory.expanduser().resolve()
    if not root.exists():
        return []
    found = []
    for metadata_path in sorted(root.glob("*.py.component.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            source = (metadata_path.parent / metadata_path.name.removesuffix(".component.json")).resolve()
            if source.parent != root or not source.is_file():
                raise ValueError("component source is missing or outside the selected library")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            found.append({"metadata": metadata, "source": str(source), "sha256": digest,
                          "stale": metadata.get("source_sha256") not in (None, digest),
                          "metadata_path": str(metadata_path)})
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            found.append({"source": str(metadata_path), "error": str(exc), "stale": True})
    return found


def _validate_schema(schema: list[dict[str, Any]], parameters: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, list) or not isinstance(parameters, dict):
        raise ValueError("parameter_schema and parameters must be a list and object")
    result = {}
    names = set()
    allowed = {"float", "integer", "boolean", "enum", "optional"}
    for item in schema:
        if not isinstance(item, dict) or not item.get("name"):
            raise ValueError("each parameter requires a name and schema")
        name, kind = item["name"], item.get("type")
        if name in names:
            raise ValueError(f"duplicate parameter: {name}")
        names.add(name)
        if kind not in allowed:
            raise ValueError(f"unsupported parameter type for {name}: {kind}")
        value = parameters.get(name, item.get("default"))
        if kind == "optional" and value is None:
            result[name] = None
            continue
        if kind == "optional":
            kind=item.get("value_type")
            if kind not in {"float","integer","boolean","enum"}:
                raise ValueError(f"optional parameter {name} requires value_type float, integer, boolean, or enum")
        if kind == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number")
            value = float(value)
        elif kind == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        elif kind == "boolean":
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
        elif kind == "enum" and value not in item.get("choices", []):
            raise ValueError(f"{name} must be one of {item.get('choices', [])}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if item.get("minimum") is not None and value < item["minimum"]:
                raise ValueError(f"{name} must be >= {item['minimum']}")
            if item.get("maximum") is not None and value > item["maximum"]:
                raise ValueError(f"{name} must be <= {item['maximum']}")
        result[name] = value
    extras = set(parameters) - names
    if extras:
        raise ValueError(f"undeclared parameters: {sorted(extras)}")
    missing = [x["name"] for x in schema if x["name"] not in result and "default" not in x]
    if missing:
        raise ValueError(f"missing parameters without defaults: {missing}")
    return result


def _worker(source: Path, entry_point: str, parameters: dict[str, Any], expected_metadata: dict[str, Any]) -> dict[str, Any]:
    import importlib.util
    spec = importlib.util.spec_from_file_location("casu_selected_reward", source)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load selected module {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata = getattr(module, "REWARD_COMPONENT", None)
    callback = getattr(module, entry_point, None)
    if not isinstance(metadata, dict):
        raise ValueError("module must declare REWARD_COMPONENT metadata")
    for key in ("id", "name", "description", "contract_version", "parameter_schema", "required_fields", "output", "deterministic"):
        if key not in metadata:
            raise ValueError(f"metadata missing {key}")
    if metadata["contract_version"] != CONTRACT_VERSION:
        raise ValueError(f"unsupported contract version: {metadata['contract_version']}")
    for key in ("id", "contract_version", "entry_point", "required_fields", "parameter_schema", "reward_terms", "output"):
        if key in expected_metadata and metadata.get(key) != expected_metadata[key]:
            raise ValueError(f"sidecar metadata field {key!r} differs from the Python module declaration")
    if metadata.get("stateful", False):
        raise ValueError("stateful reward components are not supported until the environment defines per-world reset and clone semantics")
    if not callable(callback):
        raise ValueError(f"entry point {entry_point!r} is not callable")
    if metadata["output"] != {"type": "scalar_and_terms", "shape": []}:
        raise ValueError("reward output must declare scalar_and_terms with scalar shape []")
    params = _validate_schema(metadata["parameter_schema"], parameters)
    declared_terms = metadata.get("reward_terms")
    if declared_terms is not None and (not isinstance(declared_terms, list) or any(not isinstance(x, str) or not x for x in declared_terms) or len(set(declared_terms)) != len(declared_terms)):
        raise ValueError("reward_terms must be a list of unique, non-empty names")
    import inspect
    signature = inspect.signature(callback)
    try:
        signature.bind(None, None, False, False, **params)
    except TypeError as exc:
        raise ValueError(f"entry point must accept (previous, current, terminated, truncated, **parameters): {exc}") from exc
    context=metadata.get("synthetic_context", {})
    if not isinstance(context,dict) or set(context)-{"previous","current","terminated","truncated"}:
        raise ValueError("synthetic_context must contain only previous/current/terminated/truncated")
    previous=context.get("previous",{});current=context.get("current",{})
    if not isinstance(previous,dict) or not isinstance(current,dict):
        raise ValueError("synthetic_context previous/current must be state dictionaries")
    missing=set(metadata["required_fields"])-set(previous)-set(current)
    if missing:raise ValueError(f"synthetic_context lacks declared required fields: {sorted(missing)}")
    terminated=bool(context.get("terminated",False));truncated=bool(context.get("truncated",False))
    first = callback(previous, current, terminated, truncated, **params)
    second = callback(previous, current, terminated, truncated, **params)
    for value in (first, second):
        if not isinstance(value, tuple) or len(value) != 2 or not isinstance(value[1], dict):
            raise ValueError("entry point must return (finite_scalar_reward, reward_terms_dict)")
        scalar, terms = value
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float)) or not math.isfinite(float(scalar)):
            raise ValueError("reward scalar must be finite")
        if any(not isinstance(k, str) or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for k, v in terms.items()):
            raise ValueError("reward terms must have string keys and finite numeric scalar values")
    if metadata["deterministic"] and first != second:
        raise ValueError("component declared deterministic but repeated synthetic calls differ")
    if declared_terms is not None and sorted(first[1]) != sorted(declared_terms):
        raise ValueError(f"synthetic reward term names {sorted(first[1])} do not match declared reward_terms {sorted(declared_terms)}")
    return {"metadata": metadata, "parameters": params, "synthetic_result": first,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}


def validate_component(source: Path, metadata: dict[str, Any], parameters: dict[str, Any], *, timeout: float = 5.0) -> dict[str, Any]:
    """Import and synthetic-check one explicitly selected module out of process."""
    source = source.expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"component source does not exist: {source}")
    entry_point = metadata.get("entry_point", "reward")
    payload = json.dumps({"source": str(source), "entry_point": entry_point,
                          "parameters": parameters,"metadata":metadata}, allow_nan=False)
    project_root = Path(__file__).resolve().parents[2]
    code = "import json,sys; from pathlib import Path; sys.path.insert(0, " + repr(str(project_root)) + "); from casu_grid.workbench.components import _worker; x=json.load(sys.stdin); print(json.dumps(_worker(Path(x['source']),x['entry_point'],x['parameters'],x['metadata']), allow_nan=False))"
    env = os.environ.copy()
    try:
        completed = subprocess.run([sys.executable, "-I", "-c", code], input=payload, text=True,
                                   capture_output=True, timeout=timeout, env=env, cwd=source.parent)
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"component validation timed out after {timeout:g}s") from exc
    if completed.returncode:
        raise ValueError("component worker failed: " + (completed.stderr.strip() or completed.stdout.strip()))
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"component worker returned invalid output: {completed.stdout[:300]}") from exc
    declared = metadata.get("source_sha256")
    if declared and declared != result["source_sha256"]:
        raise ValueError("component source changed since metadata was declared")
    return result


def resolve_reward_reference(config_module: Any, config_path: Path, callbacks: tuple) -> tuple:
    """Compose an explicitly declared reward reference with existing callbacks."""
    reference = getattr(config_module, "REWARD_COMPONENT_REF", None)
    if reference is None:
        return callbacks
    if not isinstance(reference, dict) or len(callbacks) != 3:
        raise ValueError("REWARD_COMPONENT_REF must be a mapping and callbacks() must return its standard triple")
    config_parent=config_path.expanduser().resolve().parent
    source = (config_parent / reference["source"]).resolve()
    if reference.get("snapshot_source"):
        snapshot_source=(config_parent / reference["snapshot_source"]).resolve()
        if snapshot_source.is_file():source=snapshot_source
    expected = reference.get("source_sha256")
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if not expected or actual != expected:
        raise ValueError(f"reward component source hash mismatch for {source}; compatibility validation is stale")
    spec = importlib.util.spec_from_file_location("casu_selected_reward_runtime", source)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load selected reward component: {source}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    metadata = getattr(module, "REWARD_COMPONENT", {})
    if metadata.get("id") != reference.get("id") or metadata.get("contract_version") != reference.get("contract_version"):
        raise ValueError("reward component identity or contract version changed since experiment selection")
    callback = getattr(module, reference.get("entry_point", "reward"), None)
    if not callable(callback): raise ValueError("selected reward entry point is missing")
    params = reference.get("parameters", {})
    expected_terms = set(reference.get("reward_terms", []))
    if not expected_terms: raise ValueError("selected component reference has no declared reward terms")

    def reward(previous, current, terminated, truncated):
        value = callback(previous, current, terminated, truncated, **params)
        if not isinstance(value, tuple) or len(value) != 2 or not isinstance(value[1], dict):
            raise ValueError("selected reward must return (scalar, reward_terms)")
        scalar, terms = value
        if not isinstance(scalar, (int, float)) or isinstance(scalar, bool) or not math.isfinite(float(scalar)):
            raise ValueError("selected reward returned a non-finite scalar")
        if set(terms) != expected_terms:
            raise ValueError(f"selected reward term names {sorted(terms)} differ from declared terms {sorted(expected_terms)}")
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)) for v in terms.values()):
            raise ValueError("selected reward returned non-finite term values")
        return float(scalar), {k: float(v) for k, v in terms.items()}

    return callbacks[0], callbacks[1], reward


def snapshot_reward_reference(reference: dict[str, Any], config_path: Path, run_dir: Path) -> dict[str, Any]:
    """Copy selected source into the immutable run directory and store provenance."""
    import shutil
    config_path=config_path.expanduser().resolve();run_dir=run_dir.expanduser().resolve()
    source=(config_path.parent/reference["source"]).resolve()
    if reference.get("snapshot_source"):
        candidate=(config_path.parent/reference["snapshot_source"]).resolve()
        if candidate.is_file():source=candidate
    actual=hashlib.sha256(source.read_bytes()).hexdigest()
    if actual!=reference.get("source_sha256"):
        raise ValueError(f"reward component changed before run snapshot: {source}")
    destination=run_dir/"components"/(actual+".py")
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():shutil.copy2(source,destination)
    result=dict(reference);result["snapshot_source"]=str(Path("components")/destination.name)
    return result
