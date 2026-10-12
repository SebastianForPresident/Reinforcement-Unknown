"""Import and validate the frozen canonical Casualties Unknown Layer 1."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import CasUGridState


FIXTURE_ID = "casu-layer1-seed-391728405-v1"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / FIXTURE_ID
MANIFEST_NAME = "canonical_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Layer1Reference:
    """Frozen arrays and source metadata; arrays use CLR-compatible ``[x,y]``."""

    manifest: dict[str, Any]
    terrain: np.ndarray
    fluid: np.ndarray

    @property
    def spawn_world_xy(self) -> tuple[float, float]:
        pos = self.manifest["runtime"]["player_world_position_xy"]
        return float(pos[0]), float(pos[1])

    @property
    def spawn_grid_xy(self) -> tuple[int, int]:
        pos = self.manifest["runtime"]["player_spawn_grid_xy"]
        return int(pos[0]), int(pos[1])

    @property
    def transition_y_exclusive(self) -> float:
        return float(self.manifest["world"]["transition_y_exclusive"])

    def new_state(self, decisions=()) -> CasUGridState:
        """Create an independent world; scheduler/camera begin at captured values."""
        world = self.manifest["world"]
        runtime = self.manifest["runtime"]
        scheduler = runtime["fluid_scheduler"]
        camera = runtime["camera_world_xy"]
        state = CasUGridState(world["width"], world["height"],
                              camera_x=int(camera[0] + world["width"] // 2),
                              camera_y=int(camera[1] + world["height"] // 2),
                              decisions=decisions)
        state.blocks[...] = self.terrain
        state.fluids[...] = self.fluid
        state.sim_index = int(scheduler["simIndex"])
        state.tile_cooldown = int(scheduler["tileCooldown"])
        # checkTime is a wall-clock scheduler accumulator, absent from V0's
        # integer-tick mechanics; retain it in manifest metadata, do not invent behavior.
        state.block_damage = []
        return state


def load_layer1_reference(fixture_dir: str | Path = FIXTURE_DIR) -> Layer1Reference:
    root = Path(fixture_dir)
    path = root / MANIFEST_NAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Layer 1 manifest {path}: {exc}") from exc
    if manifest.get("schema") != "casu.canonical-layer1.v1":
        raise ValueError(f"unsupported Layer 1 manifest schema: {manifest.get('schema')!r}")
    if manifest.get("canonical") is not True or manifest.get("fixture_id") != FIXTURE_ID:
        raise ValueError("fixture is not the approved canonical Layer 1")
    content = dict(manifest)
    content_id = content.pop("content_id", None)
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    if content_id != "sha256:" + hashlib.sha256(encoded).hexdigest():
        raise ValueError("canonical manifest content ID mismatch")
    if manifest.get("seed") != 391728405:
        raise ValueError("canonical seed mismatch")
    world = manifest.get("world", {})
    if (world.get("width"), world.get("height"), world.get("chunk_size")) != (1024, 1024, 64):
        raise ValueError("unsupported world dimensions or chunk size")
    if world.get("array_order") != "x-first; flat C-order; y fastest":
        raise ValueError("unsupported array ordering")
    loaded = {}
    expected = {"terrain": ("<u2", 2_097_152), "fluid": ("u1", 1_048_576)}
    for name, (dtype, byte_count) in expected.items():
        spec = manifest.get("payloads", {}).get(name, {})
        if spec.get("dtype") != dtype or spec.get("shape") != [1024, 1024]:
            raise ValueError(f"{name}: unsupported dtype or shape")
        if spec.get("order") != "C" or spec.get("axis_order") != "x,y":
            raise ValueError(f"{name}: unsupported byte or axis order")
        payload = root / spec.get("file", "")
        if not payload.is_file() or payload.stat().st_size != byte_count:
            raise ValueError(f"{name}: missing payload or incorrect byte length")
        digest = _sha256(payload)
        if digest != spec.get("sha256"):
            raise ValueError(f"{name}: SHA-256 mismatch ({digest})")
        arr = np.fromfile(payload, dtype=np.dtype(dtype))
        if arr.size != 1024 * 1024:
            raise ValueError(f"{name}: incorrect element count")
        loaded[name] = arr.reshape((1024, 1024), order="C").copy()
    runtime = manifest.get("runtime", {})
    if runtime.get("fluid_scheduler") != {"checkTime": 0.0, "simIndex": 0, "tileCooldown": 0}:
        raise ValueError("unexpected initial fluid scheduler state")
    if runtime.get("terrain_damage_entries") != []:
        raise ValueError("unexpected initial terrain damage records")
    if runtime.get("camera_world_xy") != [0.0, 508.0]:
        raise ValueError("unexpected camera scan origin")
    if runtime.get("player_spawn_grid_xy") != [512, 1020]:
        raise ValueError("unexpected player spawn grid position")
    collision = manifest.get("collision_metadata", {})
    collision_path = root / collision.get("file", "")
    if not collision_path.is_file() or _sha256(collision_path) != collision.get("sha256"):
        raise ValueError("collision metadata missing or checksum mismatch")
    return Layer1Reference(manifest, loaded["terrain"], loaded["fluid"])


def world_to_grid(x: float, y: float, width: int = 1024,
                  height: int = 1024) -> tuple[int, int]:
    """CasU inverse transform: add half dimensions and truncate toward zero."""
    return int(x + width / 2), int(y + height / 2)


def grid_to_world(x: int, y: int, width: int = 1024,
                  height: int = 1024) -> tuple[float, float]:
    """CasU cell-center transform."""
    return x - width / 2 + 0.5, y - height / 2 + 0.5
