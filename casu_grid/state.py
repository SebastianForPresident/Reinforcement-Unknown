"""Discrete grid rules transcribed from SearchBase WorldGeneration/FluidManager.

Grid arrays are x-first, matching C# ``array[x, y]``. Rendering, Unity
colliders, loot/audio, and rigidbody effects are intentionally out of scope.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any, Iterable

import numpy as np

from . import backend as _backend


BLOCK_HEALTH = {
    0: 0.0, 1: 100.0, 2: 25.0, 3: 60.0, 4: 20.0, 5: 800.0,
    6: 5000.0, 7: 30.0, 8: 60.0, 9: 150.0, 10: 15000.0,
    11: 150.0, 12: 15.0, 13: 90.0, 14: 420133760.0, 15: 25.0,
    16: 32.0, 17: 200.0, 18: 150.0, 19: 135.0, 20: 650.0,
    21: 200.0, 22: 250.0, 23: 35.0, 24: 150.0, 25: 20.0,
    26: 15.0, 27: 50.0, 28: 1.0, 29: 1.0, 30: 200.0,
    31: 50.0, 32: 80.0, 33: 60.0, 34: 2000.0, 35: 4000.0,
}
METALLIC_BLOCKS = frozenset((6, 10, 21))
SLIPPERY_BLOCKS = frozenset((27, 28))
TOXICITY = {22: 2.5}


@dataclass
class BlockDamage:
    x: int
    y: int
    damage: float


class DecisionStream:
    """Replayable state-affecting random decisions.

    Each decision is a bool meaning Unity Random.value > 0.5 (horizontal
    right choice) or Random.value < 0.5 (ice water drop), depending on caller.
    This abstracts UnityEngine.Random's algorithm; it does not claim seed parity.
    """

    def __init__(self, decisions: Iterable[bool] = ()):
        self.decisions = list(bool(x) for x in decisions)
        self.position = 0

    def next_bool(self) -> bool:
        if self.position >= len(self.decisions):
            raise RuntimeError("decision stream exhausted")
        result = self.decisions[self.position]
        self.position += 1
        return result

    def snapshot(self) -> dict[str, Any]:
        return {"decisions": self.decisions.copy(), "position": self.position}

    @classmethod
    def from_snapshot(cls, snap: dict[str, Any]) -> "DecisionStream":
        obj = cls(snap["decisions"])
        obj.position = snap["position"]
        return obj


class SeededDecisionStream:
    """On-demand deterministic PCG64 decisions with compact restorable state.

    This is a project-owned simulator stream, deliberately not Unity RNG parity.
    """
    def __init__(self, seed: int):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.position = 0

    def next_bool(self) -> bool:
        self.position += 1
        return bool(self.rng.integers(0, 2))

    _MASK64 = (1 << 64) - 1

    def export_state(self, out) -> None:
        """Fill a uint64[6] array ``[state_lo, state_hi, inc_lo, inc_hi, has_uint32, uinteger]`` (compiled kernels)."""
        st = self.rng.bit_generator.state
        s, inc = st["state"]["state"], st["state"]["inc"]
        out[0], out[1], out[2], out[3] = s & self._MASK64, s >> 64, inc & self._MASK64, inc >> 64
        out[4], out[5] = st["has_uint32"], st["uinteger"]

    def import_state(self, arr) -> None:
        st = self.rng.bit_generator.state
        st["state"]["state"] = int(arr[0]) | (int(arr[1]) << 64)
        st["has_uint32"], st["uinteger"] = int(arr[4]), int(arr[5])
        self.rng.bit_generator.state = st

    def snapshot(self) -> dict[str, Any]:
        return {"algorithm": "numpy.PCG64", "seed": self.seed,
                "position": self.position, "state": copy.deepcopy(self.rng.bit_generator.state)}

    @classmethod
    def from_snapshot(cls, snap: dict[str, Any]) -> "SeededDecisionStream":
        if snap.get("algorithm") != "numpy.PCG64":
            raise ValueError("unsupported seeded decision-stream algorithm")
        obj = cls(snap["seed"])
        obj.rng.bit_generator.state = copy.deepcopy(snap["state"])
        obj.position = int(snap["position"])
        return obj


class CasUGridState:
    """Discrete terrain/fluid state with configurable full-world dimensions."""

    def __init__(self, width: int = 65, height: int = 64,
                 camera_x: int | None = None, camera_y: int | None = None,
                 decisions: Iterable[bool] = (), backend: str | None = None):
        if width < 3 or height < 3:
            raise ValueError("dimensions must be >= 3 for source clamping rules")
        self.width, self.height = int(width), int(height)
        self.blocks = np.zeros((width, height), dtype=np.uint16)
        self.fluids = np.zeros((width, height), dtype=np.uint8)
        self.block_damage: list[BlockDamage] = []
        self.damage_capacity = 128
        self.sim_index = 0
        self.tile_cooldown = 0
        self.water_move_count = 0
        self.tick = 0
        self.camera_x = width // 2 if camera_x is None else int(camera_x)
        self.camera_y = height // 2 if camera_y is None else int(camera_y)
        self.rng = DecisionStream(decisions)
        self.fluid_push_enabled = False
        self.push_events = 0
        self.last_chunk_update: tuple[int, int] | None = None
        self.block_revision = 0
        self.dirty_chunks: set[tuple[int, int]] = set()
        # compiled fluid sweep (None -> readable reference implementation); see casu_grid.backend
        self._k = _backend.kernels_for(backend)
        self._counters = np.zeros(3, dtype=np.int64)
        self._rstate = np.zeros(6, dtype=np.uint64)

    @staticmethod
    def world_to_cell(x: float, y: float, width: int, height: int) -> tuple[int, int]:
        # C# cast truncates toward zero (unlike floor for negative values).
        return int(x + width // 2), int(y + height // 2)

    def block_info(self, block_id: int) -> dict[str, Any] | None:
        if block_id not in BLOCK_HEALTH:
            return None
        return {"health": BLOCK_HEALTH[block_id],
                "metallic": block_id in METALLIC_BLOCKS,
                "slippery": block_id in SLIPPERY_BLOCKS,
                "toxicity": TOXICITY.get(block_id, 0.0)}

    def get_block(self, x: int, y: int) -> int:
        return int(self.blocks[min(max(x, 0), self.width - 1),
                               min(max(y, 0), self.height - 1)])

    def set_block(self, x: int, y: int, block_id: int) -> None:
        x = min(max(int(x), 0), self.width - 1)
        y = min(max(int(y), 0), self.height - 1)
        if not 0 <= block_id <= 65535:
            raise ValueError("block ID must fit uint16")
        self.blocks[x, y] = block_id
        self.last_chunk_update = (x // 64, y // 64)
        self.dirty_chunks.add(self.last_chunk_update)
        self.block_revision += 1

    def get_damage(self, x: int, y: int) -> BlockDamage | None:
        return next((d for d in self.block_damage if d.x == x and d.y == y), None)

    def damage_block(self, x: int, y: int, amount: float,
                     bonus_metal: bool = False, *,
                     ignore_loot: bool = False,
                     ice_drop: bool | None = None) -> bool:
        """Apply discrete portion of DamageBlock; return whether destroyed.

        ``ice_drop`` represents the original loot-gated Random.value < .5
        outcome. Supply it explicitly only when reproducing that branch.
        """
        block_id = self.get_block(x, y)
        info = self.block_info(block_id)
        if info is None:
            raise ValueError(f"unknown block ID {block_id}")
        multiplier = 10.0 if bonus_metal and info["metallic"] else 1.0
        # Unity's BlockDamage.damage and input damage are C# float values.
        effective = float(np.float32(np.float32(amount) * np.float32(multiplier)))
        record = self.get_damage(x, y)
        if record is None:
            record = BlockDamage(x, y, effective)
            self.block_damage.append(record)
            if len(self.block_damage) > self.damage_capacity:
                # Source forces first record to 1e10, updates visual, then removes it.
                self.block_damage.pop(0)
        else:
            record.damage = float(np.float32(record.damage + effective))
        if record.damage >= info["health"]:
            drop_ice_water = False
            if block_id == 27 and not ignore_loot:
                # The source performs Random.value < .5 inside its loot switch.
                # Other audiovisual/loot RNG draws are intentionally abstracted.
                drop_ice_water = ice_drop if ice_drop is not None else not self.rng.next_bool()
            if drop_ice_water:
                # The original write occurs only inside !ignoreLoot's drop switch.
                self.set_liquid(x, y, 1)
            self.set_block(x, y, 0)
            self.block_damage.remove(record)
            return True
        return False

    def get_liquid(self, x: int, y: int) -> int:
        return int(self.fluids[min(max(x, 0), self.width - 1),
                               min(max(y, 0), self.height - 1)])

    def set_liquid(self, x: int, y: int, fluid_id: int) -> None:
        if not 0 <= fluid_id <= 6:
            raise ValueError("fluid ID must be in 0..6")
        self.fluids[min(max(x, 0), self.width - 1),
                    min(max(y, 0), self.height - 1)] = fluid_id

    def empty(self, x: int, y: int) -> bool:
        return self.get_liquid(x, y) == 0 and self.get_block(x, y) == 0

    def _clamp_range(self, lo: int, hi: int, size: int) -> tuple[int, int]:
        # Mirrors SimulationRangeIndex's sequential clamps exactly, including
        # max=min(size-2) and the exclusive loop endpoint.
        lo = max(lo, 1)
        hi = max(hi, 1)
        hi = min(hi, size - 2)
        lo = min(lo, size - 2)
        return lo, hi

    def simulation_range_index(self) -> tuple[tuple[int, int], tuple[int, int]]:
        xr = self._clamp_range(self.camera_x - 64, self.camera_x + 64, self.width)
        yr = self._clamp_range(self.camera_y - 64 + self.sim_index,
                               self.camera_y - 48 + self.sim_index, self.height)
        self.sim_index += 16
        if self.sim_index >= 112:
            self.sim_index = 0
        return xr, yr

    def _incr_move(self) -> None:
        if self.fluid_push_enabled:
            self.water_move_count += 1
            if self.water_move_count > 10:
                self.water_move_count = 0
                self.push_events += 1

    def simulation_step(self, *, decision_budget: int | None = None) -> None:
        """Same sweep as :meth:`simulation_step_reference`, visiting only non-empty cells.

        Exactness: columns are processed in ascending x and each column's candidates are
        re-read when the column is reached, so fluid pushed right by an earlier column is
        visited later in the same sweep exactly as in the literal loop; within a column
        every move targets a lower y or a different column, so cells that were empty at
        column start cannot become relevant before their own visit.
        """
        (xmin, xmax), (ymin, ymax) = self.simulation_range_index()
        if self._k is not None and decision_budget is None and isinstance(self.rng, SeededDecisionStream):
            counters = self._counters
            counters[0], counters[1], counters[2] = self.tile_cooldown, self.water_move_count, self.push_events
            self.rng.export_state(self._rstate)
            draws = self._k.fluid_sweep(
                self.fluids, self.blocks, xmin, xmax, ymin, ymax, self.width, self.height,
                counters, self.fluid_push_enabled, self._rstate)
            if draws:
                self.rng.import_state(self._rstate)
            self.rng.position += draws
            self.tile_cooldown, self.water_move_count, self.push_events = (
                int(counters[0]), int(counters[1]), int(counters[2]))
            self.tick += 1
            return
        if not self.fluids[xmin:xmax, ymin:ymax].any():
            self.tick += 1
            return
        for x in range(xmin, xmax):
            column = self.fluids[x, ymin:ymax]
            if not column.any():
                continue
            for y in (np.flatnonzero(column) + ymin).tolist():
                self._sweep_cell(x, y, decision_budget)
        self.tick += 1

    def _sweep_cell(self, x: int, y: int, decision_budget) -> None:
        if self.fluids[x, y] == 0:
            return
        if self.empty(x, y - 1):
            if y <= 2:
                self.fluids[x, y] = 0
            self.fluids[x, y], self.fluids[x, y - 1] = self.fluids[x, y - 1], self.fluids[x, y]
            self.tile_cooldown = (self.tile_cooldown + 1) % 256
            self._incr_move()
            if self.tile_cooldown > 16:
                self.tile_cooldown = 0
            return
        if ((self.fluids[x, y] == 2 and self.get_liquid(x, y - 1) == 1) or
                (self.fluids[x, y] == 1 and self.get_liquid(x, y - 1) == 2)):
            self.fluids[x, y - 1] = 2
            self.fluids[x, y] = 2
        right, left = self.empty(x + 1, y), self.empty(x - 1, y)
        if right and left:
            if decision_budget is not None and self.rng.position >= decision_budget:
                direction = 1  # deterministic fallback beyond replay tape budget
            else:
                direction = 1 if self.rng.next_bool() else -1
            self.fluids[x, y], self.fluids[x + direction, y] = self.fluids[x + direction, y], self.fluids[x, y]
        elif right:
            self.fluids[x, y], self.fluids[x + 1, y] = self.fluids[x + 1, y], self.fluids[x, y]
            self._incr_move()
        elif left:
            self.fluids[x, y], self.fluids[x - 1, y] = self.fluids[x - 1, y], self.fluids[x, y]
            self._incr_move()

    def simulation_step_reference(self, *, decision_budget: int | None = None) -> None:
        """Literal per-cell port of the source sweep; the oracle for :meth:`simulation_step`."""
        (xmin, xmax), (ymin, ymax) = self.simulation_range_index()
        for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                if self.fluids[x, y] == 0:
                    continue
                if self.empty(x, y - 1):
                    if y <= 2:
                        self.fluids[x, y] = 0
                    self.fluids[x, y], self.fluids[x, y - 1] = self.fluids[x, y - 1], self.fluids[x, y]
                    self.tile_cooldown = (self.tile_cooldown + 1) % 256
                    self._incr_move()
                    if self.tile_cooldown > 16:
                        self.tile_cooldown = 0
                    continue
                if ((self.fluids[x, y] == 2 and self.get_liquid(x, y - 1) == 1) or
                    (self.fluids[x, y] == 1 and self.get_liquid(x, y - 1) == 2)):
                    self.fluids[x, y - 1] = 2
                    self.fluids[x, y] = 2
                right, left = self.empty(x + 1, y), self.empty(x - 1, y)
                if right and left:
                    if decision_budget is not None and (
                            self.rng.position >= decision_budget):
                        direction = 1  # deterministic fallback beyond replay tape budget
                    else:
                        direction = 1 if self.rng.next_bool() else -1
                    self.fluids[x, y], self.fluids[x + direction, y] = self.fluids[x + direction, y], self.fluids[x, y]
                elif right:
                    self.fluids[x, y], self.fluids[x + 1, y] = self.fluids[x + 1, y], self.fluids[x, y]
                    self._incr_move()
                elif left:
                    self.fluids[x, y], self.fluids[x - 1, y] = self.fluids[x - 1, y], self.fluids[x, y]
                    self._incr_move()
        self.tick += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "width": self.width, "height": self.height,
            "blocks": self.blocks.copy(), "fluids": self.fluids.copy(),
            "damage": [(d.x, d.y, d.damage) for d in self.block_damage],
            "damage_capacity": self.damage_capacity,
            "sim_index": self.sim_index, "tile_cooldown": self.tile_cooldown,
            "water_move_count": self.water_move_count, "tick": self.tick,
            "camera_x": self.camera_x, "camera_y": self.camera_y,
            "rng": self.rng.snapshot(), "fluid_push_enabled": self.fluid_push_enabled,
            "push_events": self.push_events, "last_chunk_update": self.last_chunk_update,
            "block_revision": self.block_revision,
            "dirty_chunks": sorted(self.dirty_chunks),
        }

    def restore(self, snap: dict[str, Any]) -> None:
        if (snap["width"], snap["height"]) != (self.width, self.height):
            raise ValueError("snapshot dimensions mismatch")
        self.blocks[...] = snap["blocks"]
        self.fluids[...] = snap["fluids"]
        self.block_damage = [BlockDamage(*row) for row in snap["damage"]]
        for name in ("damage_capacity", "sim_index", "tile_cooldown", "water_move_count",
                     "tick", "camera_x", "camera_y", "fluid_push_enabled", "push_events",
                     "last_chunk_update", "block_revision"):
            setattr(self, name, snap[name])
        self.dirty_chunks = set(tuple(chunk) for chunk in snap.get("dirty_chunks", ()))
        if snap["rng"].get("algorithm") == "numpy.PCG64":
            self.rng = SeededDecisionStream.from_snapshot(snap["rng"])
        else:
            self.rng = DecisionStream.from_snapshot(snap["rng"])

    @staticmethod
    def snapshots_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
        aa, bb = copy.deepcopy(a), copy.deepcopy(b)
        for key in ("blocks", "fluids"):
            if not np.array_equal(aa.pop(key), bb.pop(key)):
                return False
        return aa == bb
