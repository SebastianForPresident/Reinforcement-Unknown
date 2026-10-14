"""Run discovery and incremental metrics reading.

Nothing here imports Qt, so it can be tested without a display. Runs are directories written by ``casu_grid.workbench.train`` (via
``lab launch``). Status is derived from the process that owns the run (the ``RUNNING`` marker holds the launcher's pid) together with
the artifacts it left, never from a command merely having returned.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:                                    # a zombie is not a running process
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        return state != "Z"
    except (OSError, IndexError):
        return True


def last_json_line(path: Path, max_back: int = 262144) -> dict | None:
    """Last complete JSON object of a JSON-lines file without reading the whole file."""
    try:
        size = path.stat().st_size
        if size == 0:
            return None
        with path.open("rb") as handle:
            handle.seek(max(0, size - max_back))
            chunk = handle.read()
    except OSError:
        return None
    for line in reversed(chunk.splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


@dataclass
class CheckpointInfo:
    path: Path
    kind: str                      # initial | scheduled | latest | other
    update: int | None
    decisions: int | None
    saved_utc: str | None
    size_bytes: int
    fingerprint: str | None
    world_partition: list[int] | None
    mtime: float
    termination_contract: str = "xy1-termination-v1"      # checkpoints written before contract ids are the reference XY1

    @property
    def label(self) -> str:
        if self.kind == "initial":
            return "initial (untrained)"
        if self.kind == "latest":
            return f"latest (update {self.update})" if self.update is not None else "latest"
        return f"update {self.update}" if self.update is not None else self.path.stem


def checkpoint_info(path: Path, kind: str | None = None) -> CheckpointInfo:
    meta = read_json(path.with_name(path.name + ".json")) or {}
    name = path.stem
    kind = kind or ("initial" if name == "initial" else "latest" if name == "latest" else "scheduled" if name.startswith("update_") else "other")
    try:
        stat = path.stat()
        size, mtime = stat.st_size, stat.st_mtime
    except OSError:
        size, mtime = 0, 0.0
    update = meta.get("update_count")
    if update is None and name.startswith("update_"):
        try:
            update = int(name.split("_")[1])
        except ValueError:
            pass
    return CheckpointInfo(path, kind, update, meta.get("total_steps"), meta.get("saved_utc"), size, meta.get("contract_fingerprint"),
                          meta.get("world_partition"), mtime, ((meta.get("environment_contract") or {}).get("termination")) or "xy1-termination-v1")


@dataclass
class RunRecord:
    path: Path
    run_id: str
    status: str = "unknown"
    experiment: str = ""
    config_path: str | None = None
    config_sha256: str | None = None
    created_utc: str | None = None
    revision: str | None = None
    seed: int | None = None
    num_envs: int | None = None
    workers: int | None = None
    physx_threads: int | None = None
    device: str | None = None
    run_kind: str | None = None
    resume_from: str | None = None
    updates_done: int = 0
    decisions: int = 0
    last_metrics: dict | None = None
    checkpoints: list[CheckpointInfo] = field(default_factory=list)
    failure: dict | None = None
    has_metrics: bool = False
    evaluations: list[Path] = field(default_factory=list)
    mtime: float = 0.0
    updates_requested: int | None = None
    legacy_spawn: bool = False
    start_update: int = 0           # >0 for a resumed run: the update of the checkpoint it continued from
    start_decisions: int = 0
    termination_contract: str = "xy1-termination-v1"

    @property
    def updates_in_run(self) -> int:
        return max(0, self.updates_done - self.start_update)

    @property
    def latest(self) -> CheckpointInfo | None:
        return next((c for c in self.checkpoints if c.kind == "latest"), None)

    @property
    def is_engineering_smoke(self) -> bool:
        return self.run_kind == "engineering_smoke" or (self.updates_done > 0 and self.updates_done < 10 and self.run_kind != "research_training")

    @property
    def title(self) -> str:
        return f"{self.experiment or 'run'} · {self.run_id}"


def derive_status(path: Path, manifest: dict | None, has_rows: bool) -> str:
    marker = path / "RUNNING"
    alive = False
    if marker.is_file():
        try:
            alive = pid_alive(int((marker.read_text().strip() or "0").splitlines()[0]))
        except (OSError, ValueError):
            alive = False
    if (path / "failure.json").exists():
        return "failed"
    if alive:
        if (path / "STOP").exists():
            return "stopping"
        return "running" if has_rows else "starting"
    if (path / "INTERRUPTED").exists():
        return "interrupted"
    if (path / "COMPLETE").exists():
        return "completed"
    if marker.is_file():                        # owner died without leaving an outcome
        return "interrupted" if has_rows else "failed"
    status = (manifest or {}).get("status")
    if status in ("failed", "interrupted", "completed"):
        return status
    if status == "starting":
        return "failed"                         # never got going and nothing owns it
    return "interrupted" if has_rows else "unknown"


def load_run(path: Path) -> RunRecord | None:
    manifest = read_json(path / "config.json")
    if not isinstance(manifest, dict) or manifest.get("schema") != "casu.training-run.v1":
        return None
    environment = read_json(path / "environment.json") or {}
    performance = environment.get("performance", {}) if isinstance(environment, dict) else {}
    execution = environment.get("execution", {}) if isinstance(environment, dict) else {}
    last = last_json_line(path / "metrics.jsonl")
    contracts_env = manifest.get("environment", {}) or {}
    experiment = (manifest.get("experiment") or {}).get("name") or Path(manifest.get("config_path") or "").stem
    run = RunRecord(
        path=path, run_id=path.name, experiment=experiment, config_path=manifest.get("config_path"), config_sha256=manifest.get("config_sha256"),
        created_utc=manifest.get("created_utc"), revision=manifest.get("git_revision"), seed=manifest.get("seed"),
        num_envs=contracts_env.get("num_envs") or environment.get("num_envs"),
        workers=performance.get("num_workers") or execution.get("effective_workers"),
        physx_threads=performance.get("physx_threads_per_worker"), device=(manifest.get("execution_requested") or {}).get("device"),
        run_kind=manifest.get("run_kind"), resume_from=manifest.get("resume_from") or None, last_metrics=last, has_metrics=last is not None,
        failure=read_json(path / "failure.json"), updates_requested=(manifest.get("launch") or {}).get("updates"))
    if last:
        run.updates_done = int(last.get("update") or 0)
        run.decisions = int(last.get("environment_steps") or 0)
    if run.resume_from:
        origin = read_json(Path(run.resume_from).with_name(Path(run.resume_from).name + ".json")) or {}
        run.start_update = int(origin.get("update_count") or 0)
        run.start_decisions = int(origin.get("total_steps") or 0)
    run.termination_contract = contracts_env.get("termination_contract") or "xy1-termination-v1"
    run.status = derive_status(path, manifest, last is not None)
    checkpoints = []
    for candidate in [path / "latest.pt"]:
        if candidate.is_file():
            checkpoints.append(checkpoint_info(candidate, "latest"))
    folder = path / "checkpoints"
    if folder.is_dir():
        for candidate in sorted(folder.glob("*.pt")):
            checkpoints.append(checkpoint_info(candidate))
    run.checkpoints = checkpoints
    evals = path / "evaluations"
    run.evaluations = sorted(evals.glob("*.json")) if evals.is_dir() else []
    launch = read_json(path / "studio_launch.json") or {}
    if launch:
        run.device = launch.get("device") or run.device
        run.physx_threads = run.physx_threads or launch.get("physx_threads")
        run.workers = run.workers or launch.get("workers")
        run.updates_requested = launch.get("updates") or run.updates_requested
        run.legacy_spawn = bool(launch.get("legacy_spawn"))
    try:
        run.mtime = max((p.stat().st_mtime for p in (path / "metrics.jsonl", path / "config.json") if p.exists()), default=path.stat().st_mtime)
    except OSError:
        run.mtime = 0.0
    return run


def discover_runs(root: Path) -> list[RunRecord]:
    runs = []
    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir():
                try:
                    run = load_run(child)
                except Exception:           # a half-written or foreign directory must never break the run list
                    run = None
                if run is not None:
                    runs.append(run)
    runs.sort(key=lambda r: r.created_utc or "", reverse=True)
    return runs


class MetricsStore:
    """Columnar, append-only view of a ``metrics.jsonl`` that is read incrementally.

    ``refresh`` reads only the bytes appended since the last call and keeps an incomplete final line for next time, so polling a
    multi-hour run costs the same as polling a new one. Rows that do not parse are counted and skipped, never fatal.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._offset = 0
        self._tail = b""
        self._inode: tuple[int, int] | None = None
        self.rows = 0
        self.columns: dict[str, list] = {}
        self.bad_lines = 0

    def reset(self):
        self._offset, self._tail, self.rows, self.columns, self.bad_lines = 0, b"", 0, {}, 0

    def refresh(self) -> int:
        try:
            stat = self.path.stat()
        except OSError:
            return 0
        ident = (stat.st_ino, stat.st_dev)
        if self._inode != ident or stat.st_size < self._offset:
            self.reset()
            self._inode = ident
        if stat.st_size == self._offset:
            return 0
        try:
            with self.path.open("rb") as handle:
                handle.seek(self._offset)
                data = handle.read()
        except OSError:
            return 0
        self._offset += len(data)
        data = self._tail + data
        lines = data.split(b"\n")
        self._tail = lines.pop()            # incomplete last line (or b"" when the file ends with a newline)
        added = 0
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                self.bad_lines += 1
                continue
            if not isinstance(row, dict):
                self.bad_lines += 1
                continue
            for key in row:
                if key not in self.columns:
                    self.columns[key] = [None] * self.rows
            for key, column in self.columns.items():
                value = row.get(key)
                column.append(value if isinstance(value, (int, float)) and not isinstance(value, bool) else (None if value is None else value))
            self.rows += 1
            added += 1
        return added

    def column(self, key: str) -> list:
        return self.columns.get(key, [None] * self.rows)

    def has(self, key: str) -> bool:
        return key in self.columns and any(v is not None for v in self.columns[key])


class SystemStore(MetricsStore):
    """Samples written by the Studio's monitor while it supervises a run (memory, GPU, process health)."""
    pass


def human_duration(seconds: float | None) -> str:
    if seconds is None or seconds != seconds:
        return "–"
    seconds = int(max(0, seconds))
    if seconds < 90:
        return f"{seconds} s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 90:
        return f"{minutes} min {sec:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


def human_count(value: float | int | None) -> str:
    if value is None:
        return "–"
    value = float(value)
    for limit, suffix in ((1e9, " G"), (1e6, " M"), (1e4, " k")):
        if abs(value) >= limit:
            return f"{value / limit:.2f}{suffix}"
    return f"{int(round(value)):,}"


def human_bytes(value: float | int | None) -> str:
    if value is None:
        return "–"
    value = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return str(value)


def format_time_ago(timestamp: float | None) -> str:
    if not timestamp:
        return "–"
    delta = time.time() - timestamp
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta / 60)} min ago"
    if delta < 86400:
        return f"{int(delta / 3600)} h ago"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(timestamp))
