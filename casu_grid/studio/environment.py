"""Where the simulator lives, how to start it, and what this checkout is.

The Studio is an ordinary PySide6 program; training and evaluation run in separate processes under the Isaac Sim bundled Python. This
module finds that Python, builds the process environment it needs (the equivalent of sourcing the project's Isaac environment script),
and answers "which source revision would a run use?" and "is anything that must not influence a real run set?".
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# the commits that make multi-world XY1 training meaningful (Optimization IV); the Studio checks they are ancestors of HEAD
REQUIRED_FIXES = {
    "a7ebcab": "XY1 mini-world spawn and goal placed per world (worlds 1-63 were mis-placed before)",
    "e6e3838": "Layer-1 collider reset restored (non-XY1)",
    "dafc767": "boundary excavation crash fixed",
    "1f5d68b": "vectorised terrain rectangle decomposition (equivalent output)",
}


@dataclass
class IsaacInstall:
    root: Path | None
    sim_root: Path | None
    python_sh: Path | None
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.python_sh is not None and not self.problems


def discover_isaac(override: str | None = None) -> IsaacInstall:
    """Locate Isaac Sim: explicit setting, then ``ISAAC_SIM``, then the project's conventional install location."""
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    if os.environ.get("ISAAC_SIM"):
        candidates.append(Path(os.environ["ISAAC_SIM"]).expanduser())
    home = Path(os.environ.get("REAL_HOME") or Path.home())
    candidates.append(home / "Applications/IsaacLab-2.1.1-IsaacSim-4.5.0/isaac-sim-4.5.0")
    for candidate in candidates:
        if candidate.is_file() and candidate.name == "python.sh":
            candidate = candidate.parent
        if (candidate / "python.sh").is_file():
            root = candidate.parent if (candidate.parent / "IsaacLab-2.1.1").is_dir() else None
            problems = []
            if root is None:
                problems.append(f"Isaac Lab 2.1.1 was not found next to {candidate}; expected {candidate.parent / 'IsaacLab-2.1.1'}")
            return IsaacInstall(root, candidate, candidate / "python.sh", problems)
    return IsaacInstall(None, None, None, ["Isaac Sim was not found. Set the Isaac Sim folder in Settings, or the ISAAC_SIM environment variable "
                                           "(see ISAAC_LAB_SETUP.md)."])


def process_environment(install: IsaacInstall, base: dict[str, str] | None = None) -> dict[str, str]:
    """The environment for simulator processes: the project's Isaac setup plus nothing else that could change the experiment."""
    env = dict(base if base is not None else os.environ)
    if install.root is None or install.sim_root is None:
        return env
    root = install.root
    env["REAL_HOME"] = env.get("REAL_HOME") or env.get("HOME", str(Path.home()))
    env["ISAAC_SIM"] = str(install.sim_root)
    env["LD_LIBRARY_PATH"] = f"{root / 'compat-libs/usr/lib64'}" + (f":{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else "")
    parts = [root / "isaaclab-deps", root / "IsaacLab-2.1.1/source/isaaclab", root / "IsaacLab-2.1.1/source/isaaclab_tasks",
             root / "IsaacLab-2.1.1/source/isaaclab_assets"]
    existing = [p for p in env.get("PYTHONPATH", "").split(":") if p and str(ROOT) != p]
    env["PYTHONPATH"] = ":".join([str(ROOT)] + [str(p) for p in parts] + existing)
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    env["HEADLESS"] = "1"
    env["HOME"] = str(root / "user-home")
    env["XDG_CACHE_HOME"] = str(root / "cache")
    env["XDG_CONFIG_HOME"] = str(root / "config")
    return env


# environment variables that change the experiment or its execution; real training must not have them set unnoticed
CASU_EXECUTION_VARIABLES = ("CASU_BACKEND", "CASU_NUM_WORKERS", "CASU_WORKER_PHYSX_THREADS", "CASU_TRAIN_DEVICE", "CASU_PPO_EXECUTION",
                            "CASU_PPO_GRAPH", "CASU_LOCAL_ROLLOUT", "CASU_SHAPE_BATCH", "CASU_INCREMENTAL_TERRAIN", "CASU_DIRECT_TENSOR_IO",
                            "CASU_DIRECT_STEP", "CASU_COMPILED_CONTROLLER", "CASU_PLAYER_COLLIDER", "CASU_SHAPE_SPEC", "CASU_PPO_CLIP_STAT")


def clean_child_environment(env: dict[str, str]) -> dict[str, str]:
    """Remove inherited execution overrides: the Studio sets every execution parameter explicitly from the visible configuration."""
    cleaned = dict(env)
    for name in CASU_EXECUTION_VARIABLES:
        cleaned.pop(name, None)
    return cleaned


def inherited_overrides(env: dict[str, str] | None = None) -> dict[str, str]:
    env = env if env is not None else os.environ
    return {name: env[name] for name in (*CASU_EXECUTION_VARIABLES, "CASU_LEGACY_MINI_SPAWN", "CASU_REPRODUCE_LEGACY") if name in env}


def _git(*args: str, cwd: Path = ROOT, timeout: float = 10.0) -> tuple[int, str]:
    git = shutil.which("git")
    if git is None:
        return 127, ""
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    try:
        result = subprocess.run([git, *args], cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
        return result.returncode, result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return 1, ""


@dataclass
class SourceInfo:
    root: Path
    revision: str | None
    short: str | None
    branch: str | None
    subject: str | None
    dirty_files: int
    fixes: dict[str, bool]

    @property
    def has_all_fixes(self) -> bool:
        return all(self.fixes.values())


def source_info(root: Path = ROOT) -> SourceInfo:
    code, revision = _git("rev-parse", "HEAD", cwd=root)
    if code != 0 or not revision:
        return SourceInfo(root, None, None, None, None, 0, {k: False for k in REQUIRED_FIXES})
    _, short = _git("rev-parse", "--short", "HEAD", cwd=root)
    _, branch = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=root)
    _, subject = _git("log", "-1", "--format=%s", cwd=root)
    _, status = _git("status", "--porcelain", "--untracked-files=no", cwd=root)
    fixes = {sha: _git("merge-base", "--is-ancestor", sha, "HEAD", cwd=root)[0] == 0 for sha in REQUIRED_FIXES}
    return SourceInfo(root, revision, short, branch, subject, len([line for line in status.splitlines() if line.strip()]), fixes)


def gpu_info() -> dict | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        name, total, used, util = [x.strip() for x in out.split(",")]
        return {"name": name, "total_mib": int(total), "used_mib": int(used), "util_percent": int(util)}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def memory_info() -> dict:
    values = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            values[key] = int(rest.split()[0]) * 1024
    except OSError:
        pass
    total, available = values.get("MemTotal", 0), values.get("MemAvailable", 0)
    return {"total": total, "available": available, "used": total - available,
            "swap_used": values.get("SwapTotal", 0) - values.get("SwapFree", 0)}


def stray_simulator_processes() -> list[tuple[int, str]]:
    """Kit/worker processes that nobody owns (a previous crashed run) would silently slow a new run down."""
    modules = {"casu_grid.isaac_shard_worker", "casu_grid.workbench.train", "casu_grid.workbench.evaluate"}
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        for i, arg in enumerate(argv[:-1]):
            if arg == "-m" and argv[i + 1] in modules and "python" in Path(argv[0]).name and Path(argv[0]).name != "python.sh":
                found.append((int(entry.name), " ".join(argv)[:200]))
                break
    return found


def disk_free(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free
