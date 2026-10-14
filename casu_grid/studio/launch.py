"""Launch plans: what a training run will do, whether it may, and how to start / stop / resume it.

No Qt here. A :class:`LaunchPlan` holds only the choices a researcher may make (execution layout, budget, seed, checkpoints); everything
scientific comes from the experiment file. :func:`validate` turns the plan plus the state of the machine into an explicit list of
checks, each ok / warning / error, so the interface can say precisely why a launch is blocked instead of failing later in a log.
"""
from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import environment as env
from .experiments import RECOMMENDED, ExperimentInfo
from .runs import CheckpointInfo, RunRecord, human_bytes, human_duration, pid_alive, read_json

OK, WARN, ERROR = "ok", "warn", "error"


@dataclass
class Check:
    level: str
    title: str
    detail: str = ""


@dataclass
class LaunchPlan:
    experiment: ExperimentInfo
    run_root: Path
    run_name: str
    worlds: int = RECOMMENDED["worlds"]
    workers: int = RECOMMENDED["workers"]
    physx_threads: int = RECOMMENDED["physx_threads"]
    device: str = RECOMMENDED["device"]
    seed: int = 7
    updates: int = 250
    checkpoint_every: int = 25
    save_initial: bool = True
    resume: CheckpointInfo | None = None
    resume_run: RunRecord | None = None
    acknowledged: set[str] = field(default_factory=set)       # warnings the researcher explicitly accepted

    @property
    def run_dir(self) -> Path:
        return self.run_root / self.run_name

    @property
    def rollout_steps(self) -> int:
        return int(self.experiment.summary["ppo"]["rollout_steps"])

    @property
    def decisions_per_update(self) -> int:
        return self.worlds * self.rollout_steps

    @property
    def total_decisions(self) -> int:
        return self.decisions_per_update * self.updates

    def estimated_seconds(self, throughput: float = RECOMMENDED["throughput"]) -> float:
        return self.total_decisions / throughput

    def estimated_memory_bytes(self) -> int:
        """Unique memory of the whole process tree, fitted to the Optimization IV measurements (17 GiB at 64 worlds x 8 workers)."""
        return int((2.0 + 1.7 * self.workers + 0.02 * self.worlds) * 2**30)

    def estimated_disk_bytes(self) -> int:
        checkpoint = 8 * 2**20            # a checkpoint is a few MiB (network + optimiser + world snapshots)
        count = 2 + (self.updates // self.checkpoint_every if self.checkpoint_every else 0)
        return int(self.total_decisions * 170 + count * checkpoint + self.updates * 6000)

    def is_recommended_layout(self) -> bool:
        return (self.worlds, self.workers, self.physx_threads, self.device) == (
            RECOMMENDED["worlds"], RECOMMENDED["workers"], RECOMMENDED["physx_threads"], RECOMMENDED["device"])


def default_run_name(experiment: ExperimentInfo, seed: int) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = "xy1mini" if experiment.is_mini else ("xy1" if experiment.is_xy1 else experiment.path.stem[:16])
    return f"{stamp}-{tag}-s{seed}"


def build_command(plan: LaunchPlan, install: env.IsaacInstall) -> list[str]:
    cfg = plan.experiment
    cmd = [str(install.python_sh), "-m", "casu_grid.workbench.lab", "--experiments", str(plan.run_root), "launch", str(cfg.path),
           "--run-dir", str(plan.run_dir), "--updates", str(plan.updates), "--num-envs", str(plan.worlds), "--workers", str(plan.workers),
           "--physx-threads", str(plan.physx_threads), "--train-device", plan.device, "--seed", str(plan.seed),
           "--terrain-mode", "independent"]
    if plan.checkpoint_every:
        cmd += ["--checkpoint-every", str(plan.checkpoint_every)]
    if plan.save_initial and plan.resume is None:
        cmd += ["--save-initial-checkpoint"]
    if plan.resume is not None:
        cmd += ["--resume", str(plan.resume.path)]
    return cmd


def command_text(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def resume_checks(plan: LaunchPlan) -> list[Check]:
    """Is this checkpoint a legitimate continuation of THIS experiment on THIS layout? Never resumes silently when it is not."""
    checks: list[Check] = []
    ck, exp = plan.resume, plan.experiment
    if ck is None:
        return checks
    if not ck.path.is_file():
        return [Check(ERROR, "Checkpoint file is missing", str(ck.path))]
    if ck.fingerprint is None:
        checks.append(Check(WARN, "Checkpoint has no compatibility record",
                            "It was written before the Studio recorded one; the trainer will still verify the learning contract when it loads it."))
    elif ck.fingerprint != exp.fingerprint:
        checks.append(Check(ERROR, "Learning contract differs",
                            "The checkpoint was trained with different actions, observations, network or PPO settings than this experiment now declares. "
                            "It cannot be resumed; start a new run (or restore the original experiment file)."))
    else:
        checks.append(Check(OK, "Learning contract matches", "Actions, observations, network and PPO settings are identical."))
    if ck.termination_contract != exp.termination_contract:
        checks.append(Check(ERROR, "Termination contract differs",
                            f"The checkpoint was trained under {ck.termination_contract}; this experiment uses {exp.termination_contract}. These are different experiments "
                            "(for example, whether leaving through the bottom of the layer ends the episode); results and checkpoints must not be mixed. Start a new run."))
    else:
        checks.append(Check(OK, "Termination contract matches", exp.termination_contract))
    run = plan.resume_run
    if run is not None and run.config_sha256 and exp.sha256 and run.config_sha256 != exp.sha256:
        checks.append(Check(WARN, "Experiment file changed since the run started",
                            "The contract is the same, but other parts of the file (environment settings, reward callbacks) differ from the source run. "
                            "Review the difference before continuing; the result is then a scientifically different run."))
    if ck.world_partition is not None:
        sizes = shard_sizes(plan.worlds, plan.workers)
        if list(ck.world_partition) != sizes:
            checks.append(Check(ERROR, "World layout differs",
                                f"The checkpoint holds the state of worlds partitioned as {list(ck.world_partition)}; this launch would use {sizes} "
                                f"({plan.worlds} worlds on {plan.workers} workers). Use the same worlds and workers as the original run."))
        else:
            checks.append(Check(OK, "World layout matches", f"{plan.worlds} worlds on {plan.workers} workers ({sizes})."))
    if run is not None and run.seed is not None and run.seed != plan.seed:
        checks.append(Check(WARN, "Seed differs from the source run", f"Source run used seed {run.seed}; the resumed run keeps the restored random state "
                                                                       "regardless, so the seed only labels it."))
    if run is not None and run.revision and plan.experiment is not None:
        src = env.source_info()
        if src.revision and run.revision != src.revision:
            checks.append(Check(WARN, "Source revision differs", f"Run was made with {run.revision[:8]}; this Studio is at {src.short}."))
    return checks


def shard_sizes(worlds: int, workers: int) -> list[int]:
    base, extra = divmod(worlds, workers)
    return [base + (1 if i < extra else 0) for i in range(workers)]


def validate(plan: LaunchPlan, install: env.IsaacInstall, *, owned_pids: set[int] | None = None) -> list[Check]:
    checks: list[Check] = []
    exp = plan.experiment
    if exp.error:
        return [Check(ERROR, "Experiment file does not load", exp.error)]
    checks.append(Check(OK, "Experiment contract is valid",
                        f"{exp.name}: {len(exp.summary['actions'])} action heads, {exp.summary['physics_ticks_per_decision']} physics ticks per decision, "
                        f"parameter count {exp.summary.get('parameter_count') or 'computed at launch'}."))
    if exp.is_xy1 and exp.mismatches:
        checks.append(Check(WARN, "Differs from the intended XY1 contract", "; ".join(exp.mismatches)))
    if not exp.is_xy1:
        checks.append(Check(ERROR, "Not an XY1 navigation experiment", "The Studio's learning workflow covers XY1 navigation experiments."))
    contract_text = {"xy1-termination-v2-layer-exit": ("Termination contract v2: bottom-of-layer exit ends the episode", "Leaving through the bottom (y < −halfHeight + 3.1, CasU's layer transition) is a true terminal event with no extra reward or penalty; the world, Timmy and the goal are re-initialised."),
                     "xy1-termination-v1": ("Reference termination contract v1", "Episodes end only by goal arrival or the 100,000-decision ceiling; an agent that digs through the floor falls out of the world and is never reset. Reproducible reference, not recommended for learning."),
                     "xy1-termination-v1x-out-of-world-truncation": ("Superseded exploratory termination variant", "Truncates the episode when the agent is outside the world; replaced by termination contract v2.")}.get(exp.termination_contract)
    if contract_text:
        checks.append(Check(OK if exp.role == "recommended" else WARN, contract_text[0], contract_text[1] + f" (id {exp.termination_contract})"))
    # execution layout
    if plan.worlds < 1 or plan.workers < 1 or plan.updates < 1:
        checks.append(Check(ERROR, "Worlds, workers and updates must be positive"))
    if plan.workers > plan.worlds:
        checks.append(Check(ERROR, "More workers than worlds", f"{plan.workers} workers for {plan.worlds} worlds leaves workers without a world."))
    if plan.physx_threads < 2:
        checks.append(Check(ERROR, "At least 2 PhysX threads per worker", "One PhysX thread makes the simulator hang."))
    cores = os.cpu_count() or 1
    if plan.workers * plan.physx_threads > cores:
        checks.append(Check(WARN, "More simulation threads than CPU threads", f"{plan.workers} × {plan.physx_threads} = {plan.workers * plan.physx_threads} "
                                                                           f"threads on {cores} logical CPUs; throughput will suffer."))
    if plan.is_recommended_layout():
        checks.append(Check(OK, "Validated execution layout", "64 worlds × 8 workers × 2 PhysX threads on CUDA: the layout measured at about 1,700 decisions/s."))
    else:
        checks.append(Check(WARN, "Not the validated execution layout",
                            f"{plan.worlds} worlds × {plan.workers} workers × {plan.physx_threads} PhysX threads on {plan.device}. Science is unchanged; "
                            "speed and memory were only validated for 64 × 8 × 2 on CUDA."))
    # simulator and code
    if not install.ok:
        checks.append(Check(ERROR, "Isaac Sim not available", "; ".join(install.problems) or "python.sh not found"))
    else:
        checks.append(Check(OK, "Isaac Sim found", str(install.sim_root)))
    src = env.source_info()
    if src.revision is None:
        checks.append(Check(ERROR, "Source revision unknown", f"{src.root} is not a readable git checkout, so the run could not record what it ran."))
    else:
        if not src.has_all_fixes:
            missing = [f"{sha} ({env.REQUIRED_FIXES[sha]})" for sha, present in src.fixes.items() if not present]
            checks.append(Check(ERROR, "Checkout lacks required Optimization IV corrections", "Missing: " + "; ".join(missing)))
        else:
            checks.append(Check(OK, f"Source revision {src.short} contains the Optimization IV corrections",
                                f"Branch {src.branch}; includes spawn/goal placement fix, reset fix, boundary-dig fix and vectorised terrain."))
        if src.dirty_files:
            checks.append(Check(WARN, f"{src.dirty_files} tracked file(s) have uncommitted changes",
                                "The run will record the commit hash, but the code that actually runs differs from it."))
    overrides = env.inherited_overrides()
    if overrides.get("CASU_LEGACY_MINI_SPAWN") == "1":
        checks.append(Check(ERROR, "Legacy spawn mode is switched on in this environment",
                            "CASU_LEGACY_MINI_SPAWN=1 reproduces the mis-placed pre-correction worlds. Unset it (and restart the Studio) for real training."))
    other = {k: v for k, v in overrides.items() if k not in ("CASU_LEGACY_MINI_SPAWN", "CASU_REPRODUCE_LEGACY")}
    if other:
        checks.append(Check(WARN, "Execution overrides in the environment are ignored",
                            "The Studio launches with explicit settings and removes: " + ", ".join(f"{k}={v}" for k, v in other.items())))
    # machine state
    gpu = env.gpu_info()
    if plan.device == "cuda":
        if gpu is None:
            checks.append(Check(ERROR, "No NVIDIA GPU visible", "CUDA training was requested; the Studio will not silently fall back to the CPU."))
        elif gpu["total_mib"] - gpu["used_mib"] < 4096:
            checks.append(Check(ERROR, "Not enough free GPU memory", f"{gpu['total_mib'] - gpu['used_mib']} MiB free; training needs about 3.6 GiB."))
        else:
            checks.append(Check(OK, "GPU available", f"{gpu['name']}, {gpu['total_mib'] - gpu['used_mib']} MiB free of {gpu['total_mib']} MiB."))
    mem = env.memory_info()
    need = plan.estimated_memory_bytes()
    if mem["available"] and mem["available"] < need * 1.1:
        checks.append(Check(ERROR, "Not enough free memory", f"About {human_bytes(need)} are needed; {human_bytes(mem['available'])} are available."))
    elif mem["available"] and mem["available"] < need + 4 * 2**30:
        checks.append(Check(WARN, "Little memory headroom", f"About {human_bytes(need)} needed, {human_bytes(mem['available'])} available."))
    elif mem["available"]:
        checks.append(Check(OK, "Enough memory", f"About {human_bytes(need)} needed, {human_bytes(mem['available'])} available."))
    strays = [(pid, cmd) for pid, cmd in env.stray_simulator_processes() if not owned_pids or pid not in owned_pids]
    if strays:
        checks.append(Check(ERROR, "Simulator processes are already running",
                            f"{len(strays)} process(es), e.g. pid {strays[0][0]}. Another run, an evaluation or an orphan from a crash would compete for the CPU "
                            "and memory. Stop it first."))
    else:
        checks.append(Check(OK, "No other simulator is running"))
    # output location
    if plan.run_dir.exists():
        checks.append(Check(ERROR, "Run folder already exists", f"{plan.run_dir} (runs are never overwritten; choose another name)."))
    if not plan.run_name or "/" in plan.run_name or plan.run_name.startswith("."):
        checks.append(Check(ERROR, "Invalid run name", "Use letters, digits, dashes and underscores."))
    free = env.disk_free(plan.run_root)
    if free < plan.estimated_disk_bytes() * 1.5 + 2 * 2**30:
        checks.append(Check(ERROR, "Not enough disk space", f"About {human_bytes(plan.estimated_disk_bytes())} will be written; {human_bytes(free)} free."))
    else:
        checks.append(Check(OK, "Disk space", f"About {human_bytes(plan.estimated_disk_bytes())} will be written to {plan.run_root}; {human_bytes(free)} free."))
    checks.extend(resume_checks(plan))
    return checks


def blocking(checks: list[Check]) -> list[Check]:
    return [c for c in checks if c.level == ERROR]


# -------------------------------------------------------------------------------------------------------------- processes
@dataclass
class Session:
    plan: LaunchPlan
    process: subprocess.Popen
    log_path: Path
    started: float
    command: list[str]
    meta_written: bool = False

    @property
    def pid(self) -> int:
        return self.process.pid

    def alive(self) -> bool:
        return self.process.poll() is None


def start(plan: LaunchPlan, install: env.IsaacInstall) -> Session:
    """Start ``lab launch`` as a detached process group; it owns the run folder, markers and the simulator child."""
    blockers = blocking(validate(plan, install))
    if blockers:
        raise RuntimeError("; ".join(c.title for c in blockers))
    plan.run_root.mkdir(parents=True, exist_ok=True)
    log_dir = plan.run_root / ".studio"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{plan.run_name}.launch.log"
    command = build_command(plan, install)
    child_env = env.clean_child_environment(env.process_environment(install))
    child_env["PYTHONUNBUFFERED"] = "1"
    handle = log_path.open("wb")
    process = subprocess.Popen(command, cwd=env.ROOT, env=child_env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    handle.close()
    return Session(plan, process, log_path, time.time(), command)


def write_launch_record(session: Session) -> None:
    """Record next to the run what the Studio asked for, once the launcher has created the run folder."""
    run_dir = session.plan.run_dir
    if session.meta_written or not run_dir.is_dir():
        return
    plan = session.plan
    src = env.source_info()
    record = {"schema": "casu.studio-launch.v1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "command": command_text(session.command), "worlds": plan.worlds, "workers": plan.workers, "physx_threads": plan.physx_threads,
              "device": plan.device, "seed": plan.seed, "updates": plan.updates, "checkpoint_every": plan.checkpoint_every,
              "save_initial": plan.save_initial, "resume": str(plan.resume.path) if plan.resume else None,
              "source_revision": src.revision, "source_branch": src.branch, "dirty_files": src.dirty_files,
              "legacy_spawn": env.inherited_overrides().get("CASU_LEGACY_MINI_SPAWN") == "1",
              "estimated_decisions": plan.total_decisions, "experiment_sha256": plan.experiment.sha256, "termination_contract": plan.experiment.termination_contract}
    try:
        (run_dir / "studio_launch.json").write_text(json.dumps(record, indent=2) + "\n")
        session.meta_written = True
    except OSError:
        pass


def request_stop(run_dir: Path) -> None:
    """Graceful stop: the trainer finishes the current update, keeps its checkpoint and marks the run interrupted."""
    (run_dir / "STOP").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")


def force_stop(run_dir: Path, session: Session | None = None) -> None:
    """Last resort: interrupt the launcher (which terminates the simulator group); the checkpoint of the last finished update remains."""
    pid = session.pid if session is not None else None
    if pid is None:
        marker = run_dir / "RUNNING"
        try:
            pid = int(marker.read_text().split()[0])
        except (OSError, ValueError, IndexError):
            return
    try:
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError:
        return
    deadline = time.time() + 20
    while time.time() < deadline and pid_alive(pid):
        time.sleep(0.25)
    if pid_alive(pid):
        try:
            import psutil
            for child in psutil.Process(pid).children(recursive=True):
                child.kill()
            psutil.Process(pid).kill()
        except Exception:
            pass


def clear_stop(run_dir: Path) -> None:
    (run_dir / "STOP").unlink(missing_ok=True)


def launch_log_tail(path: Path, lines: int = 60) -> str:
    try:
        data = path.read_bytes()[-20000:].decode(errors="replace")
    except OSError:
        return ""
    return "\n".join(data.splitlines()[-lines:])


def explain_failure(run: RunRecord | None, session_log: str = "") -> tuple[str, str]:
    """(headline, detail) for a failed run, from failure.json or the launcher log; plain language first, raw text second."""
    failure = run.failure if run is not None else None
    text = ""
    if failure:
        text = (failure.get("log_tail") or failure.get("traceback") or failure.get("message") or "")
        error_type = failure.get("error_type")
    else:
        error_type = None
    text = text or session_log
    lowered = text.lower()
    if "workerfailure" in lowered or "closed its pipe" in lowered:
        headline = "A simulation worker process died."
    elif "out of memory" in lowered or "cuda error: out of memory" in lowered:
        headline = "The GPU ran out of memory."
    elif "memoryerror" in lowered or "killed" in lowered[-400:]:
        headline = "The machine ran out of memory (the operating system killed a process)."
    elif "non-finite ppo loss" in lowered or "floatingpointerror" in lowered:
        headline = "Training diverged: a non-finite loss or gradient appeared."
    elif "legacy_mini_spawn" in lowered:
        headline = "Legacy spawn mode is switched on; it is refused for real training."
    elif "contract is incompatible" in lowered or "snapshot has wrong" in lowered or "incompatible" in lowered:
        headline = "The checkpoint does not match this experiment or layout."
    elif "isaac" in lowered and ("initialization failed" in lowered or "no module named" in lowered):
        headline = "Isaac Sim could not start."
    elif error_type:
        headline = f"Training stopped with {error_type}."
    elif text:
        headline = "Training stopped because of an error."
    else:
        headline = "The run failed without leaving an explanation."
    return headline, text[-6000:]
