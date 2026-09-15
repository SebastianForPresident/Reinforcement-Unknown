"""Research-grade fresh-process prefix replay for CB1 counterfactuals.

This runner deliberately does not use CF_CAPTURE or CF_RESTORE. Every Unity
state is reached by a fresh process, seeded reset, and the recorded action
prefix, so contacts and other engine-owned state arise naturally.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from CounterfactualReplay import (
    ACTION_PORT,
    OBS_PORT,
    Client,
    first_difference,
    observation_difference,
    snapshot_match,
)


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "PPO_Harness" / "Deploy.sh"
OBS_FIELDS = ("LayerProgress", "CurrentLayer", "BestLayerDepth", "PlayerTilePosition", "PlayerDead")


class PrefixReplayFailure(RuntimeError):
    def __init__(self, phase, message):
        super().__init__(message)
        self.phase = phase
        self.status = "STARTUP_FAIL" if phase in ("launch", "startup_handshake") else "REPLAY_FAIL"


def action_command(action):
    values = [int(value) for value in action]
    if len(values) != 7:
        raise ValueError(f"each action must contain seven values, got {len(values)}")
    return "CF_STEP " + " ".join(str(value) for value in values)


def action_tape_digest(tape):
    payload = json.dumps(tape, separators=(",", ":"), sort_keys=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def configured_environment():
    names = (
        "PPO_AUTOSTART_PRESET", "PPO_AUTOSTART_SETTINGS", "PPO_AUTOSTART_SETTINGS_FILE",
        "PPO_TIME_SCALE", "PPO_FORCE_BODY_UPDATE", "PPO_NATURAL_STAMINA",
        "PPO_PROTON_USE_XALIA", "PPO_IGNORE_XALIA_FAILURE", "PPO_UNITY_ARGS", "SDL_VIDEODRIVER",
    )
    return {name: os.environ[name] for name in names if name in os.environ}


def read_policy_rng(value):
    if not value:
        return {
            "available": False,
            "mode": "recorded_action_tape",
            "note": "No PPO sampler state supplied; the exact action tape is the reproducibility input.",
        }
    path = Path(value)
    if path.exists():
        with path.open() as stream:
            state = json.load(stream)
    else:
        state = json.loads(value)
    return {"available": True, "mode": "recorded", "state": state}


def owned_tree(root_pid):
    """Return only the exact /proc child tree below an explicitly owned PID."""
    result = []

    def visit(pid):
        if not isinstance(pid, int) or pid <= 0:
            return
        children_file = Path(f"/proc/{pid}/task/{pid}/children")
        if not children_file.is_file():
            return
        result.append(pid)
        try:
            children = [int(value) for value in children_file.read_text().split()]
        except (OSError, ValueError):
            children = []
        for child in children:
            visit(child)

    visit(root_pid)
    return result


def terminate_owned_tree(root_pid):
    """Terminate only an explicitly launched root and its current children."""
    pids = owned_tree(root_pid)
    if not pids:
        return
    print("Terminating owned prefix-replay PIDs: " + " ".join(map(str, pids)), file=sys.stderr)
    for pid in reversed(pids):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            raise RuntimeError(f"refused to terminate owned PID {pid}: {exc}") from exc
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and any(Path(f"/proc/{pid}").exists() for pid in pids):
        time.sleep(0.05)
    for pid in reversed(pids):
        if Path(f"/proc/{pid}").exists():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def process_arguments(pid):
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return [part.decode(errors="replace") for part in raw.split(b"\0") if part]
    except OSError:
        return []


def process_cmdline(pid):
    return " ".join(process_arguments(pid))


def is_launched_unity_pid(pid):
    """Match the exact Windows-mapped executable path, never a broad process name."""
    expected = "z:" + str(ROOT).replace("/", "\\") + "\\CasualtiesUnknown.exe"
    expected = expected.lower()
    return any(argument.lower() == expected for argument in process_arguments(pid))


def exact_unity_pids():
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if is_launched_unity_pid(pid):
            result.append(pid)
    return result


def terminate_exact_unity(pid):
    """Terminate only the Unity PID authenticated by its own status file."""
    if not isinstance(pid, int) or pid <= 0:
        return
    command_line = process_cmdline(pid)
    if not is_launched_unity_pid(pid):
        print(
            f"Refusing Unity cleanup: PID {pid} no longer identifies the exact launched executable",
            file=sys.stderr,
        )
        return
    print(f"Terminating exact owned Unity PID: {pid}", file=sys.stderr)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            return
        except PermissionError as exc:
            raise RuntimeError(f"refused to terminate owned Unity PID {pid}: {exc}") from exc
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and Path(f"/proc/{pid}").exists():
            time.sleep(0.05)
        if not Path(f"/proc/{pid}").exists():
            return


class FreshUnity:
    def __init__(self, root, token, label, obs_port, action_port, timeout):
        self.root = Path(root)
        self.token = token
        self.label = label
        self.session_dir = self.root / token
        self.status_file = self.session_dir / "autostart_status.json"
        self.log_path = self.session_dir / "unity_launch.log"
        self.timeout = timeout
        self.log = None
        self.process = None
        self.unity_pid = None
        self.preexisting_unity_pids = set()
        self.client = Client("127.0.0.1", self.root, timeout, obs_port, action_port)
        self.client.session_token = token
        self.client.startup_status_path = self.status_file
        self.client.failure_probe = self.launch_failure

    def launch_failure(self):
        """Report failure of this exact owned deploy root, if it has exited."""
        if self.process is not None and self.process.poll() is not None:
            return f"Deploy.sh exited with status {self.process.returncode}"
        # Xalia is a Proton-side helper.  Its fatal display initialization can
        # leave the outer Proton/Unity PID alive without ever reaching the
        # harness, so process liveness alone is not a sufficient startup
        # signal.  Match only the observed fatal signature; do not suppress
        # or manipulate the helper.
        try:
            launch_log = self.log_path.read_text(errors="replace")
        except OSError:
            launch_log = ""
        if ("FATAL UNHANDLED EXCEPTION" in launch_log and
                "No displays available" in launch_log and
                os.environ.get("PPO_IGNORE_XALIA_FAILURE", "0") != "1"):
            return "Proton/Xalia reported fatal display initialization: No displays available"
        return None

    def start(self):
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("w")
        self.preexisting_unity_pids = set(exact_unity_pids())
        environment = os.environ.copy()
        environment.update({
            "PPO_COUNTERFACTUAL": "1",
            "PPO_AUTOSTART": "1",
            "PPO_AUTOSTART_STATUS_FILE": str(self.status_file),
            "PPO_AUTOSTART_UNITY_LOG": str(self.log_path),
            "PPO_COUNTERFACTUAL_DIAGNOSTICS_ROOT": str(self.root),
            "PPO_OBS_PORT": str(self.client.obs_port),
            "PPO_ACTION_PORT": str(self.client.action_port),
        })
        self.process = subprocess.Popen(
            ["bash", str(DEPLOY)], cwd=str(ROOT), env=environment,
            stdout=self.log, stderr=subprocess.STDOUT,
        )

    def discover_unity_pid(self):
        """Authenticate the reparented Unity process from AutoStart's status."""
        deadline = time.monotonic() + min(self.timeout, 30.0)
        while time.monotonic() < deadline:
            failure = self.launch_failure()
            if failure:
                raise RuntimeError(
                    f"autostart launch failed before Unity ownership could be authenticated: {failure}"
                )
            try:
                with self.status_file.open() as stream:
                    status = json.load(stream)
                pid = status.get("process_pid")
                if isinstance(pid, int) and pid > 0:
                    if is_launched_unity_pid(pid):
                        self.unity_pid = pid
                        return
                    if status.get("status") == "failed":
                        raise RuntimeError(
                            f"autostart failed before Unity ownership could be authenticated: {status}"
                        )
            except FileNotFoundError:
                pass
            except json.JSONDecodeError:
                pass
            candidates = [
                candidate for candidate in exact_unity_pids()
                if candidate not in self.preexisting_unity_pids
            ]
            if len(candidates) == 1:
                self.unity_pid = candidates[0]
                return
            if len(candidates) > 1:
                raise RuntimeError(
                    "refusing Unity ownership: multiple newly launched exact executable PIDs: "
                    + ", ".join(map(str, candidates))
                )
            time.sleep(0.05)
        raise TimeoutError(
            f"AutoStart did not publish an authenticated Unity PID in {self.status_file}"
        )

    def hello_reset(self, seed):
        self.discover_unity_pid()
        self.client.connect()
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self.client.command("CF_HELLO " + self.token)
                break
            except RuntimeError as exc:
                if "CB1 runtime guards failed:" not in str(exc) or time.monotonic() >= deadline:
                    raise
                time.sleep(0.25)
        reset = self.client.command(f"CF_RESET {seed}")
        ready = self.client.wait_reset_ready(reset["reset_token"])
        observation = self.client.next_observation(ready["observation_id"])
        return observation

    def close(self):
        try:
            self.client.close()
        finally:
            if self.process is not None:
                try:
                    self.process.wait(timeout=15.0)
                except subprocess.TimeoutExpired:
                    pass
                if self.unity_pid is not None:
                    terminate_exact_unity(self.unity_pid)
                if self.process.poll() is None:
                    terminate_owned_tree(self.process.pid)
                    self.process.wait(timeout=5.0)
                self.process = None
            if self.log is not None:
                self.log.close()
                self.log = None


def report_path(session_dir, filename, timeout):
    path = Path(session_dir) / "unity" / filename
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            with path.open() as stream:
                return json.load(stream)
        time.sleep(0.02)
    raise TimeoutError(f"Unity report {path} was not written")


def boundary_record(step, raw, observation, report, report_file):
    return {
        "step": step,
        "observation_base64": base64.b64encode(raw).decode("ascii"),
        "observation_sha256": hashlib.sha256(raw).hexdigest(),
        "state_sha256": report["state_sha256"],
        "state_sections": report.get("state_sections", {}),
        "report": report_file,
        "navigation": {
            name: observation[name].tolist() if hasattr(observation[name], "tolist") else observation[name].item()
            for name in OBS_FIELDS
        },
    }


def run_rollout(root, token, label, seed, tape, obs_port, action_port, timeout, detail_step=None):
    process = FreshUnity(root, token, label, obs_port, action_port, timeout)
    boundaries = []
    phase = "launch"
    try:
        process.start()
        phase = "startup_handshake"
        initial = process.hello_reset(seed)
        phase = "initial_state"
        initial_report = process.client.command("CF_STATE_DETAIL" if detail_step == 0 else "CF_STATE")
        report = report_path(process.session_dir, initial_report["file"], timeout)
        boundaries.append(boundary_record(0, initial[1], initial[2], report, initial_report["file"]))
        if detail_step == 0:
            return boundaries[0], process.session_dir
        phase = "replay"
        for step, action in enumerate(tape, 1):
            process.client.command(action_command(action))
            observation_id, raw, observation = process.client.next_observation(step + initial[0])
            command = "CF_STATE_DETAIL" if detail_step == step else "CF_STATE"
            state_reply = process.client.command(command)
            report = report_path(process.session_dir, state_reply["file"], timeout)
            boundaries.append(boundary_record(step, raw, observation, report, state_reply["file"]))
            if detail_step == step:
                return boundaries[-1], process.session_dir
        return boundaries, process.session_dir
    except Exception as exc:
        failure = {
            "schema": 1,
            "label": label,
            "seed": seed,
            "detail_step": detail_step,
            "phase": phase,
            "status": "STARTUP_FAIL" if phase in ("launch", "startup_handshake") else "REPLAY_FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "owned_deploy_pid": process.process.pid if process.process is not None else None,
            "authenticated_unity_pid": process.unity_pid,
            "status_file": str(process.status_file),
            "unity_log": str(process.log_path),
        }
        try:
            with (process.session_dir / "runner_failure.json").open("w") as stream:
                json.dump(failure, stream, indent=2, sort_keys=True)
        except OSError:
            pass
        if isinstance(exc, PrefixReplayFailure):
            raise
        raise PrefixReplayFailure(phase, str(exc)) from exc
    finally:
        process.close()


def cheap_boundary_difference(left, right):
    left_raw = base64.b64decode(left["observation_base64"])
    right_raw = base64.b64decode(right["observation_base64"])
    difference = observation_difference(left_raw, right_raw)
    if difference is None:
        difference = first_difference(left["state_sections"], right["state_sections"], "unity.state_sections")
        if difference is None and left["state_sha256"] != right["state_sha256"]:
            difference = {
                "path": "unity.state_sha256",
                "left": left["state_sha256"],
                "right": right["state_sha256"],
            }
    return difference


def compare_rollouts(left, right):
    for left_boundary, right_boundary in zip(left, right):
        difference = cheap_boundary_difference(left_boundary, right_boundary)
        if difference:
            return {
                "policy_step": left_boundary["step"],
                "difference": difference,
                "prior_matching_step": left_boundary["step"] - 1,
            }
    if len(left) != len(right):
        return {
            "policy_step": min(len(left), len(right)),
            "difference": {"path": "trajectory.length", "left": len(left), "right": len(right)},
        }
    return None


def exact_difference(root, seed, tape, first, second, first_token, second_token, ports, timeout, step):
    left_boundary, left_dir = run_rollout(
        root, first_token, "detail-left", seed, tape, ports[0], ports[1], timeout, step,
    )
    right_boundary, right_dir = run_rollout(
        root, second_token, "detail-right", seed, tape, ports[2], ports[3], timeout, step,
    )
    left_report = json.load((Path(left_dir) / "unity" / left_boundary["report"]).open())
    right_report = json.load((Path(right_dir) / "unity" / right_boundary["report"]).open())
    return snapshot_match(left_report, right_report)


def outcome(boundaries):
    final = boundaries[-1]["navigation"]
    return {"final": final, "trajectory": [boundary["navigation"] for boundary in boundaries]}


def parse_steps(value, horizon):
    if not value:
        return []
    result = sorted(set(int(item) for item in value.split(",") if item.strip()))
    if any(step < 0 or step >= horizon for step in result):
        raise ValueError(f"treatment decisions must be in [0,{horizon - 1}]")
    return result


def load_tape(args):
    if not args.tape:
        from CounterfactualReplay import default_tape
        return default_tape(args.horizon)
    with Path(args.tape).open() as stream:
        tape = json.load(stream)
    if not isinstance(tape, list) or not tape:
        raise ValueError("tape must be a non-empty JSON list")
    if len(tape) < args.horizon:
        raise ValueError("tape is shorter than --horizon")
    return tape[:args.horizon]


def token(run_id, label):
    """Build a readable Unity diagnostic token within the protocol contract.

    BoundaryProtocol validates tokens as lowercase [a-z0-9-] and limits them
    to 64 characters. Detail replays add a longer label to the same run ID,
    so preserving both strings verbatim can exceed that limit. Keep a
    readable, sanitized prefix and retain a random suffix for per-process
    uniqueness after truncation.
    """
    readable = re.sub(r"[^a-z0-9-]+", "-", f"{run_id}-{label}".lower()).strip("-")
    suffix = hashlib.sha256(os.urandom(16)).hexdigest()[:8]
    prefix = "prefix-"
    room = 64 - len(prefix) - 1 - len(suffix)
    readable = readable[:max(1, room)].rstrip("-")
    return f"{prefix}{readable}-{suffix}"


def run(args):
    tape = load_tape(args)
    horizon = len(tape)
    treatment_steps = parse_steps(args.treatment_steps, horizon)
    root = Path(args.diagnostics_root)
    run_id = args.run_id or ("prefix-" + hashlib.sha256(os.urandom(16)).hexdigest()[:12])
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    ports = (args.obs_port, args.action_port, args.obs_port + 2, args.action_port + 2,
             args.obs_port + 4, args.action_port + 4, args.obs_port + 6, args.action_port + 6,
             args.obs_port + 8, args.action_port + 8)

    metadata = {
        "schema": 1, "run_id": run_id, "seed": args.seed, "horizon": horizon,
        "configuration": configured_environment(),
        "action_tape": tape, "action_tape_sha256": action_tape_digest(tape),
        "policy_rng": read_policy_rng(args.policy_rng),
        "candidate_treatment_steps": treatment_steps,
        "treatment": "attack index 6 changed to 0 in branch B",
        "state_validation": "full causal CF_STATE fingerprints at every policy boundary; CF_STATE_DETAIL on mismatch",
        "restore_used": False,
    }
    with (run_dir / "original_rollout.json").open("w") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)

    print(f"Recording original fresh-process rollout: {run_id}")
    original, _ = run_rollout(root, token(run_id, "original"), "original", args.seed, tape,
                              ports[0], ports[1], args.timeout)
    with (run_dir / "original_boundaries.json").open("w") as stream:
        json.dump(original, stream, indent=2)

    result = {"schema": 1, "run_id": run_id, "status": "failed", "original": metadata}
    print("Running independent fresh-process A/A left...")
    aa_left, _ = run_rollout(root, token(run_id, "aa-left"), "aa-left", args.seed, tape,
                             ports[2], ports[3], args.timeout)
    print("Running independent fresh-process A/A right...")
    aa_right, _ = run_rollout(root, token(run_id, "aa-right"), "aa-right", args.seed, tape,
                              ports[4], ports[5], args.timeout)
    aa_difference = compare_rollouts(aa_left, aa_right)
    original_difference = compare_rollouts(original, aa_left)
    if aa_difference or original_difference:
        cheap = aa_difference or original_difference
        detail = exact_difference(
            root, args.seed, tape,
            "aa-left" if aa_difference else "original",
            "aa-right" if aa_difference else "aa-left",
            token(run_id, "detail-left"), token(run_id, "detail-right"), ports[6:], args.timeout,
            cheap["policy_step"],
        )
        result.update({"status": "A/A_FAIL", "aa_difference": aa_difference,
                       "original_vs_aa_difference": original_difference,
                       "first_exact_difference": detail})
    else:
        result.update({"status": "A/A_PASS", "aa_difference": None,
                       "original_vs_aa_difference": None,
                       "first_exact_difference": None})
        for step in treatment_steps:
            print(f"Evaluating fresh-process A/B candidate at decision {step}...")
            branch_a_tape = tape
            branch_b_tape = [list(action) for action in tape]
            branch_b_tape[step][6] = 0
            branch_a, _ = run_rollout(root, token(run_id, f"ab-{step}-a"), f"ab-{step}-a", args.seed,
                                      branch_a_tape, ports[0], ports[1], args.timeout)
            branch_b, _ = run_rollout(root, token(run_id, f"ab-{step}-b"), f"ab-{step}-b", args.seed,
                                      branch_b_tape, ports[2], ports[3], args.timeout)
            prefix_a = branch_a[step]
            prefix_b = branch_b[step]
            prefix_check = {
                "against_original_a": cheap_boundary_difference(original[step], prefix_a),
                "against_original_b": cheap_boundary_difference(original[step], prefix_b),
                "branches_equal_before_treatment": cheap_boundary_difference(prefix_a, prefix_b) is None,
            }
            result.setdefault("candidates", {})[str(step)] = {
                "decision": step, "prefix_check": prefix_check,
                "branch_a_outcome": outcome(branch_a), "branch_b_outcome": outcome(branch_b),
                "direct_navigation_difference": first_difference(outcome(branch_a), outcome(branch_b), "navigation"),
            }
        if treatment_steps:
            result["status"] = "A/A_PASS_AB_RECORDED"

    with (run_dir / "prefix_replay_result.json").open("w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps({key: result.get(key) for key in (
        "status", "aa_difference", "original_vs_aa_difference", "first_exact_difference", "candidates")
    }, indent=2))
    return 0 if result["status"] in ("A/A_PASS", "A/A_PASS_AB_RECORDED") else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=391728405)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--tape")
    parser.add_argument("--treatment-steps", help="comma-separated decisions to evaluate after A/A")
    parser.add_argument("--policy-rng", help="JSON file or JSON value containing recorded PPO sampler state")
    parser.add_argument("--run-id")
    parser.add_argument("--diagnostics-root", default="counterfactual_prefix_diagnostics")
    parser.add_argument("--obs-port", type=int, default=int(os.environ.get("PPO_OBS_PORT", OBS_PORT)))
    parser.add_argument("--action-port", type=int, default=int(os.environ.get("PPO_ACTION_PORT", ACTION_PORT)))
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    if args.horizon < 1 or args.horizon > 10000:
        parser.error("horizon must be between 1 and 10000")
    if args.obs_port < 1024 or args.action_port < 1024:
        parser.error("ports must be >= 1024")
    try:
        return run(args)
    except Exception as exc:
        failure_path = Path(args.diagnostics_root) / (
            "prefix_replay_failure_" + str(int(time.time() * 1000)) + ".json"
        )
        try:
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            with failure_path.open("w") as stream:
                json.dump({
                    "schema": 1,
                    "status": getattr(exc, "status", "RUNNER_FAIL"),
                    "phase": getattr(exc, "phase", None),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "seed": args.seed,
                    "horizon": args.horizon,
                    "run_id": args.run_id,
                    "diagnostics_root": str(args.diagnostics_root),
                }, stream, indent=2, sort_keys=True)
            print(f"prefix replay failure artifact: {failure_path}", file=sys.stderr)
        except OSError:
            pass
        print(f"prefix replay failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
