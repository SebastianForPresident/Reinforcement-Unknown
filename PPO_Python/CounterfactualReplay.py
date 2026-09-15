"""Run the restricted CB1 A/A counterfactual replay experiment.

This is deliberately a standalone diagnostic client.  It does not import or
start Server.py, does not touch PPO checkpoints, and does not write ordinary
episode traces or training metrics.  Start the game with PPO_COUNTERFACTUAL=1
and this process will accept Unity's two TCP connections.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import queue
import secrets
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

import Reward
import Types


OBS_PORT = 45701
ACTION_PORT = 45702
PRESENTATION_CLASSIFICATIONS = {"derived-rebuildable", "demonstrably-noncausal"}
PRESENTATION_LIFECYCLE_CLASSIFICATIONS = {"presentation-ui-lifecycle"}
OBSERVATION_SIZE = Types.OBSERVATION_DTYPE.itemsize
MESSAGE_SIZE = OBSERVATION_SIZE + 8


def default_tape(length: int) -> list[list[int]]:
    """A deterministic, non-destructive tape covering CB1 action dimensions."""
    tape = []
    for i in range(length):
        tape.append([
            (i % 3),                  # move: left, neutral, right
            1 if i in (8, 40) else 0, # jump
            (i // 7) % 3,             # vertical movement
            1 if 24 <= i < 32 else 0, # crouch
            (i * 3) % 9,              # look dx, categorical
            (i * 5) % 11,             # look dy, categorical
            0,                         # attack disabled for the baseline tape
        ])
    return tape


def first_difference(left, right, path="root"):
    if type(left) is not type(right):
        return {"path": path, "left": left, "right": right}
    if isinstance(left, dict):
        keys = sorted(set(left) | set(right))
        for key in keys:
            if key not in left or key not in right:
                return {"path": f"{path}.{key}", "left": left.get(key), "right": right.get(key)}
            difference = first_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return {"path": path + ".length", "left": len(left), "right": len(right)}
        for index, (a, b) in enumerate(zip(left, right)):
            difference = first_difference(a, b, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if left != right:
        return {"path": path, "left": left, "right": right}
    return None


def observation_difference(left: bytes, right: bytes):
    if left == right:
        return None
    a = np.frombuffer(left[:OBSERVATION_SIZE], dtype=Types.OBSERVATION_DTYPE, count=1)[0]
    b = np.frombuffer(right[:OBSERVATION_SIZE], dtype=Types.OBSERVATION_DTYPE, count=1)[0]
    for name in Types.OBSERVATION_DTYPE.names:
        av = a[name]
        bv = b[name]
        if isinstance(av, np.ndarray):
            is_float = np.issubdtype(av.dtype, np.floating)
            equal = np.array_equal(av, bv, equal_nan=is_float)
        else:
            is_float = np.issubdtype(np.asarray(av).dtype, np.floating)
            equal = bool(av == bv)
            if not equal and is_float:
                equal = bool(np.isnan(av) and np.isnan(bv))
        if not equal:
            return {
                "path": "observation." + name,
                "left_sha256": hashlib.sha256(left).hexdigest(),
                "right_sha256": hashlib.sha256(right).hexdigest(),
            }
    if left[OBSERVATION_SIZE:] != right[OBSERVATION_SIZE:]:
        return {"path": "observation.observation_id"}
    return {"path": "observation.raw_bytes"}


def likely_writers(path: str):
    if "RelativeBlockMap" in path:
        return ["PPOBridge.CollectObservations", "WorldGeneration.GetBlock", "WorldGeneration.GetBlockDamage"]
    if "RelativeFluidMap" in path:
        return ["PPOBridge.CollectObservations", "FluidManager.fluid", "FluidManager.UpdateFluid"]
    if "Attack" in path:
        return ["PPOBridge.ApplyPPOActions", "Body.UseItemInHand", "Body.attackCooldown"]
    if "Velocity" in path or "Grounded" in path or "Wall" in path:
        return ["Body.FixedUpdate", "Rigidbody2D solver", "PPOBridge.ApplyPPOActions"]
    if "Player" in path or "Limb" in path:
        return ["Body.FixedUpdate", "Limb.Update", "Rigidbody2D solver"]
    return ["the component/static field named by the divergence path", "Unity engine scheduling"]


class PythonReplayState:
    """The Python reward/watchdog state used by this diagnostic client."""

    def __init__(self, initial_observation):
        self.missed_attack_penalty_mode = False
        self.attack_cost = 0.0
        self.terminal_attack_mode = False
        self.attack_budget_mode = False
        self.best_progress = float(initial_observation["LayerProgress"])
        self.completion_bonus_applied = False
        self.last_reward_terms = {}
        self.episode_steps = 0
        self.best_layer_depth = int(initial_observation["BestLayerDepth"])
        self.last_best_depth_step = 0
        self.inactive_steps = 0
        self.mobile_stagnant_steps = 0
        self.previous_tile_position = (
            int(initial_observation["PlayerTilePosition"]["X"]),
            int(initial_observation["PlayerTilePosition"]["Y"]),
        )
        self.previous_block_health = initial_observation["RelativeBlockMap"]["Health"].copy()

    def step(self, observation, action):
        reward = Reward.Reward(observation, np.asarray(action), self)
        self.episode_steps += 1
        current_depth = int(observation["BestLayerDepth"])
        depth_improved = current_depth > self.best_layer_depth
        if depth_improved:
            self.best_layer_depth = current_depth
            self.last_best_depth_step = self.episode_steps

        tile = (
            int(observation["PlayerTilePosition"]["X"]),
            int(observation["PlayerTilePosition"]["Y"]),
        )
        velocity_inactive = (
            abs(float(observation["Velocity"]["X"])) <= 0.05
            and abs(float(observation["Velocity"]["Y"])) <= 0.05
        )
        block_unchanged = np.array_equal(
            observation["RelativeBlockMap"]["Health"], self.previous_block_health
        )
        physically_inactive = tile == self.previous_tile_position and velocity_inactive and block_unchanged
        capable = (
            not bool(observation["PlayerDead"])
            and float(observation["Consciousness"]) > 0.0
            and float(observation["TimeRagdolled"]) <= 0.0
        )
        if float(self.last_reward_terms["progress_delta"]) > 0.0:
            self.mobile_stagnant_steps = 0
        else:
            self.mobile_stagnant_steps += 1
        if capable and not depth_improved and physically_inactive:
            self.inactive_steps += 1
        else:
            self.inactive_steps = 0
        self.previous_tile_position = tile
        self.previous_block_health = observation["RelativeBlockMap"]["Health"].copy()
        watchdog = None
        if self.episode_steps >= 100_000:
            watchdog = "max_episode_steps"
        elif self.episode_steps - self.last_best_depth_step >= 75_000:
            watchdog = "stagnant_best_depth"
        elif self.inactive_steps >= 75:
            watchdog = "capable_inactivity"
        elif self.mobile_stagnant_steps >= 10_000:
            watchdog = "mobile_stagnation"
        return {
            "reward": float(reward),
            "terms": self.last_reward_terms.copy(),
            "episode_steps": self.episode_steps,
            "best_progress": self.best_progress,
            "best_layer_depth": self.best_layer_depth,
            "last_best_depth_step": self.last_best_depth_step,
            "inactive_steps": self.inactive_steps,
            "mobile_stagnant_steps": self.mobile_stagnant_steps,
            "watchdog": watchdog,
        }


class Client:
    def __init__(self, host, diagnostics_root, timeout, obs_port=OBS_PORT, action_port=ACTION_PORT,
                 failure_probe=None):
        self.host = host
        self.diagnostics_root = Path(diagnostics_root)
        self.timeout = timeout
        self.startup_status_path = Path(
            os.environ.get("PPO_AUTOSTART_STATUS_FILE", "counterfactual_autostart_status.json")
        )
        self.obs_port = obs_port
        self.action_port = action_port
        self.obs_listener = self._listener(self.obs_port)
        self.action_listener = self._listener(self.action_port)
        self.obs = None
        self.action = None
        self.observations = queue.Queue()
        self.acks = queue.Queue()
        self.stop = threading.Event()
        self.session_token = None
        self.command_timings = []
        self.observation_timings = []
        # PrefixReplay supplies a probe for the exact launch it owns.  The
        # standalone client has no parent process to inspect, so it retains
        # the ordinary socket-timeout behavior.
        self.failure_probe = failure_probe
        self.connection_error = None
        self._observation_thread = None
        self._ack_thread = None

    def _listener(self, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, port))
        sock.listen(1)
        return sock

    def connect(self):
        self.obs = self._accept(self.obs_listener, "observation")
        self.action = self._accept(self.action_listener, "action")
        self.obs.settimeout(self.timeout)
        self.action.settimeout(self.timeout)
        self._observation_thread = threading.Thread(target=self._read_observations, daemon=True)
        self._ack_thread = threading.Thread(target=self._read_acks, daemon=True)
        self._observation_thread.start()
        self._ack_thread.start()

    def _close_accepted_channels_for_reconnect(self):
        """Close only accepted protocol sockets, retaining both listeners."""
        self.stop.set()
        for connection in (self.obs, self.action):
            try:
                if connection is not None:
                    connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                if connection is not None:
                    connection.close()
            except OSError:
                pass
        for thread in (self._observation_thread, self._ack_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=1.0)
        self.obs = None
        self.action = None
        self.connection_error = None
        self.observations = queue.Queue()
        self.acks = queue.Queue()
        self.stop = threading.Event()
        self._observation_thread = None
        self._ack_thread = None

    def detach_and_reconnect(self, detached_probe=None):
        """Replace transport at a held policy boundary without resetting Unity."""
        detached = self.command("CF_DETACH")
        if not detached.get("detached") or not detached.get("hold"):
            raise RuntimeError(f"Unity did not enter detach hold: {detached}")
        self._close_accepted_channels_for_reconnect()
        if detached_probe is not None:
            detached_probe()
        self.connect()
        attached = self.command("CF_ATTACH")
        if not attached.get("attached") or attached.get("reset_used"):
            raise RuntimeError(f"Unity did not attach without reset: {attached}")
        return detached, attached

    def _accept(self, listener, channel):
        """Accept Unity while surfacing an explicit autostart failure."""
        listener.settimeout(0.5)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            failure = self._autostart_failure()
            if failure:
                raise RuntimeError(f"Unity autostart failed before {channel} connection: {failure}")
            failure = self._launch_failure()
            if failure:
                raise RuntimeError(f"Unity launch failed before {channel} connection: {failure}")
            try:
                connection, _ = listener.accept()
                return connection
            except socket.timeout:
                continue
        failure = self._autostart_failure()
        if failure:
            raise RuntimeError(f"Unity autostart failed before {channel} connection: {failure}")
        failure = self._launch_failure()
        if failure:
            raise RuntimeError(f"Unity launch failed before {channel} connection: {failure}")
        raise TimeoutError(f"Unity did not connect its {channel} socket within {self.timeout:.1f}s")

    def _launch_failure(self):
        if self.failure_probe is None:
            return None
        try:
            return self.failure_probe()
        except Exception as exc:
            return f"launch-status probe failed: {exc}"

    def _autostart_failure(self):
        try:
            with self.startup_status_path.open() as stream:
                status = json.load(stream)
            if status.get("status") == "failed":
                return status.get("error") or "unspecified startup failure"
        except (OSError, ValueError, TypeError):
            pass
        return None

    def _read_observations(self):
        connection = self.obs
        try:
            while not self.stop.is_set():
                data = self._recv_exact(connection, MESSAGE_SIZE)
                payload = data[:OBSERVATION_SIZE]
                observation_id = int.from_bytes(data[OBSERVATION_SIZE:], "little")
                observation = np.frombuffer(payload, dtype=Types.OBSERVATION_DTYPE, count=1)[0].copy()
                self.observations.put((observation_id, payload + data[OBSERVATION_SIZE:], observation))
        except (OSError, ConnectionError, queue.Empty) as exc:
            self.connection_error = f"observation channel closed: {exc}"
            return

    def _read_acks(self):
        connection = self.action
        buffer = b""
        try:
            while not self.stop.is_set():
                data = connection.recv(4096)
                if not data:
                    return
                buffer += data
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if line.startswith(b"CF "):
                        self.acks.put(json.loads(line[3:].decode("utf-8")))
                    elif line.startswith(b"RESET_READY "):
                        parts = line.split()
                        self.acks.put({
                            "reset_ready": True,
                            "token": int(parts[1]),
                            "observation_id": int(parts[2]),
                            "seed": int(parts[3]),
                        })
        except (OSError, ValueError, IndexError) as exc:
            self.connection_error = f"action channel closed: {exc}"
            return

    @staticmethod
    def _recv_exact(connection, size):
        chunks = []
        while size:
            chunk = connection.recv(size)
            if not chunk:
                raise ConnectionError("Unity observation connection closed")
            chunks.append(chunk)
            size -= len(chunk)
        return b"".join(chunks)

    def command(self, command):
        started = time.perf_counter()
        outcome = "error"
        try:
            self.action.sendall((command + "\n").encode("ascii"))
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                failure = self._launch_failure()
                if failure:
                    raise RuntimeError(f"Unity launch failed while waiting for {command}: {failure}")
                if self.connection_error:
                    raise RuntimeError(f"Unity connection failed while waiting for {command}: {self.connection_error}")
                try:
                    response = self.acks.get(timeout=max(0.01, deadline - time.monotonic()))
                except queue.Empty:
                    break
                if "ok" in response:
                    if not response["ok"]:
                        raise RuntimeError(f"Unity rejected {command}: {response.get('error')}")
                    outcome = "ack"
                    return response
                # RESET_READY belongs to the reset command and is consumed by reset().
                self.acks.put(response)
            outcome = "timeout"
            raise TimeoutError(f"No acknowledgement for {command}")
        finally:
            self.command_timings.append({
                "command": command.split(" ", 1)[0],
                "outcome": outcome,
                "round_trip_ms": (time.perf_counter() - started) * 1000.0,
            })

    def next_observation(self, expected_id=None):
        started = time.perf_counter()
        outcome = "error"
        try:
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                failure = self._launch_failure()
                if failure:
                    raise RuntimeError(f"Unity launch failed while waiting for observation: {failure}")
                if self.connection_error:
                    raise RuntimeError(f"Unity connection failed while waiting for observation: {self.connection_error}")
                try:
                    item = self.observations.get(timeout=max(0.01, deadline - time.monotonic()))
                except queue.Empty:
                    break
                if expected_id is None or item[0] == expected_id:
                    outcome = "received"
                    return item
                if item[0] > expected_id:
                    raise RuntimeError(f"Observation ID jumped: expected {expected_id}, got {item[0]}")
            outcome = "timeout"
            raise TimeoutError(f"No observation {expected_id if expected_id is not None else ''}")
        finally:
            self.observation_timings.append({
                "expected_id": expected_id,
                "outcome": outcome,
                "wait_ms": (time.perf_counter() - started) * 1000.0,
            })

    def wait_reset_ready(self, token):
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            response = self.acks.get(timeout=max(0.01, deadline - time.monotonic()))
            if response.get("reset_ready"):
                if response["token"] != token:
                    raise RuntimeError(f"Unexpected reset token {response['token']}")
                return response
            self.acks.put(response)
        raise TimeoutError("No RESET_READY acknowledgement")

    def close(self):
        if self.session_token:
            path = self.diagnostics_root / self.session_token / "python_transport_timing.json"
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w") as stream:
                    json.dump({
                        "commands": self.command_timings,
                        "observations": self.observation_timings,
                    }, stream, indent=2)
            except OSError:
                pass
        self.stop.set()
        if self.action is not None:
            try:
                self.action.sendall(b"SHUTDOWN\n")
            except OSError:
                pass
        for connection in (self.obs, self.action, self.obs_listener, self.action_listener):
            try:
                if connection:
                    connection.close()
            except OSError:
                pass


def load_report(root: Path, filename: str, timeout: float):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        matches = list(root.glob(f"**/{filename}"))
        if matches:
            with matches[0].open() as stream:
                return json.load(stream)
        time.sleep(0.02)
    raise TimeoutError(f"Unity report {filename} was not found below {root}")


def snapshot_match(left, right):
    for key in ("state", "arrays", "observation_wire_sha256", "observation_sha256",
                "state_sha256", "observation_base64"):
        difference = first_difference(left.get(key), right.get(key), key)
        if difference:
            return difference
    return None


def causal_topology_keys(records):
    """Canonical topology view matching Unity's presentation exclusion rule."""
    return sorted(
        record["identity"]
        for record in records
        if record.get("classification") not in PRESENTATION_CLASSIFICATIONS
    )


def causal_lifecycle_keys(records):
    """Canonical lifecycle view; only the evidence-bound UI leaf is ignored."""
    return sorted(
        record["identity"]
        for record in records
        if record.get("classification") not in PRESENTATION_LIFECYCLE_CLASSIFICATIONS
    )


def lifecycle_values(records):
    """Values that remain causal after presentation lifecycle canonicalization."""
    return sorted(
        (record["identity"], record.get("enabled"))
        for record in records
        if record.get("classification") not in PRESENTATION_LIFECYCLE_CLASSIFICATIONS
    )


def run(args):
    tape = default_tape(args.horizon)
    if args.tape:
        with Path(args.tape).open() as stream:
            tape = json.load(stream)
        if not isinstance(tape, list) or not tape:
            raise ValueError("tape must be a non-empty JSON list")
    token = args.session or ("aa-" + secrets.token_hex(10))
    root = Path(args.diagnostics_root)
    client = Client(args.host, root, args.timeout, args.obs_port, args.action_port)
    client.session_token = token
    result = {"schema": 1, "session": token, "horizon": len(tape), "status": "failed"}
    try:
        print(f"Waiting for Unity on {args.host}:{args.obs_port}/{args.action_port}...")
        client.connect()
        hello_deadline = time.monotonic() + args.timeout
        while True:
            try:
                hello = client.command("CF_HELLO " + token)
                break
            except RuntimeError as exc:
                # Unity can accept the TCP connection before procedural
                # generation has completed. Keep the command queued at the
                # boundary, but retry the handshake until the world is safe.
                if "CB1 runtime guards failed:" not in str(exc) or time.monotonic() >= hello_deadline:
                    raise
                time.sleep(0.25)
        reset = client.command(f"CF_RESET {args.seed}")
        ready = client.wait_reset_ready(reset["reset_token"])
        baseline = client.next_observation(ready["observation_id"])
        capture = client.command("CF_CAPTURE")
        baseline_report = load_report(root, capture["file"], args.timeout)
        python_state = PythonReplayState(baseline[2])

        def branch():
            records = []
            reports = []
            expected_id = baseline[0]
            state = PythonReplayState(baseline[2])
            for step, action in enumerate(tape, 1):
                client.command("CF_STEP " + " ".join(str(value) for value in action))
                expected_id += 1
                observation_id, raw, observation = client.next_observation(expected_id)
                state_record = client.command("CF_STATE")
                unity_report = load_report(root, state_record["file"], args.timeout)
                reports.append(unity_report)
                records.append({
                    "step": step,
                    "action": action,
                    "observation_id": observation_id,
                    "observation_wire_sha256": hashlib.sha256(raw).hexdigest(),
                    "observation_sha256": hashlib.sha256(raw).hexdigest(),
                    "observation_base64": base64.b64encode(raw).decode("ascii"),
                    "python": state.step(observation, action),
                    "unity_state_sha256": unity_report["state_sha256"],
                    "unity_state_sections": unity_report.get("state_sections", {}),
                    "unity_report": state_record["file"],
                })
            return records, reports

        branch_a, reports_a = branch()
        restore = client.command("CF_RESTORE")
        restored_report = load_report(root, restore["file"], args.timeout)
        restore_difference = snapshot_match(baseline_report, restored_report)
        branch_b, reports_b = branch()
        restore_again = client.command("CF_RESTORE")
        restored_again_report = load_report(root, restore_again["file"], args.timeout)
        restore_again_difference = snapshot_match(baseline_report, restored_again_report)
        branch_c, reports_c = branch()

        first = None
        repeated_restore_first = None

        def compare(left_records, right_records, left_reports, right_reports):
            for index, (left, right, left_report, right_report) in enumerate(
                zip(left_records, right_records, left_reports, right_reports), 1
            ):
                difference = observation_difference(
                    base64.b64decode(left["observation_base64"]),
                    base64.b64decode(right["observation_base64"]),
                )
                if difference is None:
                    if left["unity_state_sha256"] != right["unity_state_sha256"]:
                        difference = first_difference(
                            left.get("unity_state_sections", {}),
                            right.get("unity_state_sections", {}),
                            "unity.state_sections",
                        )
                        if difference is None:
                            difference = {
                                "path": "unity.state_sha256",
                                "left": left["unity_state_sha256"],
                                "right": right["unity_state_sha256"],
                            }
                if difference is None:
                    difference = first_difference(left["python"], right["python"], "python")
                if difference:
                    difference["likely_writers"] = likely_writers(difference["path"])
                    difference["prior_matching_step"] = index - 1
                    return {"policy_step": index, "difference": difference}
            if len(left_records) != len(right_records):
                return {"policy_step": min(len(left_records), len(right_records)) + 1,
                        "difference": {"path": "trajectory.length",
                                       "left": len(left_records), "right": len(right_records)}}
            return None

        first = compare(branch_a, branch_b, reports_a, reports_b)
        repeated_restore_first = compare(branch_a, branch_c, reports_a, reports_c)

        coverage_gaps = baseline_report.get("coverage_gaps", [])
        trajectory_match = first is None and repeated_restore_first is None
        restore_match = restore_difference is None and restore_again_difference is None
        if not trajectory_match or not restore_match:
            status = "A/A_FAIL"
        elif coverage_gaps:
            # A semantic match is useful evidence, but the protocol forbids
            # calling it A/A_PASS while engine-owned state remains uncovered.
            status = "A/A_BLOCKED"
        else:
            status = "A/A_PASS"

        result.update({
            "status": status,
            "semantic_match": trajectory_match and restore_match,
            "scientifically_validated": False,
            "python_participates": True,
            "hello": hello,
            "reset": reset,
            "baseline_report": capture["file"],
            "restore_report": restore["file"],
            "restore_difference": restore_difference,
            "restore_again_report": restore_again["file"],
            "restore_again_difference": restore_again_difference,
            "first_divergence": first,
            "repeated_restore_first_divergence": repeated_restore_first,
            "branch_a": branch_a,
            "branch_b": branch_b,
            "branch_c": branch_c,
            "coverage_gaps": coverage_gaps,
            "tape": tape,
        })
        out = root / token / "replay_result.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
        print(json.dumps({k: result[k] for k in ("status", "first_divergence", "restore_difference", "scientifically_validated")}, indent=2))
        return 0 if result["status"] == "A/A_PASS" else 2
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=391728405)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--tape", help="JSON file containing categorical seven-action rows")
    parser.add_argument("--session", help="lowercase diagnostic session token")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--obs-port", type=int, default=int(os.environ.get("PPO_OBS_PORT", OBS_PORT)))
    parser.add_argument("--action-port", type=int, default=int(os.environ.get("PPO_ACTION_PORT", ACTION_PORT)))
    parser.add_argument("--diagnostics-root", default="counterfactual_diagnostics")
    # A full causal restore includes native component writes and collider
    # regeneration on the current CB1 scene.  Keep this bounded, but longer
    # than the ordinary command latency so a valid restore is not mislabeled
    # as a missing acknowledgement.
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    if args.horizon < 1 or args.horizon > 10000:
        parser.error("horizon must be between 1 and 10000")
    try:
        return run(args)
    except Exception as exc:
        print(f"counterfactual replay failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
