"""Bounded CB1 transport smoke: reset, categorical action tape, and state capture."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
from pathlib import Path

from CounterfactualReplay import Client


def read_session_report(session_dir, filename, timeout):
    report_path = session_dir / "unity" / filename
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if report_path.is_file():
            with report_path.open() as stream:
                return json.load(stream)
        time.sleep(0.02)
    raise TimeoutError(f"Unity report was not written: {report_path}")


def run(args):
    root = Path(args.diagnostics_root).resolve()
    token = args.session or ("cb1-smoke-" + secrets.token_hex(8))
    session = root / token
    session.mkdir(parents=True, exist_ok=False)
    client = Client("127.0.0.1", root, args.timeout, args.obs_port, args.action_port)
    client.session_token = token
    actions = json.loads(args.actions_json)
    if not isinstance(actions, list) or not actions or len(actions) > 256:
        raise ValueError("actions must be a non-empty JSON list of at most 256 action rows")
    if any(not isinstance(row, list) or len(row) != 7 or any(type(x) is not int for x in row) for row in actions):
        raise ValueError("each smoke action must contain exactly seven integer categories")
    result = {
        "schema": 1,
        "session": token,
        "seed": args.seed,
        "actions": actions,
        "ports": {"observation": args.obs_port, "action": args.action_port},
        "status": "failed",
        "listener_ready_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        # Client construction binds both listeners before this readiness marker.
        (session / "listeners_ready.json").write_text(json.dumps({
            "host": "127.0.0.1", "observation_port": args.obs_port,
            "action_port": args.action_port, "pid": os.getpid(),
        }, indent=2))
        client.connect()
        hello_deadline = time.monotonic() + args.timeout
        while True:
            try:
                hello = client.command("CF_HELLO " + token)
                break
            except RuntimeError as exc:
                if "CB1 runtime guards failed:" not in str(exc) or time.monotonic() >= hello_deadline:
                    raise
                time.sleep(0.25)
        reset = client.command(f"CF_RESET {args.seed}")
        reset_ready = client.wait_reset_ready(reset["reset_token"])
        initial_id, _, initial_obs = client.next_observation(reset_ready["observation_id"])
        if args.skip_state:
            initial_state = {"file": None}
            initial_report = {"state_sha256": None}
        else:
            initial_state = client.command("CF_STATE")
            initial_report = read_session_report(session, initial_state["file"], args.timeout)

        steps = []
        observation_ids = [initial_id]
        previous_id = initial_id
        next_obs = initial_obs
        next_report = initial_report
        next_state = initial_state
        for step_index, action in enumerate(actions, 1):
            action_ack = client.command("CF_STEP " + " ".join(map(str, action)))
            next_id, _, next_obs = client.next_observation(previous_id + 1)
            if args.skip_state:
                next_state = {"file": None}
                next_report = {"state_sha256": None}
            else:
                next_state = client.command("CF_STATE")
                next_report = read_session_report(session, next_state["file"], args.timeout)
            observation_ids.append(next_id)
            steps.append({
                "index": step_index,
                "action": action,
                "action_ack": action_ack,
                "observation_id": next_id,
                "state_file": next_state["file"],
                "state_sha256": next_report["state_sha256"],
                "player_tile_position": next_obs["PlayerTilePosition"].tolist(),
                "velocity": next_obs["Velocity"].tolist(),
                "grounded": bool(next_obs["Grounded"]),
            })
            previous_id = next_id
        result.update({
            "status": "passed",
            "hello": hello,
            "reset": reset,
            "reset_ready": reset_ready,
            "observation_ids": observation_ids,
            "steps": steps,
            "initial_state_file": initial_state["file"],
            "next_state_file": next_state["file"],
            "initial_state_sha256": initial_report["state_sha256"],
            "next_state_sha256": next_report["state_sha256"],
            "initial_observation": {
                "player_tile_position": initial_obs["PlayerTilePosition"].tolist(),
                "velocity": initial_obs["Velocity"].tolist(),
                "grounded": bool(initial_obs["Grounded"]),
            },
            "next_observation": {
                "player_tile_position": next_obs["PlayerTilePosition"].tolist(),
                "velocity": next_obs["Velocity"].tolist(),
                "grounded": bool(next_obs["Grounded"]),
            },
            "commands": client.command_timings,
            "observations": client.observation_timings,
        })
        return 0
    except Exception as exc:
        result["error"] = repr(exc)
        return 1
    finally:
        client.close()
        result["commands"] = client.command_timings
        result["observations"] = client.observation_timings
        result["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (session / "smoke_result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
        print(json.dumps({k: result.get(k) for k in ("status", "session", "observation_ids", "error")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--diagnostics-root", required=True)
    parser.add_argument("--obs-port", type=int, required=True)
    parser.add_argument("--action-port", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--actions-json", default='[[1,0,1,0,4,5,0]]')
    parser.add_argument("--skip-state", action="store_true",
                        help="do not request the (multi-second) CF_STATE semantic snapshot per step")
    args = parser.parse_args()
    if not (1024 <= args.obs_port <= 65535 and 1024 <= args.action_port <= 65535):
        parser.error("ports must be in [1024,65535]")
    if args.obs_port == args.action_port:
        parser.error("observation and action ports must differ")
    try:
        return run(args)
    except Exception as exc:
        print(f"counterfactual smoke failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
