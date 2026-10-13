#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUNTIME_ROOT="${PPO_COUNTERFACTUAL_RUNTIME_ROOT:-}"
[[ -n "$RUNTIME_ROOT" ]] || { echo "error: PPO_COUNTERFACTUAL_RUNTIME_ROOT is required; refusing canonical-runtime fallback" >&2; exit 2; }
[[ "$RUNTIME_ROOT" = /* ]] || { echo "error: staged runtime root must be absolute" >&2; exit 2; }
RUNTIME_ROOT="$(realpath -e -- "$RUNTIME_ROOT")" || { echo "error: staged runtime root does not exist" >&2; exit 2; }
[[ -f "$RUNTIME_ROOT/runtime_manifest.json" ]] || { echo "error: staged runtime manifest is missing" >&2; exit 2; }
GAME_EXE="$RUNTIME_ROOT/game/CasualtiesUnknown.exe"
PROTON="$RUNTIME_ROOT/steam/steamapps/common/Proton - Experimental/proton"
COMPAT_DATA="$RUNTIME_ROOT/steam/steamapps/compatdata/4576510"
DEPLOY_SCRIPT="$RUNTIME_ROOT/launcher/Deploy.sh"
CONTAINMENT_LAUNCHER="$RUNTIME_ROOT/launcher/LaunchContained.sh"
DIAGNOSTICS_ROOT="$RUNTIME_ROOT/user-home/diagnostics"
STATUS_FILE="${PPO_AUTOSTART_STATUS_FILE:-$RUNTIME_ROOT/user-home/counterfactual_autostart_status.json}"
UNITY_LOG="${PPO_AUTOSTART_UNITY_LOG:-$RUNTIME_ROOT/user-home/counterfactual_autostart_unity.log}"
PROGRESS_FILE="${PPO_AUTOSTART_PROGRESS_FILE:-$RUNTIME_ROOT/user-home/counterfactual_batch_progress.jsonl}"
SMOKE_CLIENT_LOG="$RUNTIME_ROOT/user-home/counterfactual_smoke_client.log"
for required_path in "$GAME_EXE" "$PROTON" "$COMPAT_DATA" "$DEPLOY_SCRIPT" "$CONTAINMENT_LAUNCHER"; do
    [[ -e "$required_path" ]] || { echo "error: required staged path is missing: $required_path" >&2; exit 2; }
done
[[ -x "$PROTON" && -x "$DEPLOY_SCRIPT" && -x "$CONTAINMENT_LAUNCHER" ]] || { echo "error: staged Proton and launchers must be executable" >&2; exit 2; }
[[ "$(realpath -e -- "$GAME_EXE")" == "$RUNTIME_ROOT/game/CasualtiesUnknown.exe" ]] || { echo "error: staged game executable resolves outside isolated runtime" >&2; exit 2; }
for output_path in "$STATUS_FILE" "$UNITY_LOG" "$PROGRESS_FILE" "$DIAGNOSTICS_ROOT"; do
    case "$(realpath -m -- "$output_path")" in "$RUNTIME_ROOT"/*) ;; *) echo "error: output path escapes staged runtime: $output_path" >&2; exit 2 ;; esac
done

if [[ $# -eq 0 ]]; then
    set -- --seed 391728405 --horizon 100
fi

repeat="${PPO_AUTOSTART_REPEAT:-1}"
replay_mode="${PPO_COUNTERFACTUAL_REPLAY_MODE:-snapshot}"
failure_policy="${PPO_AUTOSTART_BATCH_FAILURE_POLICY:-fail}"
retries="${PPO_AUTOSTART_BATCH_RETRIES:-0}"
if ! [[ "$repeat" =~ ^[0-9]+$ ]]; then
    echo "error: PPO_AUTOSTART_REPEAT must be 0 (indefinite) or a positive integer" >&2
    exit 2
fi
if [[ "$replay_mode" != "snapshot" && "$replay_mode" != "prefix" && "$replay_mode" != "smoke" ]]; then
    echo "error: PPO_COUNTERFACTUAL_REPLAY_MODE must be snapshot, prefix, or smoke" >&2
    exit 2
fi
if [[ "$repeat" -eq 0 ]]; then
    echo "Unattended counterfactual repeat mode: indefinite"
fi
if [[ "$failure_policy" != "fail" && "$failure_policy" != "skip" && "$failure_policy" != "retry" ]]; then
    echo "error: PPO_AUTOSTART_BATCH_FAILURE_POLICY must be fail, skip, or retry" >&2
    exit 2
fi
if ! [[ "$retries" =~ ^[0-9]+$ ]]; then
    echo "error: PPO_AUTOSTART_BATCH_RETRIES must be a non-negative integer" >&2
    exit 2
fi
if [[ "$failure_policy" == "retry" && "$retries" -eq 0 ]]; then
    echo "error: retry policy requires PPO_AUTOSTART_BATCH_RETRIES > 0" >&2
    exit 2
fi

cd -- "$PROJECT_ROOT"
export PPO_COUNTERFACTUAL_DIAGNOSTICS_ROOT="$DIAGNOSTICS_ROOT"
export PPO_COUNTERFACTUAL_DEPLOY_SCRIPT="$DEPLOY_SCRIPT"
export PPO_COUNTERFACTUAL_CONTAINMENT_LAUNCHER="$CONTAINMENT_LAUNCHER"
unity_pid=0
client_pid=0
owned_pids=()
iteration=0
attempt=1

record_progress() {
    local status="$1"
    local timestamp
    timestamp="$(date +%s)"
    mkdir -p -- "$(dirname -- "$PROGRESS_FILE")"
    printf '{"iteration":%d,"attempt":%d,"mode":"%s","exit_status":%d,"timestamp":%s}\n' \
        "$iteration" "$attempt" "$replay_mode" "$status" "$timestamp" >> "$PROGRESS_FILE"
}

cleanup() {
    if [[ "$unity_pid" =~ ^[1-9][0-9]*$ ]]; then
        terminate_owned_tree "$unity_pid"
    fi
    if [[ "$client_pid" =~ ^[1-9][0-9]*$ ]]; then
        terminate_owned_tree "$client_pid"
    fi
}

record_owned_tree() {
    local parent="$1"
    local child
    if ! [[ "$parent" =~ ^[1-9][0-9]*$ ]] || [[ ! -r "/proc/$parent/task/$parent/children" ]]; then
        return 0
    fi
    owned_pids+=("$parent")
    while read -r child; do
        if [[ "$child" =~ ^[1-9][0-9]*$ ]]; then
            record_owned_tree "$child"
        fi
    done < "/proc/$parent/task/$parent/children"
}

terminate_owned_tree() {
    local root="$1"
    owned_pids=()
    record_owned_tree "$root"
    if [[ "${#owned_pids[@]}" -eq 0 ]]; then
        echo "Refusing cleanup: launch PID is not a live positive owned PID: $root" >&2
        return 0
    fi
    echo "Terminating owned launch PIDs: ${owned_pids[*]}" >&2
    local index
    for ((index=${#owned_pids[@]} - 1; index >= 0; index--)); do
        local pid="${owned_pids[$index]}"
        if [[ "$pid" =~ ^[1-9][0-9]*$ ]]; then
            kill "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT INT TERM

run_once() {
    local -a launch_command=("$CONTAINMENT_LAUNCHER" "$DEPLOY_SCRIPT")
    if [[ "$replay_mode" == "smoke" ]]; then
        local obs_port action_port session_token session_dir ready_file client_status launch_status
        if [[ -n "${PPO_OBS_PORT:-}" && -n "${PPO_ACTION_PORT:-}" ]]; then
            obs_port="$PPO_OBS_PORT"
            action_port="$PPO_ACTION_PORT"
        else
            read -r obs_port action_port < <("$PROJECT_ROOT/.venv/bin/python" -c 'import socket; a=socket.socket(); b=socket.socket(); a.bind(("127.0.0.1",0)); b.bind(("127.0.0.1",0)); print(a.getsockname()[1], b.getsockname()[1]); a.close(); b.close()')
        fi
        [[ "$obs_port" =~ ^[0-9]+$ && "$action_port" =~ ^[0-9]+$ && "$obs_port" != "$action_port" ]] || {
            echo "error: smoke observation/action ports must be distinct TCP ports" >&2; return 2;
        }
        (( obs_port >= 1024 && obs_port <= 65535 && action_port >= 1024 && action_port <= 65535 )) || {
            echo "error: smoke ports must be in [1024,65535]" >&2; return 2;
        }
        command -v ss >/dev/null 2>&1 || { echo "error: ss is required for staged smoke readiness checks" >&2; return 2; }
        [[ -z "$(ss -H -ltn "sport = :$obs_port")" && -z "$(ss -H -ltn "sport = :$action_port")" ]] || {
            echo "error: smoke port is already in use; choose a fresh port pair" >&2; return 2;
        }
        session_token="cb1-smoke-${iteration}-${attempt}-$(date +%s)"
        session_dir="$DIAGNOSTICS_ROOT/$session_token"
        player_ticks_output_wine="S:\\user-home\\diagnostics\\${session_token}\\player-fixed-tick.jsonl"
        player_post_ticks_output_wine="S:\\user-home\\diagnostics\\${session_token}\\player-post-physics.jsonl"
        occ_output_wine="S:\\user-home\\diagnostics\\${session_token}\\occupancy-region.jsonl"
        occ_full_output_wine="S:\\user-home\\diagnostics\\${session_token}\\occupancy-full.bin"
        player_prefixed_output_wine="S:\\user-home\\diagnostics\\${session_token}\\player-pre-fixed.jsonl"
        ready_file="$session_dir/listeners_ready.json"
        rm -f -- "$STATUS_FILE"
        echo "Starting CB1 smoke client listeners first on 127.0.0.1:$obs_port/$action_port..."
        PPO_COUNTERFACTUAL_DIAGNOSTICS_ROOT="$DIAGNOSTICS_ROOT" \
        PPO_OBS_PORT="$obs_port" PPO_ACTION_PORT="$action_port" \
        PYTHONPATH="$PROJECT_ROOT/PPO_Python${PYTHONPATH:+:$PYTHONPATH}" \
            "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/PPO_Python/CounterfactualSmoke.py" \
            --seed "${PPO_COUNTERFACTUAL_SMOKE_SEED:-391728405}" \
            --session "$session_token" --diagnostics-root "$DIAGNOSTICS_ROOT" \
            --obs-port "$obs_port" --action-port "$action_port" \
            --timeout "${PPO_COUNTERFACTUAL_SMOKE_TIMEOUT:-90}" \
            --actions-json "${PPO_COUNTERFACTUAL_SMOKE_ACTIONS:-[[1,0,1,0,4,5,0]]}" \
            $([[ "${PPO_COUNTERFACTUAL_SMOKE_SKIP_STATE:-0}" == "1" ]] && echo --skip-state) \
            >"$SMOKE_CLIENT_LOG" 2>&1 &
        client_pid=$!
        local ready_deadline=$((SECONDS + 20))
        while [[ ! -f "$ready_file" && "$SECONDS" -lt "$ready_deadline" ]]; do
            if [[ ! -e "/proc/$client_pid" ]]; then
                wait "$client_pid" || client_status=$?
                client_pid=0
                cat -- "$SMOKE_CLIENT_LOG" >&2
                echo "error: CB1 smoke client failed before both listeners were ready" >&2
                return "${client_status:-1}"
            fi
            sleep 0.1
        done
        if [[ ! -f "$ready_file" ]]; then
            terminate_owned_tree "$client_pid"; client_pid=0
            echo "error: CB1 smoke listeners did not become ready" >&2
            return 1
        fi
        echo "Starting exact staged game only after both client listeners are ready..."
        PPO_COUNTERFACTUAL=1 PPO_AUTOSTART=1 \
        PPO_COUNTERFACTUAL_RUNTIME_ROOT="$RUNTIME_ROOT" PPO_COUNTERFACTUAL_SKIP_BUILD=1 \
        PPO_OBS_PORT="$obs_port" PPO_ACTION_PORT="$action_port" \
        PPO_AUTOSTART_STATUS_FILE="$STATUS_FILE" \
        PPO_COUNTERFACTUAL_DIAGNOSTICS_ROOT="$DIAGNOSTICS_ROOT" \
        PPO_COUNTERFACTUAL_SESSION="$session_token" \
        PPO_DIAG_RAGDOLL_CALLS="${PPO_DIAG_RAGDOLL_CALLS:-0}" \
        PPO_DIAG_STANDING_FIXTURE="${PPO_DIAG_STANDING_FIXTURE:-0}" \
        PPO_DIAG_PLAYER_FIXTURE_GRID_X="${PPO_DIAG_PLAYER_FIXTURE_GRID_X:-}" \
        PPO_DIAG_PLAYER_FIXTURE_GRID_Y="${PPO_DIAG_PLAYER_FIXTURE_GRID_Y:-}" \
        PPO_DIAG_PLAYER_FIXTURE_TERRAIN="${PPO_DIAG_PLAYER_FIXTURE_TERRAIN:-}" \
        PPO_DIAG_PLAYER_TICKS="${PPO_DIAG_PLAYER_TICKS:-0}" \
        PPO_DIAG_PLAYER_TICKS_OUTPUT="${PPO_DIAG_PLAYER_TICKS_OUTPUT:-$player_ticks_output_wine}" \
        PPO_DIAG_PLAYER_POST_PHYSICS="${PPO_DIAG_PLAYER_POST_PHYSICS:-0}" \
        PPO_DIAG_PLAYER_POST_PHYSICS_OUTPUT="${PPO_DIAG_PLAYER_POST_PHYSICS_OUTPUT:-$player_post_ticks_output_wine}" \
        PPO_DIAG_OCC_FULL="${PPO_DIAG_OCC_FULL:-0}" PPO_DIAG_OCC_REGION="${PPO_DIAG_OCC_REGION:-}" \
        PPO_DIAG_OCC_EVERY="${PPO_DIAG_OCC_EVERY:-0}" PPO_DIAG_OCC_FULL_AT="${PPO_DIAG_OCC_FULL_AT:-0}" \
        PPO_DIAG_OCC_FULL_OUTPUT="${PPO_DIAG_OCC_FULL_OUTPUT:-$occ_full_output_wine}" \
        PPO_DIAG_OCC_OUTPUT="${PPO_DIAG_OCC_OUTPUT:-$occ_output_wine}" \
        PPO_DIAG_PLAYER_PREFIXED="${PPO_DIAG_PLAYER_PREFIXED:-0}" \
        PPO_DIAG_PLAYER_PREFIXED_OUTPUT="${PPO_DIAG_PLAYER_PREFIXED_OUTPUT:-$player_prefixed_output_wine}" \
        PPO_DIAG_PLAYER_PHYSICS_TRACE="${PPO_DIAG_PLAYER_PHYSICS_TRACE:-0}" \
            "${launch_command[@]}" >"$UNITY_LOG" 2>&1 &
        unity_pid=$!
        set +e
        wait "$client_pid"
        client_status=$?
        set -e
        client_pid=0
        if [[ "$client_status" -ne 0 ]]; then
            echo "error: CB1 smoke client failed; see $SMOKE_CLIENT_LOG" >&2
            cleanup
            return "$client_status"
        fi
        # SHUTDOWN is sent by the established Client.close() path. Give the
        # exact launch root a bounded graceful exit, then clean only its tree.
        local exit_deadline=$((SECONDS + 15))
        while [[ -e "/proc/$unity_pid" && "$SECONDS" -lt "$exit_deadline" ]]; do sleep 0.1; done
        if [[ -e "/proc/$unity_pid" ]]; then terminate_owned_tree "$unity_pid"; fi
        set +e
        wait "$unity_pid"
        launch_status=$?
        set -e
        unity_pid=0
        [[ "$launch_status" -eq 0 ]] || { echo "error: staged launch exited $launch_status; see $UNITY_LOG" >&2; return "$launch_status"; }
        cat -- "$session_dir/smoke_result.json"
        return 0
    fi
    if [[ "$replay_mode" == "prefix" ]]; then
        echo "Launching unattended fresh-process prefix replay..."
        local -a prefix_args=("$@")
        if [[ "$repeat" -ne 1 ]]; then
            # PrefixReplay creates an immutable artifact directory keyed by
            # --run-id. Repeated/indefinite batches must never collide, even
            # when the caller supplied a convenient base run ID.
            prefix_args+=(--run-id "batch-iteration-${iteration}-attempt-${attempt}")
            echo "Prefix batch artifact ID: batch-iteration-${iteration}-attempt-${attempt}"
        fi
        set +e
        PPO_COUNTERFACTUAL_DIAGNOSTICS_ROOT="$DIAGNOSTICS_ROOT" \
        PYTHONPATH="$PROJECT_ROOT/PPO_Python${PYTHONPATH:+:$PYTHONPATH}" \
            "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/PPO_Python/PrefixReplay.py" "${prefix_args[@]}"
        local replay_status=$?
        set -e
        return "$replay_status"
    fi
    rm -f -- "$STATUS_FILE"
    echo "Launching unattended counterfactual Unity process..."
    PPO_COUNTERFACTUAL=1 \
    PPO_AUTOSTART=1 \
    PPO_COUNTERFACTUAL_RUNTIME_ROOT="$RUNTIME_ROOT" \
    PPO_COUNTERFACTUAL_SKIP_BUILD=1 \
    PPO_AUTOSTART_STATUS_FILE="$STATUS_FILE" \
    PPO_COUNTERFACTUAL_DIAGNOSTICS_ROOT="$DIAGNOSTICS_ROOT" \
    "${launch_command[@]}" >"$UNITY_LOG" 2>&1 &
    unity_pid=$!

    set +e
    PYTHONPATH="$PROJECT_ROOT/PPO_Python${PYTHONPATH:+:$PYTHONPATH}" \
        "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/PPO_Python/CounterfactualReplay.py" "$@"
    local replay_status=$?
    set -e

    if [[ "$replay_status" -ne 0 ]]; then
        cleanup
    fi
    wait "$unity_pid" 2>/dev/null || true
    unity_pid=0
    return "$replay_status"
}

while [[ "$repeat" -eq 0 || "$iteration" -lt "$repeat" ]]; do
    iteration=$((iteration + 1))
    echo "Counterfactual iteration $iteration"
    attempt=1
    while true; do
        if run_once "$@"; then
            iteration_status=0
        else
            iteration_status=$?
        fi
        record_progress "$iteration_status"
        if [[ "$iteration_status" -eq 0 ]]; then
            break
        fi
        if [[ "$failure_policy" == "retry" && "$attempt" -le "$retries" ]]; then
            attempt=$((attempt + 1))
            echo "Retrying failed iteration $iteration (attempt $attempt/$((retries + 1)))" >&2
            continue
        fi
        if [[ "$repeat" -eq 1 || "$failure_policy" == "fail" ]]; then
            exit "$iteration_status"
        fi
        echo "Skipping failed iteration $iteration (exit $iteration_status)" >&2
        break
    done
done
