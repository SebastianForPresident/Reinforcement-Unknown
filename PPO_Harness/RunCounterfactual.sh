#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
STATUS_FILE="${PPO_AUTOSTART_STATUS_FILE:-$PROJECT_ROOT/counterfactual_autostart_status.json}"
UNITY_LOG="${PPO_AUTOSTART_UNITY_LOG:-$PROJECT_ROOT/counterfactual_autostart_unity.log}"
PROGRESS_FILE="${PPO_AUTOSTART_PROGRESS_FILE:-$PROJECT_ROOT/counterfactual_batch_progress.jsonl}"

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
if [[ "$replay_mode" != "snapshot" && "$replay_mode" != "prefix" ]]; then
    echo "error: PPO_COUNTERFACTUAL_REPLAY_MODE must be snapshot or prefix" >&2
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
unity_pid=0
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
        PYTHONPATH="$PROJECT_ROOT/PPO_Python${PYTHONPATH:+:$PYTHONPATH}" \
            ./.venv/bin/python PPO_Python/PrefixReplay.py "${prefix_args[@]}"
        local replay_status=$?
        set -e
        return "$replay_status"
    fi
    rm -f -- "$STATUS_FILE"
    echo "Launching unattended counterfactual Unity process..."
    PPO_COUNTERFACTUAL=1 \
    PPO_AUTOSTART=1 \
    PPO_AUTOSTART_STATUS_FILE="$STATUS_FILE" \
    "$SCRIPT_DIR/Deploy.sh" >"$UNITY_LOG" 2>&1 &
    unity_pid=$!

    set +e
    ./.venv/bin/python PPO_Python/CounterfactualReplay.py "$@"
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
