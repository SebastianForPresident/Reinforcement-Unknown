#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="$ROOT/PPO_Harness/RunCounterfactual.sh"
DEPLOY="$ROOT/PPO_Harness/Deploy.sh"

if rg -n '(^|[[:space:];|&(])(pkill|killall|loginctl|systemctl[[:space:]]+--user|setsid|plasmalogin|kwin|plasmashell)([[:space:]]|$)|kill[[:space:]]+0|kill[[:space:]]+--[[:space:]]+-|kill[[:space:]]+-[0-9]+' "$RUNNER" "$DEPLOY"; then
    echo "unsafe session/process termination primitive found" >&2
    exit 1
fi

rg -q 'prefix_args=\("\$@"\)' "$RUNNER"
rg -q 'prefix_args\+=\(--run-id "batch-iteration-\$\{iteration\}-attempt-\$\{attempt\}"\)' "$RUNNER"

if PPO_AUTOSTART_REPEAT=invalid "$RUNNER" >/tmp/counterfactual-safety-test.out 2>&1; then
    echo "invalid repeat mode unexpectedly launched" >&2
    exit 1
fi
rg -q 'PPO_AUTOSTART_REPEAT must be 0' /tmp/counterfactual-safety-test.out

if PPO_COUNTERFACTUAL_REPLAY_MODE=invalid "$RUNNER" >/tmp/counterfactual-mode-test.out 2>&1; then
    echo "invalid replay mode unexpectedly launched" >&2
    exit 1
fi
rg -q 'PPO_COUNTERFACTUAL_REPLAY_MODE must be snapshot or prefix' /tmp/counterfactual-mode-test.out

if PPO_AUTOSTART_BATCH_FAILURE_POLICY=invalid "$RUNNER" >/tmp/counterfactual-policy-test.out 2>&1; then
    echo "invalid batch failure policy unexpectedly launched" >&2
    exit 1
fi
rg -q 'PPO_AUTOSTART_BATCH_FAILURE_POLICY must be fail, skip, or retry' /tmp/counterfactual-policy-test.out

if PPO_AUTOSTART_BATCH_FAILURE_POLICY=retry PPO_AUTOSTART_BATCH_RETRIES=0 "$RUNNER" >/tmp/counterfactual-retry-test.out 2>&1; then
    echo "zero-attempt retry policy unexpectedly launched" >&2
    exit 1
fi
rg -q 'retry policy requires PPO_AUTOSTART_BATCH_RETRIES > 0' /tmp/counterfactual-retry-test.out
echo "process safety checks passed"
