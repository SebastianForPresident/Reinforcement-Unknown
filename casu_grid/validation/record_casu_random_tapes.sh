#!/usr/bin/env bash
# Record random-action tapes in real CasU from the benchmark spawn (grid 483,1010) with the staged, isolated runtime.
#
#   bash casu_grid/validation/record_casu_random_tapes.sh OUT_DIR SEED [SEED ...]
#
# Uses the existing instrumented harness (Reinforcement-Unknown/PPO_Harness/RunCounterfactual.sh, smoke mode, per-tick
# pre/post-FixedUpdate and post-Physics2D traces, Body.Ragdoll call log) unchanged, except that the smoke *client* is
# replaced by source_harness/CounterfactualSmokeRecord.py (same protocol, richer result). The launcher's own owned-process
# cleanup is used; nothing else is signalled. No game physics or fixture is modified.
set -u
OUT=${1:?output directory}; shift
RT=${CASU_RUNTIME_ROOT:-/var/tmp/reinforcement-unknown-cb1-runtime-mission12-20261012}
SRC=${CASU_SOURCE_ROOT:-"/home/sebastian/Documents/Casualties Unknown Projects/Reinforcement-Unknown"}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LENGTH=${TAPE_LENGTH:-200}
mkdir -p "$OUT"
RUNNER="$OUT/RunCounterfactualRecord.sh"
sed -e "s|^PROJECT_ROOT=.*|PROJECT_ROOT=\"$SRC\"|" \
    -e "s|\$PROJECT_ROOT/PPO_Python/CounterfactualSmoke.py|$HERE/source_harness/CounterfactualSmokeRecord.py|" \
    "$SRC/PPO_Harness/RunCounterfactual.sh" > "$RUNNER"
chmod +x "$RUNNER"
grep -q CounterfactualSmokeRecord "$RUNNER" || { echo "runner patch failed" >&2; exit 2; }
for SEED in "$@"; do
  NAME=rand_$SEED${TAPE_SUFFIX:-}
  DEST="$OUT/$NAME"; mkdir -p "$DEST"
  ACTIONS=$(python3 "$HERE/validation/random_tapes.py" --seed "$SEED" --length "$LENGTH")
  echo "$ACTIONS" > "$DEST/tape.json"
  RAG="$RT/user-home/diagnostics/ragdoll-calls/ragdoll-calls.jsonl"
  BEFORE=$( [ -f "$RAG" ] && wc -l < "$RAG" || echo 0 )
  echo "== $(date +%H:%M:%S) recording $NAME ($LENGTH actions)"
  (
    cd "$SRC"
    export PPO_COUNTERFACTUAL_RUNTIME_ROOT=$RT PPO_COUNTERFACTUAL_REPLAY_MODE=smoke
    export PPO_DIAG_STANDING_FIXTURE=1 PPO_DIAG_PLAYER_FIXTURE_GRID_X=483 PPO_DIAG_PLAYER_FIXTURE_GRID_Y=1010
    export PPO_DIAG_PLAYER_TICKS=1 PPO_DIAG_PLAYER_POST_PHYSICS=1 PPO_DIAG_PLAYER_PREFIXED=1 PPO_DIAG_RAGDOLL_CALLS=1
    export PPO_COUNTERFACTUAL_SMOKE_SKIP_STATE=1 PPO_COUNTERFACTUAL_SMOKE_TIMEOUT=${TIMEOUT:-60}
    export PPO_COUNTERFACTUAL_SMOKE_ACTIONS="$ACTIONS"
    timeout ${RUN_TIMEOUT:-1500} "$RUNNER" > "$DEST/runner.out" 2>&1
    echo "exit=$?" >> "$DEST/runner.out"
  )
  SESSION=$(ls -td "$RT"/user-home/diagnostics/cb1-smoke-* | head -1)
  for f in smoke_result.json player-fixed-tick.jsonl player-post-physics.jsonl player-pre-fixed.jsonl; do
    cp "$SESSION/$f" "$DEST/" 2>/dev/null
  done
  echo "$SESSION" > "$DEST/session.txt"
  [ -f "$RAG" ] && tail -n +$((BEFORE + 1)) "$RAG" > "$DEST/ragdoll-calls.jsonl" || : > "$DEST/ragdoll-calls.jsonl"
  tail -1 "$DEST/runner.out"
  python3 -c "import json;d=json.load(open('$DEST/smoke_result.json'));print('  status',d['status'],'steps',len(d.get('steps',[])),'error',d.get('error'))"
done
