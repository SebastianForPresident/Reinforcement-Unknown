#!/bin/bash
# XY1 v2 three-seed validation: for each seed, train 250 updates through the Studio launch path (64 worlds x 8 workers x 2 PhysX threads, CUDA), then evaluate the
# initial and the final checkpoint on held-out terrains. Sequential; a finished seed (marker file) is never rerun.
#   CASU_SCRATCH=<dir containing st/env_le.sh> casu_grid/perf/xy1_v2_campaign.sh [seed ...]        default: 7 17 27
cd "$(dirname "$0")/../.." || exit 1
ROOT="$PWD"
SP="${CASU_SCRATCH:-/tmp}"
source "$SP/st/env_le.sh" >/dev/null 2>&1 || true      # Isaac environment (PY, PYTHONPATH, ...) for the evaluation processes
OUT="$ROOT/casu_grid/results/xy1_v2_validation"
CFG="$ROOT/casu_grid/workbench/experiments/xy1_point_navigation_mini_layer_exit.py"
cd "$ROOT"
for SEED in ${@:-7 17 27}; do
  D="$OUT/seed$SEED"; mkdir -p "$D"
  if [ -f "$D/TRAINING_DONE" ]; then echo "seed $SEED: training already done"; else
    RUN="v2-seed$SEED-$(date +%Y%m%d-%H%M%S)"
    echo "[$(date +%T)] seed $SEED: training $RUN"
    PYTHONPATH="$ROOT" QT_QPA_PLATFORM=offscreen python3 -m casu_grid.studio.tests.journey diagnostic --experiment layer_exit --seed "$SEED" --run-name "$RUN" --tag "_v2s$SEED" > "$D/training_journey.log" 2>&1
    RD="$ROOT/experiments/$RUN"
    if [ -f "$RD/COMPLETE" ] && [ "$(wc -l < "$RD/metrics.jsonl")" = "250" ]; then
      mkdir -p "$D/training" "$D/checkpoints"
      cp "$RD/metrics.jsonl" "$RD/config.json" "$RD/studio_launch.json" "$RD/environment.json" "$RD/launcher.log" "$D/training/" 2>/dev/null; cp "$RD/system.jsonl" "$D/training/" 2>/dev/null
      cp "$RD/checkpoints/initial.pt" "$RD/checkpoints/update_000250.pt" "$RD/checkpoints/"*.json "$D/checkpoints/" 2>/dev/null
      echo "$RD" > "$D/RUN_DIR"; date -u +%FT%TZ > "$D/TRAINING_DONE"
    else echo "seed $SEED: TRAINING INCOMPLETE ($RD)"; echo "$RD" > "$D/RUN_DIR_INCOMPLETE"; continue; fi
  fi
  if [ -f "$D/EVAL_DONE" ]; then echo "seed $SEED: evaluation already done"; else
    RD="$(cat "$D/RUN_DIR")"; echo "[$(date +%T)] seed $SEED: evaluation"
    CASU_NUM_WORKERS=4 CASU_WORKER_PHYSX_THREADS=2 CASU_TRAIN_DEVICE=cuda timeout 3000 "$PY" -m casu_grid.workbench.evaluate --config "$CFG" \
      --policy "checkpoint=$RD/checkpoints/initial.pt" --policy "checkpoint=$RD/checkpoints/update_000250.pt" --worlds 16 --rounds 8 --horizon 1000 \
      --eval-seed 20261014 --init-seed "$SEED" --modes deterministic,stochastic --output "$D/evaluation.json" > "$D/evaluation.log" 2>&1
    if [ -s "$D/evaluation.json" ]; then date -u +%FT%TZ > "$D/EVAL_DONE"; else echo "seed $SEED: EVALUATION FAILED"; fi
  fi
done
echo CAMPAIGN_FINISHED
