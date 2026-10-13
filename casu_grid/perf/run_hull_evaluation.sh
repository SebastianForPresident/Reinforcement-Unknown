#!/bin/bash
# Reproduce the hull-vs-compound evaluation (see casu_grid/HULL_EVALUATION_REPORT.md). Run from the repository root with the
# Isaac environment active (PY = Isaac python.sh). About 60 minutes on an idle machine; one Isaac process at a time.
#
#   bash casu_grid/perf/run_hull_evaluation.sh OUTPUT_DIR
set -u
OUT=${1:?output directory}
PY=${PY:?set PY to the Isaac python.sh}
export PYTHONPATH=$PWD:${PYTHONPATH:-}
mkdir -p "$OUT/paired"

# 1. recorded CasU traces: all three comparison modes, per-tick rows kept for paired statistics
for coll in compound convex_hull; do
  for t in flat_master course_v1 ice_run immersed_standing; do
    "$PY" casu_grid/isaac_player_ab.py --trace $t --modes segment,teacher,free --player-collider $coll --detail \
      --step-clear 0.02 --output "$OUT/ab_${t}_$coll.json"
  done
  for d in 31 0; do
    "$PY" casu_grid/isaac_player_ab.py --trace destroy_support --modes free,segment,teacher --player-collider $coll --detail \
      --collider-delay $d --step-clear 0.02 --output "$OUT/ab_destroy_support_d${d}_$coll.json"
  done
  # 2. Mission-12 CasU collision probes (walls, chunk seam, tile junctions, ledge), free-run only
  for pr in wall glass_wall chunk_seam_cross chunk_seam isolated_tile adjacent_tiles natural_flat_ledge; do
    "$PY" casu_grid/isaac_player_ab.py --legacy-probe $pr --player-collider $coll --detail --step-clear 0.02 \
      --output "$OUT/probe_${pr}_$coll.json"
  done
done

# 3. sensitivity to the hull facet count (geometry parameter, 8 is the default)
for n in 4 30; do
  for t in flat_master course_v1 ice_run immersed_standing; do
    "$PY" casu_grid/isaac_player_ab.py --trace $t --modes segment,teacher,free --player-collider convex_hull --hull-segments $n \
      --detail --step-clear 0.02 --output "$OUT/ab_hull_${t}_s$n.json"
  done
done

# 4. robustness on real terrain without a CasU reference (embedding depth, implausible jumps, terminations)
for tape in random run climb; do for seed in 1 2; do for coll in compound convex_hull; do
  "$PY" casu_grid/perf/collider_stress.py --player-collider $coll --envs 64 --decisions 1200 --tape $tape --seed $seed \
    --output "$OUT/stress_${tape}_s${seed}_$coll.json"
done; done; done

# 5. paired statistics
for t in flat_master course_v1 ice_run immersed_standing destroy_support_d31 destroy_support_d0; do
  python casu_grid/tools/compare_ab_paired.py "$OUT/ab_${t}_compound.json" "$OUT/ab_${t}_convex_hull.json" --seams --json "$OUT/paired/$t.json"
done
python casu_grid/tools/summarize_paired.py "$OUT"/paired/{flat_master,course_v1,ice_run,immersed_standing,destroy_support_d31}.json
