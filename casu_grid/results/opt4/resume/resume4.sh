#!/bin/bash
source /tmp/claude-1000/-home-sebastian-Documents-Casualties-Unknown-Projects-Reinforcement-Unknown-Isaac/919ee011-e014-4f02-9c35-5b7de75a3190/scratchpad/o4/env4.sh
R=$SP/o4/resume; rm -rf $R; mkdir -p $R
export CASU_NUM_WORKERS=2
CFG=casu_grid/workbench/experiments/xy1_point_navigation_mini.py
tr() { bounded 900 $PY -m casu_grid.workbench.train --config $CFG --num-envs ${NENV:-16} --terrain-mode independent --seed 7 "$@"; }
ins() { $PY casu_grid/perf/checkpoint_inspect.py "$1" 2>&1 | tail -1; }
echo "== R1 fresh 4 updates (cuda, worker-local)"; tr --updates 4 --run-dir $R/a > $R/a.log 2>&1; echo "rc=$?"; ins $R/a/latest.pt; cp $R/a/latest.pt $R/a_ckpt.pt
echo "== R2 resume +4"; tr --updates 4 --run-dir $R/b --resume $R/a_ckpt.pt > $R/b.log 2>&1; echo "rc=$?"; ins $R/b/latest.pt
echo "== R3 uninterrupted 8"; tr --updates 8 --run-dir $R/c > $R/c.log 2>&1; echo "rc=$?"; ins $R/c/latest.pt
echo MARK_R3
