#!/bin/bash
# checkpoint/resume compatibility: baseline(CPU, central) -> optimized(GPU, local) and optimized -> optimized, plus uninterrupted reference
SP=/tmp/claude-1000/-home-sebastian-Documents-Casualties-Unknown-Projects-Reinforcement-Unknown-Isaac/919ee011-e014-4f02-9c35-5b7de75a3190/scratchpad
OPT="/home/sebastian/Documents/Casualties Unknown Projects/Reinforcement-Unknown-Isaac-opt3"
source $SP/isaac_env.sh >/dev/null 2>&1
BASEPP=$PYTHONPATH
R=$SP/o3/resume; rm -rf $R; mkdir -p $R
export CASU_NUM_WORKERS=2
CFG=casu_grid/workbench/experiments/xy1_point_navigation_mini.py
run_opt()  { cd "$OPT"; PYTHONPATH="$OPT:$BASEPP" timeout 900 $PY -m casu_grid.workbench.train --config $CFG --num-envs 8 --terrain-mode independent --seed 7 "$@"; }
run_base() { cd $SP/o3/base_src; PYTHONPATH="$SP/o3/base_src:$BASEPP" timeout 900 $PY -m casu_grid.workbench.train --config $CFG --num-envs 8 --terrain-mode independent --seed 7 "$@"; }
# 1. optimized 3 updates, then resume to 6
run_opt --updates 3 --run-dir $R/opt_a > $R/opt_a.log 2>&1; echo "opt_a rc=$?"
cp $R/opt_a/latest.pt $R/opt_a_ckpt.pt
run_opt --updates 3 --run-dir $R/opt_a2 --resume $R/opt_a_ckpt.pt > $R/opt_a2.log 2>&1; echo "opt_a resume rc=$?"
# 2. baseline 3 updates, resume in optimized to 6
run_base --updates 3 --run-dir $R/base_b > $R/base_b.log 2>&1; echo "base_b rc=$?"
cp $R/base_b/latest.pt $R/base_b_ckpt.pt
run_opt --updates 3 --run-dir $R/mix_b --resume $R/base_b_ckpt.pt > $R/mix_b.log 2>&1; echo "base->opt resume rc=$?"
# 3. optimized checkpoint resumed by the baseline code (informational: forward compatibility)
run_base --updates 3 --run-dir $R/mix_c --resume $R/opt_a_ckpt.pt > $R/mix_c.log 2>&1; echo "opt->base resume rc=$?"
