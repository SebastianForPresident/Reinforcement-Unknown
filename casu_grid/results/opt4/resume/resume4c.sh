#!/bin/bash
source /tmp/claude-1000/-home-sebastian-Documents-Casualties-Unknown-Projects-Reinforcement-Unknown-Isaac/919ee011-e014-4f02-9c35-5b7de75a3190/scratchpad/o4/env4.sh
R=$SP/o4/resume; export CASU_NUM_WORKERS=2
CFG=casu_grid/workbench/experiments/xy1_point_navigation_mini.py
ins() { $PY casu_grid/perf/checkpoint_inspect.py "$1" 2>&1 | tail -1; }
KIT=$(dirname $PY)/kit/python/bin/python3; [ -x $KIT ] || KIT=$PY
tr() { bounded 900 $PY -m casu_grid.workbench.train --config $CFG --num-envs ${NENV:-16} --terrain-mode independent --seed 7 "$@"; }
echo "== R4 SIGINT"; python3 casu_grid/perf/resume_fault_test.py sigint $R/d --python $PY
ins $R/d/latest.pt; cp $R/d/latest.pt $R/d_ckpt.pt
echo "== R4b resume +2"; tr --updates 2 --run-dir $R/d2 --resume $R/d_ckpt.pt > $R/d2.log 2>&1; echo "rc=$?"; ins $R/d2/latest.pt
echo "== R5 worker SIGKILL"; python3 casu_grid/perf/resume_fault_test.py killworker $R/e --python $PY
ps -eo pid,cmd | grep -E "[k]it/python" | wc -l
ins $R/e/latest.pt; cp $R/e/latest.pt $R/e_ckpt.pt
echo "== R5b resume +2"; tr --updates 2 --run-dir $R/e2 --resume $R/e_ckpt.pt > $R/e2.log 2>&1; echo "rc=$?"; ins $R/e2/latest.pt
echo "== R6 num-envs 8 vs 16-env ckpt"; NENV=8 tr --updates 1 --run-dir $R/f --resume $R/a_ckpt.pt > $R/f.log 2>&1; echo "rc=$?"; grep -E "Error|error" $R/f.log | tail -3 | cut -c1-300
echo "== R6b terrain-mode shared"; bounded 600 $PY -m casu_grid.workbench.train --config $CFG --num-envs 16 --terrain-mode shared --seed 7 --updates 1 --run-dir $R/g --resume $R/a_ckpt.pt > $R/g.log 2>&1; echo "rc=$?"; grep -E "Error|error" $R/g.log | tail -3 | cut -c1-300
echo "== R6c other config"; bounded 600 $PY -m casu_grid.workbench.train --config casu_grid/workbench/experiments/xy1_point_navigation.py --num-envs 16 --terrain-mode independent --seed 7 --updates 1 --run-dir $R/h --resume $R/a_ckpt.pt > $R/h.log 2>&1; echo "rc=$?"; grep -E "Error|error" $R/h.log | tail -3 | cut -c1-300
echo "== R7 determinism of resume: second resume from a_ckpt"; tr --updates 4 --run-dir $R/b2 --resume $R/a_ckpt.pt > $R/b2.log 2>&1; echo "rc=$?"
python3 - <<PY
import json
L=lambda p:[json.loads(l) for l in open(p)]
b,b2=L("$R/b/metrics.jsonl"),L("$R/b2/metrics.jsonl")
ks=("policy_loss","value_loss","entropy","approx_kl","grad_norm","mean_completed_return","environment_steps")
print("resume vs resume identical:", all(x[k]==y[k] for x,y in zip(b,b2) for k in ks), [round(x["policy_loss"],6) for x in b],[round(x["policy_loss"],6) for x in b2])
PY
echo MARK_R7
