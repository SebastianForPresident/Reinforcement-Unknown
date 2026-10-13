#!/bin/bash
# usage: run_gpu_scenarios.sh OUTDIR [--cuda]   (Isaac environment already set up; one process per scenario)
out=$1; shift
mkdir -p "$out"
for sc in ${SCENARIOS:-crouch_tunnel support_removal landing wall junction_run}; do
  timeout 600 "${PY:-python.sh}" casu_grid/perf/gpu_scenarios.py --only $sc --output "$out/$sc.json" "$@" > "$out/$sc.log" 2>&1
  echo "$sc exit=$? $(grep -E "^$sc " "$out/$sc.log" | head -1)"
done
