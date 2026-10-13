#!/usr/bin/env bash
# Vendor the opt-in CasU source-harness instrumentation (kept uncommitted in the
# historical Reinforcement-Unknown checkout) so the evidence is reproducible
# from this branch. Run from the Isaac worktree root.
set -euo pipefail
SRC="${1:-$HOME/Documents/Casualties Unknown Projects/Reinforcement-Unknown}"
DST="casu_grid/source_harness"
mkdir -p "$DST"
cp "$SRC/PPO_Harness/Counterfactual/CollisionProbe.cs" "$DST/"
cp "$SRC/PPO_Harness/RunCounterfactual.sh" "$SRC/PPO_Harness/NamespaceChild.sh" "$DST/"
cp "$SRC/PPO_Python/CounterfactualSmoke.py" "$DST/"
git -C "$SRC" diff HEAD -- PPO_Harness/Plugin.cs > "$DST/Plugin.cs.diff-vs-main-HEAD"
echo "synced from $SRC"
