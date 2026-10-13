# Research workbench navigation

## Current Layer 1 research

Start the graphical studio with `python -m casu_grid.workbench.gui`, or read the
[five-minute quick start](casu_grid/workbench/README.md). The original CLI remains
available through `python -m casu_grid.workbench.lab --help` or its guided menu. New Isaac Layer 1
experiments use `casu_grid/workbench/` and the canonical seed 391728405. Read
[`casu_grid/ACCEPTANCE_REPORT.md`](casu_grid/ACCEPTANCE_REPORT.md) for measured
source fidelity and known limitations.

## Historical Unity/CasU research

`PPO_Python/` and `PPO_Harness/` preserve the earlier Unity/BepInEx experiments,
including PPO/REINFORCE training, independent critics, counterfactual replay,
attack studies and matched-pair evaluation. Use their local guides when
reproducing that work. These files and results are intentionally retained.

## Machine setup

The local Isaac installation and invocation are documented in
[`ISAAC_LAB_SETUP.md`](ISAAC_LAB_SETUP.md). This is machine-specific and is not
part of the experiment configuration.
