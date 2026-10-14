# Research Studio V2

A desktop application for the XY1 point-navigation experiment: review the configuration, validate, train, monitor, inspect learning, evaluate on
held-out terrains, compare runs, resume checkpoints.

## Start

```bash
cd <this checkout>              # e.g. Reinforcement-Unknown-Isaac-studio
python3 -m casu_grid.studio     # or: casu_grid/studio/run_studio.sh
```

* Requirements for the Studio itself: **Python 3 with PySide6 ≥ 6.5 and psutil** (`pip install -r casu_grid/studio/requirements.txt`). Tested with
  Python 3.14, PySide6 6.11.2, Fedora 44 (Wayland and X11).
* The simulator is **not** started inside the Studio. Training and evaluation run as separate processes under the Isaac Sim bundled Python, found
  automatically in `~/Applications/IsaacLab-2.1.1-IsaacSim-4.5.0/isaac-sim-4.5.0` (or set it under *Help → Settings*, or `ISAAC_SIM`).
  The Studio sets up the Isaac environment variables itself; you do not need to source anything.
* The code revision that runs is the **checkout the Studio was started from**; the Overview and the validation list show it, and refuse to launch
  if the Optimization IV corrections are missing.
* Runs are written to `<checkout>/experiments/` (git-ignored) unless you change the run folder in *Help → Settings* or set `CASU_STUDIO_RUN_ROOT`.
* `CASU_LEGACY_MINI_SPAWN=1` must not be set: the Studio (and the trainer) refuse to train with it.

## Layout of the code

| file | role |
|---|---|
| `app.py`, `context.py` | main window, navigation, shared state, background jobs, system monitor |
| `pages/` | Overview, Experiments, Training, Metrics, Runs & checkpoints, Evaluation, Help |
| `charts.py`, `catalog.py` | the interactive chart widget; which recorded metric feeds which chart |
| `launch.py`, `environment.py`, `experiments.py` | launch plan, validation, process control; Isaac discovery; experiment contracts |
| `runs.py`, `analysis.py` | run discovery/status, incremental metric reading; the "is it learning?" readout |
| `help.py` | the glossary behind every (i) button |
| `tests/` | unit and widget tests (`test_*.py`), the real-simulator journey harness (`journey.py`), synthetic-run helpers |

## Tests

```bash
# unit + widget tests (no simulator; needs PySide6, so use the system Python)
QT_QPA_PLATFORM=offscreen python3 -m unittest discover -s casu_grid/studio/tests -t . -p "test_*.py"
# trainer-side tests (Isaac Python)
"$ISAAC_SIM/python.sh" -m unittest discover -s casu_grid/tests -p "test_*.py"
# the full journey against the real simulator (a few minutes each stage)
QT_QPA_PLATFORM=offscreen python3 -m casu_grid.studio.tests.journey smoke|live|stop_resume|failure|evaluation|diagnostic|gallery
```
