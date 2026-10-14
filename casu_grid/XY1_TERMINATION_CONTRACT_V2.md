# XY1 termination contract v2: bottom-of-layer exit

Decision of the researcher: when Timmy exits through the bottom of the 128 × 128 mini-world, the current navigation episode **terminates** and the world, Timmy and the goal are
re-initialised. This models CasU's bottom-of-layer transition; it is **not** player death. Branch `xy1-layer-exit` (from Research Studio V2 `studio-v2` / `d2b1cc3`).

## Source rule and boundary definition

CasU `SearchBase/WorldGeneration.cs` (`Update`, `ContinueRun`): the layer transition opens when
`PlayerCamera.main.body.transform.position.y < (float)(0 - halfHeight) + 3.1`, with `halfHeight = (uint)(height * 0.5)`.

* World-relative bottom-exit threshold for the 128-tile mini-world: **`y < −64 + 3.1 = −60.9`** (slot-local coordinates, world origin at the centre; `layer_exit_margin = 3.1`,
  `halfHeight = grid.height // 2`). The floor occupies rows y ∈ [−64, −62]; Timmy standing on the intact floor rests at about y = −60.3, so the rule fires only after the floor has been
  dug through (verified: lowest y over 12 decisions on an intact floor is −1.4 at spawn height, and the exit fires at the first decision whose end state is below −60.9: y −56.6 → −61.9).
* It is evaluated once per decision (after the ten physics ticks), exactly like arrival.

## Semantics

| aspect | contract v2 |
|---|---|
| transition type | **true terminal** (`terminated = True`, `truncated = False`), reason `layer_exit`; PPO's GAE does not bootstrap through it (`terminated` masks the value bootstrap; `done` stops the recursion) |
| reward | **none added**: no failure reward, no penalty, no arrival bonus; the exit decision pays only the ordinary `0.1 × new-best-distance improvement` term (0 when falling away) |
| goal arrival | unchanged: reason `arrival` (the existing goal-arrival record; `nav/arrivals`), +10 bonus, terminal |
| time limit | unchanged: 100,000-decision ceiling, truncation, reason `time_limit` |
| precedence inside one decision | **goal arrival > layer exit > decision ceiling** (verified for both pairs) |
| after an exit | `reset_done` re-initialises that world only: terrain restored, Timmy at the world centre (0.5, 0.5), new goal drawn in the 64 × 64 viewport, counters cleared; other worlds untouched |
| floor, physics, reward formula, observation, actions, PPO, world distribution | unchanged (the floor stays destructible) |

## Identification and versioning

* Environment flag `bottom_exit_terminates` (default **off** = the reference). Experiment file **`xy1_point_navigation_mini_layer_exit.py`**
  (id `xy1-point-navigation-mini-128x128-layer-exit-v2`); the reference `xy1_point_navigation_mini.py` is byte-identical to before (trajectory gate `284e8dde…` unchanged) and is
  termination contract **`xy1-termination-v1`**. The superseded exploratory variant (`terminate_out_of_world`, truncation when |y| > half the world) is `xy1-termination-v1x-out-of-world-truncation`.
  `bottom_exit_terminates` and `terminate_out_of_world` are mutually exclusive.
* Contract id: `xy1-termination-v2-layer-exit` (`workbench.contracts.termination_contract_id`). It is written to the run manifest (`config.json → environment.termination_contract`),
  `studio_launch.json`, every checkpoint (`environment_contract`) and its sidecar `*.pt.json`, and to evaluation results.
* The learning-contract fingerprint (actions, observations, network, PPO) is identical for v1 and v2, so it cannot tell them apart; **`PPOTrainer.load` therefore refuses to resume a checkpoint
  under a different termination contract** ("different experiments: start a new run"), checkpoints without an id count as v1, and the Studio does not offer such checkpoints and shows an error.
* `.gitignore` previously ignored the *directory* `casu_grid/workbench/experiments/` (pattern `experiments/`), so new experiment definitions were silently untracked — including the Studio-era
  variant file; the pattern is now `/experiments/` (run folders at the repository root only), and both experiment files are committed here.

## Metrics (Training / Metrics pages, `metrics.jsonl`)

`termination/layer_exit` (count per update), `nav/layer_exits`, **`nav/layer_exits_per_1k_decisions`** (chart *Layer exits per 1,000 decisions*), `nav/decisions_to_layer_exit_mean`;
arrivals, timeouts and layer exits are separate series in *Episodes finished per update*; `nav/arrival_rate` counts layer exits among finished episodes that did not arrive;
`nav/active_outside_world_fraction` stays 0 under v2. The Experiments page, launch validation, run records and the Runs table show the contract; help topics *Layer exit* and *Termination contract*.

## Validation (all on this branch)

* Readiness audit `perf/xy1_readiness_audit.py --layer-exit`: **32 of 32** checks (the 19 reference checks plus L0–L7: threshold, intact floor never exits, exact first-crossing, terminal not truncation,
  no reward/penalty, other worlds unaffected, full re-initialisation incl. terrain/goal/observation, next episode valid, arrival > exit, exit > ceiling, ceiling still truncates on an intact floor, repeated
  exits in two worlds at once). The reference audit still passes 19 of 19 (`results/layer_exit/`).
* Unit tests: GAE masks (true terminal ignores the next value, truncation bootstraps, no leakage across the episode boundary), navigation metrics with layer exits, checkpoint termination-contract
  refusal both ways, contract ids; Studio tests (85) incl. no resume across contracts and run/checkpoint identification. Trainer suite 157 OK, Kit test OK.
* Regressions: XY1 reference trajectory hash `284e8dde…` unchanged; the v2 dump is **identical** to it up to the first terminal event (no layer exit in 300 random decisions); 8 of 8 trajectory gates
  bit-identical (`perf/run_perf_check.py --quick`).
* Engineering smoke through the Studio (16 worlds × 2 workers, 60 updates = 61,440 decisions; not a learning run; `results/layer_exit/smoke_analysis.txt`): 13 layer exits (0.21 per 1,000 decisions),
  7 arrivals, 0 timeouts; in the raw transition records every one of the 13 exits is `terminated = True, truncated = False`, pays no bonus and no penalty, and in all 13 the next decision of that world
  starts at the world centre (y between −0.71 and 4.49); no agent ever left the world (lowest y −64.4, against −10⁷ in the reference); outside-world fraction 0 throughout.
