# Reinforcement-Unknown

## Current research workbench

New Layer 1 Isaac research starts at [RESEARCH_WORKBENCH.md](RESEARCH_WORKBENCH.md). For commands and experiment configuration, see [the five-minute workbench guide](casu_grid/workbench/README.md). Historical Unity/BepInEx work remains under `PPO_Python/` and `PPO_Harness/`.
A test to measure how PPO would fare in the complex learning problem that is Casualties: Unknown.

## Overview

Reinforcement-Unknown is my attempt at training AI-powered NPCs in *Casualties: Unknown* using reinforcement learning (RL), beginning with Proximal Policy Optimization (PPO). It stemmed from my mod [Project Mechanism](https://www.nexusmods.com/scavprototype/mods/45) needing competent NPC behavior rather than structures holed up in pods

## Releases

Every GitHub Release represents a trained checkpoint.

Each release contains:

- Trained PPO checkpoint
- TensorBoard logs
- Matching plugin binary
- Training summary
- Behavior preview

## Goal

The long-term goal is to produce an agent capable of surviving and eventually completing an entire layer of Casualties: Unknown. 

The stretch goal is to complete multiple.

## Current Progress

| Checkpoint | Reward | Behavior |
|------------|--------|----------|
| Test | None | Random thrashing |
| Checkpoint 1 | V1 | Holds right, attacks upward, avoids ragdolling |
| Checkpoint 2 | V2 | Thrashing for mobility and terrain destruction |
| Checkpoint 3 | V3 | Lighter, non-destructive seizures |
| Checkpoint 4 | V4 | Wiggle to fall |
| Checkpoint 5 | V5 | Leave pod to try to descend |
| Checkpoint 6 | V5 | Exercise and avoid ledges + overall descent |
| Checkpoint 6.5 | V5 | Descend as much as possible |
| Checkpoint 7 | V6 | Unsafe, brute-force descent |
| Checkpoint 8 | V7 | Unproductive self-damage |
| Checkpoint 9 | V7 | Unproductive self-damage |
| Checkpoint 10 | V8 & V9 | Unproductive self-damage |
| Checkpoint 11 | V10 | Preserve health, stay safe, and stay stagnant |
| Checkpoint 12 | V10 | Minimal movement/exertion while trying to descend |
| Checkpoint 13 | V11, V12, & V13 | Blind descent and attack farming |
| Checkpoint 14 | V14 & V15 | Careful descent |
| Checkpoint 15 | V15 | High-exertion "descent" |
| Checkpoint 16 | V16 | Aggressive, high-entropy descent |
| Checkpoint 17 | V17 | Searching for places to fall |

## B Series

A jump from the original checkpoints that heavily modified the learning problem

| Checkpoint | Reward | Behavior |
|------------|--------|----------|
| CB1 | VB1 | Validation in progress |

## Reproducible Cave Seeds

CB1 resets send an explicit non-negative 32-bit world seed to the Unity
harness. Training selects a fresh seed for every procedural cave, while
inference and validation can load a specific cave:

```bash
./.venv/bin/python PPO_Python/Server.py inference checkpoint.zip 123456
./.venv/bin/python PPO_Python/Server.py validate 1500 123456
```

The applied seed is acknowledged by Unity and recorded in every episode trace
under `world_seed`. Calling `env.reset(seed=123456)` or
`env.reset(options={"world_seed": 123456})` loads the same requested seed.

## Restricted CB1 counterfactual replay

Counterfactual diagnostics use a separate client and never start PPO training.
Build the harness, launch the game with `PPO_COUNTERFACTUAL=1`, then run the
standalone A/A test from another terminal:

```bash
dotnet build PPO_Harness/PPO_Harness.csproj
PPO_COUNTERFACTUAL=1 PPO_FORCE_BODY_UPDATE=1 PPO_TIME_SCALE=10 \
  PPO_Harness/Deploy.sh
# in another terminal, while the game is running:
./.venv/bin/python PPO_Python/CounterfactualReplay.py --seed 391728405 --horizon 100
```

For a fresh, completely unattended launch, use the combined runner. It builds
and launches Unity, waits for the autostart status/socket, runs the replay, and
lets the protocol shut Unity down:

```bash
PPO_Harness/RunCounterfactual.sh --seed 391728405 --horizon 1
```

The fresh-process prefix-replay architecture is selected explicitly with
`PPO_COUNTERFACTUAL_REPLAY_MODE=prefix`; this mode lets `PrefixReplay.py` own
each exact Unity launch and never uses `CF_RESTORE`:

```bash
PPO_COUNTERFACTUAL_REPLAY_MODE=prefix \
  PPO_Harness/RunCounterfactual.sh --seed 391728405 --horizon 1
```

`PPO_COUNTERFACTUAL=1` automatically enables autostart; `PPO_AUTOSTART=1`
also enables it for a non-counterfactual harness session. With no preset or
overrides, the initialized menu defaults are used. `PPO_AUTOSTART_PRESET`
selects the same named preset as the menu, while
`PPO_AUTOSTART_SETTINGS` is a JSON object whose keys are the exact run-setting
names exposed by the game. The equivalent command-line forms are
`--ppo-autostart-preset`, `--ppo-autostart-settings`, and
`--ppo-autostart-settings-file` when supplied to the Unity executable directly.
Unity logs the complete initialized setting
key list in `AUTO_START_SETTINGS_APPLIED` and writes startup state to
`counterfactual_autostart_status.json` (override with
`PPO_AUTOSTART_STATUS_FILE`). Invalid names, values, failed initialization, and
startup timeouts hard-fail with `AUTO_START_FAILED`; successful startup emits
`AUTO_START_CONFIG`, `AUTO_START_READY`, `AUTO_START_TRIGGERED`, and
`AUTO_START_COMPLETE`.

During an opt-in startup, `AUTO_START_HEARTBEAT` is emitted every five seconds
from the plugin `Update` loop with the current startup phase and scene state.
Its absence after `Chainloader startup complete` indicates a native Unity
player-loop/display startup stall rather than a replay or state-digest delay.

For a Proton display-helper diagnostic, `PPO_PROTON_USE_XALIA=0` or `1` is
translated to Proton's `PROTON_USE_XALIA` variable. It is unset by default, so
normal Proton behavior is preserved; disabling the helper is diagnostic-only
and does not by itself certify Unity startup.

`PPO_UNITY_ARGS` is an optional whitespace-separated list passed only to the
explicitly launched Unity executable for startup diagnostics; it is unset by
default.

Set `PPO_AUTOSTART_REPEAT=0` to launch a fresh Unity process for every
iteration indefinitely; a positive value runs that many fresh iterations.
For a bounded unattended batch, set it to the desired positive iteration
count and pass `--treatment-steps` to `PrefixReplay.py`; each candidate remains
gated on strict fresh-process A/A and direct navigation outcomes. The current
supported fallback for fast replay is documented in
`PPO_Harness/CRIU_FEASIBILITY.md`; CRIU capability checks failed read-only, so
no CRIU path is enabled.

Example bounded prefix batch (each iteration receives an isolated artifact ID;
failed trials are explicitly skipped and recorded):

```bash
PPO_COUNTERFACTUAL_REPLAY_MODE=prefix PPO_AUTOSTART_REPEAT=3 \
  PPO_AUTOSTART_BATCH_FAILURE_POLICY=skip \
  PPO_Harness/RunCounterfactual.sh --seed 391728405 --horizon 1 \
  --treatment-steps 0
```

Progress is appended as JSONL to `counterfactual_batch_progress.jsonl`
(override with `PPO_AUTOSTART_PROGRESS_FILE`). The explicit policy is
`fail` (default), `skip`, or `retry`; retry mode also requires the positive
`PPO_AUTOSTART_BATCH_RETRIES` count. If a base `--run-id` is supplied with
repeat mode, the runner appends a unique iteration/attempt ID so later
iterations cannot overwrite the first result.

The runner writes isolated artifacts under `counterfactual_diagnostics/` and
reports the first differing observation, Unity state path, or Python reward /
watchdog field. Prefix replay reports `A/A_FAIL` when independent fresh
processes diverge and records the first exact difference; it does not proceed
to A/B. Snapshot replay may report `A/A_BLOCKED` when repeated restores match
semantically but the report still contains uncovered engine-owned state;
`A/A_PASS` is reserved for a complete accepted validation. Neither status
authorizes A/B branching by itself. Engine-private clock,
coroutine, and Physics2D solver gaps remain explicit in the artifact until
they are empirically neutralized under CB1.

### Observation state classification

The serialized observation at the snapshot boundary is preserved as an
independent `observation_wire_sha256`/base64 baseline and restored directly.
After restore, `CollectObservations()` reconstructs the preallocated `obs` DTO
before `WriteObservation()` emits the next packet; A/A compares those emitted
wire bytes independently of the managed-state digest.

The managed snapshot does not traverse `obs`, `RecipeDatabase`, or
`CurrentObservation`. Under the CB1 guards, `obs` is observation-builder
scratch written by `CollectObservations()` and read by `WriteObservation()`;
`RecipeDatabase` belongs to the disabled unused-observation branch; and
`CurrentObservation` has no readers beyond its declaration. Causal observation
bookkeeping remains explicit in the harness roots, including observation IDs,
latched actions, macrostep/depth state, sound candidates, and damage tracking.
The managed digest records unique reference identities and reference reuses;
aliases are emitted as references and are not recursively traversed or charged
against the node budget.

Restore is differential but capture remains broad. Each restore compares
captured tilemap cells, matrices, flags, native properties, polygon paths, and
animator state before writing. It rewrites only changed tilemaps, rebuilds only
colliders whose tilemap or causal collider inputs changed, and performs at most
one final physics synchronization. `timing.log` reports phase timings and
`restore_diff_####.json` records the changed tilemaps/cells/components and
skipped no-op work for audit.

The normal full-coverage `CF_STATE` path caches immutable reflection and
causal-topology metadata and canonical component-root paths, reuses managed
field suffixes and digest maps, buffers hash input without changing its
canonical bytes, uses allocation-free float bit formatting, and streams
primitive arrays through a reusable hash buffer and SHA instance. These are
representation optimizations only: the latest fresh-process probe still
examined 6,413 GameObjects, 23,055 components, 283 tilemaps/1,059,937 cells,
238,115 managed values, and 99,229,045 fingerprint bytes. Measured normal
`CF_STATE` time improved from approximately 4.7--4.8 s to approximately
3.55--3.70 s on a boundary after the causal topology plan was established;
the remaining dominant stages are full managed value/path emission, native
component reads, tilemap reads, and array hashing. The latest probe artifact
is `counterfactual_prefix_root-path-cache-20260914/root-path-cache-20260914/prefix_replay_result.json`.
This is not an A/A pass: the run remains correctly blocked at step 1 by the
documented Unity/Box2D contact-impulse limitation.

Topology invariants use the same exclusion in both directions: a proven
presentation object present at capture but gone at restore is ignored exactly
as a newly-created presentation object is ignored. Unknown and causal
GameObject/component changes remain blocking; presentation objects are not
recreated or included in the causal component/native fingerprint. The harness
debug dot is likewise excluded from the explicit harness roots after its
source-level noncausal proof.

The lifecycle exception is narrower than the topology exclusion: the exact
`TMPro.TMP_SubMeshUI` leaf at
`GlobalDark(Clone)/MouseOver/DescBack/Description/TMP UI SubObject` is a
renderer-owned description subcomponent with no gameplay registry or physics
identity. Its volatile active/enabled bits are canonicalized symmetrically,
but the component remains in topology, so creation or destruction still
blocks. Other TMP/UI components and all unclassified lifecycle changes remain
causal by default.
