# CasU research workbench

`contracts.py` is the user-owned interface for action heads and timing,
observation fields and normalization, reward-term mappings, model layers, and
PPO settings. `model.py` builds a separate spatial encoder, scalar encoder,
fusion stack, policy heads, and value head from those contracts. `ppo.py`
collects vectorized Gymnasium rollouts, performs clipped PPO updates, records
reward-term diagnostics, and saves resumable checkpoints with a strict learning
contract fingerprint.

The contracts in `engineering_smoke.py` are temporary plumbing examples only.
They are not an approved Timmy action space, observation representation,
reward, architecture, or experiment. Copy that file to a run-specific config
and edit the contract fields directly; simulator code should not need to
change when those research choices change. Action names are the adapter keys
passed to the environment. Observation names are keys returned by it, with
each shape written per environment (batch dimension omitted).

Observation normalization is applied by the model before its spatial or scalar
encoder. `none` passes values through; `unit` applies `(x - low) / (high - low)`
with defaults 0 and 1; `standardize` applies `(x - mean) / std`; `custom` is
passed through for transformation by the configured `observation_encoder`
callback. `normalization_params` may contain scalar or broadcastable tensor-like
values. A spatial field may name another configured observation in `mask`;
the mask is multiplied into that field after normalization and must share its
height and width, with one channel or a channel count matching the field. This
can zero padded or invalid cells before the spatial encoder.

Feed-forward models are the default. Setting `MODEL.recurrent=True` enables a
GRU after feature fusion. PPO preserves one hidden state per environment across
rollout calls, resets it when an episode ends, and trains on contiguous full
rollout sequences with backpropagation through time. Set `recurrent_size` to
choose the GRU width. Recurrent sequence minibatches are grouped by environment;
the configured minibatch count therefore counts environment sequences.

The PPO implementation accepts vectorized `reset()` and `step()` methods:

```python
observation, info = env.reset(seed=run_seed)
observation, reward, terminated, truncated, info = env.step(action_dict)
```

Observations and rewards are batched by environment. `info` may include
`reward_terms`, a mapping from the configured term names to per-environment
values. The trainer reports each term separately. Checkpoints reject a load if
action, observation, model, or PPO contracts differ; the selected config and
source revision should be stored alongside each experiment.

The unit smoke uses a tiny deterministic goal environment to verify PPO
optimizer updates and checkpoint continuation. `train.py` accepts a Python
config file exporting `ACTION`, `OBSERVATION`, `REWARD`, `MODEL`, `PPO`, and
optionally `callbacks() -> (action_decoder, observation_encoder,
reward_function)`.

Run one environment from the project root after setting up Isaac Lab as shown
in `ISAAC_LAB_SETUP.md`:

```bash
"$ISAAC_SIM/python.sh" -m casu_grid.workbench.train \
  --config casu_grid/workbench/engineering_smoke.py \
  --updates 10 --run-dir experiments/engineering-smoke
```

The sample contract is only plumbing and must not be interpreted as an
approved Timmy experiment. Each run saves `config.json`, `metrics.jsonl`,
`latest.pt`, and TensorBoard event data. Monitor it with:

```bash
"$ISAAC_SIM/python.sh" -m tensorboard.main --logdir experiments/engineering-smoke/tensorboard
```

`train.py` requires `--geometry-scope full` for training, rejects nonpositive
update/environment counts and records the config file hash plus
`environment.json` (canonical content ID, collision geometry counts/build
time, timestep, action cadence and fluid settings). Use `spawn_chunk` only with
the focused diagnostic smokes; it is not a training world.

## Launch options

```bash
"$ISAAC_SIM/python.sh" -m casu_grid.workbench.train --config <your_config.py> \
  --num-envs 8 --updates 500 --run-dir experiments/my-run
```

* `--num-envs N` runs N independent worlds in one process (independent grids,
  fluid RNG streams, controllers, doors and episode state). `--terrain-mode shared`
  (default) shares one terrain; it is much cheaper but cannot edit terrain.
  `independent` gives every world its own editable terrain (about N times the
  build time and memory; see `ACCEPTANCE_REPORT.md` section 10).
* `--episode-steps N` truncates episodes after N policy decisions (default 500).
* `--resume <run>/latest.pt` continues a compatible run (same contracts).

## Inspecting a run

`metrics.jsonl` and TensorBoard hold losses, entropy, KL, gradient norm, reward
terms, action frequencies, completed-episode returns/lengths, `termination/<reason>`
counts (`time_limit`, `ragdoll:*`, `world_bounds`) and timing split into simulation
(`physics_step_seconds`), rollout and optimisation seconds.
`trajectories.jsonl` records every step's actions, rewards, poses and terminations
(plus a marker where a world restarts).

```bash
"$ISAAC_SIM/python.sh" -m casu_grid.workbench.replay \
  --config <your_config.py> --run-dir experiments/my-run
```

re-executes the recorded actions in a fresh environment, checks they reproduce the
logged poses (`replay/summary.json`) and writes `replay/replay.csv` and
`replay/paths.png`.

## What the environment guarantees (and does not)

The environment is the canonical Layer 1 world with a source-validated standing
controller on PhysX: 0.02 s physics steps, ten per decision, exact reset and
snapshot/restore, per-world isolation. It is not a bit-exact CasU replica;
landings, rare tile-junction bumps and long open-loop trajectories differ. See
`ACCEPTANCE_REPORT.md` section 6 before drawing conclusions about transfer to the
real game. Fluid decisions use a seeded PCG64 stream (replayable, not Unity's RNG);
the Body/LiquidAffect callback order is exposed as `liquid_affect_phase`.
Terrain-edit collider refresh is immediate by default; `terrain_collider_delay_ticks=31`
reproduces CasU's delayed refresh.
