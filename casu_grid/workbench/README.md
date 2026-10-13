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

The Isaac launcher supports a single world and a batched `--num-envs N` path.
The batched path authors one isolated canonical terrain namespace and rigid
player instance per world, with independent grids, fluid RNG streams,
controllers, episode state, reset, snapshot and restore. `full` geometry uses
one static triangle mesh per nonempty source chunk; `spawn_chunk` is a
diagnostic-only local geometry path and must not be used as the canonical full
world. Full-world vector throughput and replay measurements are written by
`casu_grid/isaac_vector_smoke.py`. The trainer reports aggregate environment
decisions/ticks per second and per-environment decision throughput.
Snapshots include the complete source grid and scheduler/RNG state; chunk
collider regeneration is wired for explicit terrain edits. Fluid grid updates
run once per 20 ms source tick. Unity and Isaac callback order between
`Body.FixedUpdate` and `LiquidAffect.FixedUpdate` is not pinned, so the
environment exposes the phase as a config variable. Its PCG64 decisions are
deterministic and replayable, but do not claim Unity RNG seed parity.

The optional `rounded_player_collider` setting expands player collision
dimensions by the measured Unity `edgeRadius` as an engineering approximation.
The Isaac navigation smoke checks the standing/crouched collider switching,
reset and branch replay. Source crouch scale changes smoothly over fixed ticks;
the currently authored PhysX shape selection changes when the controller's
smoothed height first crosses the standing threshold. This transition remains
an identified source-parity limitation.
