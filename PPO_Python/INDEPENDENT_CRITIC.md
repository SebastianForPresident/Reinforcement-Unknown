# CB1 independent critic

This is a continuous CB1 training phase, not a new observation/reward
protocol. The existing actor, timestep count, action space, VB1 objective, and
inference format remain unchanged.

## Update order

1. Collect the normal 2,048-step on-policy PPO rollout.
2. Pause Unity using the existing acknowledged pause.
3. Train a copied, independent value network on a rotating selection from the
   prior 32-rollout window.
4. Evaluate it on the current held-out rollout.
5. In `shadow` mode, run the ordinary CB1 PPO update unchanged.
6. In `active` mode, recompute current-rollout values and GAE with the
   independent critic, then run PPO with the shared value loss disabled.
7. Save the current rollout into the bounded window and resume Unity.

The current rollout is deliberately excluded from critic training until after
its held-out metrics are recorded.

## Storage and VRAM

Rollouts store exact normalized inputs without dense one-hot expansion:

- float32 health and toxicity
- packed extreme-health bits
- uint8 sleep/fluid categories
- cursor index and constant player-center reconstruction
- float32 general observations and training targets

Only the newest 32 rollout files are retained. Files are uncompressed to keep
the Unity pause short. The independent critic and its Adam state live on CPU
during rollout collection and the PPO update, moving to CUDA only for the
critic phase.

Rollout-aligned `casu_ppo_*.zip` files remain normal-size actor checkpoints
and are saved after every five complete PPO updates. The
independent critic has one atomic snapshot:

`independent_critic_cb1_latest.pt`

## Migration

Do not run migration while the ordinary trainer owns CUDA. After stopping the
current run and confirming its latest ZIP is complete, start shadow mode:

```bash
./.venv/bin/python PPO_Python/Server.py critic-train \
  checkpoints/CB1_2026-08-08_02-13-52 shadow
```

Shadow mode must run first. Activation is an explicit later restart using
`active`; it is never selected automatically from metrics.

The new trainer loads the newest ordinary CB1 ZIP, copies its value path into
the independent critic, and preserves `num_timesteps`. If a critic snapshot
already exists, it restores that snapshot after loading the actor.

## Metrics

TensorBoard receives:

- `critic_independent/train_loss`
- `critic_independent/heldout_loss`
- `critic_independent/heldout_explained_variance`
- `critic_independent/shared_heldout_loss`
- `critic_independent/shared_heldout_explained_variance`
- `critic_independent/window_rollouts`

Do not activate based on one rollout. Cave returns are sparse and
explained-variance readings can be pathological when target variance is tiny.
