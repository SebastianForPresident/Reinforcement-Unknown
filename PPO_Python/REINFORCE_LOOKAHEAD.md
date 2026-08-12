# CB1 critic-free lookahead

`reinforce-train` collects 2,506 decisions under one frozen policy. Only the
first 2,048 transitions are eligible for an optimizer update. The remaining
458 transitions provide observed future rewards; they are never trained on or
reused after the policy changes.

At gamma 0.99, the first omitted reward has weight `0.99 ** 459`, below one
percent. Episode resets stop return accumulation, so rewards from a new cave
cannot leak into the previous cave.

Shadow mode retains ordinary CB1 GAE for the actor and logs comparisons under
the `reinforce/` TensorBoard namespace. Active mode uses the observed returns
as PPO advantages and sets the value-loss coefficient to zero for the update.
Neither mode changes the checkpoint/protocol lineage.

Run shadow mode:

```text
./.venv/bin/python PPO_Python/Server.py reinforce-train checkpoints/CB1_2026-08-08_02-13-52 shadow
```

Active mode is deliberately separate and should only be selected after the
shadow metrics have been inspected.
