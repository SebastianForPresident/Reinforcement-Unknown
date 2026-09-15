# CRIU feasibility result

CRIU was investigated as a possible fast fresh-process substitute for the
counterfactual replay loop. The installed version is 4.2.1, but the
read-only capability check fails before any checkpoint operation:

```text
$ criu check --all
CRIU needs to have the CAP_SYS_ADMIN or the CAP_CHECKPOINT_RESTORE capability:
setcap cap_checkpoint_restore+eip criu
```

The observed exit status is `1`. No capabilities were added and no system or
user-session state was changed. In particular, no checkpoint, restore, process
kill, namespace mutation, or graphical-session operation was attempted.

This is an evidenced fallback, not a claim that CRIU is semantically suitable
for CB1. Even if the missing capability is made available later, a CRIU path
would need independent A/A validation of Unity native physics/contact state
and the same strict PID/session safety audit. Until then, the supported
counterfactual path is deterministic fresh-process prefix replay through
`RunCounterfactual.sh`.

The captured shell evidence is retained at
`/tmp/cb1-criu-check.Y4qLsj/check.log` on the test machine when available.
