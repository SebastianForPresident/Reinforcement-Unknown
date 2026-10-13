# Researcher reward components

The GUI component library recognizes explicitly imported reward modules with an
adjacent JSON declaration named `<module>.py.component.json`. Discovery reads
this JSON without importing Python. Importing a component copies the selected
source into `casu_grid/workbench/components/`; it never overwrites a file.
Validation imports only that selected module in a short-lived subprocess with a
five-second timeout. Python is executable trusted code; subprocess isolation
limits the GUI process lifetime and accidental namespace effects, but is not a
security sandbox.

## Contract version 1

```json
{
  "id": "researcher.progress_reward",
  "name": "Progress reward",
  "description": "Reward change in a declared progress field.",
  "entry_point": "reward",
  "contract_version": 1,
  "required_fields": ["progress"],
  "synthetic_context": {"previous": {"progress": 0.0}, "current": {"progress": 1.0}},
  "environment_compatibility": "Isaac Layer 1 state dictionaries containing progress",
  "parameter_schema": [
    {"name": "weight", "type": "float", "default": 1.0,
     "minimum": 0.0, "maximum": 100.0, "units": "reward/progress",
     "description": "Researcher-selected scale"}
  ],
  "output": {"type": "scalar_and_terms", "shape": []},
  "stateful": false,
  "reset_entry_point": null,
  "deterministic": true,
  "supported_context": "previous/current state plus terminated/truncated flags"
}
```

The Python module declares matching metadata as `REWARD_COMPONENT` and exposes
the selected `entry_point` with signature
`(previous, current, terminated, truncated, **parameters)`. The callback
returns `(finite_scalar_reward, {term_name: finite_scalar, ...})`. Current
workbench runtime validation then checks the callback at the environment
boundary against the experiment's declared reward term names. A component
must use the fields actually supplied by the configured environment adapter;
metadata alone does not prove those fields exist.

Example module:

```python
REWARD_COMPONENT = {
    "id": "researcher.progress_reward",
    "name": "Progress reward",
    "description": "Reward change in a declared progress field.",
    "entry_point": "reward",
    "contract_version": 1,
    "required_fields": ["progress"],
    "synthetic_context": {"previous": {"progress": 0.0}, "current": {"progress": 1.0}},
    "parameter_schema": [{"name": "weight", "type": "float", "default": 1.0}],
    "reward_terms": ["progress"],
    "output": {"type": "scalar_and_terms", "shape": []},
    "stateful": False,
    "deterministic": True,
}

def reward(previous, current, terminated, truncated, *, weight):
    value = weight * (current["progress"] - previous["progress"])
    return value, {"progress": value}
```

The current worker performs schema checks and two bounded synthetic calls with
the author-declared `synthetic_context` (or empty dictionaries when omitted).
Such a probe validates basic interface, parameter and output properties only;
environment-state compatibility, physics behavior,
reset semantics, and stochastic/stateful behavior require a bounded Isaac
probe and are not inferred from synthetic validation. Stateful components are
currently not accepted as run-integrated callbacks because the environment
callback contract does not define per-world reset/clone semantics. Authors
must declare `stateful: false` for this initial implementation.

The GUI records a `REWARD_COMPONENT_REF` with source path, entry point, source
hash, parameters, term names and validation level after showing a source diff
and receiving explicit save confirmation. Training and replay resolve that
reference through the same helper, preserving the experiment's action decoder
and observation encoder. The ordinary runtime callback validator then enforces
the configured observation/reward contract at each environment step. Run
manifests preserve the exact config snapshot/hash; imported source files must
also be retained alongside results for independent reproduction.

Before selecting a component, the experiment's Python `REWARD.terms` names must
match the component's declared `reward_terms`. The GUI checks this before
writing the explicit component reference. It does not rewrite arbitrary Python
reward contract expressions on the researcher's behalf.

Observation encoders/producers, action decoders, model factories and evaluation
metrics remain future registry categories. Their presence in the extension
architecture does not imply they are currently loadable plugins.
