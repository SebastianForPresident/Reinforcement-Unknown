# Counterfactual profile and acceptance boundary

Latest unattended fresh-process probe:

- run: `root-path-cache-20260914`
- artifact: `counterfactual_prefix_root-path-cache-20260914/root-path-cache-20260914/prefix_replay_result.json`
- mode: fresh-process prefix replay; no `CF_RESTORE`
- result: `A/A_FAIL` at policy step 1; step 0 matched

The exact first mismatch is:

```text
state.scene/scene:Assets/Scenes/SampleScene.unity|hierarchy:World[0]/Grid[0]/Chunk[109]|component_ordinal:4|component_type:UnityEngine.Rigidbody2D/contacts/2/m_NormalImpulse
```

The two detailed values were `f32:00082269` and `f32:0003d187`. Contact
membership and geometric contact values remain covered; only the engine-owned
solver impulse differs. This remains a hard A/A failure. No A/B treatment
branch is permitted until this engine limitation is independently resolved.

## CF_STATE timing

On the original process, policy step 1 after the topology cache was established:

| Measure | Value |
| --- | ---: |
| total `CF_STATE` | 3547.243 ms |
| `SemanticSnapshot` | 3544.166 ms |
| managed digest traversal | 1167.529 ms |
| fingerprint add | 1952.238 ms |
| fingerprint finalization | 2.534 ms |
| managed digest roots | 57,641 |
| managed values visited | 238,115 |
| scene GameObjects/components | 6,413 / 23,055 |
| causal GameObjects/components | 6,326 / 22,795 |
| tilemaps/cells | 283 / 1,059,937 |
| primitive array bytes hashed | 79,461,381 |
| fingerprint input bytes | 99,229,045 |
| tilemap/native hash segment | 957.423 ms |
| other native hash segment | 1036.490 ms |
| other managed hash segment | 761.299 ms |
| array hash segment | 645.687 ms |
| topology fingerprint (cache hit) | 49.721 ms |

The timings are nested: category timings and `fingerprint add` both include
parts of the same emitted input. The optimizations currently deployed are
representation-only: cached reflection plans, cached causal topology and
component-root paths, one managed field path per read, reference-identity
deduplication, buffered streaming, primitive-array buffer/SHA reuse, and
reused per-value digest-map storage. Coverage counts and canonical fingerprint
bytes remain unchanged.

## Batch readiness

The unattended launcher supports bounded or indefinite prefix batches:

```bash
PPO_COUNTERFACTUAL_REPLAY_MODE=prefix PPO_AUTOSTART_REPEAT=3 \
  PPO_Harness/RunCounterfactual.sh --seed 391728405 --horizon 1 \
  --treatment-steps 0
```

Every repeated iteration gets a unique `batch-iteration-N` artifact ID. Unity
cleanup is limited to exact authenticated launch PIDs and their exact child
trees. Batch failures are recorded in JSONL and follow the explicit `fail`
(default), `skip`, or bounded `retry` policy. No session-management,
display-manager, process-group, or broad-name termination is used.
