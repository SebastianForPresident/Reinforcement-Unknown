# Convex-hull vs compound player collider — CasU evidence

Follow-up: [`HULL_QUALIFICATION_III_REPORT.md`](HULL_QUALIFICATION_III_REPORT.md) answers the open item of §8–§9 with real CasU random-tape recordings.

Status: **evidence only.** The compound collider stays the default, the convex hull stays an opt-in
(`CASU_PLAYER_COLLIDER=convex_hull`), no default, baseline or fixture was changed, and no hull parameter was tuned.
The physics-contract decision is separate. Raw data: `results/hull_evaluation/` (per-tick rows included); commands:
`perf/run_hull_evaluation.sh`; tools: `tools/compare_ab_paired.py`, `tools/summarize_paired.py`, `perf/collider_stress.py`.

## 1. Questions and answers in one table

| question | answer |
|---|---|
| Does the hull match recorded CasU motion at least as well as the validated compound? | **Mostly better, not uniformly.** Better or equal in 4 of 5 recorded traces and 7 of 7 recorded collision probes; worse in two isolated high-speed events (§4, §5) and marginally worse in one trace (immersed standing, 2 cm). |
| Is the `flat_master` landing regression systematic? | **No.** It is one decision (action 39: a 26 m/s impact on a ledge corner that CasU resolves with a Box2D position correction neither representation reproduces). The other 7 landing decisions of that trace and all 12 teacher-forced landing ticks are identical. |
| Edges and seams | CasU shows the compound **snags** where the hull does not: crossing a chunk seam (probe) and passing the top corner of a wall (`course_v1`). The hull reproduces both within 1–6 cm. Static rests on isolated/adjacent tiles and seams are identical (0 mm). No seam-specific regression of the hull in 2,530 teacher-forced ticks, apart from one ceiling tick (§6). |
| Were any corrections found? | **None applied, none supported.** The only hull geometry parameter (facets per corner) is insensitive (4 / 8 / 30 give the same aggregates within 4 %). A fix for the `flat_master` event would be fitting a single Box2D artefact. §7. |
| Open issue | On the real canonical terrain the two colliders produce **different outcome distributions for random play** (e.g. episode ends by leaving the world at y > 512: 598 vs 267; by hard landings: 274 vs 946 per 64 worlds × 1,200 decisions). There is no CasU population data to say which is closer. §8. |

## 2. Evidence and method

* **CasU references** (recorded with the instrumented source build, A/A repeatable: 700/700 and 720/720 ticks identical):
  five packed traces — `flat_master` (70 decisions, 25 m/s flat run with walls/steps/ceiling blocks, crosses the chunk seam at x = 64),
  `course_v1` (73, steps/wall/ceiling course with a 13-tile wall at the seam), `ice_run` (42), `immersed_standing` (45, in water against a wall),
  `destroy_support` (24, terrain removed under the player, collider delay 31 and 0) — and seven Mission-12 collision probes
  (wall, glass wall, chunk-seam crossing, chunk seam standing, isolated tile, adjacent tiles, natural flat ledge).
* **Three comparisons per trace** (`isaac_player_ab.py`): *teacher-forced* (each CasU tick seeds Isaac, one PhysX step compared; no compounding),
  *decision-level* (each 10-tick decision seeded from CasU), *free-run* (action tape from the first tick). Probes: free-run only
  (they have no pre-FixedUpdate controller state; `legacy_probe_traces.py`, initial state = standing at rest at the recorded pose).
* **Paired statistics** (`compare_ab_paired.py`): every tick/decision is compared between compound and hull on identical source state;
  *better/worse* counts use tolerances of 0.5 mm (position) and 0.01 m/s (velocity); the largest regressions are listed, so one event cannot hide in an average.
* **Settings are identical for both colliders** (`--step-clear 0.02`, rest offset 0.0155, contact offset 0.04, terrain contact offset 0.001, CCD on).
  The compound runs reproduce the accepted results of the earlier acceptance exactly (all modes of all traces, byte-identical statistics).
* Isaac is deterministic, so there is no run-to-run variance; CasU has one recorded run per scenario (plus the A/A check).

## 3. Results

### 3.1 Teacher-forced (engine fidelity), 2,530 ticks over 5 traces
mean error compound → hull (position m / velocity m/s); better/worse = tick counts

| source contact class | n | position | velocity | pos better/worse | vel better/worse |
|---|---:|---:|---:|---:|---:|
| ground | 1,154 | 0.0040 → 0.0037 | 0.100 → 0.082 | 4/0 | 4/0 |
| wall, grounded | 663 | 0.0016 → 0.0014 | 0.077 → 0.071 | 29/0 | 2/38 (all ≤ 0.012 m/s) |
| wall, airborne | 249 | 0.0088 → 0.0086 | 0.529 → 0.528 | 5/2 | 14/6 |
| airborne | 416 | 0.0056 → 0.0055 | 0.138 → 0.133 | 1/0 | 1/1 |
| landing | 34 | 0.0480 → 0.0416 | 4.96 → 4.58 | 1/1 | 1/1 |
| ceiling | 14 | 0.0701 → 0.0509 | 4.52 → 2.59 | 2/3 | 3/2 |

### 3.2 Decision-level (resynchronised each decision), 249 decisions

| decision kind | n | position | velocity | pos better/worse |
|---|---:|---:|---:|---:|
| ground only | 94 | 0.1085 → 0.1048 | 0.897 → 0.873 | 2/0 |
| air only / air→ground | 27 | unchanged | unchanged | 0/0 |
| landing | 21 | 0.256 → 0.168 | 1.72 → 1.21 | 5/5 |
| wall or ceiling | 107 | 0.0292 → 0.0118 | 0.553 → 0.238 | 12/1 |

### 3.3 Free-run (end-to-end), mean |position error| x / y in m

| trace | compound | hull | first error > 0.5 m (decision) compound / hull |
|---|---:|---:|---|
| `course_v1` (73 decisions) | 0.712 / 1.059 | **0.009 / 0.011** | 13 / never (y: 12 / never) |
| `ice_run` | 2.145 / 0.204 | 1.369 / 0.014 | 9 / 9 (y: 26 / 29) |
| `flat_master` | 8.107 / 0.886 | 8.001 / 0.861 | 8 / 8 |
| `destroy_support` (delay 31 / 0) | 0.192 / 0.207 (1.294) | identical | 22 / 22 |
| `immersed_standing` | 0.060 / 0.013 | 0.078 / 0.020 | never / never |

`flat_master` and `ice_run` diverge from CasU after a high-speed wall impact in decision 8–9 with either collider (a shared, previously documented residual).

### 3.4 Collision probes (free-run, ticks 10–70)
| probe | compound max err (x / y) | hull max err (x / y) |
|---|---|---|
| chunk seam, **crossing** | **0.425 / 0.213 m** (velocity error 4.3 m/s) | **0.004 / 0.006 m** |
| wall (2 decisions), glass wall | 0.097 / 0.009 m | 0.097 / 0.008 m (identical) |
| chunk seam standing, isolated tile, adjacent tiles | 0.000 / 0.000 m | 0.000 / 0.000 m |
| natural flat ledge (70 ticks) | 0.087 / 0.046 m | 0.087 / 0.046 m (identical) |

In the seam-crossing probe CasU itself shows a small bump at the seam (+1.05 then −1.15 m/s vertical, 2 cm); the hull reproduces it (position within 6 mm throughout, horizontal speed within 0.04 m/s; the vertical bump is matched up to a one-tick timing difference, 1.1 m/s at tick 7),
the compound loses 4.3 m/s horizontal speed and gains 3.3 m/s vertical speed, ending 0.43 m behind CasU.

## 4. Where the hull wins and loses

**Wins (CasU-backed)**
1. *Wall-top corner passing* (`course_v1` decision 29, the player climbs along the wall at the seam at 22 m/s): compound snags and drops from 21.6 to 5.1 m/s vertical speed at the corner (final error 1.64 m); hull passes as CasU does (0.06 m). Same signature in `destroy_support` decision 22 (0.87 m → 0.03 m). Drives the `course_v1` free-run result.
2. *Chunk-seam crossing* (probe above).
3. *Wall/ceiling decisions* are 2.5× closer (0.029 → 0.012 m) and *ground-contact velocity* error is 18 % lower (0.100 → 0.082 m/s).
4. *Ceiling impact ticks at 25 m/s* in `flat_master` (12.95 → 1.46 m/s velocity error) and `ice_run` (12.77 → 1.54).

**Losses**
1. `flat_master` decision 39 (26 m/s corner impact): hull error 1.13 m vs compound 0.59 m (§5).
2. `immersed_standing` decision 20 (vertical jump past the end of a wall in water): both get a ≈ 3 m/s sideways kick where CasU shows 0.3; hull 3.7 vs compound 2.9 m/s, so the persistent offset is 0.37 vs 0.28 m.
3. Ceiling impacts are mixed (14 ticks: position better 2 / worse 3, velocity better 3 / worse 2; e.g. `course_v1` tick 348 velocity error 0.03 → 1.49 m/s).
4. A 1 cm/s residual horizontal speed when standing against a wall (compound 0.001 m/s); position unaffected.

Everything else is equal: of the 21 landing decisions, 17 agree to within 1.5 cm (hull clearly better in 2, clearly worse in 2).

## 5. The `flat_master` landing regression

* The aggregate (landing decisions mean error 0.072 → 0.152 m in the earlier summary) comes from **one** of 8 decisions; 7 are identical to ≤ 0.015 m, and the 12 teacher-forced landing ticks are identical (position 0.0285 m, velocity 6.03 m/s for both).
* Tick-by-tick (decision 39): the player falls at −9 m/s along a ledge at x ≈ 73 while moving at 26 m/s into it. CasU keeps scraping the wall for 3 ticks (vy −9.1…−9.6), then at tick 6 **its position jumps up 0.74 m onto the ledge with zero vertical speed** — a Box2D depenetration/time-of-impact outcome, not motion.
  Compound's vertical speed collapses to ≈ 0 on tick 3 (it stays near y = 128.3–128.5 while CasU keeps falling, ending 0.33 m below CasU); hull loses 5.4 m/s at the impact and keeps sliding down (ending 1.04 m below). Neither reproduces the jump. With CCD off both pass through the corner and end on the ledge at the CasU height (128.615 m), consistent with the corner being hit ≈ 0.7 m below the ledge top.
* Not a faceting artefact: 4 / 16 / 30 facets per corner give nearly the same final error (1.04 / 1.16 / 1.14 m) and a 4–5.4 m/s vertical-speed loss at the impact.
* Conclusion: a CCD corner-impact event whose CasU outcome is an engine artefact. No correction is supported; making the hull "fit" it would be fitting one event.

## 6. Edges and seams

CasU evidence available and what it shows:

| feature | CasU evidence | result |
|---|---|---|
| chunk seam, crossing at speed | probe + `flat_master`, `course_v1`, `ice_run` (≈ 520 teacher-forced ticks within 1 m of a seam) | hull equal or better in all; compound snag in the probe |
| seam, standing | probe, 0 mm | identical |
| isolated tile, adjacent tiles | probes, 0 mm | identical |
| wall contact, grounded and airborne, glass wall | probe + 3 traces | hull equal or better (position 29 ticks better / 0 worse) |
| wall top corner (climbing past) | `course_v1`, `destroy_support`, `immersed_standing` | hull better in 2, worse by 0.9 m/s in 1 |
| ceiling impacts | 5 traces, 14 ticks | mixed |
| step / ledge | `flat_master`, `course_v1`, probe | equal |
| landing on tops of tiles | 21 decisions | equal in 17, hull clearly better in 2, worse in 2 (§4) |

Not covered by any CasU recording (so unverified for both colliders): walking off a ledge into a pit, one-tile gaps and crouch passage, slopes/diagonals, doors, repeated wall jumps, objects. Isaac-only checks (7-scenario suite, 20/50/80/100 m falls, vector and sharded validation, the three hull trajectory gates) show no hull-specific defect there, but without a CasU reference they say nothing about fidelity.

## 7. Corrections: what is and is not supported

* **Applied: none.** This round changed no physics parameter. New code only exposes the existing hull facet count as a config field (`player_hull_segments`, default 8, validated 2–30), adds the paired tools, the legacy probe loader and `--detail` output.
* **Facet count** (the one hull geometry parameter; 8 was the value chosen when the collider was authored, before any of these comparisons; its geometric error is ≈ 0.5 mm): 4 / 8 / 30 facets give teacher position error 0.0047 / 0.0047 / 0.0047 m, decision-level 0.060 / 0.060 / 0.062 m, landing 0.167 / 0.168 / 0.177 m. The probe result for the seam crossing varies non-monotonically (0.033 / 0.004 / 0.040 m): micro-bump timing, not a trend. There is no basis to retune it; selecting 8 on that probe would be tuning to one benchmark and is **not** claimed.
* **Offsets** (rest 0.0155, contact 0.04, terrain 0.001) were validated for the compound collider and are shared; no hull-specific value is supported by CasU data.
* A correction would be CasU-supported only if it improved an *independent* held-out recording after being fixed on others. With five traces and seven short probes there is not enough held-out data for that, so no correction is recommended.

## 8. Population behaviour on the real terrain (no CasU reference)

`perf/collider_stress.py`: 64 worlds × 1,200 decisions of random play from the benchmark spawn (a chamber under an overhang; the exits through the top cluster at x ≈ −44), two seeds per collider, 3 tapes.

| tape | seed | collider | episode ends: world top / hard landing | embedded ≥ 0.15 m at two consecutive decisions |
|---|---:|---|---|---:|
| random | 1 | compound | 598 / 274 | 1,007 of 75,927 |
| random | 1 | hull | 267 / 946 | 131 of 75,587 |
| random | 2 | compound | 532 / 290 | 966 of 75,976 |
| random | 2 | hull | 248 / 843 | 677 of 75,707 |
| run, climb | 1, 2 | both | time limit only, no embedding, no non-finite state, no implausible jump | 0 |

* No non-finite states and no implausible per-decision jumps with either collider.
* The "embedded" measure is only indicative (it counts the controller's collider extent against solid tiles, and auto-crouch under overhangs produces transient overlap); it is **lower for the hull**, so the hull is not more prone to tunnelling or sticking.
* The **outcome distributions differ systematically** (same direction on both seeds): the compound exits through the top shaft 2.2× more often, the hull dies by hard landing 2.9–3.5× more often. The `course_v1` evidence suggests the hull's wall-top behaviour is the faithful one, but the population effect is not explained by that mechanism alone, and CasU has not been run on these tapes. This is the main unresolved item.

## 9. What would close the remaining gaps

1. **CasU random-tape recordings** from the benchmark spawn (the same relocation/tape mechanism used for the five traces): e.g. 10 tapes × 150 decisions replayed in both colliders; compare episode ends and free-run divergence time. This is the direct test of §8 and also covers shaft climbing and wall jumping. It needs the staged CasU runtime to be launched; it was not done in this round.
2. Short targeted recordings for the uncovered features of §6 (ledge walk-off, 1-tile gap with crouch, ceiling bump while crouched).
3. After (1), re-check the 21 landing decisions and 14 ceiling ticks with the larger sample; the current counts are too small for per-class significance.
4. A trained-policy sanity run (same checkpoint evaluated under both colliders) before any change of default.

## 10. Recommendation

On the evidence available, **the hull is the better-supported representation of CasU's collision behaviour**: it is equal or better in every class of recorded motion except two isolated high-speed events, it removes a documented compound defect (snagging at wall-top corners and chunk seams, both CasU-contradicted), it is 1.4–2.8× faster, and nothing in 12 stress runs or the existing validation suites shows a hull-specific pathology. No correction to the hull is supported or needed.

It should nevertheless not become the default on tick-level evidence alone, because (a) the CasU recordings are few and short, (b) the population-level outcome distributions differ markedly (§8) and nothing yet says which is closer, and (c) changing the default changes every future comparison against earlier runs. I recommend: **record the CasU random-tape set (§9.1), then decide**. If the population data agree with the hull (or are inconclusive but not contradictory), promote it as a new, separately baselined default with `compound` kept selectable as the reference; if they favour the compound, keep the hull as the experimental fast mode and use the evidence above to scope the compound's wall-top/seam snag as a known limitation.

## 11. Files

`results/hull_evaluation/`: `ab_<trace>_{compound,convex_hull}.json` (per-tick rows), `ab_hull_<trace>_s{4,30}.json` (facet sensitivity),
`probe_<name>_{compound,convex_hull}.json`, `paired/*.json`, `stress_*.json`. Code: `legacy_probe_traces.py`, `isaac_player_ab.py` (`--detail`, `--hull-segments`, `--legacy-probe`),
`tools/compare_ab_paired.py`, `tools/summarize_paired.py`, `perf/collider_stress.py`, `perf/run_hull_evaluation.sh`, `tests/test_legacy_probes.py`.
