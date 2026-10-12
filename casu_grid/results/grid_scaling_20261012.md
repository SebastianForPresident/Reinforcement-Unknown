# CasU grid-state full-world baseline

Run: 2026-10-12, revision `631bcec6aad44749e29069c514d57c06aa1ac38a` on
`isaac-lab`. Full raw samples and setup metadata are in
[`grid_scaling_20261012.json`](grid_scaling_20261012.json).

## Correctness gate

The existing V0 suite and the new full-world fixtures passed: **35 tests, 0
failures**. The 1024×1024 cases cover typed allocation, all four corners,
chunk seams on both axes, block clamping, damage thresholds and eviction,
both ice outcomes, interior flow and mixing, scan limits at world edges,
camera bands and wrap, full snapshots, a repeated action tape, independence
among four states, and a 65×64 case embedded in a full grid under matching
local camera/band context.

Two fresh processes replayed the same full-world actions and explicit random
decisions and produced the same SHA-256 fingerprint over both grids and all
snapshot metadata. The 65×64 embedding comparison also matched its block and
fluid subarrays exactly after 14 ticks. No V0 mechanics were changed.

These are tests against hand-constructed expectations and the decompiled
source contract; they are not yet executable-versus-surrogate parity tests.

## Method

- Grid: 1024×1024, x-first NumPy `uint16` block array and `uint8` fluid array;
  64×64 logical chunks.
- One `simulation_step()` is one physics-equivalent tick. Each call uses the
  actual camera-relative 128×16 scan, x outer/y inner, and seven-band schedule.
  No artificial full-grid traversal was used.
- Normal source timing is 0.02 seconds per fixed tick. Policy-equivalent
  throughput is reported as ten ticks per macrostep.
- Single-world runs: 20 untimed warm-up ticks, then 101 individually timed
  ticks. Multi-world runs: fresh process per count, 10 warm-up cycles then 51
  measured cycles; each cycle advances each world sequentially.
- State construction and workload setup are outside the simulation-rate
  samples. Their durations are retained in the JSON. Fluid densities are
  reported as both fluid cells / 1,048,576 world cells and fluid cells / the
  initial 2,048-cell scan band. Dynamic workload end density is also recorded.
- This used the Isaac-bundled Python 3.10 interpreter, but did **not** start
  Isaac Kit, Torch, or CUDA. Transition loops and arrays ran on CPU in one
  Python thread. Process CPU use was about 99.8% of one logical core.
- `nvidia-smi` samples were 28% GPU / 456 MiB before and 31% / 456 MiB after.
  The benchmark code did not submit GPU work; those point samples are
  device-wide and do not attribute activity to this process.
- Raw per-tick and per-cycle nanosecond samples, memory readings, allocation
  times, workload densities, and snapshot latency vectors are preserved in
  the JSON artifact.

## Single-world performance

Latency values are median / p95 / p99 milliseconds for one simulation tick.
Rates count full source-equivalent ticks, including any per-tick damage action
listed in the workload name/definition.

| Workload | Fluid density (world; initial band) | Solid density | ticks/s | 10-tick macrosteps/s | median / p95 / p99 ms |
|---|---:|---:|---:|---:|---:|
| Empty | 0%; 0% | 0% | 503.65 | 50.37 | 1.905 / 2.757 / 3.156 |
| Static terrain | 0%; 0% | 25% | 502.60 | 50.26 | 1.886 / 2.667 / 3.052 |
| Sparse, shielded fluids | 0.0916%; 0.1953% | 0.275% | 511.29 | 51.13 | 1.884 / 2.326 / 3.173 |
| Moderate, dynamic fluids | 11.1546%; 10.2539% → 18.3105% | 0% | 398.80 | 39.88 | 2.434 / 3.112 / 3.631 |
| Dense fluid ID 3 | 100%; 100% | 0% | 89.59 | 8.96 | 11.166 / 11.721 / 11.989 |
| Stable mixed grid | 25%; 25% | 75% | 185.57 | 18.56 | 5.320 / 5.972 / 6.172 |
| Partial damage | 0%; 0% | one block | 518.44 | 51.84 | 1.884 / 2.176 / 2.220 |
| Repeated destruction | 0%; 0% | 20 transient blocks/tick | 483.13 | 48.31 | 2.056 / 2.328 / 2.500 |

Workload construction details: sparse fluids are isolated by surrounding blocks;
the moderate case places ID 3 on every third x and y index in empty terrain; the
dense case is an all-ID-3 stable array; stable mixed uses ID 3 on 25% of cells
and blocks on the other 75%, leaving no empty destinations. Partial damage
applies ten 0.001-damage hits per tick to one block after an initial 3000 damage.
Repeated destruction sets and destroys twenty ID-1 blocks each tick. These
damage workloads measure only the discrete state core; there is no Tilemap or
collider work.

The moderate workload preserved its total 116,964 fluid cells but the initial
active-band occupancy rose from 210 to 375 cells during measurement. The dense
and stable-mixed configurations remained unchanged.

## Multi-world scaling

Each world uses the stable mixed workload (25% ID-3 fluid, 75% solid blocks).
Worlds have independent arrays and are advanced sequentially by one process.
Rates are aggregate over all environments; per-environment throughput falls as
world count increases while aggregate throughput remains near one CPU core's
capacity.

| Worlds | Aggregate ticks/s | Aggregate 10-tick macrosteps/s | ticks/s per world | Raw grid arrays | Process VmRSS / VmHWM |
|---:|---:|---:|---:|---:|---:|
| 1 | 187.44 | 18.74 | 187.44 | 3 MiB | 39.3 / 39.3 MiB |
| 2 | 186.32 | 18.63 | 93.16 | 6 MiB | 42.2 / 42.2 MiB |
| 4 | 187.83 | 18.78 | 46.96 | 12 MiB | 48.2 / 48.2 MiB |
| 8 | 188.18 | 18.82 | 23.52 | 24 MiB | 60.3 / 60.3 MiB |
| 16 | 187.59 | 18.76 | 11.72 | 48 MiB | 84.3 / 84.3 MiB |
| 32 | 188.02 | 18.80 | 5.88 | 96 MiB | 132.4 / 132.4 MiB |
| 64 | 189.30 | 18.93 | 2.96 | 192 MiB | 228.5 / 228.4 MiB |

VmRSS is the process resident memory after world allocation/setup; VmHWM is
the kernel's process high-water reading. The 64-world raw arrays account for exactly 192 MiB;
the remainder is Python, NumPy, interpreter and process overhead. Scaling
stopped at 64 worlds, well below the configured 512 MiB raw-array guard.

## Snapshot and resource accounting

- One world’s two arrays use **3,145,728 bytes (3 MiB)**. A snapshot copies
  those arrays and serializes to 3,146,222 bytes with Python pickle protocol 5.
- Across 51 capture/restore samples: capture median / p95 / p99 was
  **0.094 / 0.168 / 0.884 ms**; restore was **0.097 / 0.182 / 0.243 ms**.
  The p99 capture value is a single high outlier; raw samples are retained.
- Snapshot-run process RSS/high-water reached about **49.3 MiB**. These figures
  include interpreter and NumPy overhead.
- No grid state was allocated on GPU. The 8 GB RTX 3070 was only observed via
  device-wide `nvidia-smi` samples; its utilization was nonzero, so those
  readings cannot be treated as this benchmark’s GPU utilization.

## Bottlenecks and baseline comparison

The scan visits the same 2,048 positions per tick regardless of the 1024²
world’s total cell count. An empty tick took about 1.90 ms median. Dense fluid
cost about 5.9× the empty median; the 25% fluid / 75% solid case cost about
2.8×. The added cost comes from the source-preserving Python branches and
repeated indexed reads at occupied fluid cells. The damage-only actions add
little relative to the scan. Snapshot copies are small and fast compared with
the 5–11 ms occupied-fluid ticks.

Against the stated approximate 50-tick/s CasU real-time baseline, the measured
core-only ratios range from **1.79×** (dense fluid) to **10.37×** (partial
damage). This is a ratio to a nominal game rate, not an end-to-end speedup:
Unity rendering, physics, collider updates, audio, and game scripts are absent.
All tested single-world cases exceeded 50 core ticks/s, but that does not prove
the complete surrogate will run faster than CasU.

The prior 8.96-million trivial Isaac environment-steps/s figure measures a
simple rigid-body scene and is not comparable to these ordered grid ticks.
Dividing those numbers would produce a misleading throughput claim.

## Ranked optimization candidates (not implemented)

1. **Compile the exact ordered inner loop (Cython/C++ first).** This targets
   Python dispatch and scalar NumPy indexing while retaining x-major/y-minor,
   in-place movement and branch order. Risk: high if typed arithmetic or loop
   bounds differ. Required gate: exact full-array, damage, scheduler and RNG
   position equality after every tick for all 35 fixtures plus randomized
   explicit decision tapes and all density workloads.
2. **Batch independent worlds across a compiled/GPU kernel.** This targets the
   flat aggregate rate at increasing environment count while preserving a
   serial ordered loop within each world. Risk: high branch divergence and
   random-stream coupling; no GPU workload was measured here. Required gate:
   compare every environment’s exact state and decision-stream position across
   1/2/4/8/16/32/64 worlds and repeated process launches.
3. **Profile and reduce snapshot overhead only if replay volume grows.** Current
   snapshot latency is under 0.2 ms at p95 and data size is about 3 MiB, so this
   is lower priority. Copy-on-write or delta snapshots risk restoration
   semantics. Required gate: exact full-state restore and deterministic
   continuation after arbitrary mutation/restore sequences.
4. **Consider sparse active-fluid indexes only after a semantic proof.** It
   could reduce work for sparse occupancy, but building a list before an
   in-place sweep can miss fluids moved into later cells and change the
   rightward scan bias. Risk is very high. Required gate: compare exact state
   after every source-ordered cell visit, including newly moved cells, boundary
   loss, random two-sided moves, and all seven camera bands.

No optimization, collider integration or player physics was performed.
