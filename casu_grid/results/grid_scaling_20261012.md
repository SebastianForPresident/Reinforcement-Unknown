# CasU grid-state full-world baseline

Run: 2026-10-12, benchmark revision `337e8b4c83c6803848437e1ef5add316f4568521` on
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
- `nvidia-smi` samples were 33% GPU / 456 MiB before and 32% / 457 MiB after.
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
| Empty | 0%; 0% | 0% | 499.77 | 49.98 | 1.880 / 2.805 / 3.422 |
| Static terrain | 0%; 0% | 25% | 521.69 | 52.17 | 1.879 / 2.135 / 2.668 |
| Sparse, shielded fluids | 0.0916%; 0.1953% | 0.275% | 508.10 | 50.81 | 1.952 / 2.191 / 2.297 |
| Moderate, dynamic fluids | 11.1546%; 10.2539% → 18.3105% | 0% | 402.30 | 40.23 | 2.419 / 3.033 / 3.601 |
| Dense fluid ID 3 | 100%; 100% | 0% | 89.53 | 8.95 | 11.173 / 11.886 / 12.270 |
| Stable mixed grid | 25%; 25% | 75% | 188.19 | 18.82 | 5.307 / 5.658 / 5.777 |
| Partial damage | 0%; 0% | one block | 520.34 | 52.03 | 1.892 / 2.152 / 2.333 |
| Repeated destruction | 0%; 0% | 20 transient blocks/tick | 488.92 | 48.89 | 2.034 / 2.246 / 2.379 |

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
| 1 | 188.88 | 18.89 | 188.88 | 3 MiB | 39.2 / 39.2 MiB |
| 2 | 189.63 | 18.96 | 94.81 | 6 MiB | 42.2 / 42.2 MiB |
| 4 | 185.91 | 18.59 | 46.48 | 12 MiB | 48.2 / 48.2 MiB |
| 8 | 185.04 | 18.50 | 23.13 | 24 MiB | 60.2 / 60.2 MiB |
| 16 | 188.17 | 18.82 | 11.76 | 48 MiB | 84.2 / 84.2 MiB |
| 32 | 188.64 | 18.86 | 5.89 | 96 MiB | 132.4 / 132.4 MiB |
| 64 | 188.39 | 18.84 | 2.94 | 192 MiB | 228.5 / 228.5 MiB |

VmRSS is the process resident memory after world allocation/setup; VmHWM is
the kernel's process high-water reading. The 64-world raw arrays account for exactly 192 MiB;
the remainder is Python, NumPy, interpreter and process overhead. Scaling
stopped at 64 worlds, well below the configured 512 MiB raw-array guard.

## Snapshot and resource accounting

- One world’s two arrays use **3,145,728 bytes (3 MiB)**. A snapshot copies
  those arrays and serializes to 3,146,222 bytes with Python pickle protocol 5.
- Across 51 capture/restore samples: capture median / p95 / p99 was
  **0.092 / 0.130 / 0.723 ms**; restore was **0.095 / 0.135 / 0.232 ms**.
  The p99 capture value is a single high outlier; raw samples are retained.
- Snapshot-run process RSS/high-water reached about **49.2 MiB**. These figures
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
core-only ratios range from **1.79×** (dense fluid) to **10.43×** (static
terrain). This is a ratio to a nominal game rate, not an end-to-end speedup:
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
