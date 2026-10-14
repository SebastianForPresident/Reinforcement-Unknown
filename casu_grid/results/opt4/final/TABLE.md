**Legacy workload (worlds ≥ 1 mis-placed, as measured in Optimization III)** — Optimization III revision `e78cfe2` vs Optimization IV with `CASU_LEGACY_MINI_SPAWN=1`

| worlds | workers | Opt III end-to-end (runs) | Opt IV end-to-end (runs) | speed-up | PSS GiB (III → IV) | peak RSS GiB (III → IV) | GPU MiB (III → IV) |
|---:|---:|---|---|---:|---|---|---|
| 16 | 8 | 1,259 (1,251, 1,267) | **1,273** (1,266, 1,279) | **1.011×** | 15.4 → 15.4 | 23.0 → 23.0 | 3401 → 3401 |
| 32 | 8 | 1,471 (1,407, 1,535) | **1,537** (1,537, 1,537) | **1.045×** | 15.9 → 15.9 | 23.5 → 23.5 | 3541 → 3405 |
| 64 | 8 | 1,762 (1,768, 1,756) | **1,771** (1,761, 1,782) | **1.005×** | 16.9 → 16.9 | 24.5 → 24.5 | 3429 → 3429 |

**Corrected workload (every world at its own spawn)** — Optimization III code + spawn fix (`a7ebcab`) vs Optimization IV HEAD

| worlds | workers | Opt III code (runs) | Opt IV (runs) | speed-up | collection s/update (III → IV) | optimiser s/update | PSS GiB (III → IV) | peak RSS GiB (III → IV) |
|---:|---:|---|---|---:|---|---|---|---|
| 16 | 8 | 1,245 (1,236, 1,254) | **1,267** (1,263, 1,272) | **1.018×** | 0.65 → 0.64 | 0.17 → 0.17 | 15.4 → 15.4 | 23.6 → 23.0 |
| 32 | 8 | 1,468 (1,420, 1,515) | **1,545** (1,546, 1,543) | **1.052×** | 1.06 → 1.00 | 0.34 → 0.33 | 15.9 → 15.9 | 23.5 → 23.5 |
| 64 | 8 | 1,702 (1,699, 1,704) | **1,740** (1,740, 1,739) | **1.022×** | 1.75 → 1.70 | 0.66 → 0.65 | 16.9 → 16.9 | 24.5 → 24.5 |

**Effect of the workload fix alone (same code)**

| worlds | Opt III code legacy → fixed | Opt IV code legacy → fixed |
|---:|---|---|
| 16 | 1,259 → 1,245 (-1.1%) | 1,273 → 1,267 (-0.5%) |
| 32 | 1,471 → 1,468 (-0.2%) | 1,537 → 1,545 (+0.5%) |
| 64 | 1,762 → 1,702 (-3.4%) | 1,771 → 1,740 (-1.8%) |
