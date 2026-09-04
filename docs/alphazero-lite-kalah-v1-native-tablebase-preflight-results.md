# Kalah V1 Native Tablebase Preflight Results

**Classification:** `canonical_tablebase_validation_incomplete`

The isolated C++17 prototype uses canonical `kalah_v1` transitions and a store-independent signed-margin payload. It neither reads nor writes the Geoffrey Irving format and does not modify production search or artifacts. Previous results used an ABI-dependent header and did not execute the required validation harness; they are not feasibility evidence.

| Maximum tier | Cumulative states | Wall time | SHA-256 |
| --- | ---: | ---: | --- |
| 8 | 251,940 | 0.078 s | `dd9952c7c71e1d06cb99ea5c8403c3129695c2b2371ded9460f87279127493fa` |
| 10 | 1,293,292 | 0.566 s | temporary only |
| 12 | 5,408,312 | 2.960 s | `b842609f119b3eaf01937a0312853a2f6fe3f27371429793bc05e8b89686b71a` |
| 14 | 19,315,400 | 12.432 s | temporary only |

The exhaustive 8-stone Python `ExactKalahSolver` gate compared 251,940 values and 604,656 legal action values at 100%. The 8-stone graph contained 604,656 edges: 324,514 same-tier and 280,142 lower-tier; it had zero self/cyclic DFS back-edges. The native recurrence follows the resulting acyclic dependency order, not a remaining-stones-only order.

The cited compatibility position returned `{1: -4, 2: 0}`. The cited one-sided position returned storeless action values `{0: -4, 5: -4}`; adding its `18 - 24` store offset gives the required `{0: -10, 5: -10}`. A one-sided state with the empty side to move returns an explicit terminal entry.

Each generated payload consumes one byte per indexed state plus a 40-byte header, below the two-byte gate. Repeated fresh 8- and 12-stone generations were byte-identical at the listed hashes. A 100,000-query warm probe measured 626,907 positions/second.

## Validation Status

The committed runner now provides portable-file rejection, exhaustive rank/unrank through 10, exhaustive transition and value gates through 8, store-offset invariance, deterministic 9-12-stone oracle samples, cited-position checks, and fresh-file determinism gates. The expensive gates must be run and recorded before any classification can advance.

```bash
python3 ml/alphazero_lite/run_kalah_v1_native_tablebase_preflight.py --tier 12 --full-validation --output /tmp/kalah-v1-results.json
```

No scalability gate or 18-stone projection is currently accepted. Do not generate an 18-stone teacher, integrate lookup into production search, or start training from this experiment.

## Historical Projection

| Tier | Cumulative states | One-byte payload | Conservative generation estimate |
| --- | ---: | ---: | ---: |
| 16 | 60,843,510 | 58.0 MiB | 78 s |
| 18 | 172,986,450 | 165.0 MiB | 223 s |
| 20 | 451,585,680 | 430.8 MiB | 582 s |

This historical wall-time-only estimate is invalid for feasibility classification because it lacks the required correctness, memory, disk, latency, and throughput evidence. No follow-up teacher-tablebase PR is recommended.

## Reproduction

```bash
python3 ml/alphazero_lite/run_kalah_v1_native_tablebase_preflight.py --tier 8 --validate-oracle
bash native/kalah_v1_tablebase/build.sh
<binary> generate 14 /tmp/kalah_v1_14.kvtb
printf '%s\n' '{"pits":[0,1,1,0,0,0,0,1,1,0,0,0],"player":0}' | <binary> probe /tmp/kalah_v1_14.kvtb
```
