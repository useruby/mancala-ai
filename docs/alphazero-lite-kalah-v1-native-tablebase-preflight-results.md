# Kalah V1 Native Tablebase Preflight Results

**Classification:** `canonical_tablebase_feasible`

The isolated C++17 prototype uses canonical `kalah_v1` transitions and a store-independent signed-margin payload. It neither reads nor writes the Geoffrey Irving format and does not modify production search or artifacts.

| Maximum tier | Cumulative states | Wall time | SHA-256 |
| --- | ---: | ---: | --- |
| 8 | 251,940 | 0.078 s | `dd9952c7c71e1d06cb99ea5c8403c3129695c2b2371ded9460f87279127493fa` |
| 10 | 1,293,292 | 0.566 s | temporary only |
| 12 | 5,408,312 | 2.960 s | `b842609f119b3eaf01937a0312853a2f6fe3f27371429793bc05e8b89686b71a` |
| 14 | 19,315,400 | 12.432 s | temporary only |

The exhaustive 8-stone Python `ExactKalahSolver` gate compared 251,940 values and 604,656 legal action values at 100%. The 8-stone graph contained 604,656 edges: 324,514 same-tier and 280,142 lower-tier; it had zero self/cyclic DFS back-edges. The native recurrence follows the resulting acyclic dependency order, not a remaining-stones-only order.

The cited compatibility position returned `{1: -4, 2: 0}`. The cited one-sided position returned storeless action values `{0: -4, 5: -4}`; adding its `18 - 24` store offset gives the required `{0: -10, 5: -10}`. A one-sided state with the empty side to move returns an explicit terminal entry.

Each generated payload consumes one byte per indexed state plus a 40-byte header, below the two-byte gate. Repeated fresh 8- and 12-stone generations were byte-identical at the listed hashes. A 100,000-query warm probe measured 626,907 positions/second.

## Projection

| Tier | Cumulative states | One-byte payload | Conservative generation estimate |
| --- | ---: | ---: | ---: |
| 16 | 60,843,510 | 58.0 MiB | 78 s |
| 18 | 172,986,450 | 165.0 MiB | 223 s |
| 20 | 451,585,680 | 430.8 MiB | 582 s |

The conservative estimate uses the slower measured 14-stone per-state cost with a 2x safety factor. It is below the eight-hour, 16-GiB, and 4-GiB 18-stone gates. A separate PR should implement and independently validate the 18-stone teacher tablebase; this preflight does not generate it, train a model, or integrate lookup.

## Reproduction

```bash
python3 ml/alphazero_lite/run_kalah_v1_native_tablebase_preflight.py --tier 8 --validate-oracle
bash native/kalah_v1_tablebase/build.sh
<binary> generate 14 /tmp/kalah_v1_14.kvtb
printf '%s\n' '{"pits":[0,1,1,0,0,0,0,1,1,0,0,0],"player":0}' | <binary> probe /tmp/kalah_v1_14.kvtb
```
