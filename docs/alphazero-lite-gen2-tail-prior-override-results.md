# Gen-2 Tail Prior Override Results

**Primary classification:** `moderately_sparse_tail_causal`

**Next experiment:** train Gen-2 with a broader selective per-state output constraint

## Frozen Calibration

- Gen-2 trust-state SHA256: `fef39b0bf5c3ecd3fa93c8cd7cbe688e96bc4ae2285448488ee2234608a20e9f`
- P1 state hash: `a86acb54b97c860289530fcb7ca64194724d43667580a52e2948359bfa3ebdf4`
- P2 state hash: `336496d5fb33331240178c4b834b8faf9548e3915b45c9b5f7e4b7aad6626870`

| Quantile | Legal-policy L1 threshold |
| --- | ---: |
| q50 | 0.0056215204 |
| q75 | 0.0084935703 |
| q90 | 0.0110328559 |
| q95 | 0.0126585270 |
| q99 | 0.0160649136 |

## Canonical Arena (384:256, 128 openings, seat swapped)

| Lane | Effect | 95% CI | P0 | P1 | W/D/L | Recovery | P0 Recovery |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: |
| candidate_all | -0.0957 | [-0.1270, -0.0645] | -0.1914 | +0.0000 | 354/32/126 | 0.0 | 0.0 |
| tail_q99 | -0.0957 | [-0.1270, -0.0645] | -0.1914 | +0.0000 | 354/32/126 | 0.0 | 0.0 |
| tail_q95 | -0.0410 | [-0.0625, -0.0234] | -0.0820 | +0.0000 | 382/32/98 | 0.5714285714285714 | 0.5714285714285714 |
| tail_q90 | -0.0410 | [-0.0625, -0.0234] | -0.0820 | +0.0000 | 382/32/98 | 0.5714285714285714 | 0.5714285714285714 |
| tail_q75 | -0.0176 | [-0.0293, -0.0078] | -0.0352 | +0.0000 | 382/56/74 | 0.8163265306122449 | 0.8163265306122449 |
| incumbent_all | +0.0000 | [+0.0000, +0.0000] | +0.0000 | +0.0000 | 400/38/74 | 1.0 | 1.0 |

## Override Coverage (frozen replay-state PUCT probe)

| Lane | Expanded | Overridden | Fraction | L1 mass removed |
| --- | ---: | ---: | ---: | ---: |
| tail_q99 | 77339 | 624 | 0.0081 | 0.0257 |
| tail_q95 | 77364 | 3076 | 0.0398 | 0.1042 |
| tail_q90 | 77372 | 6618 | 0.0855 | 0.2000 |
| tail_q75 | 77370 | 17855 | 0.2308 | 0.4482 |
| incumbent_all | 77373 | 77373 | 1.0000 | 1.0000 |

High-budget gate: not run. No selective lane simultaneously recovered at least 70% and overrode at most 10% of expanded nodes.
