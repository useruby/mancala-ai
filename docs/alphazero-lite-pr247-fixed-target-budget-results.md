# PR #247 Fixed Policy-Target Budget

**Classification:** `fixed_768_sufficient`

## Frozen Replay

- Reused the PR #247 seed-45 trajectory: 27,858 eligible rows and policy-excluded SHA-256 `45e9b0a76879fd0e92f6536107073ecbe16a59832125e74fb6d12fb6230c527c`.
- Every view used batch-plan SHA-256 `a4e859f1340078e21c9ad2b4e0c04bac5c69c7463d8b8b859f83f79f915382fe`.
- Reconstructed fresh controls matched exactly: fresh-384 `74203f35df72c25d4b02f25605926fb34e36edc114cb0a4e2c2cd8f2771c84c6`; fresh-equivalent `e1bc3b17c165b8aee210fc8267e4477c510abe467f47225c3c4f874f74005bab`.
- The A16 snapshot and pristine Adam state matched their frozen hashes. The repeated fixed-1024 lane was byte-identical.

## Compute

| Target lane | Target simulations | Gameplay plus target | Multiplier vs gameplay-only |
| --- | ---: | ---: | ---: |
| fresh-384 | 384 | 768 | 2.000x |
| fixed-768 | 768 | 1152 | 3.000x |
| fixed-1024 | 1024 | 1408 | 3.667x |
| fixed-1280 | 1280 | 1664 | 4.333x |
| fixed-1536 | 1536 | 1920 | 5.000x |
| fresh-equivalent | empirical mean 1228.42 | empirical mean 1612.42 | empirical mean 4.199x |

Fresh-equivalent target simulations were p50 1106, p90 2071, p99 3301.43, and max 5773.

## Target Geometry

Fixed-target mean JS versus fresh-equivalent was .021203 (768), .014845 (1024), .017780 (1280), and .023544 (1536). It was not monotonic with budget, so target-space convergence was not used for selection. The fraction of positions closer to fresh-equivalent than to fresh-384 was .700, .762, .778, and .782 respectively.

The full JSON artifact records each fixed lane's legal L1 distribution, JS, top-1 disagreement, and entropy difference against fresh-384, fresh-equivalent, and the PR #247 reused target, plus all requested per-state strata.

## Step-16 Arena

All evaluations used ordinary PUCT, the current canonical 128 openings, seat swapping, seed 42, matched P1 controls, and 10,000 opening bootstraps.

| Lane | 384:256 effect / W-D-L | 1200:1200 effect / CI / W-D-L |
| --- | --- | --- |
| fresh-384 | -.019531 / 390-38-84 | -.019531 / [-.039062,-.003906] / 180-132-200 |
| fixed-768 | -.019531 / 390-38-84 | +.041016 / [+.013672,+.066406] / 242-70-200 |
| fixed-1024 | -.019531 / 390-38-84 | -.019531 / [-.039062,-.003906] / 180-132-200 |
| fixed-1280 | -.019531 / 390-38-84 | -.019531 / [-.039062,-.003906] / 180-132-200 |
| fixed-1536 | -.019531 / 390-38-84 | -.019531 / [-.039062,-.003906] / 180-132-200 |
| fresh-equivalent | -.019531 / 390-38-84 | +.041016 / [+.013672,+.066406] / 242-70-200 |

Fixed-768 minus fresh-384 at 1200:1200 was +.060547 with paired CI [+.042969,+.080078]. Fixed-768 minus fresh-equivalent was 0 with paired CI [0,0]. Its fit fraction was at least .25, so it is the smallest passing preregistered budget. The low-budget result remained the established -.019531 regression for every lane.

## Interpretation

A fixed fresh policy-target search of 768 simulations retains the PR #247 deep-search gain at 3.0x gameplay-only per-move compute, replacing the inherited-mass rule without carrying subtree or Q statistics.

Recommended next: prospectively generate with fixed-768 targets, then address shallow-search safety separately.
