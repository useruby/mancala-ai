# Uniform1200 Midgame Replay-Exposure x Policy-Target 2x2

Classification: `midgame_2x2_no_gain`. No model was promoted and no self-play was generated.

## Preflight

PR #289's committed audit selected `mid` under the registered rule: median matched-state policy TV was 0.230506 early, 0.311059 mid, and 0.036521 late. Shared-state counts for early/mid/late were 1359/347/510 (seed 44), 1362/398/551 (seed 45), and 1400/361/508 (seed 46), all above 200. `mid` is move indices 10-29.

The verified parent SHA was `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`; fixed selected/control replays were `787cca93c7db454c2daebe1fdab001e93df0b9f80a1e7005598a0324774c427b` and `ffac24dc8fb773fddd4dc14f397317d7c5e0a973631ca71adb216b5e68b444a5`.

## Design And Exposure

`A` reused PR #288 `uniform_exposure__sharpened`. `B` was control-like exposure with stored sharpened policies. `C` was uniform exposure with midgame-only square-root reconstructed policies. `D` combined both interventions. Dynamic rows, fixed rows, weights `1,1,2`, and total effective examples were identical for all cells of each seed.

| Seed | Control-like stratum-TV reduction |
| ---: | ---: |
| 44 | 99.736% |
| 45 | 99.711% |
| 46 | 99.765% |

Only selected-phase policies changed in C/D; zeros and values were preserved. `sharpen(unsharpen(q))` round-tripped within floating-point tolerance.

## Arena And Forensics

Arena used the frozen 128-opening suite, seed 42, two games per opening, seat balancing, and the registered budget pairs. The table reports standard paired effect versus incumbent with 95% bootstrap CI.

| Lane | Seed 44 | Seed 45 | Seed 46 |
| --- | --- | --- | --- |
| B | +0.168 [+0.131,+0.205] | -0.021 [-0.080,+0.035] | -0.082 [-0.158,-0.006] |
| C | +0.063 [-0.002,+0.127] | +0.113 [+0.059,+0.168] | -0.088 [-0.160,-0.016] |
| D | -0.074 [-0.152,+0.004] | -0.117 [-0.195,-0.039] | -0.109 [-0.189,-0.031] |

Frozen-reference forensic evaluation at 384 simulations and `c_puct=1.25` found an overall blunder-rate regression against matched control in every modified cell.

| Lane | Seed 44 | Seed 45 | Seed 46 |
| --- | ---: | ---: | ---: |
| B | +0.0759 | +0.0669 | +0.0581 |
| C | +0.0625 | +0.0625 | +0.0491 |
| D | +0.0669 | +0.0669 | +0.0268 |

## Hard Classification

`midgame_2x2_no_gain`

Every modified lane fails the production forensic condition; B and C additionally each have a statistically credible standard-arena regression, while D has one in every seed. Therefore no lane meets the pre-registered success rule.

## Next Action

Expand/strengthen the frozen forensic training-distribution diagnostic around PR #289's consistent regression families before another ML change.
