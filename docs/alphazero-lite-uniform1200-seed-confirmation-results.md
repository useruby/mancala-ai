# Uniform1200 Fresh-Seed Confirmation Results

Classification: `uniform1200_not_confirmed`.

PR #286's regenerated baseline was reused byte-identically: parent weights `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`, selected replay `787cca93c7db454c2daebe1fdab001e93df0b9f80a1e7005598a0324774c427b`, and controls replay `ffac24dc8fb773fddd4dc14f397317d7c5e0a973631ca71adb216b5e68b444a5`. No artifact was promoted.

Fresh matched training/self-play seeds were 44, 45, and 46, with seed sweeps `43,44,45`, `44,45,46`, and `45,46,47`; frozen evaluation seed was 42.

| seed | direct uniform-control 1200:1200 [95% CI] | disadvantaged seat | control rows | uniform rows | compute multiplier | uniform gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 44 | +0.164 [+0.127, +0.201] | 0.883 | 63,352 | 71,166 | 5.840x | fail: arena score 0.500 |
| 45 | +0.232 [+0.193, +0.273] | 0.891 | 64,108 | 70,975 | 5.768x | fail: forensic overall |
| 46 | +0.082 [+0.037, +0.129] | 0.625 | 63,562 | 70,584 | 5.777x | fail: arena score 0.500 |

The frozen 128-opening suite was SHA-256 `57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04`. Effects use the repository opening-bootstrap 95% CI with 10,000 samples. The mean direct effect is `+0.160`, so uniform1200 beat its matched control in all three seeds, but it did not pass the unchanged production gate often enough to meet the pre-registered success rule.

Exact audit labels were excluded from training. Each lane sampled 128 opening and 128 midgame states deterministically. With the one-second exact-solver limit, no opening state completed; uniform midgame solved counts were 51, 55, and 55, with optimal-action rates 0.843, 0.800, and 0.782. This secondary audit is therefore incomplete for opening states and not used as a rejection criterion.

Next action: investigate replay/state-distribution variance between successful direct arena results and production-gate failures.
