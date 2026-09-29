# Seed467 Matched Parent-Anchor Control

## Decision

`parent_anchor_transfer_confirmed`

On the exact frozen seed467 replay and primary optimizer trajectory, the 0.10
parent-policy anchor improved E4 diagnostic arena strength: H4 scored `1.0000`
against seed455 and U4 scored `0.5000`. The preregistered paired H4-U4
opening-bootstrap effect is `+0.5000`, 95% CI `[+0.5000, +0.5000]` (10,000
samples, seed 378). H4 also beat U4 directly, `1.0000` on the same frozen
non-promotion suite. Neither sibling is promotion or canonical-gate eligible.

## Fixed Inputs

- Semantic identity: `seed467-parent-anchor-w010-vs-unanchored`.
- Parent weights: `f06e3e1e46815e674bd64a9437a8d8eb3a76f62632f3cbaab19771454551d00c`.
- Fresh self-play: `6f7038dccf282fb9365a031a06f8c5a8af4a2b8a6eda32725fe125b06d654a76`.
- Frozen diagnostic suite: `5835a907a712d8680000215c91e2b26d7c2bb5ce51289b60cbd141ec490a3f3f`.
- Controlled difference only: `behavior_loss_weight: 0.10 -> 0.0`; U supplied no anchor artifact or auxiliary behavior stream.
- Both arms: seed 467, replay weights `1,4,1,8,4`, four epochs, batch 512, 1,052 updates, no LR scheduler, and `best_validation` selection.
- H checkpoint SHA256: H1 `4a0728597eed9fc4f60dc9f9656ae448cc98b1b7f492d467c60d56493b4765bf`; H2 `184b9c5dbc6cfe0de517be0987bce35503946c6a7762e427c23b700df88018c7`; H3 `498852a556fa9e4cc5f2abb0631e6542aa14d169385c1d30e76fa7c0b0aee8a1`; H4 `ced2eb4b35ed4593408baaf6176832735e6d1667ee33b4f0d769ec91d80de3b9`.
- U checkpoint SHA256: U1 `0ea38854e5610997080a419f6e6db1a181967a130eb5494e8ec31676c18ac3fe`; U2 `584b4861c818fcbdcd17ee0f95beb572b4db4a5eb15384149711f66fd608f7f3`; U3 `cc4534f980507d2332ef1cee59ce3abcdd860ac2513b010aa1305846e7d28360`; U4 `798b4654a5e16ceb4dd675193074b8150da83e1887d7c78202c1890d2ff71f2f`.

## Sampling And Training

The primary replay vector, train and validation positions, train replay
indexes, all four primary permutations, update count, and LR sequence match.
U's hashes are recorded in
`.tmp/seed467-unanchored-control/runs/seed467-unanchored-control-iter1/primary_sampling_audit.json`;
the anchor's auxiliary stream is independent and therefore cannot advance the
primary Torch RNG.

| Epoch | H policy/value/val total | U policy/value/val total |
| --- | --- | --- |
| 1 | 0.93998 / 0.20581 / 1.13219 | 0.94337 / 0.20570 / 1.01754 |
| 2 | 0.92069 / 0.20192 / 1.12470 | 0.92283 / 0.20190 / 1.01305 |
| 3 | 0.90513 / 0.19991 / 1.14383 | 0.90600 / 0.20018 / 1.02854 |
| 4 | 0.89339 / 0.19917 / 1.11934 | 0.89378 / 0.19930 / 1.00511 |

Both `best_validation` selectors chose E4. The differing validation totals are
expected because H includes its anchor validation term; they were not used for
cross-arm model selection.

## Retention And Fresh Fit

Retention columns are raw argmax agreement, raw JS, PUCT agreement, PUCT visit
JS, and absolute value drift from seed455.

| Epoch | H retention | U retention |
| --- | --- | --- |
| 1 | .7478 / .0244 / .6795 / .0624 / .0770 | .7211 / .0319 / .6410 / .0691 / .0829 |
| 2 | .7567 / .0276 / .6380 / .0791 / .0974 | .7181 / .0394 / .5846 / .0883 / .1038 |
| 3 | .7448 / .0252 / .6499 / .0829 / .1415 | .6973 / .0323 / .6053 / .0895 / .1340 |
| 4 | .7656 / .0242 / .5757 / .0922 / .1474 | .7122 / .0328 / .5727 / .0957 / .1355 |

On fresh seed467 validation states with active stones above 32, H/U E4 policy
CE is `0.92941/0.92552`, argmax agreement `0.71245/0.70884`, JS
`0.20963/0.20605`; value MAE is `0.68565/0.69236` and sign accuracy is
`0.72169/0.72530`. The small policy-fit cost is not a material learning block;
value MAE is slightly better for H. Source-conditioned policy CE/value-MAE
diagnostics are retained in the two frozen trajectory JSON reports.

## Arena And Transfer

| Epoch | H vs parent | U vs parent | H - U paired 95% CI |
| --- | ---: | ---: | --- |
| 1 | .5000 | .5000 | .0000 [.0000, .0000] |
| 2 | .5000 | .5000 | .0000 [.0000, .0000] |
| 3 | .5000 | .5000 | .0000 [.0000, .0000] |
| 4 | 1.0000 | .5000 | +.5000 [+.5000, +.5000] |

| Dataset | No-anchor E4 vs parent | Anchor E4 vs parent | Anchor - no-anchor |
| --- | ---: | ---: | ---: |
| Frozen seed461 | .5117 | .6133 | +.1016 (PR #376 paired CI [.0664, .1406]) |
| Fresh seed467 | .5000 | 1.0000 | +.5000 (paired CI [.5000, .5000]) |

Seed467 is weak/flat through E3 in both arms, then the anchor produces the E4
gain; this is not the seed461 early-gain-then-decline pattern. The next axis is
one new prospective anchored generation with a preregistered matched
unanchored sibling, evaluated only on non-promotion diagnostics first.

## Guardrails

- New self-play games: `0`.
- Canonical games: `0`.
- Promotions: `0`.
