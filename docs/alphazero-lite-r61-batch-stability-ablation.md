# R61 Batch Stability Ablation

Inherited PR #309 classification: `reduced_lr_no_stability_gain`.

## Frozen Inputs

- G0 SHA-256: `4cd77f12319935d776b68c5e597b8d399fd68f15ba7ea25b3d66bbe786292ed4`
- R61 replay SHA-256: `6218a63b4817a58ce41df9c701c3816739976d26a1629aa32b158f3d0650aa87`
- Frozen set SHA-256: `5f86367fdb75af02294a1ace1c08dc2938919f0a5f0419376bdf89d127b6257a`
- Adam, LR `0.001`, schedule `none`, epochs `4`, replay weights `1,1,2`; physical batch size is the only variable.
- Preflight: CPU execution, `61041` train examples per epoch; B2048 completed without accumulation or replay mutation.

## Resource And Run Table

| seed | batch | anchor delta | forgotten | P(0) | steps | examples | p90 step delta |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| T61 | 512 | -0.2289 | True | 0.1583 | 432 | 244164 | 0.11565 |
| T61 | 1024 | -0.2547 | True | 0.1325 | 216 | 244164 | 0.09922 |
| T61 | 2048 | -0.3109 | True | 0.0763 | 108 | 244164 | 0.11044 |
| T63 | 512 | 0.1592 | False | 0.5464 | 432 | 244164 | 0.13754 |
| T63 | 1024 | -0.3671 | True | 0.0201 | 216 | 244164 | 0.10398 |
| T63 | 2048 | -0.3578 | True | 0.0294 | 108 | 244164 | 0.08450 |

## Baseline Reproduction

| seed | expected B512 delta | observed | within tolerance |
| --- | ---: | ---: | --- |
| T61 | -0.2289 | -0.2289 | True |
| T63 | 0.1592 | 0.1592 | True |

## Noise, Convergence, And Safety

| run | median abs step | variance | delta / step | delta / 10k examples | policy loss | value loss | cluster mass | control mass | arena effect (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| R61-T61-b512 | 0.03713 | 0.0055357 | -0.000530 | -0.009376 | 0.7907 | 0.0417 | 0.6771 | 0.7434 | 0.5000 (0.3865, 0.5000) |
| R61-T61-b1024 | 0.03476 | 0.0056093 | -0.001179 | -0.010432 | 0.7849 | 0.0428 | 0.6584 | 0.8070 | 0.5000 (0.3865, 0.5000) |
| R61-T61-b2048 | 0.03892 | 0.0064660 | -0.002879 | -0.012735 | 0.7867 | 0.0439 | 0.6355 | 0.8316 | 0.5000 (0.3865, 0.5000) |
| R61-T63-b512 | 0.04664 | 0.0068578 | 0.000369 | 0.006522 | 0.7899 | 0.0421 | 0.7089 | 0.7779 | 0.5000 (0.3865, 0.5000) |
| R61-T63-b1024 | 0.02788 | 0.0042343 | -0.001700 | -0.015035 | 0.7864 | 0.0427 | 0.6387 | 0.8264 | 0.5000 (0.3865, 0.5000) |
| R61-T63-b2048 | 0.02574 | 0.0042148 | -0.003313 | -0.014655 | 0.7825 | 0.0444 | 0.6182 | 0.7796 | 0.5000 (0.3865, 0.5000) |

The machine-readable result contains every optimizer-step anchor before/after value, losses, and the matched 25/50/75/100% example-exposure curve. The exact-outcome frozen-set safety check found no newly critical regression. The fixed arena uses the inherited deterministic seed contract; no checkpoint was promoted.

## Undertraining Diagnostic

All lanes consume the same examples over four epochs. B1024 and B2048 receive half and quarter of B512's optimizer updates respectively; their loss curves remain descending, so this result is `no_stability_gain`, not the pre-registered undertraining classification.

## Classification

`larger_batch_no_stability_gain`

Exactly one next experiment: test gradient clipping at B512/LR0 on R61.
