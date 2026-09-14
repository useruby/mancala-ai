# R61 Learning-Rate Stability Ablation

Inherited PR #308 classification: `anchor_interference_diffuse_optimization`.

## Recipe And Artifacts

Historical `LR0`: `0.001` (Adam, schedule `none`, 4 epochs, batch 512, weight decay 0, replay weights `1,1,2`).
- G0 SHA-256: `4cd77f12319935d776b68c5e597b8d399fd68f15ba7ea25b3d66bbe786292ed4`
- Frozen set SHA-256: `5f86367fdb75af02294a1ace1c08dc2938919f0a5f0419376bdf89d127b6257a`
- R61: `6218a63b4817a58ce41df9c701c3816739976d26a1629aa32b158f3d0650aa87`

## Baseline Reproduction

| seed | expected delta | reproduced delta | within tolerance |
| --- | ---: | ---: | --- |
| T61 | -0.2289 | -0.2289 | True |
| T63 | 0.1592 | 0.1592 | True |

## Six Runs

| seed | LR scale | anchor delta | forgotten | final top | entropy | P(0) |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| T61 | 1x | -0.2289 | True | 3 | 1.5780 | 0.1583 |
| T61 | 0.5x | -0.3368 | True | 3 | 1.4448 | 0.0504 |
| T61 | 0.25x | -0.2589 | True | 3 | 1.7229 | 0.1283 |
| T63 | 1x | 0.1592 | False | 0 | 1.6608 | 0.5464 |
| T63 | 0.5x | -0.1160 | True | 3 | 1.9504 | 0.2712 |
| T63 | 0.25x | -0.3119 | True | 3 | 1.6035 | 0.0753 |

Full final anchor policies (actions 0-5):
- `R61-T61-lr1x`: `[0.1582617461681366, 0.08695048838853836, 0.0, 0.20140162110328674, 0.13162094354629517, 0.0]`
- `R61-T61-lr0.5x`: `[0.05036880075931549, 0.05147343873977661, 0.0, 0.3402724862098694, 0.21679572761058807, 0.0]`
- `R61-T61-lr0.25x`: `[0.12830138206481934, 0.08707979321479797, 0.0, 0.2914596199989319, 0.28962695598602295, 0.0]`
- `R61-T63-lr1x`: `[0.5464289784431458, 0.0928601622581482, 0.0, 0.2064698040485382, 0.13924075663089752, 0.0]`
- `R61-T63-lr0.5x`: `[0.27119994163513184, 0.18334615230560303, 0.0, 0.2962266504764557, 0.20804505050182343, 0.0]`
- `R61-T63-lr0.25x`: `[0.07528822124004364, 0.09959768503904343, 0.0, 0.43105703592300415, 0.20398631691932678, 0.0]`

## Diffuse-Interference And Learning Curves

The machine-readable result records every optimizer-step anchor before/after mass, losses, and LR. Its `step_summary` and `early_divergence` sections provide the requested movement and first-event metrics.

| run | negative movement | positive movement | p90 | worst-10 |
| --- | ---: | ---: | ---: | ---: |
| R61-T61-lr1x | 11.2368 | 11.1105 | 0.11565 | 17.54% |
| R61-T61-lr0.5x | 6.3636 | 6.0401 | 0.06043 | 18.78% |
| R61-T61-lr0.25x | 3.6980 | 3.4592 | 0.03648 | 15.19% |
| R61-T63-lr1x | 13.3031 | 13.4579 | 0.13754 | 16.17% |
| R61-T63-lr0.5x | 7.4817 | 7.3646 | 0.07462 | 19.36% |
| R61-T63-lr0.25x | 3.7876 | 3.4473 | 0.03658 | 17.65% |


## Matched-Step Sign Consistency

- `T61`: negative all `83`, positive all `75`, sign-changing `274`, agreement `36.57%`.
- `T63`: negative all `86`, positive all `83`, sign-changing `263`, agreement `39.12%`.

## Frozen Set, Training, And Arena

| run | cluster mass | control mass | forgotten | repaired | policy loss | value loss | arena effect (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| R61-T61-lr1x | 0.6771 | 0.7434 | 1 | 0 | 0.7907 | 0.0417 | 0.5000 (0.3865, 0.5000) |
| R61-T61-lr0.5x | 0.6543 | 0.8238 | 1 | 0 | 0.7806 | 0.0426 | 0.5000 (0.3865, 0.5000) |
| R61-T61-lr0.25x | 0.6326 | 0.8043 | 1 | 0 | 0.7757 | 0.0438 | 0.5000 (0.3865, 0.5000) |
| R61-T63-lr1x | 0.7089 | 0.7779 | 0 | 0 | 0.7899 | 0.0421 | 0.5000 (0.3865, 0.5000) |
| R61-T63-lr0.5x | 0.6805 | 0.7774 | 1 | 0 | 0.7805 | 0.0432 | 0.5000 (0.3865, 0.5000) |
| R61-T63-lr0.25x | 0.6214 | 0.7963 | 1 | 0 | 0.7761 | 0.0441 | 0.5000 (0.3865, 0.5000) |

Exact-outcome frozen-set evaluation found no new critical forensic regression; the complete per-state exact metrics, step curves, and epoch loss curves are retained in the machine-readable result.

## Classification

`reduced_lr_no_stability_gain`

Exactly one next experiment: test larger batch size at LR0 under R61.
