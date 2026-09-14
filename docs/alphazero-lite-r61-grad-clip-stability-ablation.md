# R61 Gradient Clip Strength Ablation

Inherited PR #310 classification: `larger_batch_no_stability_gain`.

Historical B512/LR0 already used global `grad_clip=1.0`; this is threshold relaxation, not clipping off -> on.

## Artifacts

- G0 SHA-256: `4cd77f12319935d776b68c5e597b8d399fd68f15ba7ea25b3d66bbe786292ed4`
- R61 replay SHA-256: `6218a63b4817a58ce41df9c701c3816739976d26a1629aa32b158f3d0650aa87`
- Frozen set SHA-256: `5f86367fdb75af02294a1ace1c08dc2938919f0a5f0419376bdf89d127b6257a`
- Adam, LR `0.001`, B512, four epochs, schedule `none`, and replay weights `1,1,2` are fixed.
- G0 anchor optimal mass: `0.3872`.

## C1 Baseline Reproduction

| seed | expected delta | observed | within tolerance |
| --- | ---: | ---: | --- |
| T61 | -0.2289 | -0.2289 | True |
| T63 | 0.1592 | 0.1592 | True |

## Eight Runs And Clip Exposure

| run | clip | anchor delta | forgotten | top | P(0) | entropy | clipped | p50/p90 preclip | p50/p90 scale |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| R61-T61-c1 | 1.0 | -0.2289 | True | 3 | 0.1583 | 1.5780 | 100.0% | 3.119/4.465 | 0.321/0.443 |
| R61-T61-c2 | 2.0 | -0.1944 | False | 0 | 0.1928 | 1.3985 | 97.7% | 3.059/4.486 | 0.654/0.879 |
| R61-T61-c4 | 4.0 | -0.3242 | True | 1 | 0.0630 | 1.4494 | 15.7% | 3.017/4.208 | 1.000/1.000 |
| R61-T61-cnone | None | -0.3196 | True | 3 | 0.0676 | 1.5638 | 0.0% | 3.065/4.531 | 1.000/1.000 |
| R61-T63-c1 | 1.0 | 0.1592 | False | 0 | 0.5464 | 1.6608 | 100.0% | 3.196/4.519 | 0.313/0.432 |
| R61-T63-c2 | 2.0 | 0.1630 | False | 0 | 0.5502 | 1.6527 | 97.5% | 3.102/4.462 | 0.645/0.879 |
| R61-T63-c4 | 4.0 | 0.0964 | False | 0 | 0.4836 | 1.7251 | 17.4% | 3.098/4.454 | 1.000/1.000 |
| R61-T63-cnone | None | 0.0674 | False | 0 | 0.4546 | 1.7466 | 0.0% | 3.022/4.295 | 1.000/1.000 |

## Full Clipping Exposure

| run | p50/p90/p95/p99 preclip | p50/p90 scale | mean postclip | total preclip | total postclip |
| --- | --- | --- | ---: | ---: | ---: |
| R61-T61-c1 | 3.119/4.465/5.221/6.016 | 0.321/0.443 | 1.000 | 1443.5 | 432.0 |
| R61-T61-c2 | 3.059/4.486/4.882/5.878 | 0.654/0.879 | 1.998 | 1427.6 | 863.1 |
| R61-T61-c4 | 3.017/4.208/4.773/6.134 | 1.000/1.000 | 3.039 | 1371.0 | 1312.7 |
| R61-T61-cnone | 3.065/4.531/4.923/5.879 | 1.000/1.000 | 3.304 | 1427.1 | 1427.1 |
| R61-T63-c1 | 3.196/4.519/4.928/6.371 | 0.313/0.432 | 1.000 | 1469.5 | 432.0 |
| R61-T63-c2 | 3.102/4.462/4.840/6.548 | 0.645/0.879 | 1.996 | 1430.8 | 862.3 |
| R61-T63-c4 | 3.098/4.454/4.831/6.346 | 1.000/1.000 | 3.125 | 1425.6 | 1349.8 |
| R61-T63-cnone | 3.022/4.295/4.860/5.912 | 1.000/1.000 | 3.227 | 1394.3 | 1394.3 |

## Step Dynamics

| run | harmful/protective | negative/positive movement | median/p90 abs delta | variance | delta/postclip norm | first flip | <0.20 | >0.50 | longest incorrect |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R61-T61-c1 | 215/217 | 11.237/11.111 | 0.0371/0.1157 | 0.005536 | -0.000292 | 1 | 1 | 3 | 57 |
| R61-T61-c2 | 217/215 | 10.996/11.001 | 0.0379/0.1066 | 0.005259 | 0.000005 | 1 | 1 | 3 | 65 |
| R61-T61-c4 | 219/213 | 11.201/11.107 | 0.0370/0.1151 | 0.005464 | -0.000072 | 1 | 1 | 3 | 78 |
| R61-T61-cnone | 220/212 | 9.496/9.325 | 0.0315/0.0946 | 0.003824 | -0.000120 | 1 | 1 | 2 | 150 |
| R61-T63-c1 | 220/212 | 13.303/13.458 | 0.0466/0.1375 | 0.006858 | 0.000358 | 1 | 1 | 12 | 50 |
| R61-T63-c2 | 217/215 | 11.355/11.521 | 0.0356/0.1258 | 0.005433 | 0.000192 | 1 | 1 | 12 | 57 |
| R61-T63-c4 | 220/212 | 10.842/10.954 | 0.0376/0.1038 | 0.004985 | 0.000083 | 1 | 1 | 12 | 123 |
| R61-T63-cnone | 209/223 | 10.132/10.229 | 0.0313/0.1050 | 0.004700 | 0.000069 | 1 | 1 | 4 | 94 |

## Clipped Versus Unclipped (Descriptive)

| run | group | mean delta | harmful rate | protective rate | policy loss | value loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| R61-T61-c1 | clipped (432) | -0.00029 | 49.8% | 50.2% | 0.8186 | 0.0454 |
| R61-T61-c1 | unclipped (0) | 0.00000 | 0.0% | 0.0% | 0.0000 | 0.0000 |
| R61-T61-c2 | clipped (422) | 0.00003 | 50.2% | 49.8% | 0.8195 | 0.0455 |
| R61-T61-c2 | unclipped (10) | -0.00062 | 50.0% | 50.0% | 0.7767 | 0.0448 |
| R61-T61-c4 | clipped (68) | 0.00526 | 44.1% | 55.9% | 0.8435 | 0.0459 |
| R61-T61-c4 | unclipped (364) | -0.00124 | 51.9% | 48.1% | 0.8101 | 0.0451 |
| R61-T61-cnone | clipped (0) | 0.00000 | 0.0% | 0.0% | 0.0000 | 0.0000 |
| R61-T61-cnone | unclipped (432) | -0.00040 | 50.9% | 49.1% | 0.8181 | 0.0452 |
| R61-T63-c1 | clipped (432) | 0.00036 | 50.9% | 49.1% | 0.8168 | 0.0451 |
| R61-T63-c1 | unclipped (0) | 0.00000 | 0.0% | 0.0% | 0.0000 | 0.0000 |
| R61-T63-c2 | clipped (421) | 0.00042 | 50.1% | 49.9% | 0.8160 | 0.0451 |
| R61-T63-c2 | unclipped (11) | -0.00089 | 54.5% | 45.5% | 0.7807 | 0.0435 |
| R61-T63-c4 | clipped (75) | 0.00763 | 49.3% | 50.7% | 0.8447 | 0.0465 |
| R61-T63-c4 | unclipped (357) | -0.00129 | 51.3% | 48.7% | 0.8065 | 0.0446 |
| R61-T63-cnone | clipped (0) | 0.00000 | 0.0% | 0.0% | 0.0000 | 0.0000 |
| R61-T63-cnone | unclipped (432) | 0.00022 | 48.4% | 51.6% | 0.8169 | 0.0453 |

## Matched Steps

Same-seed minibatch order was verified across all lanes. A clip-sensitive step is pre-registered as absolute anchor-step-delta range >= `0.05`. Full matched trajectories and future masses are in the JSON result; clipped-versus-unclipped associations are descriptive only.
- `T61`: `243` clip-sensitive steps.
- `T63`: `248` clip-sensitive steps.

## Adam State By Epoch

The JSON records compact first-moment, second-moment, and parameter-update norms for shared trunk, policy head, and value head after every epoch. The final policy-head/trunk values are:

| run | trunk m1/m2/update | policy m1/m2/update |
| --- | --- | --- |
| R61-T61-c1 | 0.00025/0.01111/0.87148 | 0.00008/0.00015/0.30476 |
| R61-T61-c2 | 0.00045/0.04427/0.89490 | 0.00015/0.00058/0.33087 |
| R61-T61-c4 | 0.00071/0.10826/0.83133 | 0.00020/0.00143/0.31418 |
| R61-T61-cnone | 0.00087/0.14741/0.78572 | 0.00026/0.00217/0.26876 |
| R61-T63-c1 | 0.00021/0.01141/0.94054 | 0.00006/0.00014/0.33649 |
| R61-T63-c2 | 0.00040/0.04441/0.87033 | 0.00012/0.00055/0.36207 |
| R61-T63-c4 | 0.00062/0.11828/0.83566 | 0.00018/0.00147/0.38277 |
| R61-T63-cnone | 0.00070/0.14429/0.80616 | 0.00022/0.00204/0.35115 |

## Frozen Set, Convergence, And Safety

| run | cluster mass | control mass | cluster-control | forgotten | repaired | policy/value loss | validation policy/value | arena effect (95% CI) | stable |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| R61-T61-c1 | 0.6771 | 0.7434 | -0.0663 | 1 | 0 | 0.7907/0.0417 | 0.7937/0.0417 | 0.5000 (0.3865, 0.5000) | True |
| R61-T61-c2 | 0.6867 | 0.7422 | -0.0555 | 0 | 0 | 0.7892/0.0413 | 0.8052/0.0425 | 0.5000 (0.3865, 0.5000) | True |
| R61-T61-c4 | 0.6130 | 0.7411 | -0.1281 | 1 | 0 | 0.7862/0.0412 | 0.8109/0.0434 | 0.5000 (0.3865, 0.5000) | True |
| R61-T61-cnone | 0.6191 | 0.7493 | -0.1301 | 1 | 0 | 0.7867/0.0414 | 0.8018/0.0423 | 0.5000 (0.3865, 0.5000) | True |
| R61-T63-c1 | 0.7089 | 0.7779 | -0.0690 | 0 | 0 | 0.7899/0.0421 | 0.7936/0.0418 | 0.5000 (0.3865, 0.5000) | True |
| R61-T63-c2 | 0.7086 | 0.7800 | -0.0714 | 0 | 0 | 0.7878/0.0419 | 0.7982/0.0409 | 0.0000 (-0.1685, 0.1685) | True |
| R61-T63-c4 | 0.6968 | 0.7683 | -0.0715 | 0 | 0 | 0.7861/0.0422 | 0.7863/0.0405 | 0.5000 (0.3865, 0.5000) | True |
| R61-T63-cnone | 0.7045 | 0.7542 | -0.0497 | 0 | 0 | 0.7874/0.0422 | 0.7852/0.0406 | 0.0000 (-0.1685, 0.1685) | True |

The unchanged frozen exact-outcome set is the forensic safety evaluation; no new critical correct-to-incorrect state relative to C1 is permitted. Safety limits were preclip norm <= `100` and loss <= `10`; every lane records NaN/infinity and spike status in JSON. Per-step norms, losses, anchor trajectories, clipped/unclipped summaries, and epoch Adam summaries are retained in the machine-readable artifact. No self-play, replay mutation, or promotion occurred.

## Classification

`grad_clip_relaxation_unstable`

Exactly one next experiment: keep clip=1.0 and audit Adam moment/parameter-subspace drift; do not tighten clipping further.
