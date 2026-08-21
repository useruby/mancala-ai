# PR #212 Isolated Policy Sublayer Training

**Classification:** `both_single_sublayers_unsafe`

**Recommended follow-up:** move to a parent-preserving additive policy adapter rather than further mutation of the existing head

## Training Metrics

| Lane | Step | CE(search) | CE(P1) | CE(beta095) | Improvement | Fit | L1 mean/p99/max | JS | Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| full_policy | 1 | 1.087671 | 0.917013 | 0.925546 | +0.000106 | 0.0225 | 0.000613/0.001589/0.001862 | 8.94e-08 | 0.0000 |
| full_policy | 4 | 1.087365 | 0.917016 | 0.925534 | +0.000412 | 0.0874 | 0.002029/0.005172/0.005866 | 9.60e-07 | 0.0020 |
| full_policy | 16 | 1.086442 | 0.917030 | 0.925501 | +0.001335 | 0.2830 | 0.004400/0.010867/0.013056 | 4.42e-06 | 0.0039 |
| full_policy | 46 | 1.085398 | 0.917041 | 0.925459 | +0.002379 | 0.5044 | 0.005438/0.014529/0.016301 | 7.26e-06 | 0.0039 |
| hidden_only | 1 | 1.087696 | 0.917013 | 0.925547 | +0.000081 | 0.0172 | 0.000478/0.001212/0.001463 | 5.43e-08 | 0.0000 |
| hidden_only | 4 | 1.087457 | 0.917015 | 0.925537 | +0.000320 | 0.0678 | 0.001528/0.003878/0.004573 | 5.51e-07 | 0.0020 |
| hidden_only | 16 | 1.086644 | 0.917026 | 0.925507 | +0.001133 | 0.2402 | 0.003873/0.009247/0.010808 | 3.37e-06 | 0.0020 |
| hidden_only | 46 | 1.085675 | 0.917038 | 0.925470 | +0.002102 | 0.4458 | 0.005105/0.013174/0.015394 | 6.30e-06 | 0.0039 |
| readout_only | 1 | 1.087752 | 0.917013 | 0.925549 | +0.000025 | 0.0053 | 0.000138/0.000379/0.000410 | 4.65e-09 | 0.0000 |
| readout_only | 4 | 1.087678 | 0.917013 | 0.925546 | +0.000099 | 0.0210 | 0.000572/0.001478/0.001779 | 7.51e-08 | 0.0000 |
| readout_only | 16 | 1.087398 | 0.917016 | 0.925535 | +0.000379 | 0.0804 | 0.001930/0.004971/0.006357 | 8.50e-07 | 0.0020 |
| readout_only | 46 | 1.086913 | 0.917025 | 0.925519 | +0.000864 | 0.1831 | 0.003680/0.009604/0.011140 | 3.09e-06 | 0.0020 |

## Arena Matrix

| Lane | Step | Budget | Effect | 95% CI | Seat A | Seat B | W/D/L | Safe | Recovery |
| --- | ---: | --- | ---: | --- | ---: | ---: | --- | ---: | ---: |
| full_policy | 16 | 384:256 | -0.0742 | [-0.1055, -0.0430] | -0.1484 | +0.0000 | 362/38/112 | False | - |
| full_policy | 16 | 1200:1200 | -0.0234 | [-0.0430, -0.0078] | -0.0391 | -0.0078 | 180/128/204 | False | - |
| full_policy | 46 | 384:256 | -0.0820 | [-0.1133, -0.0508] | -0.1641 | +0.0000 | 354/46/112 | False | - |
| full_policy | 46 | 1200:1200 | -0.0234 | [-0.0430, -0.0078] | -0.0391 | -0.0078 | 180/128/204 | False | - |
| hidden_only | 16 | 384:256 | -0.0195 | [-0.0391, -0.0039] | -0.0391 | +0.0000 | 390/38/84 | False | 73.7% |
| hidden_only | 16 | 1200:1200 | -0.0234 | [-0.0430, -0.0078] | -0.0391 | -0.0078 | 180/128/204 | False | - |
| hidden_only | 46 | 384:256 | -0.0820 | [-0.1133, -0.0508] | -0.1641 | +0.0000 | 354/46/112 | False | 0.0% |
| readout_only | 16 | 384:256 | -0.0176 | [-0.0293, -0.0078] | -0.0195 | -0.0156 | 382/56/74 | True | 76.3% |
| readout_only | 16 | 1200:1200 | -0.0195 | [-0.0391, -0.0039] | -0.0391 | +0.0000 | 180/132/200 | False | - |
| readout_only | 46 | 384:256 | -0.0195 | [-0.0391, -0.0039] | -0.0391 | +0.0000 | 390/38/84 | False | 76.2% |
| readout_only | 46 | 1200:1200 | -0.0195 | [-0.0391, -0.0039] | -0.0391 | +0.0000 | 180/132/200 | False | - |

## Trajectory Vs Graft

| Lane | Step | Param cosine | Norm ratio | Output-delta cosine | Independent-vs-graft L1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| hidden_only | 16 | 0.9939 | 1.0529 | 0.9931 | 0.000753 |
| hidden_only | 46 | 0.9953 | 1.0307 | 0.9893 | 0.000835 |
| readout_only | 16 | 0.9453 | 1.1999 | 0.9550 | 0.002751 |
| readout_only | 46 | 0.8676 | 1.2374 | 0.8282 | 0.003281 |

## Parameter Drift

| Lane | Step | Hidden | Readout | Trunk | Value stack |
| --- | ---: | ---: | ---: | ---: | ---: |
| full_policy | 16 | 7.417027e-03 | 1.840658e-03 | 0.00e+00 | 0.00e+00 |
| full_policy | 46 | 1.743933e-02 | 4.222616e-03 | 0.00e+00 | 0.00e+00 |
| hidden_only | 16 | 7.809535e-03 | 0.000000e+00 | 0.00e+00 | 0.00e+00 |
| hidden_only | 46 | 1.797394e-02 | 0.000000e+00 | 0.00e+00 | 0.00e+00 |
| readout_only | 16 | 0.000000e+00 | 2.208520e-03 | 0.00e+00 | 0.00e+00 |
| readout_only | 46 | 0.000000e+00 | 5.225224e-03 | 0.00e+00 | 0.00e+00 |

## Search Diagnostics

| Candidate | Budget | Move changes | Visit JS | Q-rank changes | Root-value delta | Visit-margin delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| hidden_only_16 | 384:256 | 0.0117 | 2.228847e-04 | -0.0273 | +0.000073 | +0.001485 |
| hidden_only_16 | 1200:1200 | 0.0117 | 3.484694e-04 | +0.0039 | +0.000266 | -0.003405 |
| hidden_only_46 | 384:256 | 0.0195 | 1.563466e-04 | -0.0273 | -0.000106 | +0.000315 |
| hidden_only_46 | 1200:1200 | 0.0117 | 6.014666e-04 | +0.0117 | -0.000008 | -0.003027 |
| readout_only_16 | 384:256 | 0.0039 | 1.549594e-04 | -0.0352 | +0.000097 | +0.002075 |
| readout_only_16 | 1200:1200 | 0.0000 | 2.553142e-04 | +0.0156 | -0.000101 | -0.003822 |
| readout_only_46 | 384:256 | 0.0117 | 2.207675e-04 | -0.0156 | +0.000003 | +0.000977 |
| readout_only_46 | 1200:1200 | 0.0039 | 2.988239e-04 | +0.0000 | +0.000023 | -0.003021 |

The JSON summary retains the full and graft diagnostic matrices, including the depth-0 legal-policy L1/JS values and independent-minus-graft diagnostic differences.

## Invariants

- Full-policy reproduction: `true`; its step-16 and step-46 hashes match PR #212 exactly.
- Hidden and readout first-batch gradients match full-policy exactly (maximum absolute difference `0.0`).
- Trunk and value gradients were absent; frozen families are byte-identical to P1.
- No isolated checkpoint was both fit-eligible and safe against P1 at both budgets, so no P0 cumulative gate was eligible or run.
