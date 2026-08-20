# PR #212 Policy Sublayer Causal Decomposition

**Classification:** `joint_policy_sublayer_interaction`

**Next experiment:** train one sublayer at a time, beginning with whichever isolated graft has better teacher fit/search behavior

## Exact Reconstruction

| Model | State hash |
| --- | --- |
| P1 | `a86acb54b97c860289530fcb7ca64194724d43667580a52e2948359bfa3ebdf4` |
| C16 | `933ed483da1db3a8d17c45e9ace65ae88dbb6b13c22c18de3d13f36863fc7ca1` |
| C46 | `58bd2ca08a4329573432313384e1a2915e61f7de6304f9e9b08e36689a0bf66f` |

All C16/C46 trunk and value tensors are byte-identical to P1; only the four policy-hidden/readout tensors differ. Full grafts are state-identical to their source candidates.

## Parameter Deltas

| Family | Step | L2 | Relative | Weight L2 | Bias L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| hidden | 16 | 7.417027e-03 | 7.332092e-04 | 7.376149e-03 | 7.776329e-04 |
| hidden | 46 | 1.743933e-02 | 1.723962e-03 | 1.736516e-02 | 1.606614e-03 |
| hidden alignment | 16 vs 46 | cosine 0.9074 | step-46 fraction 0.3859 | | |
| readout | 16 | 1.840658e-03 | 4.847543e-04 | 1.828669e-03 | 2.097413e-04 |
| readout | 46 | 4.222616e-03 | 1.112065e-03 | 4.207091e-03 | 3.617647e-04 |
| readout alignment | 16 vs 46 | cosine 0.8889 | step-46 fraction 0.3875 | | |

## Paired Arenas

| Step | Graft | Budget | Effect | 95% CI | Seat A | Seat B | W/D/L | Recovery |
| ---: | --- | --- | ---: | --- | ---: | ---: | --- | ---: |
| 16 | full | 384:256 | -0.0742 | [-0.1055, -0.0430] | -0.1484 | +0.0000 | 362/38/112 | - |
| 16 | full | 1200:1200 | -0.0234 | [-0.0430, -0.0078] | -0.0391 | -0.0078 | 180/128/204 | - |
| 16 | hidden | 384:256 | -0.0195 | [-0.0391, -0.0039] | -0.0391 | +0.0000 | 390/38/84 | 73.7% |
| 16 | hidden | 1200:1200 | +0.0410 | [+0.0137, +0.0664] | -0.0391 | +0.1211 | 242/70/200 | - |
| 16 | readout | 384:256 | -0.0176 | [-0.0293, -0.0078] | -0.0195 | -0.0156 | 382/56/74 | 76.3% |
| 16 | readout | 1200:1200 | -0.0195 | [-0.0391, -0.0039] | -0.0391 | +0.0000 | 180/132/200 | - |
| 46 | full | 384:256 | -0.0820 | [-0.1133, -0.0508] | -0.1641 | +0.0000 | 354/46/112 | - |
| 46 | full | 1200:1200 | -0.0234 | [-0.0430, -0.0078] | -0.0391 | -0.0078 | 180/128/204 | - |
| 46 | hidden | 384:256 | -0.0273 | [-0.0469, -0.0117] | -0.0547 | +0.0000 | 382/46/84 | 66.7% |
| 46 | readout | 384:256 | -0.0215 | [-0.0352, -0.0098] | -0.0195 | -0.0234 | 378/60/74 | 73.8% |
| 46 | readout | 1200:1200 | -0.0195 | [-0.0391, -0.0039] | -0.0391 | +0.0000 | 180/132/200 | - |

## Network Decomposition

| Probe | Candidate | Mean L1 | P99 L1 | Mean JS | Top-1 disagreement | Max action change |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fresh_replay | full_16 | 0.004701 | 0.010833 | 4.839876e-06 | 0.0043 | 0.008161 |
| fresh_replay | hidden_16 | 0.003503 | 0.008001 | 2.657107e-06 | 0.0033 | 0.005693 |
| fresh_replay | readout_16 | 0.001316 | 0.004106 | 4.065839e-07 | 0.0010 | 0.002820 |
| fresh_replay | full_46 | 0.005565 | 0.014585 | 7.381671e-06 | 0.0044 | 0.011571 |
| fresh_replay | hidden_46 | 0.004540 | 0.012012 | 4.912761e-06 | 0.0041 | 0.009743 |
| fresh_replay | readout_46 | 0.001579 | 0.003921 | 5.803634e-07 | 0.0011 | 0.003289 |
| deterministic_puct_roots | full_16 | 0.004602 | 0.011083 | 4.643868e-06 | 0.0039 | 0.006528 |
| deterministic_puct_roots | hidden_16 | 0.003421 | 0.007850 | 2.550714e-06 | 0.0039 | 0.004588 |
| deterministic_puct_roots | readout_16 | 0.001281 | 0.003635 | 3.844372e-07 | 0.0039 | 0.001948 |
| deterministic_puct_roots | full_46 | 0.005588 | 0.014791 | 7.415387e-06 | 0.0039 | 0.008151 |
| deterministic_puct_roots | hidden_46 | 0.004549 | 0.011323 | 4.865821e-06 | 0.0039 | 0.006863 |
| deterministic_puct_roots | readout_46 | 0.001523 | 0.004128 | 5.603285e-07 | 0.0078 | 0.002381 |
| canonical_opening_roots | full_16 | 0.006160 | 0.010229 | 6.990409e-06 | 0.0000 | 0.006017 |
| canonical_opening_roots | hidden_16 | 0.004357 | 0.007841 | 3.605047e-06 | 0.0000 | 0.004297 |
| canonical_opening_roots | readout_16 | 0.001884 | 0.003408 | 6.718013e-07 | 0.0000 | 0.001730 |
| canonical_opening_roots | full_46 | 0.005974 | 0.012162 | 8.267367e-06 | 0.0000 | 0.006081 |
| canonical_opening_roots | hidden_46 | 0.004704 | 0.009791 | 5.203610e-06 | 0.0000 | 0.005391 |
| canonical_opening_roots | readout_46 | 0.002040 | 0.003842 | 8.057420e-07 | 0.0000 | 0.002534 |

The JSON summary retains the requested player, legal-move-count, and ply-bucket splits and full-vs-additive output alignment.

## Search Decomposition

| Candidate | Budget | Move changes | Visit JS | Q-rank change | Root-value delta | Visit-margin delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| full_16 | 384:256 | 0.0078 | 2.374258e-04 | -0.0273 | -0.000032 | +0.001333 |
| full_16 | 1200:1200 | 0.0117 | 3.604692e-04 | +0.0078 | +0.000220 | -0.003861 |
| hidden_16 | 384:256 | 0.0156 | 1.683500e-04 | -0.0156 | +0.000090 | +0.000559 |
| hidden_16 | 1200:1200 | 0.0078 | 2.953455e-04 | +0.0078 | +0.000166 | -0.003337 |
| readout_16 | 384:256 | 0.0078 | 1.141301e-04 | -0.0273 | +0.000037 | +0.001099 |
| readout_16 | 1200:1200 | 0.0039 | 1.958686e-04 | -0.0039 | -0.000120 | -0.002135 |
| full_46 | 384:256 | 0.0156 | 1.796287e-04 | -0.0312 | -0.000117 | +0.000315 |
| full_46 | 1200:1200 | 0.0117 | 5.068587e-04 | -0.0039 | -0.000044 | -0.003717 |
| hidden_46 | 384:256 | 0.0156 | 1.494477e-04 | -0.0312 | -0.000265 | +0.000722 |
| hidden_46 | 1200:1200 | 0.0117 | 4.621935e-04 | -0.0156 | +0.000174 | -0.004469 |
| readout_46 | 384:256 | 0.0078 | 3.151469e-05 | -0.0352 | +0.000043 | +0.000498 |
| readout_46 | 1200:1200 | 0.0039 | 2.058307e-04 | -0.0039 | -0.000210 | -0.001270 |

The deterministic PUCT probe evaluates root states, so per-depth policy reporting is depth 0; the corresponding L1/JS is retained in JSON.

## Caveat

A graft is a causal parameter intervention, not proof that independently training a sublayer follows the same trajectory. No trainable scope is changed here.
