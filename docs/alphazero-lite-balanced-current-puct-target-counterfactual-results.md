# AlphaZero-Lite Balanced Current PUCT Target Counterfactual Results

**Date**: 2026-06-25

**Classification**: `puct_targets_good_but_training_harmful`

PUCT-forced continuations are better than raw-forced, so the regression likely comes from the update format rather than the targets.

## Inputs

| Field | Value |
|---|---|
| Current artifact weights SHA256 | 8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a |
| Consensus mining profile SHA256 | d6957cb560782a2d8cf645d5cb04bc24e5daf1ae8b0460eaebf3b3f08f939ac7 |
| Continuation search profile SHA256 | b9e50f4c78b621a35b1fa96c9a4ca5fa9f286c62a0d6b6d20b434afb34972cae |

## Sampled State Counts

| Bucket | Count |
|---|---|
| Valid disagreement candidates | 27697 |
| Sampled disagreement states | 1024 |
| Valid stability candidates | 42447 |
| Sampled stability states | 512 |

## Disagreement-State Stratification

| Group | Count |
|---|---|
| fixed_large | 328 |
| heldout | 696 |

| Seat Context | Count |
|---|---|
| challenger | 387 |
| current | 325 |
| mixed | 312 |

| Phase | Count |
|---|---|
| late | 334 |
| mid | 504 |
| opening | 186 |

| Raw Margin Bucket | Count |
|---|---|
| 0.02 <= margin < 0.05 | 223 |
| 0.05 <= margin < 0.10 | 260 |
| margin < 0.02 | 197 |
| margin >= 0.10 | 344 |

| PUCT Visit Share Bucket | Count |
|---|---|
| 0.55 <= share < 0.65 | 245 |
| 0.65 <= share < 0.75 | 228 |
| 0.75 <= share < 0.85 | 271 |
| share >= 0.85 | 280 |

| PUCT Top Move | Count |
|---|---|
| 0 | 171 |
| 1 | 162 |
| 2 | 148 |
| 3 | 178 |
| 4 | 177 |
| 5 | 188 |

| Poor 384:256 P0 Context | Count |
|---|---|
| False | 792 |
| True | 232 |

## PUCT-Forced Versus Raw-Forced

| Budget | N | Wins | Ties | Losses | Mean | Lower 95% | Upper 95% |
|---|---|---|---|---|---|---|---|
| 384 | 1024 | 367 | 590 | 67 | +0.4541 | +0.3955 | +0.5127 |
| 1200 | 1024 | 344 | 617 | 63 | +0.4092 | +0.3535 | +0.4648 |

Primary-budget mean outcome delta (`PUCT - raw`) at `1200`: +0.4092

## Outcome Delta By Phase

| Phase | N | Wins | Ties | Losses | Mean | Lower 95% | Upper 95% |
|---|---|---|---|---|---|---|---|
| late | 334 | 91 | 242 | 1 | +0.3563 | +0.2904 | +0.4251 |
| mid | 504 | 186 | 289 | 29 | +0.4921 | +0.4107 | +0.5734 |
| opening | 186 | 67 | 86 | 33 | +0.2796 | +0.1075 | +0.4516 |

## Outcome Delta By Seat Context

| Seat Context | N | Wins | Ties | Losses | Mean | Lower 95% | Upper 95% |
|---|---|---|---|---|---|---|---|
| challenger | 387 | 132 | 230 | 25 | +0.4031 | +0.3127 | +0.4935 |
| current | 325 | 104 | 207 | 14 | +0.4215 | +0.3292 | +0.5138 |
| mixed | 312 | 108 | 180 | 24 | +0.4038 | +0.2981 | +0.5096 |

## Outcome Delta By Raw Margin Bucket

| Raw Margin Bucket | N | Wins | Ties | Losses | Mean | Lower 95% | Upper 95% |
|---|---|---|---|---|---|---|---|
| 0.02 <= margin < 0.05 | 223 | 75 | 133 | 15 | +0.3677 | +0.2556 | +0.4798 |
| 0.05 <= margin < 0.10 | 260 | 95 | 150 | 15 | +0.4500 | +0.3385 | +0.5615 |
| margin < 0.02 | 197 | 65 | 123 | 9 | +0.4924 | +0.3604 | +0.6244 |
| margin >= 0.10 | 344 | 109 | 211 | 24 | +0.3576 | +0.2587 | +0.4564 |

## Outcome Delta By PUCT Visit-Share Bucket

| Visit Share Bucket | N | Wins | Ties | Losses | Mean | Lower 95% | Upper 95% |
|---|---|---|---|---|---|---|---|
| 0.55 <= share < 0.65 | 245 | 49 | 178 | 18 | +0.1878 | +0.0857 | +0.2939 |
| 0.65 <= share < 0.75 | 228 | 71 | 135 | 22 | +0.3333 | +0.2061 | +0.4561 |
| 0.75 <= share < 0.85 | 271 | 101 | 153 | 17 | +0.4502 | +0.3432 | +0.5609 |
| share >= 0.85 | 280 | 123 | 151 | 6 | +0.6250 | +0.5250 | +0.7250 |

## Outcome Delta By 384:256 P0 Context

| Poor 384:256 P0 | N | Wins | Ties | Losses | Mean | Lower 95% | Upper 95% |
|---|---|---|---|---|---|---|---|
| False | 792 | 252 | 492 | 48 | +0.3902 | +0.3245 | +0.4533 |
| True | 232 | 92 | 125 | 15 | +0.4741 | +0.3621 | +0.5905 |

## Outcome Delta By PUCT Top Move

| Top Move | N | Wins | Ties | Losses | Mean | Lower 95% | Upper 95% |
|---|---|---|---|---|---|---|---|
| 0 | 171 | 63 | 103 | 5 | +0.5322 | +0.4035 | +0.6667 |
| 1 | 162 | 49 | 102 | 11 | +0.3148 | +0.1852 | +0.4444 |
| 2 | 148 | 34 | 106 | 8 | +0.2568 | +0.1216 | +0.3919 |
| 3 | 178 | 48 | 116 | 14 | +0.3202 | +0.1798 | +0.4607 |
| 4 | 177 | 69 | 96 | 12 | +0.4576 | +0.3277 | +0.5932 |
| 5 | 188 | 81 | 94 | 13 | +0.5372 | +0.3989 | +0.6755 |

## Continuation Budget Comparison

| Budget | N | Wins | Ties | Losses | Mean | Lower 95% | Upper 95% |
|---|---|---|---|---|---|---|---|
| 384 | 1024 | 367 | 590 | 67 | +0.4541 | +0.3955 | +0.5127 |
| 1200 | 1024 | 344 | 617 | 63 | +0.4092 | +0.3535 | +0.4648 |

Primary-minus-lower continuation mean delta change: -0.0449 [-0.1055, +0.0166]

States with any continuation-budget result difference: 352

## Stability-Control Sanity

| Budget | N | Wins | Ties | Losses | Mean | Lower 95% | Upper 95% |
|---|---|---|---|---|---|---|---|
| 384 | 512 | 139 | 316 | 57 | +0.2734 | +0.1875 | +0.3594 |
| 1200 | 512 | 124 | 348 | 40 | +0.2461 | +0.1680 | +0.3223 |

Primary-budget shared-top1 control mean delta (`shared - alternative`) at `1200`: +0.2461

## Largest Harmful PUCT Disagreements

| State Hash | Suite | Phase | Seat | Poor P0 | Legal | Raw Policy | PUCT Visits 1200 | Raw | PUCT | Raw Margin | PUCT Share | Raw Outcome | PUCT Outcome | Raw Margin Final | PUCT Margin Final | Delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 36ca85b6219cd306a8d8b6a77caa4b31bfd2f4a448aa168a869d604d201da58a | heldout_seed43_large | mid | challenger | False | 1,2,3,4,5 | 0.000,0.034,0.232,0.306,0.245,0.183 | 1:320,2:661,3:93,4:101,5:25 | 3 | 2 | +0.0601 | +0.5508 | +1.0000 | -1.0000 | 8 | -10 | -2.0000 |
| 6d2aecec3f87f13be8576c240b2ae9ea041d8e99e3a319a5988cfd49fda9a412 | heldout_seed43_large | mid | mixed | False | 1,2,4,5 | 0.000,0.367,0.123,0.000,0.120,0.390 | 1:672,2:83,4:71,5:374 | 5 | 1 | +0.0224 | +0.5600 | +1.0000 | -1.0000 | 4 | -2 | -2.0000 |
| 3ed6636bb91135751122b899b284798ba6a3d363020198c3812b07c654a59dbd | fixed_large | mid | challenger | True | 0,1,3,4,5 | 0.074,0.399,0.000,0.405,0.066,0.056 | 0:39,1:676,3:117,4:134,5:234 | 3 | 1 | +0.0058 | +0.5633 | +1.0000 | -1.0000 | 8 | -2 | -2.0000 |
| ee2676f3b4bee3c50bf7988a8f4846a514990a9ecad771814787a8dbab6b4ea8 | fixed_large | mid | challenger | False | 0,1,3,5 | 0.069,0.284,0.000,0.289,0.000,0.358 | 0:48,1:677,3:250,5:225 | 5 | 1 | +0.0682 | +0.5642 | +1.0000 | -1.0000 | 6 | -4 | -2.0000 |
| bd7ac48d9680694a08df09a8c3a0117d2eff756e7e590fb9f91c77e26d7d66e4 | heldout_seed48_large | opening | current | False | 0,1,2,3,5 | 0.024,0.044,0.499,0.418,0.000,0.016 | 0:21,1:16,2:467,3:689,5:7 | 2 | 3 | +0.0811 | +0.5742 | +1.0000 | -1.0000 | 2 | -4 | -2.0000 |
| b0f1a2322579007a22122ad6c755901d7e5c8cf392686372dfa7ba41d10e6e8c | fixed_large | opening | mixed | True | 0,1,2,4 | 0.494,0.296,0.157,0.000,0.054,0.000 | 0:165,1:696,2:321,4:18 | 0 | 1 | +0.1984 | +0.5800 | +1.0000 | -1.0000 | 6 | -2 | -2.0000 |
| c8c8953121324ed35067f3ef8db3cdc16fb5f0661fd359d0b5f87348078e5731 | heldout_seed43_large | opening | challenger | False | 2,4,5 | 0.000,0.000,0.393,0.000,0.258,0.349 | 2:316,4:698,5:186 | 2 | 4 | +0.0438 | +0.5817 | +1.0000 | -1.0000 | 8 | -4 | -2.0000 |
| ea0e1b5c12d89080c3541d3724136e80928a3ae98c59c46b7376f8e09c562d81 | heldout_seed43_large | opening | challenger | False | 1,2,3,4,5 | 0.000,0.151,0.077,0.497,0.257,0.018 | 1:184,2:707,3:221,4:84,5:4 | 3 | 2 | +0.2399 | +0.5892 | +1.0000 | -1.0000 | 8 | -6 | -2.0000 |
| 3fb2e9774b1c57e4beb66355617e6c11878427330dd829ff4e03966679d539a8 | heldout_seed47_large | opening | mixed | False | 0,1,2,3,5 | 0.016,0.359,0.344,0.070,0.000,0.211 | 0:4,1:227,2:748,3:102,5:119 | 1 | 2 | +0.0147 | +0.6233 | +1.0000 | -1.0000 | 12 | -6 | -2.0000 |
| bebb3aadd9fa74d1c1f51f744397ea6713e886f64710b43e063d65d0233c361d | heldout_seed46_large | opening | challenger | False | 0,1,2,3,4 | 0.072,0.298,0.060,0.277,0.293,0.000 | 0:15,1:105,2:101,3:228,4:751 | 1 | 4 | +0.0050 | +0.6258 | +1.0000 | -1.0000 | 6 | -2 | -2.0000 |
| d2e4d9473936e24b6547ff5533b45778ebda93800214cab4e329456409416e14 | fixed_large | opening | challenger | False | 0,1,2,3,4,5 | 0.071,0.348,0.207,0.233,0.050,0.090 | 0:23,1:205,2:89,3:753,4:73,5:57 | 1 | 3 | +0.1151 | +0.6275 | +1.0000 | -1.0000 | 2 | -8 | -2.0000 |
| 69c1a5f4b2a9129f3706e8ada5dccdc2b7e4e4fc68d2c2a02eaa237dfc7cdaeb | heldout_seed45_large | opening | mixed | False | 0,2,3,4,5 | 0.021,0.000,0.447,0.303,0.133,0.096 | 0:13,2:229,3:794,4:142,5:22 | 2 | 3 | +0.1448 | +0.6617 | +1.0000 | -1.0000 | 4 | -6 | -2.0000 |
| 2922571a0c8307c815fb2453a665a6539b886440b4d99ee510b92d13d98d264d | heldout_seed46_large | opening | mixed | False | 0,1,2,4,5 | 0.209,0.127,0.540,0.000,0.122,0.003 | 0:68,1:829,2:247,4:55,5:1 | 2 | 1 | +0.3312 | +0.6908 | +1.0000 | -1.0000 | 16 | -6 | -2.0000 |
| e1b066b3c915bac10f5288dfdc21ba02efe91a6f1810b7f134e44000f32c4b53 | fixed_large | opening | mixed | False | 0,1,2,3,5 | 0.121,0.295,0.272,0.034,0.000,0.278 | 0:41,1:212,2:103,3:10,5:834 | 1 | 5 | +0.0171 | +0.6950 | +1.0000 | -1.0000 | 6 | -6 | -2.0000 |
| df3658a0cb53c9ff0c68e0e42e2bac1315fa42c4610dce0bbb83e2c1e5bdece0 | heldout_seed46_large | opening | challenger | False | 0,1,2,3,4,5 | 0.042,0.022,0.171,0.390,0.248,0.127 | 0:7,1:6,2:842,3:243,4:68,5:34 | 3 | 2 | +0.1422 | +0.7017 | +1.0000 | -1.0000 | 12 | -6 | -2.0000 |
| 0739967446f13f253352551fb3b76e9f91c2255f193dbcbdf71b676ff492ef8b | heldout_seed43_large | opening | challenger | False | 0,1,3,4,5 | 0.258,0.107,0.000,0.111,0.486,0.038 | 0:57,1:57,3:845,4:222,5:19 | 4 | 3 | +0.2273 | +0.7042 | +1.0000 | -1.0000 | 10 | -16 | -2.0000 |
| 9aef6ccb7b56f4d662764179a33338ddc41469e768c224ec728e5cd03544f52f | heldout_seed48_large | mid | current | False | 0,1,2,3,5 | 0.040,0.166,0.178,0.324,0.000,0.292 | 0:55,1:45,2:102,3:142,5:856 | 3 | 5 | +0.0315 | +0.7133 | +1.0000 | -1.0000 | 2 | -4 | -2.0000 |
| ee6646b2e94af2c7b1ce60f8ecf7356ff5f576fd99c3b27fe91680d68a2b3a6f | fixed_large | opening | mixed | True | 0,1,2,3,5 | 0.676,0.143,0.062,0.047,0.000,0.072 | 0:227,1:74,2:18,3:862,5:19 | 0 | 3 | +0.5335 | +0.7183 | +1.0000 | -1.0000 | 8 | -6 | -2.0000 |
| da5aabb037c761d1d878b76ebc1323dab59b7343fddeec4ab1ff4d7372343c36 | heldout_seed45_large | mid | challenger | False | 1,3,5 | 0.000,0.333,0.000,0.268,0.000,0.399 | 1:248,3:864,5:88 | 5 | 3 | +0.0663 | +0.7200 | +1.0000 | -1.0000 | 6 | -4 | -2.0000 |
| 9baade714245cf92e3846665f278ad37804ed5fbf2997989684c948cce225ce5 | heldout_seed46_large | opening | current | False | 0,1,2,3,5 | 0.073,0.106,0.133,0.125,0.000,0.563 | 0:34,1:36,2:97,3:873,5:160 | 5 | 3 | +0.4300 | +0.7275 | +1.0000 | -1.0000 | 10 | -8 | -2.0000 |
| fcd0a86f5fc258431ece53d494d2859ed51460dfa5c1cbbe7ecfd6489aa46e81 | fixed_large | opening | mixed | True | 1,2,3,4,5 | 0.000,0.178,0.045,0.455,0.201,0.121 | 1:61,2:14,3:173,4:67,5:885 | 3 | 5 | +0.2547 | +0.7375 | +1.0000 | -1.0000 | 4 | -2 | -2.0000 |
| e4196998cc44c29f4b0ef5af41b3dc7859111c80119c8678422a6e8ec29ef148 | heldout_seed45_large | mid | mixed | False | 0,1,3,4,5 | 0.291,0.030,0.000,0.183,0.332,0.164 | 0:894,1:3,3:56,4:212,5:35 | 4 | 0 | +0.0416 | +0.7450 | +1.0000 | -1.0000 | 2 | -8 | -2.0000 |
| d14b9691a220aad1bb3ac78fb109a47749f7f13745875261807ed962f40ebf10 | heldout_seed45_large | mid | current | False | 0,1,2,3,4,5 | 0.018,0.086,0.158,0.514,0.180,0.044 | 0:6,1:89,2:917,3:100,4:69,5:19 | 3 | 2 | +0.3334 | +0.7642 | +1.0000 | -1.0000 | 6 | -6 | -2.0000 |
| 9ec4b251f0c4b09571dfaa119f62c8edb707916d81771da44671f2a4a1240084 | fixed_large | mid | current | False | 1,2,3,4 | 0.000,0.039,0.348,0.366,0.247,0.000 | 1:8,2:105,3:158,4:929 | 3 | 4 | +0.0174 | +0.7742 | +1.0000 | -1.0000 | 10 | -4 | -2.0000 |
| a752b384fdd8183061cffb013eba557121eabeade2e63c55eaead4fa0fdf936e | heldout_seed47_large | opening | current | False | 0,1,2,3,4,5 | 0.037,0.008,0.471,0.062,0.086,0.336 | 0:9,1:5,2:178,3:37,4:32,5:939 | 2 | 5 | +0.1346 | +0.7825 | +1.0000 | -1.0000 | 10 | -12 | -2.0000 |

## Largest Helpful PUCT Disagreements

| State Hash | Suite | Phase | Seat | Poor P0 | Legal | Raw Policy | PUCT Visits 1200 | Raw | PUCT | Raw Margin | PUCT Share | Raw Outcome | PUCT Outcome | Raw Margin Final | PUCT Margin Final | Delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 17d895bb510e77b4f594acc02bb4bb80a9563942e71be05bec0f210dc598b79c | heldout_seed43_large | late | challenger | False | 2,5 | 0.000,0.000,0.489,0.000,0.000,0.511 | 2:1188,5:12 | 5 | 2 | +0.0219 | +0.9900 | -1.0000 | +1.0000 | -12 | 2 | +2.0000 |
| dd124fbc698da9bf6669e7ece24779eb8ea9f388edbc32045206f98a60c1f950 | heldout_seed47_large | late | current | False | 1,2,3,5 | 0.000,0.381,0.324,0.152,0.000,0.143 | 1:10,2:8,3:1178,5:4 | 1 | 3 | +0.0563 | +0.9817 | -1.0000 | +1.0000 | -6 | 2 | +2.0000 |
| 9cec884ab6762affb51959fb056a8aba651a1ffc447237ed0026fb2ccbba7b3a | heldout_seed47_large | mid | mixed | False | 1,2,3,4 | 0.000,0.310,0.124,0.253,0.314,0.000 | 1:9,2:5,3:1174,4:12 | 4 | 3 | +0.0036 | +0.9783 | -1.0000 | +1.0000 | -2 | 10 | +2.0000 |
| b03f2c74ba2e6e541d614c61c3f375a38e075023fa27dfbb7adc4453da0739b1 | fixed_large | late | mixed | True | 1,2,3 | 0.000,0.386,0.366,0.248,0.000,0.000 | 1:15,2:1173,3:12 | 1 | 2 | +0.0206 | +0.9775 | -1.0000 | +1.0000 | -4 | 6 | +2.0000 |
| 28aaef763f2597ac4af542cc3373474e11abea56dc41d2069006170bad7941b9 | heldout_seed43_large | mid | challenger | False | 0,1,2,4,5 | 0.307,0.085,0.144,0.000,0.104,0.361 | 0:1173,1:3,2:6,4:5,5:13 | 5 | 0 | +0.0533 | +0.9775 | -1.0000 | +1.0000 | -12 | 20 | +2.0000 |
| 34925e620546eeb651efff72811bea277da985331026c8209d4ddc92f0b389d4 | heldout_seed47_large | late | challenger | False | 2,5 | 0.000,0.000,0.797,0.000,0.000,0.203 | 2:29,5:1171 | 2 | 5 | +0.5934 | +0.9758 | -1.0000 | +1.0000 | -2 | 2 | +2.0000 |
| ad31c93efeeeed169417ce5d8449f16f09c2898458b94155b2572c16fa8cdab4 | heldout_seed46_large | mid | challenger | False | 0,1,3 | 0.140,0.437,0.000,0.423,0.000,0.000 | 0:10,1:20,3:1170 | 1 | 3 | +0.0139 | +0.9750 | -1.0000 | +1.0000 | -12 | 8 | +2.0000 |
| f8c8b3595f6620d19db4482910f61b95dd39605628e0f2231a5e67b6e87b984c | heldout_seed47_large | opening | mixed | False | 0,2,3,5 | 0.233,0.000,0.404,0.322,0.000,0.041 | 0:1169,2:15,3:15,5:1 | 2 | 0 | +0.0816 | +0.9742 | -1.0000 | +1.0000 | -18 | 22 | +2.0000 |
| 96539c3d74c123682e3a9e78f5b46219c6a0fdf9f7993bc3a4eadf257abd0eb0 | heldout_seed45_large | late | challenger | False | 1,2,3,4 | 0.000,0.122,0.391,0.341,0.147,0.000 | 1:4,2:17,3:11,4:1168 | 2 | 4 | +0.0501 | +0.9733 | -1.0000 | +1.0000 | -2 | 2 | +2.0000 |
| 0b3c240e89b17c41ec53672cf46930fb0b0433a3185bf01a9a82ca66181165ba | fixed_large | late | current | False | 1,2,3,4,5 | 0.000,0.412,0.064,0.128,0.104,0.291 | 1:11,2:5,3:1166,4:4,5:14 | 1 | 3 | +0.1209 | +0.9717 | -1.0000 | +1.0000 | -4 | 8 | +2.0000 |
| 9c2d443fd9ea97c360db603ed804de1ef78402877f7f95e2e4f30882c167f280 | heldout_seed44_large | mid | challenger | False | 0,1,2,3,5 | 0.264,0.153,0.281,0.103,0.000,0.199 | 0:1165,1:4,2:12,3:9,5:10 | 2 | 0 | +0.0178 | +0.9708 | -1.0000 | +1.0000 | -8 | 10 | +2.0000 |
| 13f63ad968773c6afde426036355006da2f6ffbeae02008d344c3b8c7d78f2d0 | fixed_large | mid | mixed | True | 0,1,2,3,4 | 0.233,0.085,0.289,0.215,0.177,0.000 | 0:12,1:5,2:12,3:1164,4:7 | 2 | 3 | +0.0557 | +0.9700 | -1.0000 | +1.0000 | -6 | 24 | +2.0000 |
| 004115d1921873b51c63506b613ad7e3883b339219c8c91351ec1c7dabdf41aa | heldout_seed44_large | mid | challenger | False | 0,1,2,4,5 | 0.189,0.271,0.369,0.000,0.077,0.095 | 0:16,1:1156,2:16,4:8,5:4 | 2 | 1 | +0.0981 | +0.9633 | -1.0000 | +1.0000 | -2 | 12 | +2.0000 |
| a218d4d9dc7b6166b6b0356f06266300bf33e7260f83ddf3bc682128078ba5e5 | heldout_seed48_large | mid | current | False | 0,1,2,3,5 | 0.068,0.424,0.150,0.214,0.000,0.144 | 0:4,1:19,2:7,3:1155,5:15 | 1 | 3 | +0.2096 | +0.9625 | -1.0000 | +1.0000 | -2 | 6 | +2.0000 |
| a09f26250b2580a0de244b6a708cb101fce20c246e86504629f7d640a74507d2 | heldout_seed46_large | mid | challenger | False | 2,3,5 | 0.000,0.000,0.175,0.252,0.000,0.572 | 2:22,3:1149,5:29 | 5 | 3 | +0.3196 | +0.9575 | -1.0000 | +1.0000 | -4 | 6 | +2.0000 |
| 9c4ea2d49f9c1d3c4a62619ba0b57e8a31fd898583f4eaede10a1237a83e7124 | heldout_seed45_large | mid | current | False | 0,2,3,4,5 | 0.073,0.000,0.292,0.288,0.164,0.182 | 0:6,2:16,3:21,4:16,5:1141 | 2 | 5 | +0.0041 | +0.9508 | -1.0000 | +1.0000 | -6 | 18 | +2.0000 |
| 2dc3ce489218e9a143344f4576a2b614c6387884e136066e7de0715f782895bf | heldout_seed47_large | mid | mixed | False | 0,1,2,4 | 0.267,0.264,0.276,0.000,0.193,0.000 | 0:29,1:14,2:16,4:1141 | 2 | 4 | +0.0093 | +0.9508 | -1.0000 | +1.0000 | -4 | 20 | +2.0000 |
| 3d87f18eb784b0b284c8b59af749ea63161e1f0a020c3d5ee753672cd411e8bd | fixed_large | mid | challenger | False | 0,1,3,4,5 | 0.095,0.333,0.000,0.334,0.047,0.190 | 0:7,1:1139,3:14,4:3,5:37 | 3 | 1 | +0.0013 | +0.9492 | -1.0000 | +1.0000 | -4 | 4 | +2.0000 |
| 3691a26360e71d70d7266394f722d6e3eb2ee6ef7b5d16c99689ce653b91f09e | heldout_seed48_large | mid | challenger | False | 0,2,3,5 | 0.129,0.000,0.124,0.406,0.000,0.342 | 0:20,2:8,3:33,5:1139 | 3 | 5 | +0.0635 | +0.9492 | -1.0000 | +1.0000 | -4 | 6 | +2.0000 |
| 5190655d804b5aa2e6626a6899c2b14fdd4b8f26046fece103067e4390cdcbd9 | heldout_seed48_large | late | challenger | False | 0,1,3,4,5 | 0.082,0.148,0.000,0.284,0.313,0.174 | 0:1138,1:21,3:14,4:21,5:6 | 4 | 0 | +0.0288 | +0.9483 | -1.0000 | +1.0000 | -2 | 6 | +2.0000 |
| a4bd330e08e60878d90e245ec12419eb75dfa8cdca44ddb6593abddd3d7b23e4 | fixed_large | mid | current | False | 0,2,5 | 0.256,0.000,0.317,0.000,0.000,0.427 | 0:26,2:1136,5:38 | 5 | 2 | +0.1104 | +0.9467 | -1.0000 | +1.0000 | -4 | 4 | +2.0000 |
| 5848f85bd8a6d4c850cb685b94fbfcddf217de80146365ad2b52e4fb8a52a9e5 | fixed_large | opening | mixed | True | 0,1,2,3,4,5 | 0.035,0.137,0.369,0.098,0.184,0.178 | 0:1,1:19,2:21,3:1136,4:9,5:14 | 2 | 3 | +0.1850 | +0.9467 | -1.0000 | +1.0000 | -16 | 6 | +2.0000 |
| dcfe4fa91cf59d0ddb3397c84ec895529d5d21d25c82c684785d1af48c77f90c | heldout_seed47_large | late | current | False | 0,2,4 | 0.159,0.000,0.408,0.000,0.433,0.000 | 0:29,2:1128,4:43 | 4 | 2 | +0.0251 | +0.9400 | -1.0000 | +1.0000 | -4 | 6 | +2.0000 |
| f1467676abfbc0c42a84369c345632eb462b1f251884aa057249c0c90a274be0 | heldout_seed45_large | late | current | False | 0,3 | 0.509,0.000,0.000,0.491,0.000,0.000 | 0:73,3:1127 | 0 | 3 | +0.0173 | +0.9392 | -1.0000 | +1.0000 | -2 | 2 | +2.0000 |
| 2d4456b2376af9928e1da2a543e242166849b3975647515ed5cc32c08b16ffed | fixed_large | late | current | True | 1,2,3 | 0.000,0.637,0.226,0.137,0.000,0.000 | 1:42,2:1124,3:34 | 1 | 2 | +0.4118 | +0.9367 | -1.0000 | +1.0000 | -4 | 4 | +2.0000 |

