# AlphaZero-Lite Balanced Current PUCT Consensus Filter Results

**Date**: 2026-06-25

**Classification**: `consensus_targets_still_harmful`

## Inputs

- Current artifact weights SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- Current source checkpoint SHA256: `18dacb5ef38602a77abf3927dc8748ac167c2ca91237385ec010ab27a123adc9`
- Search-profile SHA256: `d6957cb560782a2d8cf645d5cb04bc24e5daf1ae8b0460eaebf3b3f08f939ac7`

## Consensus Audit

- Total states analyzed: `115413`
- 384/768/1200 top-1 agreement rate: `0.8460`
- Rejected unstable-disagreement count: `1641`
- Consensus high-confidence disagreements: `27697`
- Consensus stability states: `42447`
- Duplicate state count: `36902`
- Duplicate trajectory count: `1075`
- Duplicate trajectory rate: `0.2000`

| Budget | Raw-vs-search top-1 disagree | Mean KL(search||raw) | Top-1 share mean | Top-1 share p50 | Top-1 share p90 |
|---|---|---|---|---|---|
| 384 | +0.3718 | +0.2962 | +0.6961 | +0.6979 | +0.9896 |
| 768 | +0.4043 | +0.3831 | +0.7254 | +0.7474 | +0.9909 |
| 1200 | +0.4195 | +0.4398 | +0.7433 | +0.7792 | +0.9925 |

## Replay

- Final disagreement rows: `2000`
- Final stability rows: `2000`
- Effective replay sampling fractions: `{"consensus_disagreement": 0.24855527247871745, "consensus_stability": 0.12427763623935872, "generic_bootstrap": 0.5958491269496055, "random_teacher": 0.0313179643323184}`

## Audit Breakdown

| Phase | 384 | 768 | 1200 |
|---|---|---|---|
| late | +0.2814 | +0.2869 | +0.2882 |
| mid | +0.4535 | +0.5013 | +0.5233 |
| opening | +0.3073 | +0.3766 | +0.4179 |

| Seat Context | 384 | 768 | 1200 |
|---|---|---|---|
| challenger | +0.3817 | +0.4082 | +0.4203 |
| current | +0.3722 | +0.4017 | +0.4143 |
| mixed | +0.3499 | +0.4011 | +0.4287 |

| Raw Margin Bucket | 384 | 768 | 1200 |
|---|---|---|---|
| 0.02 <= margin < 0.05 | +0.6122 | +0.6284 | +0.6314 |
| 0.05 <= margin < 0.10 | +0.5659 | +0.5879 | +0.5982 |
| margin < 0.02 | +0.6405 | +0.6596 | +0.6706 |
| margin >= 0.10 | +0.3087 | +0.3448 | +0.3621 |

| Top Move | States |
|---|---|
| 0 | 12157 |
| 1 | 14416 |
| 2 | 16564 |
| 3 | 18707 |
| 4 | 24366 |
| 5 | 29203 |

## Candidate Aggregate Table

| Candidate | Mean DS 384:256 | Delta 384:256 | Delta 768:768 | Delta 1200:1200 | Delta norm | Mined top-1 changed | Stability preserved |
|---|---|---|---|---|---|---|---|
| balanced_current_ref | -0.5484 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +1.0000 |
| pr128_iter2_e1_ref | -0.7394 | -0.1910 | -0.0962 | -0.0532 | +0.0289 | +0.0255 | +1.0000 |
| consensus_w8s4_policy_head_e1 | -0.8953 | -0.3469 | -0.3363 | +0.0651 | +0.0332 | +0.0300 | +1.0000 |
| consensus_w8s4_policy_head_e2 | -0.8705 | -0.3222 | -0.1682 | -0.0045 | +0.0605 | +0.0445 | +0.9990 |

## Losses And Artifacts

| Candidate | Policy loss | Value loss | Validation loss | Checkpoint SHA256 | Artifact weights SHA256 |
|---|---|---|---|---|---|
| balanced_current_ref | n/a | n/a | n/a | 18dacb5ef38602a77abf3927dc8748ac167c2ca91237385ec010ab27a123adc9 | 8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a |
| pr128_iter2_e1_ref | n/a | n/a | n/a | 6891bf9c6aa7f631928b8962f663fdcc76c9c7756bae31a3103f57e5a2c9fefd | e191f2db6c3ae4c40db5a19799bb6031944daa37adc94e286ce45ea2d9fa6299 |
| consensus_w8s4_policy_head_e1 | +1.1021 | +0.2225 | +1.2547 | cba534f05fe8098a5f4ef8eceddd798a69b5773d9ff494476a46a11e797e7bc0 | 270a381610377bb03ebe65f5f474d7c7f8b70ce77ab44aadf2c881d0d7b8e503 |
| consensus_w8s4_policy_head_e2 | +1.1011 | +0.2224 | +1.2542 | 7021d2fb77cd1922a71eb3c22b4cf47bf2af4d68f28de882bdf192c7f9fcba87 | 3900764c95607d09a4d837358f2970848d64da30048308187c749873f337e0f2 |

## Fixed Large DS Table

| Candidate | 384:256 | 768:256 | 768:768 | 1200:1200 | 1200:256 | 256:768 |
|---|---|---|---|---|---|---|
| balanced_current_ref | -0.5456 | -0.4648 | -0.2943 | -0.1380 | -0.1797 | -0.4648 |
| pr128_iter2_e1_ref | -0.7370 | -0.3594 | -0.4023 | -0.1927 | -0.2643 | -0.5260 |
| consensus_w8s4_policy_head_e1 | -0.8789 | -0.3281 | -0.6328 | -0.0768 | -0.3359 | -0.4388 |
| consensus_w8s4_policy_head_e2 | -0.8529 | -0.4128 | -0.4635 | -0.1458 | -0.3359 | -0.4167 |

## Held-Out Mean/Worst DS Table

| Candidate | Held-out mean 384:256 | Held-out worst-suite 384:256 |
|---|---|---|
| balanced_current_ref | -0.5488 | -0.5664 |
| pr128_iter2_e1_ref | -0.7398 | -0.7669 |
| consensus_w8s4_policy_head_e1 | -0.8980 | -0.9284 |
| consensus_w8s4_policy_head_e2 | -0.8735 | -0.9023 |

## Bootstrap CI Table

| Comparison | Mean | Lower 95% | Upper 95% | Openings |
|---|---|---|---|---|
| balanced_current_ref_minus_balanced_current_ref_1200_1200 | +0.0000 | +0.0000 | +0.0000 | 2688 |
| balanced_current_ref_minus_balanced_current_ref_1200_256 | +0.0000 | +0.0000 | +0.0000 | 2688 |
| balanced_current_ref_minus_balanced_current_ref_384_256 | +0.0000 | +0.0000 | +0.0000 | 2688 |
| balanced_current_ref_minus_balanced_current_ref_768_768 | +0.0000 | +0.0000 | +0.0000 | 2688 |
| pr128_iter2_e1_ref_minus_balanced_current_ref_1200_1200 | -0.0532 | -0.0618 | -0.0446 | 2688 |
| pr128_iter2_e1_ref_minus_balanced_current_ref_1200_256 | -0.0841 | -0.0911 | -0.0770 | 2688 |
| pr128_iter2_e1_ref_minus_balanced_current_ref_384_256 | -0.1910 | -0.2052 | -0.1769 | 2688 |
| pr128_iter2_e1_ref_minus_balanced_current_ref_768_768 | -0.0962 | -0.1202 | -0.0718 | 2688 |
| consensus_w8s4_policy_head_e1_minus_balanced_current_ref_1200_1200 | +0.0651 | +0.0560 | +0.0740 | 2688 |
| consensus_w8s4_policy_head_e1_minus_balanced_current_ref_1200_256 | -0.1507 | -0.1600 | -0.1412 | 2688 |
| consensus_w8s4_policy_head_e1_minus_balanced_current_ref_384_256 | -0.3469 | -0.3646 | -0.3298 | 2688 |
| consensus_w8s4_policy_head_e1_minus_balanced_current_ref_768_768 | -0.3363 | -0.3646 | -0.3080 | 2688 |
| consensus_w8s4_policy_head_e2_minus_balanced_current_ref_1200_1200 | -0.0045 | -0.0160 | +0.0071 | 2688 |
| consensus_w8s4_policy_head_e2_minus_balanced_current_ref_1200_256 | -0.1507 | -0.1600 | -0.1412 | 2688 |
| consensus_w8s4_policy_head_e2_minus_balanced_current_ref_384_256 | -0.3222 | -0.3400 | -0.3047 | 2688 |
| consensus_w8s4_policy_head_e2_minus_balanced_current_ref_768_768 | -0.1682 | -0.1823 | -0.1540 | 2688 |

## P0/P1 Split At 384:256

| Candidate | Mean P0 | Mean P1 | Gap | Mean duplicates |
|---|---|---|---|---|
| balanced_current_ref | +0.4107 | +0.9591 | +0.5484 | +1536.0000 |
| pr128_iter2_e1_ref | +0.1968 | +0.9362 | +0.7394 | +1536.0000 |
| consensus_w8s4_policy_head_e1 | +0.0409 | +0.9362 | +0.8953 | +1536.0000 |
| consensus_w8s4_policy_head_e2 | +0.0409 | +0.9115 | +0.8705 | +1536.0000 |

## Gate

- balanced_current_ref: `high_search_breakthrough`
- pr128_iter2_e1_ref: `not_run`
- consensus_w8s4_policy_head_e1: `not_run`
- consensus_w8s4_policy_head_e2: `not_run`
