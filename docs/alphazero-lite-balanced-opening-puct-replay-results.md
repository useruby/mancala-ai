# AlphaZero-Lite Balanced Opening PUCT Replay Results

**Date**: 2026-06-23

**Classification**: `balanced_replay_success`

## Completion

- original PR #125 balanced experiment: `completed`
- fixed-suite evaluation complete: `True`
- held-out evaluation complete: `True`
- gate complete: `True`

## Decision

- best balanced lane: `balanced_w8s4_policy_head_e1`
- mean delta vs promoted current at `384:256`: `+0.1899`
- mean delta vs promoted current at `768:768`: `-0.0638`
- mean delta vs promoted current at `1200:1200`: `+0.0532`

## Stability Replay

- selected rows: `2000`
- unique rows: `2000`
- overlap with mined disagreement before exclusion: `4512`
- stability entropy mean: `0.3531`

## Candidate Aggregate Table

| Candidate | Mean DS 384:256 | Delta 384:256 | Delta 768:768 | Delta 1200:1200 | Stability preserved | Mined top-1 changed |
|---|---|---|---|---|---|---|
| promoted_current_ref | -0.3984 | +0.0000 | +0.0000 | +0.0000 | +1.0000 | +0.0000 |
| pr123_w8_e1_ref | -0.1382 | +0.2602 | -0.4001 | +0.1153 | +1.0000 | +0.0710 |
| pr123_w4_e2_ref | -0.1856 | +0.2128 | -0.2320 | +0.1682 | +1.0000 | +0.0695 |
| balanced_w8s4_policy_head_e1 | -0.2085 | +0.1899 | -0.0638 | +0.0532 | +0.9995 | +0.0680 |
| balanced_w8s4_policy_head_e2 | -0.5458 | -0.1473 | -0.2511 | +0.1682 | +0.9990 | +0.0940 |
| balanced_w8s8_policy_head_e1 | -0.2846 | +0.1138 | -0.0867 | +0.1060 | +0.9995 | +0.0710 |
| balanced_w8s8_policy_head_e2 | -0.5705 | -0.1721 | -0.4459 | +0.1060 | +0.9990 | +0.0970 |
| balanced_w4s8_policy_head_e1 | -0.3890 | +0.0095 | -0.0867 | +0.1060 | +1.0000 | +0.0575 |
| balanced_w4s8_policy_head_e2 | -0.6029 | -0.2044 | -0.1708 | +0.1060 | +0.9990 | +0.0790 |

## Bootstrap CI Table

| Comparison | Mean | Lower 95% | Upper 95% | Openings |
|---|---|---|---|---|
| promoted_current_ref_minus_pr123_w8_e1_ref_384_256 | -0.2602 | -0.2891 | -0.2321 | 2688 |
| promoted_current_ref_minus_pr123_w8_e1_ref_768_768 | +0.4001 | +0.3718 | +0.4284 | 2688 |
| promoted_current_ref_minus_promoted_current_ref_1200_1200 | +0.0000 | +0.0000 | +0.0000 | 2688 |
| promoted_current_ref_minus_promoted_current_ref_384_256 | +0.0000 | +0.0000 | +0.0000 | 2688 |
| promoted_current_ref_minus_promoted_current_ref_768_768 | +0.0000 | +0.0000 | +0.0000 | 2688 |
| pr123_w8_e1_ref_minus_promoted_current_ref_1200_1200 | +0.1153 | +0.1062 | +0.1246 | 2688 |
| pr123_w8_e1_ref_minus_promoted_current_ref_384_256 | +0.2602 | +0.2325 | +0.2889 | 2688 |
| pr123_w8_e1_ref_minus_promoted_current_ref_768_768 | -0.4001 | -0.4288 | -0.3718 | 2688 |
| pr123_w4_e2_ref_minus_pr123_w8_e1_ref_384_256 | -0.0474 | -0.0554 | -0.0394 | 2688 |
| pr123_w4_e2_ref_minus_pr123_w8_e1_ref_768_768 | +0.1682 | +0.1540 | +0.1823 | 2688 |
| pr123_w4_e2_ref_minus_promoted_current_ref_1200_1200 | +0.1682 | +0.1540 | +0.1823 | 2688 |
| pr123_w4_e2_ref_minus_promoted_current_ref_384_256 | +0.2128 | +0.1929 | +0.2336 | 2688 |
| pr123_w4_e2_ref_minus_promoted_current_ref_768_768 | -0.2320 | -0.2476 | -0.2163 | 2688 |
| balanced_w8s4_policy_head_e1_minus_pr123_w8_e1_ref_384_256 | -0.0703 | -0.0770 | -0.0636 | 2688 |
| balanced_w8s4_policy_head_e1_minus_pr123_w8_e1_ref_768_768 | +0.3363 | +0.3080 | +0.3646 | 2688 |
| balanced_w8s4_policy_head_e1_minus_promoted_current_ref_1200_1200 | +0.0532 | +0.0446 | +0.0618 | 2688 |
| balanced_w8s4_policy_head_e1_minus_promoted_current_ref_384_256 | +0.1899 | +0.1682 | +0.2121 | 2688 |
| balanced_w8s4_policy_head_e1_minus_promoted_current_ref_768_768 | -0.0638 | -0.0722 | -0.0552 | 2688 |
| balanced_w8s4_policy_head_e2_minus_pr123_w8_e1_ref_384_256 | -0.4076 | -0.4371 | -0.3780 | 2688 |
| balanced_w8s4_policy_head_e2_minus_pr123_w8_e1_ref_768_768 | +0.1490 | +0.1339 | +0.1642 | 2688 |
| balanced_w8s4_policy_head_e2_minus_promoted_current_ref_1200_1200 | +0.1682 | +0.1540 | +0.1823 | 2688 |
| balanced_w8s4_policy_head_e2_minus_promoted_current_ref_384_256 | -0.1473 | -0.1637 | -0.1308 | 2688 |
| balanced_w8s4_policy_head_e2_minus_promoted_current_ref_768_768 | -0.2511 | -0.2852 | -0.2167 | 2688 |
| balanced_w8s8_policy_head_e1_minus_pr123_w8_e1_ref_384_256 | -0.1464 | -0.1611 | -0.1319 | 2688 |
| balanced_w8s8_policy_head_e1_minus_pr123_w8_e1_ref_768_768 | +0.3134 | +0.2846 | +0.3426 | 2688 |
| balanced_w8s8_policy_head_e1_minus_promoted_current_ref_1200_1200 | +0.1060 | +0.0975 | +0.1148 | 2688 |
| balanced_w8s8_policy_head_e1_minus_promoted_current_ref_384_256 | +0.1138 | +0.0978 | +0.1306 | 2688 |
| balanced_w8s8_policy_head_e1_minus_promoted_current_ref_768_768 | -0.0867 | -0.0939 | -0.0794 | 2688 |
| balanced_w8s8_policy_head_e2_minus_pr123_w8_e1_ref_384_256 | -0.4323 | -0.4619 | -0.4031 | 2688 |
| balanced_w8s8_policy_head_e2_minus_pr123_w8_e1_ref_768_768 | -0.0458 | -0.0536 | -0.0379 | 2688 |
| balanced_w8s8_policy_head_e2_minus_promoted_current_ref_1200_1200 | +0.1060 | +0.0975 | +0.1148 | 2688 |
| balanced_w8s8_policy_head_e2_minus_promoted_current_ref_384_256 | -0.1721 | -0.1875 | -0.1564 | 2688 |
| balanced_w8s8_policy_head_e2_minus_promoted_current_ref_768_768 | -0.4459 | -0.4732 | -0.4183 | 2688 |
| balanced_w4s8_policy_head_e1_minus_pr123_w8_e1_ref_384_256 | -0.2507 | -0.2799 | -0.2225 | 2688 |
| balanced_w4s8_policy_head_e1_minus_pr123_w8_e1_ref_768_768 | +0.3134 | +0.2846 | +0.3426 | 2688 |
| balanced_w4s8_policy_head_e1_minus_promoted_current_ref_1200_1200 | +0.1060 | +0.0975 | +0.1148 | 2688 |
| balanced_w4s8_policy_head_e1_minus_promoted_current_ref_384_256 | +0.0095 | +0.0071 | +0.0121 | 2688 |
| balanced_w4s8_policy_head_e1_minus_promoted_current_ref_768_768 | -0.0867 | -0.0939 | -0.0794 | 2688 |
| balanced_w4s8_policy_head_e2_minus_pr123_w8_e1_ref_384_256 | -0.4647 | -0.4926 | -0.4368 | 2688 |
| balanced_w4s8_policy_head_e2_minus_pr123_w8_e1_ref_768_768 | +0.2294 | +0.2076 | +0.2515 | 2688 |
| balanced_w4s8_policy_head_e2_minus_promoted_current_ref_1200_1200 | +0.1060 | +0.0975 | +0.1148 | 2688 |
| balanced_w4s8_policy_head_e2_minus_promoted_current_ref_384_256 | -0.2044 | -0.2202 | -0.1884 | 2688 |
| balanced_w4s8_policy_head_e2_minus_promoted_current_ref_768_768 | -0.1708 | -0.1797 | -0.1616 | 2688 |

## Large-Suite Budgets

### `384:256`

| Candidate | Mean DS | Worst-suite DS | Mean P0 | Mean P1 | Mean duplicates |
|---|---|---|---|---|---|
| promoted_current_ref | -0.3984 | -0.4349 | +0.4115 | +0.8099 | +1536.0000 |
| pr123_w8_e1_ref | -0.1382 | -0.1654 | +0.5311 | +0.6693 | +1536.0000 |
| pr123_w4_e2_ref | -0.1856 | -0.2161 | +0.5539 | +0.7396 | +1536.0000 |
| balanced_w8s4_policy_head_e1 | -0.2085 | -0.2370 | +0.5311 | +0.7396 | +1536.0000 |
| balanced_w8s4_policy_head_e2 | -0.5458 | -0.5638 | +0.2641 | +0.8099 | +1536.0000 |
| balanced_w8s8_policy_head_e1 | -0.2846 | -0.3073 | +0.3847 | +0.6693 | +1536.0000 |
| balanced_w8s8_policy_head_e2 | -0.5705 | -0.5885 | +0.2394 | +0.8099 | +1536.0000 |
| balanced_w4s8_policy_head_e1 | -0.3890 | -0.4219 | +0.4209 | +0.8099 | +1536.0000 |
| balanced_w4s8_policy_head_e2 | -0.6029 | -0.6224 | +0.2070 | +0.8099 | +1536.0000 |

### `768:256`

| Candidate | Mean DS | Worst-suite DS | Mean P0 | Mean P1 | Mean duplicates |
|---|---|---|---|---|---|
| promoted_current_ref | -0.4016 | -0.4167 | +0.5874 | +0.9890 | +1536.0000 |
| pr123_w8_e1_ref | -0.3906 | -0.4102 | +0.5874 | +0.9781 | +1536.0000 |
| pr123_w4_e2_ref | -0.3203 | -0.3398 | +0.6577 | +0.9781 | +1536.0000 |
| balanced_w8s4_policy_head_e1 | -0.3906 | -0.4102 | +0.5874 | +0.9781 | +1536.0000 |
| balanced_w8s4_policy_head_e2 | -0.3203 | -0.3398 | +0.6577 | +0.9781 | +1536.0000 |
| balanced_w8s8_policy_head_e1 | -0.3906 | -0.4102 | +0.5874 | +0.9781 | +1536.0000 |
| balanced_w8s8_policy_head_e2 | -0.3203 | -0.3398 | +0.6577 | +0.9781 | +1536.0000 |
| balanced_w4s8_policy_head_e1 | -0.3787 | -0.3958 | +0.5874 | +0.9661 | +1536.0000 |
| balanced_w4s8_policy_head_e2 | -0.3175 | -0.3372 | +0.5874 | +0.9049 | +1536.0000 |

### `768:768`

| Candidate | Mean DS | Worst-suite DS | Mean P0 | Mean P1 | Mean duplicates |
|---|---|---|---|---|---|
| promoted_current_ref | -0.1894 | -0.2188 | +0.4053 | +0.5947 | +1536.0000 |
| pr123_w8_e1_ref | -0.5895 | -0.6328 | +0.2600 | +0.8495 | +1536.0000 |
| pr123_w4_e2_ref | -0.4213 | -0.4531 | +0.4282 | +0.8495 | +1536.0000 |
| balanced_w8s4_policy_head_e1 | -0.2532 | -0.2839 | +0.4282 | +0.6814 | +1536.0000 |
| balanced_w8s4_policy_head_e2 | -0.4405 | -0.4857 | +0.3778 | +0.8183 | +1536.0000 |
| balanced_w8s8_policy_head_e1 | -0.2760 | -0.3047 | +0.4282 | +0.7042 | +1536.0000 |
| balanced_w8s8_policy_head_e2 | -0.6352 | -0.6823 | +0.2372 | +0.8724 | +1536.0000 |
| balanced_w4s8_policy_head_e1 | -0.2760 | -0.3047 | +0.4282 | +0.7042 | +1536.0000 |
| balanced_w4s8_policy_head_e2 | -0.3601 | -0.3880 | +0.4282 | +0.7883 | +1536.0000 |

### `1200:1200`

| Candidate | Mean DS | Worst-suite DS | Mean P0 | Mean P1 | Mean duplicates |
|---|---|---|---|---|---|
| promoted_current_ref | -0.4405 | -0.4635 | +0.2798 | +0.7202 | +1536.0000 |
| pr123_w8_e1_ref | -0.3251 | -0.3451 | +0.2798 | +0.6049 | +1536.0000 |
| pr123_w4_e2_ref | -0.2723 | -0.2917 | +0.2798 | +0.5521 | +1536.0000 |
| balanced_w8s4_policy_head_e1 | -0.3873 | -0.4193 | +0.3017 | +0.6890 | +1536.0000 |
| balanced_w8s4_policy_head_e2 | -0.2723 | -0.2917 | +0.3638 | +0.6362 | +1536.0000 |
| balanced_w8s8_policy_head_e1 | -0.3344 | -0.3607 | +0.3017 | +0.6362 | +1536.0000 |
| balanced_w8s8_policy_head_e2 | -0.3344 | -0.3607 | +0.3017 | +0.6362 | +1536.0000 |
| balanced_w4s8_policy_head_e1 | -0.3344 | -0.3607 | +0.3017 | +0.6362 | +1536.0000 |
| balanced_w4s8_policy_head_e2 | -0.3344 | -0.3607 | +0.3017 | +0.6362 | +1536.0000 |

### `1200:256`

| Candidate | Mean DS | Worst-suite DS | Mean P0 | Mean P1 | Mean duplicates |
|---|---|---|---|---|---|
| promoted_current_ref | -0.1674 | -0.1797 | +0.8326 | +1.0000 | +1536.0000 |
| pr123_w8_e1_ref | -0.1674 | -0.1797 | +0.8326 | +1.0000 | +1536.0000 |
| pr123_w4_e2_ref | -0.1674 | -0.1797 | +0.8326 | +1.0000 | +1536.0000 |
| balanced_w8s4_policy_head_e1 | -0.0833 | -0.1003 | +0.8326 | +0.9159 | +1536.0000 |
| balanced_w8s4_policy_head_e2 | -0.1674 | -0.1797 | +0.8326 | +1.0000 | +1536.0000 |
| balanced_w8s8_policy_head_e1 | -0.1674 | -0.1797 | +0.8326 | +1.0000 | +1536.0000 |
| balanced_w8s8_policy_head_e2 | -0.1674 | -0.1797 | +0.8326 | +1.0000 | +1536.0000 |
| balanced_w4s8_policy_head_e1 | -0.1674 | -0.1797 | +0.8326 | +1.0000 | +1536.0000 |
| balanced_w4s8_policy_head_e2 | -0.1674 | -0.1797 | +0.8326 | +1.0000 | +1536.0000 |

### `256:768`

| Candidate | Mean DS | Worst-suite DS | Mean P0 | Mean P1 | Mean duplicates |
|---|---|---|---|---|---|
| promoted_current_ref | -0.4016 | -0.4167 | +0.0110 | +0.4126 | +1536.0000 |
| pr123_w8_e1_ref | -0.2701 | -0.2982 | +0.0409 | +0.3110 | +1536.0000 |
| pr123_w4_e2_ref | -0.2796 | -0.3034 | +0.0314 | +0.3110 | +1536.0000 |
| balanced_w8s4_policy_head_e1 | -0.2811 | -0.3047 | +0.0299 | +0.3110 | +1536.0000 |
| balanced_w8s4_policy_head_e2 | -0.2701 | -0.2982 | +0.0409 | +0.3110 | +1536.0000 |
| balanced_w8s8_policy_head_e1 | -0.3084 | -0.3242 | +0.0339 | +0.3423 | +1536.0000 |
| balanced_w8s8_policy_head_e2 | -0.2662 | -0.2839 | +0.0448 | +0.3110 | +1536.0000 |
| balanced_w4s8_policy_head_e1 | -0.4408 | -0.4505 | +0.0770 | +0.5179 | +1536.0000 |
| balanced_w4s8_policy_head_e2 | -0.2881 | -0.2969 | +0.0229 | +0.3110 | +1536.0000 |
