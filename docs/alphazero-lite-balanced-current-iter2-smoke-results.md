# AlphaZero-Lite Balanced Current Iter2 Smoke Results

**Date**: 2026-06-24

**Classification**: `iter2_overfit_or_destructive`

## Inputs

- Current artifact weights SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- Current source checkpoint SHA256: `18dacb5ef38602a77abf3927dc8748ac167c2ca91237385ec010ab27a123adc9`

## Replay Audit

- Disagreement analyzed states: `115413`
- Disagreement top-1 rate: `0.3718`
- High-confidence disagreements: `24362`
- Disagreement replay rows: `2000`
- Stability candidate rows: `294530`
- Stability replay rows: `2000`
- Replay overlap: `0`

## PR126 Comparison

- Disagreement-rate delta vs PR126 audit: `+0.0144`
- High-confidence disagreement delta vs PR126 audit: `10999`
- High-margin KL delta vs PR126 audit: `+0.0088`

## Candidate Aggregate Table

| Candidate | Mean DS 384:256 | Delta 384:256 | Delta 768:768 | Delta 1200:1200 | Stability preserved | Mined top-1 changed |
|---|---|---|---|---|---|---|
| balanced_current_ref | -0.5484 | +0.0000 | +0.0000 | +0.0000 | +1.0000 | +0.0000 |
| balanced_iter2_w8s4_policy_head_e1 | -0.7394 | -0.1910 | -0.0962 | -0.0532 | +1.0000 | +0.0340 |
| balanced_iter2_w8s4_policy_head_e2 | -0.8705 | -0.3222 | -0.3821 | -0.0532 | +1.0000 | +0.0450 |

## Bootstrap CI Table

| Comparison | Mean | Lower 95% | Upper 95% | Openings |
|---|---|---|---|---|
| balanced_current_ref_minus_balanced_current_ref_1200_1200 | +0.0000 | +0.0000 | +0.0000 | 2688 |
| balanced_current_ref_minus_balanced_current_ref_1200_256 | +0.0000 | +0.0000 | +0.0000 | 2688 |
| balanced_current_ref_minus_balanced_current_ref_384_256 | +0.0000 | +0.0000 | +0.0000 | 2688 |
| balanced_current_ref_minus_balanced_current_ref_768_768 | +0.0000 | +0.0000 | +0.0000 | 2688 |
| balanced_iter2_w8s4_policy_head_e1_minus_balanced_current_ref_1200_1200 | -0.0532 | -0.0618 | -0.0446 | 2688 |
| balanced_iter2_w8s4_policy_head_e1_minus_balanced_current_ref_1200_256 | -0.0841 | -0.0911 | -0.0770 | 2688 |
| balanced_iter2_w8s4_policy_head_e1_minus_balanced_current_ref_384_256 | -0.1910 | -0.2052 | -0.1769 | 2688 |
| balanced_iter2_w8s4_policy_head_e1_minus_balanced_current_ref_768_768 | -0.0962 | -0.1202 | -0.0718 | 2688 |
| balanced_iter2_w8s4_policy_head_e2_minus_balanced_current_ref_1200_1200 | -0.0532 | -0.0618 | -0.0446 | 2688 |
| balanced_iter2_w8s4_policy_head_e2_minus_balanced_current_ref_1200_256 | -0.0841 | -0.0911 | -0.0770 | 2688 |
| balanced_iter2_w8s4_policy_head_e2_minus_balanced_current_ref_384_256 | -0.3222 | -0.3400 | -0.3047 | 2688 |
| balanced_iter2_w8s4_policy_head_e2_minus_balanced_current_ref_768_768 | -0.3821 | -0.4103 | -0.3538 | 2688 |

## Fixed Large DS Table

| Candidate | 384:256 | 768:256 | 768:768 | 1200:1200 | 1200:256 | 256:768 |
|---|---|---|---|---|---|---|
| balanced_current_ref | -0.5456 | -0.4648 | -0.2943 | -0.1380 | -0.1797 | -0.4648 |
| balanced_iter2_w8s4_policy_head_e1 | -0.7370 | -0.3594 | -0.4023 | -0.1927 | -0.2643 | -0.5260 |
| balanced_iter2_w8s4_policy_head_e2 | -0.8529 | -0.3802 | -0.6771 | -0.1927 | -0.2643 | -0.5286 |

## Held-Out Mean/Worst DS Table

| Candidate | Held-out mean 384:256 | Held-out worst-suite 384:256 |
|---|---|---|
| balanced_current_ref | -0.5488 | -0.5664 |
| balanced_iter2_w8s4_policy_head_e1 | -0.7398 | -0.7669 |
| balanced_iter2_w8s4_policy_head_e2 | -0.8735 | -0.9023 |

## P0/P1 Split At 384:256

| Candidate | Mean P0 | Mean P1 | Gap | Mean duplicates |
|---|---|---|---|---|
| balanced_current_ref | +0.4107 | +0.9591 | +0.5484 | +1536.0000 |
| balanced_iter2_w8s4_policy_head_e1 | +0.1968 | +0.9362 | +0.7394 | +1536.0000 |
| balanced_iter2_w8s4_policy_head_e2 | +0.0657 | +0.9362 | +0.8705 | +1536.0000 |

## Gate

- balanced_current_ref: `high_search_breakthrough`
- balanced_iter2_w8s4_policy_head_e1: `not_run`
- balanced_iter2_w8s4_policy_head_e2: `not_run`

## Conclusion

- The balanced-current replay recipe is not saturated: fresh mining still produced `2000` disagreement rows, `2000` stability rows, and materially more high-confidence disagreements than the PR #126 audit.
- The resulting policy-head-only iter2 candidates were uniformly worse than `balanced_current_ref` on fixed and held-out `384:256`, and also regressed `768:768` and high-budget robustness.
- The limiting factor now appears to be search/mining quality rather than the promotion step, so this PR does not support another balanced self-improvement promotion pass.
