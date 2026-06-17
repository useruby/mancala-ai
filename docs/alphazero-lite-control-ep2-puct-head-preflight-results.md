# AlphaZero-Lite Control EP2 PUCT Head Preflight Results

**Date**: 2026-06-17
**Classification**: `select_e1_for_promotion_followup`
**Schema**: `azlite_control_ep2_puct_head_preflight_v1`

## Summary

Recommendation: `select_e1_for_promotion_followup`.

## Artifact Hashes

| Candidate | Checkpoint | Checkpoint SHA256 | Artifact | Weights SHA256 |
|---|---|---|---|---|
| canonical_ref | /tmp/azlite_control_continuation_sweep/lr3e5/checkpoint_epoch2.npz | 619376dba87b968cc85bfc401cc230aca64e444dc3570216b0630c2c29a388f9 | /tmp/azlite_control_ep2_seed_harvest/control_ep2_eval_only/artifact_control_ep2 | 34b3697f95c3c63e092838d10668ba1361dab2a68a0bfe12305a294c45f765ad |
| puct_policy_head_e1 | /tmp/azlite_control_ep2_puct_smoke/puct_policy_head_e1/checkpoint_epoch1.npz | a793f32565b0c706c4228e4de3bc00aea5c471089ec940c4fe85e726fe4f9357 | /tmp/azlite_control_ep2_puct_smoke/puct_policy_head_e1/artifact_puct_policy_head_e1 | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece |
| puct_policy_head_e2 | /tmp/azlite_control_ep2_puct_smoke/puct_policy_head_e2/checkpoint_epoch2.npz | 6e13745f102e097d2d7215e4ed642324345b18ec4fcc2dfd87670863fa8b7a9c | /tmp/azlite_control_ep2_puct_smoke/artifacts/puct_policy_head_e2 | 40d6db6d9f63074f8e0ac4247bda893c051834cd00f885b14473a5d6c712a35f |

## Suite Hashes

| Suite | Path | SHA256 | Rows |
|---|---|---|---|
| fixed_large | /tmp/azlite_opening_suite/large_eval.jsonl | ebc86f053a8a4c12f3e937b5acfe37df7b0d2f1f0002969dcac077a9f99d65a4 | 384 |
| heldout_seed43_large | /tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed43_large.jsonl | 5e0ed96ba56f99318c32a139692309759659da16a0a98c590e35f7e2496cade9 | 384 |
| heldout_seed44_large | /tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed44_large.jsonl | 323f7abec32fb00d3b7b6b153ddd5692435755a3bc5340d8aa5aec32ca639620 | 384 |
| heldout_seed45_large | /tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed45_large.jsonl | ca72c8b7fe1adf7229f81183321475033f8548dcdeb1c71fd16c85c35ce89cda | 384 |

## Fixed-Suite Results

| Suite | Candidate | 384:256 DS | 384:256 P0 / P1 | Duplicate trajectories |
|---|---|---|---|---|
| fixed_large | canonical_ref | -0.1784 | 0.2878 / 0.4661 | 1536 |
| fixed_large | puct_policy_head_e1 | -0.0208 | 0.2760 / 0.2969 | 1536 |
| fixed_large | puct_policy_head_e2 | -0.0091 | 0.2878 / 0.2969 | 1536 |

## Held-Out-Suite Results

| Suite | Candidate | 384:256 DS | 384:256 P0 / P1 | Duplicate trajectories |
|---|---|---|---|---|
| heldout_seed43_large | canonical_ref | -0.1497 | 0.3060 / 0.4557 | 1536 |
| heldout_seed43_large | puct_policy_head_e1 | -0.0039 | 0.2956 / 0.2995 | 1536 |
| heldout_seed43_large | puct_policy_head_e2 | +0.0065 | 0.3060 / 0.2995 | 1536 |
| heldout_seed44_large | canonical_ref | -0.1641 | 0.3021 / 0.4661 | 1536 |
| heldout_seed44_large | puct_policy_head_e1 | -0.0143 | 0.2930 / 0.3073 | 1536 |
| heldout_seed44_large | puct_policy_head_e2 | -0.0052 | 0.3021 / 0.3073 | 1536 |
| heldout_seed45_large | canonical_ref | -0.1784 | 0.2956 / 0.4740 | 1536 |
| heldout_seed45_large | puct_policy_head_e1 | -0.0326 | 0.2826 / 0.3151 | 1536 |
| heldout_seed45_large | puct_policy_head_e2 | -0.0195 | 0.2956 / 0.3151 | 1536 |

## Aggregate Robustness Table

| Candidate | Mean DS 384:256 | Worst-suite DS 384:256 | Stddev DS 384:256 | Mean delta vs canonical 384:256 | Mean DS 1200:1200 | Mean delta vs canonical 1200:1200 |
|---|---|---|---|---|---|---|
| canonical_ref | -0.1676 | -0.1784 | 0.0137 | +0.0000 | +0.1009 | +0.0000 |
| puct_policy_head_e1 | -0.0179 | -0.0326 | 0.0120 | +0.1497 | +0.0781 | -0.0228 |
| puct_policy_head_e2 | -0.0068 | -0.0195 | 0.0108 | +0.1608 | +0.0078 | -0.0931 |

## Paired Bootstrap CI Table

| Comparison | Mean | Lower 95% | Upper 95% | Openings |
|---|---|---|---|---|
| e1_minus_canonical_384_256 | +0.1497 | +0.1305 | +0.1686 | 1536 |
| e1_minus_canonical_1200_1200 | -0.0228 | -0.0283 | -0.0176 | 1536 |
| e2_minus_canonical_384_256 | +0.1608 | +0.1432 | +0.1790 | 1536 |
| e2_minus_canonical_1200_1200 | -0.0931 | -0.1029 | -0.0833 | 1536 |
| e2_minus_e1_384_256 | +0.0111 | +0.0075 | +0.0150 | 1536 |
| e2_minus_e1_1200_1200 | -0.0703 | -0.0791 | -0.0618 | 1536 |

## Direct e1/e2 Head-to-Head Table

| Matchup | Budget | Arena score | Disadvantaged-seat score | Starts 0 | Starts 1 | Duplicate trajectories |
|---|---|---|---|---|---|---|
| puct_policy_head_e1_vs_puct_policy_head_e2 | 256:384 | +0.5000 | +0.0000 | +1.0000 | +0.0000 | 240 |
| puct_policy_head_e1_vs_puct_policy_head_e2 | 384:384 | +0.5000 | +0.0000 | +1.0000 | +0.0000 | 240 |
| puct_policy_head_e1_vs_puct_policy_head_e2 | 1200:1200 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 240 |
| puct_policy_head_e1_vs_puct_policy_head_e2 | 384:256 | +0.5000 | +0.0000 | +1.0000 | +0.0000 | 240 |
| puct_policy_head_e2_vs_puct_policy_head_e1 | 256:384 | +0.5000 | +0.0000 | +1.0000 | +0.0000 | 240 |
| puct_policy_head_e2_vs_puct_policy_head_e1 | 384:384 | +0.5000 | +0.0000 | +1.0000 | +0.0000 | 240 |
| puct_policy_head_e2_vs_puct_policy_head_e1 | 1200:1200 | +1.0000 | +1.0000 | +1.0000 | +1.0000 | 240 |
| puct_policy_head_e2_vs_puct_policy_head_e1 | 384:256 | +0.5000 | +0.0000 | +1.0000 | +0.0000 | 240 |

## Default Gate Classification Table

| Candidate | Classification | High-search preserved | 384:256 disadvantaged | 1200:1200 disadvantaged |
|---|---|---|---|---|
| canonical_ref | high_search_breakthrough | yes | +0.0000 | +1.0000 |
| puct_policy_head_e1 | high_search_breakthrough | yes | +0.0000 | +1.0000 |
| puct_policy_head_e2 | high_search_breakthrough | yes | +0.0000 | +1.0000 |

## Opening-Level e1 vs e2 Count

| Budget | e1 better | e2 better | Tie |
|---|---|---|---|
| 384:256 disadvantaged seat | 0 | 0 | 1536 |

## Final Classification

- Classification: `select_e1_for_promotion_followup`
- Recommendation: `select_e1_for_promotion_followup`
- Rationale: Uses fixed plus held-out opening robustness, paired per-opening bootstrap CIs, default gate behavior, and direct e1/e2 head-to-head checks without any training or promotion step.
