# AlphaZero-lite Incumbent Proxy Disagreement Residual Reference Adjudication Results

## 1. Context

- No training was run.
- No arena was run.
- No model was promoted.
- No replay artifacts were created.
- Active references stayed read-only at `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- Current artifact: `storage/ai/alphazero_lite/current`.

## 2. Why PR #44 blocked training

- PR #44 fixed row `021`, reran the post-adjudication rebaseline, and showed that `residual_reference_suspicious` still had 9 rows.
- That suspicious-reference bucket remained larger than the usable `stable_value_head_miscalibration` bucket, so training would risk baking incorrect labels into any value-calibration pass.
- This run adjudicates only those residual suspicious rows and keeps the active fixture unchanged.

## 3. Suspicious row validation

| row_id | active_reference_move | legal | reference_unstable | canonical_state_match | current_selected_384 | current_selected_1200 | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-007 | 0 | true | false | true | 5 | 3 | ok | validated against active references |
| incumbent_proxy_disagreement-009 | 0 | true | false | true | 3 | 3 | ok | validated against active references |
| incumbent_proxy_disagreement-010 | 4 | true | false | true | 5 | 5 | ok | validated against active references |
| incumbent_proxy_disagreement-018 | 2 | true | false | true | 5 | 3 | ok | validated against active references |
| incumbent_proxy_disagreement-021 | 2 | true | false | true | 5 | 5 | ok | validated against active references |
| incumbent_proxy_disagreement-023 | 2 | true | false | true | 1 | 1 | ok | validated against active references |
| incumbent_proxy_disagreement-024 | 2 | true | false | true | 1 | 1 | ok | validated against active references |
| incumbent_proxy_disagreement-032 | 4 | true | false | true | 0 | 1 | ok | validated against active references |
| incumbent_proxy_disagreement-033 | 4 | true | false | true | 0 | 0 | ok | validated against active references |

## 4. Root ClassicMCTS adjudication

| row_id | budget | seeds | active_reference_move | observed_top_moves | majority_move | majority_fraction | reference_selected_fraction | top1_margin_mean | decision | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-007 | 1200 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.1416 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-007 | 2400 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.1987 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-007 | 5000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.2044 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-007 | 10000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.1941 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-007 | 30000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.2010 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-009 | 1200 | 7 | 0 | 0:6, 3:1 | 0 | 0.8571 | 0.8571 | 0.0170 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-009 | 2400 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.1208 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-009 | 5000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.2171 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-009 | 10000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.2882 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-009 | 30000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.3570 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-010 | 1200 | 7 | 4 | 4:1, 5:6 | 5 | 0.8571 | 0.1429 | 0.0190 | supports_flip_candidate | high-budget majority prefers a different move |
| incumbent_proxy_disagreement-010 | 2400 | 7 | 4 | 5:7 | 5 | 1.0000 | 0.0000 | 0.0360 | supports_flip_candidate | high-budget majority prefers a different move |
| incumbent_proxy_disagreement-010 | 5000 | 7 | 4 | 5:7 | 5 | 1.0000 | 0.0000 | 0.0200 | supports_flip_candidate | high-budget majority prefers a different move |
| incumbent_proxy_disagreement-010 | 10000 | 7 | 4 | 0:7 | 0 | 1.0000 | 0.0000 | 0.2357 | supports_flip_candidate | high-budget majority prefers a different move |
| incumbent_proxy_disagreement-010 | 30000 | 7 | 4 | 0:7 | 0 | 1.0000 | 0.0000 | 0.3787 | supports_flip_candidate | high-budget majority prefers a different move |
| incumbent_proxy_disagreement-018 | 1200 | 7 | 2 | 2:4, 3:3 | 2 | 0.5714 | 0.5714 | 0.0302 | unstable_or_mixed | seed majorities remain mixed at this budget |
| incumbent_proxy_disagreement-018 | 2400 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 0.2029 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-018 | 5000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 0.3035 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-018 | 10000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 0.3561 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-018 | 30000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 0.3996 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-021 | 1200 | 7 | 2 | 3:7 | 3 | 1.0000 | 0.0000 | 0.0837 | supports_flip_candidate | high-budget majority prefers a different move |
| incumbent_proxy_disagreement-021 | 2400 | 7 | 2 | 0:1, 2:1, 3:5 | 3 | 0.7143 | 0.1429 | 0.0274 | supports_flip_candidate | high-budget majority prefers a different move |
| incumbent_proxy_disagreement-021 | 5000 | 7 | 2 | 2:6, 3:1 | 2 | 0.8571 | 0.8571 | 0.2015 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-021 | 10000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 0.3384 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-021 | 30000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 0.4356 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-023 | 1200 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 0.5397 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-023 | 2400 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 0.8420 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-023 | 5000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 0.9808 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-023 | 10000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.1469 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-023 | 30000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.3458 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-024 | 1200 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 0.3235 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-024 | 2400 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 0.4803 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-024 | 5000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 0.7500 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-024 | 10000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 0.9575 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-024 | 30000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.1439 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-032 | 1200 | 7 | 4 | 4:7 | 4 | 1.0000 | 1.0000 | 0.0877 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-032 | 2400 | 7 | 4 | 4:7 | 4 | 1.0000 | 1.0000 | 0.1721 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-032 | 5000 | 7 | 4 | 4:7 | 4 | 1.0000 | 1.0000 | 0.2008 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-032 | 10000 | 7 | 4 | 4:7 | 4 | 1.0000 | 1.0000 | 0.2304 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-032 | 30000 | 7 | 4 | 4:7 | 4 | 1.0000 | 1.0000 | 0.2465 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-033 | 1200 | 7 | 4 | 4:7 | 4 | 1.0000 | 1.0000 | 0.1087 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-033 | 2400 | 7 | 4 | 4:7 | 4 | 1.0000 | 1.0000 | 0.1189 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-033 | 5000 | 7 | 4 | 4:7 | 4 | 1.0000 | 1.0000 | 0.1626 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-033 | 10000 | 7 | 4 | 4:7 | 4 | 1.0000 | 1.0000 | 0.1902 | supports_active_reference | highest visit majority stays on the active reference |
| incumbent_proxy_disagreement-033 | 30000 | 7 | 4 | 4:7 | 4 | 1.0000 | 1.0000 | 0.2213 | supports_active_reference | highest visit majority stays on the active reference |

## 5. Child-afterstate adjudication

| row_id | child_from_move | budget | child_value_root_mean | child_value_root_std | child_selected_moves | root_perspective_value_delta_vs_reference | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-007 | 0 | 1200 | 0.0136 | 0.1404 | 2:4, 3:1 | 0.0000 | reference child baseline |
| incumbent_proxy_disagreement-007 | 3 | 1200 | 0.3109 | 0.0714 | 3:3, 4:2 | 0.2973 | negative means worse than active reference |
| incumbent_proxy_disagreement-007 | 0 | 2400 | -0.4880 | 0.0676 | 2:4, 3:1 | 0.0000 | reference child baseline |
| incumbent_proxy_disagreement-007 | 3 | 2400 | -0.1592 | 0.1476 | 3:4, 4:1 | 0.3288 | negative means worse than active reference |
| incumbent_proxy_disagreement-007 | 0 | 5000 | -0.7299 | 0.0356 | 2:4, 3:1 | 0.0000 | reference child baseline |
| incumbent_proxy_disagreement-007 | 3 | 5000 | -0.4757 | 0.1642 | 3:4, 4:1 | 0.2542 | negative means worse than active reference |
| incumbent_proxy_disagreement-009 | 0 | 1200 | -0.3330 | 0.0290 | 1:5 | 0.0000 | reference child baseline |
| incumbent_proxy_disagreement-009 | 3 | 1200 | -0.0653 | 0.1223 | 1:5 | 0.2677 | negative means worse than active reference |
| incumbent_proxy_disagreement-009 | 0 | 2400 | -0.5681 | 0.0256 | 1:5 | 0.0000 | reference child baseline |
| incumbent_proxy_disagreement-009 | 3 | 2400 | -0.4216 | 0.0423 | 1:5 | 0.1465 | negative means worse than active reference |
| incumbent_proxy_disagreement-009 | 0 | 5000 | -0.7494 | 0.0129 | 1:5 | 0.0000 | reference child baseline |
| incumbent_proxy_disagreement-009 | 3 | 5000 | -0.6223 | 0.0200 | 1:5 | 0.1271 | negative means worse than active reference |
| incumbent_proxy_disagreement-010 | 4 | 1200 | -0.2837 | 0.0454 | 0:1, 3:4 | 0.0000 | reference child baseline |
| incumbent_proxy_disagreement-010 | 0 | 1200 | -0.8218 | 0.0048 | 0:5 | -0.5381 | negative means worse than active reference |
| incumbent_proxy_disagreement-010 | 5 | 1200 | -0.2529 | 0.0339 | 3:3, 4:2 | 0.0308 | negative means worse than active reference |
| incumbent_proxy_disagreement-010 | 4 | 2400 | -0.5606 | 0.0627 | 0:1, 3:4 | 0.0000 | reference child baseline |
| incumbent_proxy_disagreement-010 | 0 | 2400 | -0.8332 | 0.0039 | 0:5 | -0.2726 | negative means worse than active reference |
| incumbent_proxy_disagreement-010 | 5 | 2400 | -0.3863 | 0.0812 | 3:3, 4:2 | 0.1743 | negative means worse than active reference |
| incumbent_proxy_disagreement-010 | 4 | 5000 | -0.7495 | 0.0297 | 0:1, 3:4 | 0.0000 | reference child baseline |
| incumbent_proxy_disagreement-010 | 0 | 5000 | -0.8633 | 0.0016 | 0:5 | -0.1138 | negative means worse than active reference |
| incumbent_proxy_disagreement-010 | 5 | 5000 | -0.5343 | 0.0970 | 3:4, 4:1 | 0.2152 | negative means worse than active reference |
| incumbent_proxy_disagreement-018 | 2 | 1200 | -0.4265 | 0.0170 | 4:5 | 0.0000 | reference child baseline |
| incumbent_proxy_disagreement-018 | 3 | 1200 | -0.1638 | 0.0524 | 3:4, 4:1 | 0.2627 | negative means worse than active reference |
| incumbent_proxy_disagreement-018 | 2 | 2400 | -0.4787 | 0.0136 | 3:4, 4:1 | 0.0000 | reference child baseline |
| incumbent_proxy_disagreement-018 | 3 | 2400 | -0.4261 | 0.0747 | 3:4, 4:1 | 0.0526 | negative means worse than active reference |
| incumbent_proxy_disagreement-018 | 2 | 5000 | -0.6873 | 0.0153 | 3:5 | 0.0000 | reference child baseline |
| incumbent_proxy_disagreement-018 | 3 | 5000 | -0.6131 | 0.1082 | 3:4, 4:1 | 0.0742 | negative means worse than active reference |
| incumbent_proxy_disagreement-021 | 2 | 1200 | -0.6437 | 0.0264 | 0:5 | 0.0000 | reference child baseline |
| incumbent_proxy_disagreement-021 | 5 | 1200 | -0.6567 | 0.0177 | 2:5 | -0.0130 | negative means worse than active reference |
| incumbent_proxy_disagreement-021 | 2 | 2400 | -0.7746 | 0.0071 | 0:5 | 0.0000 | reference child baseline |
| incumbent_proxy_disagreement-021 | 5 | 2400 | -0.7271 | 0.0065 | 2:5 | 0.0475 | negative means worse than active reference |
| incumbent_proxy_disagreement-021 | 2 | 5000 | -0.8469 | 0.0036 | 0:5 | 0.0000 | reference child baseline |
| incumbent_proxy_disagreement-021 | 5 | 5000 | -0.7862 | 0.0049 | 2:5 | 0.0607 | negative means worse than active reference |
| incumbent_proxy_disagreement-023 | 2 | 1200 | -0.9649 | 0.0053 | 2:5 | 0.0000 | reference child baseline |
| incumbent_proxy_disagreement-023 | 1 | 1200 | -0.5984 | 0.0284 | 3:5 | 0.3665 | negative means worse than active reference |
| incumbent_proxy_disagreement-023 | 2 | 2400 | -0.9452 | 0.0013 | 2:5 | 0.0000 | reference child baseline |
| incumbent_proxy_disagreement-023 | 1 | 2400 | -0.6146 | 0.0095 | 3:5 | 0.3306 | negative means worse than active reference |
| incumbent_proxy_disagreement-023 | 2 | 5000 | -0.9275 | 0.0021 | 2:5 | 0.0000 | reference child baseline |
| incumbent_proxy_disagreement-023 | 1 | 5000 | -0.7183 | 0.0189 | 2:1, 3:4 | 0.2092 | negative means worse than active reference |
| incumbent_proxy_disagreement-024 | 2 | 1200 | -0.6759 | 0.0117 | 4:5 | 0.0000 | reference child baseline |
| incumbent_proxy_disagreement-024 | 1 | 1200 | -0.5613 | 0.0130 | 2:1, 3:4 | 0.1146 | negative means worse than active reference |
| incumbent_proxy_disagreement-024 | 2 | 2400 | -0.7561 | 0.0031 | 4:5 | 0.0000 | reference child baseline |
| incumbent_proxy_disagreement-024 | 1 | 2400 | -0.5314 | 0.0080 | 2:2, 3:3 | 0.2247 | negative means worse than active reference |
| incumbent_proxy_disagreement-024 | 2 | 5000 | -0.8357 | 0.0020 | 4:5 | 0.0000 | reference child baseline |
| incumbent_proxy_disagreement-024 | 1 | 5000 | -0.6645 | 0.0345 | 3:5 | 0.1712 | negative means worse than active reference |
| incumbent_proxy_disagreement-032 | 4 | 1200 | -0.0717 | 0.0285 | 5:5 | 0.0000 | reference child baseline |
| incumbent_proxy_disagreement-032 | 1 | 1200 | -0.1604 | 0.0779 | 2:5 | -0.0887 | negative means worse than active reference |
| incumbent_proxy_disagreement-032 | 4 | 2400 | -0.3090 | 0.0538 | 1:1, 5:4 | 0.0000 | reference child baseline |
| incumbent_proxy_disagreement-032 | 1 | 2400 | -0.3255 | 0.0307 | 2:5 | -0.0165 | negative means worse than active reference |
| incumbent_proxy_disagreement-032 | 4 | 5000 | -0.5447 | 0.0334 | 1:1, 5:4 | 0.0000 | reference child baseline |
| incumbent_proxy_disagreement-032 | 1 | 5000 | -0.5070 | 0.0132 | 2:5 | 0.0377 | negative means worse than active reference |
| incumbent_proxy_disagreement-033 | 4 | 1200 | 0.0815 | 0.0177 | 4:5 | 0.0000 | reference child baseline |
| incumbent_proxy_disagreement-033 | 0 | 1200 | 0.0930 | 0.0852 | 0:3, 1:1, 2:1 | 0.0115 | negative means worse than active reference |
| incumbent_proxy_disagreement-033 | 4 | 2400 | -0.1308 | 0.0314 | 4:5 | 0.0000 | reference child baseline |
| incumbent_proxy_disagreement-033 | 0 | 2400 | -0.1607 | 0.0575 | 0:4, 1:1 | -0.0299 | negative means worse than active reference |
| incumbent_proxy_disagreement-033 | 4 | 5000 | -0.4255 | 0.0212 | 4:5 | 0.0000 | reference child baseline |
| incumbent_proxy_disagreement-033 | 0 | 5000 | -0.3597 | 0.0420 | 0:4, 1:1 | 0.0658 | negative means worse than active reference |

- Perspective conversion used the repo's existing convention: `+1` when `child.current_player == root.current_player`, otherwise sign-flipped with `-1` to convert child values back to the root player's perspective.

## 6. Tablebase availability

| row_id | state_label | remaining_seed_count | tablebase_available | tablebase_value_root | tablebase_preferred_move | agrees_with_classic_majority | agrees_with_active_reference | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-007 | root | 44 | false | - | - | - | - | not solvable under the repo threshold |
| incumbent_proxy_disagreement-007 | child_from_0 | 36 | false | - | - | - | - | not solvable under the repo threshold |
| incumbent_proxy_disagreement-007 | child_from_3 | 43 | false | - | - | - | - | not solvable under the repo threshold |
| incumbent_proxy_disagreement-009 | root | 44 | false | - | - | - | - | not solvable under the repo threshold |
| incumbent_proxy_disagreement-009 | child_from_0 | 44 | false | - | - | - | - | not solvable under the repo threshold |
| incumbent_proxy_disagreement-009 | child_from_3 | 43 | false | - | - | - | - | not solvable under the repo threshold |
| incumbent_proxy_disagreement-010 | root | 43 | false | - | - | - | - | not solvable under the repo threshold |
| incumbent_proxy_disagreement-010 | child_from_4 | 42 | false | - | - | - | - | not solvable under the repo threshold |
| incumbent_proxy_disagreement-010 | child_from_0 | 43 | false | - | - | - | - | not solvable under the repo threshold |
| incumbent_proxy_disagreement-010 | child_from_5 | 42 | false | - | - | - | - | not solvable under the repo threshold |
| incumbent_proxy_disagreement-018 | root | 43 | false | - | - | - | - | not solvable under the repo threshold |
| incumbent_proxy_disagreement-018 | child_from_2 | 43 | false | - | - | - | - | not solvable under the repo threshold |
| incumbent_proxy_disagreement-018 | child_from_3 | 42 | false | - | - | - | - | not solvable under the repo threshold |
| incumbent_proxy_disagreement-021 | root | 43 | false | - | - | - | - | not solvable under the repo threshold |
| incumbent_proxy_disagreement-021 | child_from_2 | 43 | false | - | - | - | - | not solvable under the repo threshold |
| incumbent_proxy_disagreement-021 | child_from_5 | 42 | false | - | - | - | - | not solvable under the repo threshold |
| incumbent_proxy_disagreement-023 | root | 42 | false | - | - | - | - | not solvable under the repo threshold |
| incumbent_proxy_disagreement-023 | child_from_2 | 41 | false | - | - | - | - | not solvable under the repo threshold |
| incumbent_proxy_disagreement-023 | child_from_1 | 41 | false | - | - | - | - | not solvable under the repo threshold |
| incumbent_proxy_disagreement-024 | root | 42 | false | - | - | - | - | not solvable under the repo threshold |
| incumbent_proxy_disagreement-024 | child_from_2 | 41 | false | - | - | - | - | not solvable under the repo threshold |
| incumbent_proxy_disagreement-024 | child_from_1 | 41 | false | - | - | - | - | not solvable under the repo threshold |
| incumbent_proxy_disagreement-032 | root | 42 | false | - | - | - | - | not solvable under the repo threshold |
| incumbent_proxy_disagreement-032 | child_from_4 | 41 | false | - | - | - | - | not solvable under the repo threshold |
| incumbent_proxy_disagreement-032 | child_from_1 | 42 | false | - | - | - | - | not solvable under the repo threshold |
| incumbent_proxy_disagreement-033 | root | 42 | false | - | - | - | - | not solvable under the repo threshold |
| incumbent_proxy_disagreement-033 | child_from_4 | 41 | false | - | - | - | - | not solvable under the repo threshold |
| incumbent_proxy_disagreement-033 | child_from_0 | 42 | false | - | - | - | - | not solvable under the repo threshold |

## 7. PUCT/artifact teacher comparison

| row_id | budget | active_reference_move | puct_selected_move | puct_reference_visit_share | puct_selected_visit_share | puct_agrees_with_classic_majority | puct_agrees_with_active_reference | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-007 | 384 | 0 | 5 | 0.0052 | 0.4219 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-007 | 1200 | 0 | 3 | 0.0025 | 0.5167 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-007 | 2400 | 0 | 3 | 0.0021 | 0.4967 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-007 | 5000 | 0 | 3 | 0.0020 | 0.7574 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-009 | 384 | 0 | 3 | 0.0026 | 0.8516 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-009 | 1200 | 0 | 3 | 0.0008 | 0.9500 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-009 | 2400 | 0 | 3 | 0.0004 | 0.9592 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-009 | 5000 | 0 | 3 | 0.0002 | 0.8984 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-010 | 384 | 4 | 5 | 0.0417 | 0.9453 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-010 | 1200 | 4 | 5 | 0.1183 | 0.8733 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-010 | 2400 | 4 | 4 | 0.5396 | 0.5396 | false | true | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-010 | 5000 | 4 | 4 | 0.7782 | 0.7782 | false | true | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-018 | 384 | 2 | 5 | 0.0182 | 0.5339 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-018 | 1200 | 2 | 3 | 0.0108 | 0.8142 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-018 | 2400 | 2 | 3 | 0.0079 | 0.9042 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-018 | 5000 | 2 | 3 | 0.0070 | 0.9500 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-021 | 384 | 2 | 5 | 0.0078 | 0.9167 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-021 | 1200 | 2 | 5 | 0.0042 | 0.9692 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-021 | 2400 | 2 | 5 | 0.0042 | 0.9808 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-021 | 5000 | 2 | 5 | 0.0034 | 0.7600 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-023 | 384 | 2 | 1 | 0.0130 | 0.9557 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-023 | 1200 | 2 | 1 | 0.0050 | 0.9792 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-023 | 2400 | 2 | 1 | 0.1154 | 0.7950 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-023 | 5000 | 2 | 1 | 0.0554 | 0.8634 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-024 | 384 | 2 | 1 | 0.0104 | 0.9479 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-024 | 1200 | 2 | 1 | 0.0067 | 0.9725 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-024 | 2400 | 2 | 1 | 0.0121 | 0.9742 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-024 | 5000 | 2 | 1 | 0.1802 | 0.7940 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-032 | 384 | 4 | 0 | 0.0859 | 0.3724 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-032 | 1200 | 4 | 1 | 0.0275 | 0.7533 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-032 | 2400 | 4 | 1 | 0.0146 | 0.8554 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-032 | 5000 | 4 | 1 | 0.0108 | 0.7566 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-033 | 384 | 4 | 0 | 0.2578 | 0.4271 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-033 | 1200 | 4 | 0 | 0.0825 | 0.8133 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-033 | 2400 | 4 | 0 | 0.0537 | 0.8692 | false | false | deterministic artifact PUCT with PR #44 audit settings |
| incumbent_proxy_disagreement-033 | 5000 | 4 | 0 | 0.0258 | 0.9348 | false | false | deterministic artifact PUCT with PR #44 audit settings |

## 8. Row decisions

| row_id | active_reference_move | adjudicated_decision | proposed_reference_move | proposed_unstable | evidence_summary | recommended_use | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-007 | 0 | puct_teacher_divergence | - | false | highest_budget_majority=0 fraction=1.0000; reference_fraction=1.0000; highest_puct_selected=3; child_reference_beats_majority=true; root_puct_agrees_reference=false | diagnostic-only until teacher/reference policy is chosen | ClassicMCTS supports the active reference while deterministic PUCT keeps selecting away |
| incumbent_proxy_disagreement-009 | 0 | puct_teacher_divergence | - | false | highest_budget_majority=0 fraction=1.0000; reference_fraction=1.0000; highest_puct_selected=3; child_reference_beats_majority=true; root_puct_agrees_reference=false | diagnostic-only until teacher/reference policy is chosen | ClassicMCTS supports the active reference while deterministic PUCT keeps selecting away |
| incumbent_proxy_disagreement-010 | 4 | reference_unstable | - | true | highest_budget_majority=0 fraction=1.0000; reference_fraction=0.0000; highest_puct_selected=4; child_reference_beats_majority=true; root_puct_agrees_reference=true | exclude from hard gates and training targets | seeds or budgets do not converge on one stable move |
| incumbent_proxy_disagreement-018 | 2 | puct_teacher_divergence | - | false | highest_budget_majority=2 fraction=1.0000; reference_fraction=1.0000; highest_puct_selected=3; child_reference_beats_majority=true; root_puct_agrees_reference=false | diagnostic-only until teacher/reference policy is chosen | ClassicMCTS supports the active reference while deterministic PUCT keeps selecting away |
| incumbent_proxy_disagreement-021 | 2 | puct_teacher_divergence | - | false | highest_budget_majority=2 fraction=1.0000; reference_fraction=1.0000; highest_puct_selected=5; child_reference_beats_majority=true; root_puct_agrees_reference=false | diagnostic-only until teacher/reference policy is chosen | ClassicMCTS supports the active reference while deterministic PUCT keeps selecting away |
| incumbent_proxy_disagreement-023 | 2 | puct_teacher_divergence | - | false | highest_budget_majority=2 fraction=1.0000; reference_fraction=1.0000; highest_puct_selected=1; child_reference_beats_majority=true; root_puct_agrees_reference=false | diagnostic-only until teacher/reference policy is chosen | ClassicMCTS supports the active reference while deterministic PUCT keeps selecting away |
| incumbent_proxy_disagreement-024 | 2 | puct_teacher_divergence | - | false | highest_budget_majority=2 fraction=1.0000; reference_fraction=1.0000; highest_puct_selected=1; child_reference_beats_majority=true; root_puct_agrees_reference=false | diagnostic-only until teacher/reference policy is chosen | ClassicMCTS supports the active reference while deterministic PUCT keeps selecting away |
| incumbent_proxy_disagreement-032 | 4 | puct_teacher_divergence | - | false | highest_budget_majority=4 fraction=1.0000; reference_fraction=1.0000; highest_puct_selected=1; child_reference_beats_majority=true; root_puct_agrees_reference=false | diagnostic-only until teacher/reference policy is chosen | ClassicMCTS supports the active reference while deterministic PUCT keeps selecting away |
| incumbent_proxy_disagreement-033 | 4 | puct_teacher_divergence | - | false | highest_budget_majority=4 fraction=1.0000; reference_fraction=1.0000; highest_puct_selected=0; child_reference_beats_majority=true; root_puct_agrees_reference=false | diagnostic-only until teacher/reference policy is chosen | ClassicMCTS supports the active reference while deterministic PUCT keeps selecting away |

## 9. Proposed non-mutating patch artifact

- Proposed review artifact: `/tmp/azlite_incumbent_proxy_residual_reference_adjudication/incumbent_proxy_residual_reference_review_patch_v1.json`.
- `do_not_auto_apply` is set for every proposed row.
- The active fixture was not edited by this run.

## 10. Projected clean mechanism buckets

| bucket | row_count | rows | train_target_eligible_count | risks | next_action |
| --- | --- | --- | --- | --- | --- |
| confirmed_value_head_miscalibration_candidates | 7 | incumbent_proxy_disagreement-003, incumbent_proxy_disagreement-012, incumbent_proxy_disagreement-020, incumbent_proxy_disagreement-022, incumbent_proxy_disagreement-025, incumbent_proxy_disagreement-027, incumbent_proxy_disagreement-035 | 7 | teacher-confirmed labels can still overfit if the set stays small | usable for small train-only value calibration if references are otherwise clean |
| confirmed_root_selection_pressure_candidates | 9 | incumbent_proxy_disagreement-007, incumbent_proxy_disagreement-009, incumbent_proxy_disagreement-014, incumbent_proxy_disagreement-018, incumbent_proxy_disagreement-021, incumbent_proxy_disagreement-023, incumbent_proxy_disagreement-024, incumbent_proxy_disagreement-032, incumbent_proxy_disagreement-033 | 0 | root-prior pressure may need search-stack fixes before training | diagnose root selection pressure separately from training |
| confirmed_puct_child_mismatch_candidates | 1 | incumbent_proxy_disagreement-011 | 0 | child search disagreement may indicate backup/search defects rather than label quality | diagnose child search mismatch before treating as training targets |
| reference_flip_candidates | 0 | - | 0 | fixture changes need explicit review before they are safe to use | review and apply only via a separate explicit fixture patch |
| unstable_or_excluded | 1 | incumbent_proxy_disagreement-010 | 0 | unstable labels would contaminate any hard target set | exclude from hard pass/fail gates and training targets |
| stable_controls | 4 | incumbent_proxy_disagreement-008, incumbent_proxy_disagreement-026, incumbent_proxy_disagreement-028, incumbent_proxy_disagreement-029 | 0 | controls must stay unchanged during any follow-up | preserve unchanged as regression checks |

- Projected family decision: `teacher_family_divergence`.
- Value-calibration rows still eligible after excluding or patching suspicious rows: `7`.
- Stable value-head remains the largest usable mechanism bucket: `true`.

## 11. Exactly one recommended next action

Recommendation: **decide which teacher should define references before training**
