# AlphaZero-lite Incumbent Proxy Classic Teacher Low-LR Non-Regression Lane Results

## 1. Context

- This run implements the exact next branch recommended by the interference attribution audit.
- It trains one low-LR Classic-only lane from the current checkpoint.
- No arena was run.
- No model was promoted.
- No active references were mutated.
- No PUCT or excluded rows were used as training targets.

## 2. Lane Definition

- Artifact: `/tmp/azlite_incumbent_proxy_teacher_interference_attribution/artifacts/classic_all_low_lr_lane.jsonl`.
- LR: `0.00025`.
- Epoch checkpoints: `1, 2, 4`.
- Attribution prerequisite status: `loaded` with classification `update_size_sensitive_interference`.

## 3. Strict Gate

- Gate requires at least one Classic target gain, zero PUCT lost selections at 384 and 1200, zero heavy PUCT regressions, and zero excluded drift.

| epoch | classic_gain_count | classic_strict_pass_count | puct_full_match_count | puct_row_count | excluded_drift_count | heavy_regression_count | gate_pass | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 2 | 5 | 7 | 0 | 2 | false | puct_lost_selection_384,puct_lost_selection_1200,heavy_puct_regression |
| 2 | 4 | 3 | 2 | 7 | 0 | 5 | false | puct_lost_selection_384,puct_lost_selection_1200,heavy_puct_regression |
| 4 | 4 | 4 | 1 | 7 | 1 | 6 | false | puct_lost_selection_384,puct_lost_selection_1200,heavy_puct_regression,excluded_drift |

## 4. Classic Results

| epoch | row_id | selected_384 | selected_1200 | active_reference_move | reference_visit_share_384 | reference_visit_share_1200 | improved_vs_current | strict_pass | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9375 | 0.9683 | true | true | reference_share_up_384 |
| 1 | incumbent_proxy_disagreement-014 | 2 | 2 | 4 | 0.0521 | 0.0167 | true | false | reference_share_up_384 |
| 1 | incumbent_proxy_disagreement-022 | 5 | 3 | 3 | 0.0078 | 0.4758 | true | false | selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| 1 | incumbent_proxy_disagreement-024 | 1 | 1 | 2 | 0.0234 | 0.0300 | true | false | reference_share_up_384,reference_share_up_1200 |
| 1 | incumbent_proxy_disagreement-025 | 4 | 4 | 2 | 0.0000 | 0.0000 | false | false | no_change_vs_current |
| 1 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.9375 | 0.9683 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| 2 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9688 | 0.9692 | true | true | reference_share_up_384 |
| 2 | incumbent_proxy_disagreement-014 | 5 | 2 | 4 | 0.2396 | 0.0850 | true | false | reference_share_up_384 |
| 2 | incumbent_proxy_disagreement-022 | 5 | 3 | 3 | 0.3229 | 0.5267 | true | false | selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| 2 | incumbent_proxy_disagreement-024 | 2 | 2 | 2 | 0.9349 | 0.7942 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| 2 | incumbent_proxy_disagreement-025 | 4 | 4 | 2 | 0.0000 | 0.0000 | false | false | no_change_vs_current |
| 2 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.6146 | 0.8700 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| 4 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9661 | 0.9867 | true | true | reference_share_up_384,reference_share_up_1200 |
| 4 | incumbent_proxy_disagreement-014 | 4 | 4 | 4 | 0.7969 | 0.6592 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| 4 | incumbent_proxy_disagreement-022 | 3 | 3 | 3 | 0.7500 | 0.7575 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| 4 | incumbent_proxy_disagreement-024 | 2 | 2 | 2 | 0.9635 | 0.8608 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| 4 | incumbent_proxy_disagreement-025 | 4 | 4 | 2 | 0.0000 | 0.0000 | false | false | no_change_vs_current |
| 4 | incumbent_proxy_disagreement-035 | 4 | 4 | 3 | 0.4740 | 0.3067 | true | false | reference_share_up_384,reference_share_up_1200 |

## 5. PUCT Results

| epoch | row_id | puct_preferred_move | selected_384 | selected_1200 | puct_preferred_visit_share_384 | puct_preferred_visit_share_1200 | selected_equals_puct_preferred_1200 | cross_teacher_regression | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | incumbent_proxy_disagreement-007 | 3 | 3 | 3 | 0.6172 | 0.6483 | true | false | no_heavy_puct_regression |
| 1 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.8724 | 0.9450 | true | false | no_heavy_puct_regression |
| 1 | incumbent_proxy_disagreement-021 | 5 | 5 | 5 | 0.8906 | 0.8642 | true | false | no_heavy_puct_regression |
| 1 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.9557 | 0.9625 | true | false | no_heavy_puct_regression |
| 1 | incumbent_proxy_disagreement-028 | 5 | 5 | 5 | 0.8672 | 0.7692 | true | false | no_heavy_puct_regression |
| 1 | incumbent_proxy_disagreement-032 | 1 | 2 | 0 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| 1 | incumbent_proxy_disagreement-033 | 0 | 0 | 1 | 0.4635 | 0.3625 | false | true | lost_puct_selection_1200,large_preferred_share_drop_1200 |
| 2 | incumbent_proxy_disagreement-007 | 3 | 2 | 2 | 0.0391 | 0.0142 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| 2 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.7656 | 0.9233 | true | false | no_heavy_puct_regression |
| 2 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.1615 | 0.0517 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| 2 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.8776 | 0.9592 | true | false | no_heavy_puct_regression |
| 2 | incumbent_proxy_disagreement-028 | 5 | 5 | 3 | 0.8490 | 0.3175 | false | true | lost_puct_selection_1200,large_preferred_share_drop_1200 |
| 2 | incumbent_proxy_disagreement-032 | 1 | 2 | 2 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| 2 | incumbent_proxy_disagreement-033 | 0 | 4 | 4 | 0.0286 | 0.0125 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| 4 | incumbent_proxy_disagreement-007 | 3 | 0 | 0 | 0.0599 | 0.0192 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| 4 | incumbent_proxy_disagreement-009 | 3 | 5 | 3 | 0.3828 | 0.7892 | true | true | lost_puct_selection_384,large_preferred_share_drop_384 |
| 4 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.0026 | 0.0550 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| 4 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.8802 | 0.9608 | true | false | no_heavy_puct_regression |
| 4 | incumbent_proxy_disagreement-028 | 5 | 3 | 3 | 0.2786 | 0.0892 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| 4 | incumbent_proxy_disagreement-032 | 1 | 2 | 2 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| 4 | incumbent_proxy_disagreement-033 | 0 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |

## 6. Decision

- Final classification: `low_lr_lane_gate_failed`.
- best low-LR checkpoint was epoch 1 but still failed: puct_lost_selection_384,puct_lost_selection_1200,heavy_puct_regression

## 7. Exactly One Recommended Next Action

Recommendation: **do not advance this lane; keep the branch diagnostic-only because even low LR still fails the strict PUCT non-regression gate.**
