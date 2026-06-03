# AlphaZero-lite Incumbent Proxy Teacher Interference Attribution Results

## 1. Context

- No arena was run.
- No MCTS1200 benchmark lane was run.
- No model was promoted.
- Active references stayed read-only at `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- PUCT rows were never used as Classic training targets.
- Excluded rows were never used as training targets or gates.
- Temporary artifacts and exports stayed under `/tmp/azlite_incumbent_proxy_teacher_interference_attribution/`.

## 2. Why PR #48 blocked production training

- PR #48 showed local Classic improvement but broad PUCT-bucket regression under tiny traces.
- Its final classification was `cross_teacher_interference`.
- This audit isolates which Classic row or training component causes that interference.

## 3. Input and bucket validation

- Validation status: `ok`.
- PUCT rows present in Classic artifact: `[]`.
- Excluded rows present in Classic artifact: `[]`.
- Conflicting duplicate canonical roles: `0`.

## 4. State-similarity analysis

| classic_row | nearest_puct_rows | distance | shared_legal_mask | shared_consequence_pattern | notes |
| --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-008 | incumbent_proxy_disagreement-007 (0.650791), incumbent_proxy_disagreement-009 (1.874601), incumbent_proxy_disagreement-021 (2.045927) | 0.650791 | 1.0000 | 0.8333 | nearest_puct_neighbors |
| incumbent_proxy_disagreement-014 | incumbent_proxy_disagreement-021 (1.372224), incumbent_proxy_disagreement-032 (1.44462), incumbent_proxy_disagreement-009 (1.842357) | 1.372224 | 0.6667 | 0.6667 | nearest_puct_neighbors |
| incumbent_proxy_disagreement-022 | incumbent_proxy_disagreement-021 (0.668608), incumbent_proxy_disagreement-009 (0.959617), incumbent_proxy_disagreement-026 (1.656692) | 0.668608 | 1.0000 | 1.0000 | nearest_puct_neighbors |
| incumbent_proxy_disagreement-024 | incumbent_proxy_disagreement-032 (2.891253), incumbent_proxy_disagreement-033 (3.1833), incumbent_proxy_disagreement-007 (3.498515) | 2.891253 | 1.0000 | 0.1667 | nearest_puct_neighbors |
| incumbent_proxy_disagreement-025 | incumbent_proxy_disagreement-032 (0.635953), incumbent_proxy_disagreement-033 (1.36624), incumbent_proxy_disagreement-021 (1.541778) | 0.635953 | 1.0000 | 1.0000 | nearest_puct_neighbors |
| incumbent_proxy_disagreement-035 | incumbent_proxy_disagreement-009 (3.029516), incumbent_proxy_disagreement-021 (3.402908), incumbent_proxy_disagreement-032 (3.450123) | 3.029516 | 0.6667 | 0.1667 | nearest_puct_neighbors |

## 5. Artifact variants

| artifact_name | included_classic_rows | excluded_classic_rows | row_count | status | notes |
| --- | --- | --- | --- | --- | --- |
| best_leave_one_out_low_lr | incumbent_proxy_disagreement-008,incumbent_proxy_disagreement-014,incumbent_proxy_disagreement-022,incumbent_proxy_disagreement-024,incumbent_proxy_disagreement-025 | incumbent_proxy_disagreement-035 | 5 | completed | same rows as classic_without_035 with lower learning rate |
| classic_all | incumbent_proxy_disagreement-008,incumbent_proxy_disagreement-014,incumbent_proxy_disagreement-022,incumbent_proxy_disagreement-024,incumbent_proxy_disagreement-025,incumbent_proxy_disagreement-035 |  | 6 | completed | same rows as PR #48 full Classic artifact |
| classic_all_low_lr | incumbent_proxy_disagreement-008,incumbent_proxy_disagreement-014,incumbent_proxy_disagreement-022,incumbent_proxy_disagreement-024,incumbent_proxy_disagreement-025,incumbent_proxy_disagreement-035 |  | 6 | completed | same rows as classic_all with lower learning rate |
| classic_all_policy_only | incumbent_proxy_disagreement-008,incumbent_proxy_disagreement-014,incumbent_proxy_disagreement-022,incumbent_proxy_disagreement-024,incumbent_proxy_disagreement-025,incumbent_proxy_disagreement-035 |  | 6 | completed | same rows as classic_all with value loss weight forced to 0.0 |
| classic_only_008 | incumbent_proxy_disagreement-008 | incumbent_proxy_disagreement-014,incumbent_proxy_disagreement-022,incumbent_proxy_disagreement-024,incumbent_proxy_disagreement-025,incumbent_proxy_disagreement-035 | 1 | completed | preservation control 008 only |
| classic_only_014 | incumbent_proxy_disagreement-014 | incumbent_proxy_disagreement-008,incumbent_proxy_disagreement-022,incumbent_proxy_disagreement-024,incumbent_proxy_disagreement-025,incumbent_proxy_disagreement-035 | 1 | completed | single-row Classic target 014 only |
| classic_only_022 | incumbent_proxy_disagreement-022 | incumbent_proxy_disagreement-008,incumbent_proxy_disagreement-014,incumbent_proxy_disagreement-024,incumbent_proxy_disagreement-025,incumbent_proxy_disagreement-035 | 1 | completed | single-row Classic target 022 only |
| classic_only_024 | incumbent_proxy_disagreement-024 | incumbent_proxy_disagreement-008,incumbent_proxy_disagreement-014,incumbent_proxy_disagreement-022,incumbent_proxy_disagreement-025,incumbent_proxy_disagreement-035 | 1 | completed | single-row Classic target 024 only |
| classic_only_025 | incumbent_proxy_disagreement-025 | incumbent_proxy_disagreement-008,incumbent_proxy_disagreement-014,incumbent_proxy_disagreement-022,incumbent_proxy_disagreement-024,incumbent_proxy_disagreement-035 | 1 | completed | single-row Classic target 025 only |
| classic_only_035 | incumbent_proxy_disagreement-035 | incumbent_proxy_disagreement-008,incumbent_proxy_disagreement-014,incumbent_proxy_disagreement-022,incumbent_proxy_disagreement-024,incumbent_proxy_disagreement-025 | 1 | completed | single-row Classic target 035 only |
| classic_without_008 | incumbent_proxy_disagreement-014,incumbent_proxy_disagreement-022,incumbent_proxy_disagreement-024,incumbent_proxy_disagreement-025,incumbent_proxy_disagreement-035 | incumbent_proxy_disagreement-008 | 5 | completed | leave out the preservation control 008 |
| classic_without_014 | incumbent_proxy_disagreement-008,incumbent_proxy_disagreement-022,incumbent_proxy_disagreement-024,incumbent_proxy_disagreement-025,incumbent_proxy_disagreement-035 | incumbent_proxy_disagreement-014 | 5 | completed | leave out Classic target row 014 |
| classic_without_022 | incumbent_proxy_disagreement-008,incumbent_proxy_disagreement-014,incumbent_proxy_disagreement-024,incumbent_proxy_disagreement-025,incumbent_proxy_disagreement-035 | incumbent_proxy_disagreement-022 | 5 | completed | leave out Classic target row 022 |
| classic_without_024 | incumbent_proxy_disagreement-008,incumbent_proxy_disagreement-014,incumbent_proxy_disagreement-022,incumbent_proxy_disagreement-025,incumbent_proxy_disagreement-035 | incumbent_proxy_disagreement-024 | 5 | completed | leave out Classic target row 024 |
| classic_without_025 | incumbent_proxy_disagreement-008,incumbent_proxy_disagreement-014,incumbent_proxy_disagreement-022,incumbent_proxy_disagreement-024,incumbent_proxy_disagreement-035 | incumbent_proxy_disagreement-025 | 5 | completed | leave out Classic target row 025 |
| classic_without_035 | incumbent_proxy_disagreement-008,incumbent_proxy_disagreement-014,incumbent_proxy_disagreement-022,incumbent_proxy_disagreement-024,incumbent_proxy_disagreement-025 | incumbent_proxy_disagreement-035 | 5 | completed | leave out Classic target row 035 |
| preservation_only | incumbent_proxy_disagreement-008 | incumbent_proxy_disagreement-014,incumbent_proxy_disagreement-022,incumbent_proxy_disagreement-024,incumbent_proxy_disagreement-025,incumbent_proxy_disagreement-035 | 1 | completed | preservation control only |
| target_only | incumbent_proxy_disagreement-014,incumbent_proxy_disagreement-022,incumbent_proxy_disagreement-024,incumbent_proxy_disagreement-025,incumbent_proxy_disagreement-035 | incumbent_proxy_disagreement-008 | 5 | completed | all five Classic target rows without preservation control |

## 6. Training traces

- Phase-2 epoch-4 reruns: `classic_all, classic_without_035, classic_only_014, target_only, preservation_only`.
- Learning-rate ablations: `classic_all_low_lr`, `best_leave_one_out_low_lr` with lr=0.00025.
- Value-only ablation skipped because train.py does not disable policy loss cleanly.

## 7. Classic improvement results

| artifact_name | epoch | row_id | selected_384 | selected_1200 | active_reference_move | reference_visit_share_384 | reference_visit_share_1200 | improved_vs_current | strict_pass | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_leave_one_out_low_lr | 1 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9427 | 0.9733 | true | true | reference_share_up_384,reference_share_up_1200 |
| best_leave_one_out_low_lr | 1 | incumbent_proxy_disagreement-014 | 2 | 1 | 4 | 0.0469 | 0.0150 | true | false | reference_share_up_384 |
| best_leave_one_out_low_lr | 1 | incumbent_proxy_disagreement-022 | 5 | 3 | 3 | 0.0078 | 0.5033 | true | false | selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| best_leave_one_out_low_lr | 1 | incumbent_proxy_disagreement-024 | 1 | 1 | 2 | 0.0130 | 0.0067 | true | false | reference_share_up_384 |
| best_leave_one_out_low_lr | 1 | incumbent_proxy_disagreement-025 | 4 | 2 | 2 | 0.3776 | 0.5542 | true | false | selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| best_leave_one_out_low_lr | 1 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.9375 | 0.8450 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| best_leave_one_out_low_lr | 2 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9714 | 0.9842 | true | true | reference_share_up_384,reference_share_up_1200 |
| best_leave_one_out_low_lr | 2 | incumbent_proxy_disagreement-014 | 2 | 2 | 4 | 0.0547 | 0.0175 | true | false | reference_share_up_384 |
| best_leave_one_out_low_lr | 2 | incumbent_proxy_disagreement-022 | 5 | 3 | 3 | 0.0078 | 0.4567 | true | false | selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| best_leave_one_out_low_lr | 2 | incumbent_proxy_disagreement-024 | 1 | 1 | 2 | 0.0078 | 0.0058 | false | false | no_change_vs_current |
| best_leave_one_out_low_lr | 2 | incumbent_proxy_disagreement-025 | 3 | 3 | 2 | 0.2760 | 0.2150 | true | false | reference_share_up_384 |
| best_leave_one_out_low_lr | 2 | incumbent_proxy_disagreement-035 | 4 | 4 | 3 | 0.0208 | 0.0208 | true | false | reference_share_up_384,reference_share_up_1200 |
| classic_all | 1 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9583 | 0.9767 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_all | 1 | incumbent_proxy_disagreement-014 | 4 | 4 | 4 | 0.4974 | 0.7433 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_all | 1 | incumbent_proxy_disagreement-022 | 2 | 2 | 3 | 0.1068 | 0.0342 | true | false | reference_share_up_384,reference_share_up_1200 |
| classic_all | 1 | incumbent_proxy_disagreement-024 | 2 | 2 | 2 | 0.7917 | 0.9100 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_all | 1 | incumbent_proxy_disagreement-025 | 4 | 4 | 2 | 0.0000 | 0.0000 | false | false | no_change_vs_current |
| classic_all | 1 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.9609 | 0.8733 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_all | 2 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9792 | 0.9875 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_all | 2 | incumbent_proxy_disagreement-014 | 4 | 4 | 4 | 0.8177 | 0.8983 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_all | 2 | incumbent_proxy_disagreement-022 | 2 | 2 | 3 | 0.0469 | 0.2158 | true | false | reference_share_up_384,reference_share_up_1200 |
| classic_all | 2 | incumbent_proxy_disagreement-024 | 2 | 2 | 2 | 0.9896 | 0.9867 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_all | 2 | incumbent_proxy_disagreement-025 | 3 | 3 | 2 | 0.0000 | 0.0000 | false | false | no_change_vs_current |
| classic_all | 2 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.9844 | 0.9942 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_all | 4 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9349 | 0.9767 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_all | 4 | incumbent_proxy_disagreement-014 | 4 | 4 | 4 | 0.9922 | 0.9967 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_all | 4 | incumbent_proxy_disagreement-022 | 3 | 3 | 3 | 0.9661 | 0.8917 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_all | 4 | incumbent_proxy_disagreement-024 | 2 | 2 | 2 | 0.9661 | 0.9767 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_all | 4 | incumbent_proxy_disagreement-025 | 4 | 4 | 2 | 0.0000 | 0.0000 | false | false | no_change_vs_current |
| classic_all | 4 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.9922 | 0.9950 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_all_low_lr | 1 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9505 | 0.9725 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_all_low_lr | 1 | incumbent_proxy_disagreement-014 | 2 | 2 | 4 | 0.0495 | 0.0167 | true | false | reference_share_up_384 |
| classic_all_low_lr | 1 | incumbent_proxy_disagreement-022 | 5 | 3 | 3 | 0.0078 | 0.5400 | true | false | selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_all_low_lr | 1 | incumbent_proxy_disagreement-024 | 1 | 1 | 2 | 0.0130 | 0.0067 | true | false | reference_share_up_384 |
| classic_all_low_lr | 1 | incumbent_proxy_disagreement-025 | 4 | 2 | 2 | 0.2318 | 0.7158 | true | false | selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_all_low_lr | 1 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.9375 | 0.9667 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_all_low_lr | 2 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9714 | 0.9717 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_all_low_lr | 2 | incumbent_proxy_disagreement-014 | 2 | 2 | 4 | 0.0729 | 0.0233 | true | false | reference_share_up_384 |
| classic_all_low_lr | 2 | incumbent_proxy_disagreement-022 | 5 | 3 | 3 | 0.0807 | 0.4667 | true | false | selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_all_low_lr | 2 | incumbent_proxy_disagreement-024 | 1 | 1 | 2 | 0.0260 | 0.0217 | true | false | reference_share_up_384,reference_share_up_1200 |
| classic_all_low_lr | 2 | incumbent_proxy_disagreement-025 | 3 | 3 | 2 | 0.3750 | 0.1525 | true | false | reference_share_up_384 |
| classic_all_low_lr | 2 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.9271 | 0.9600 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_all_policy_only | 1 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9766 | 0.9875 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_all_policy_only | 1 | incumbent_proxy_disagreement-014 | 1 | 1 | 4 | 0.0312 | 0.0100 | true | false | reference_share_up_384 |
| classic_all_policy_only | 1 | incumbent_proxy_disagreement-022 | 5 | 5 | 3 | 0.0365 | 0.3950 | true | false | reference_share_up_384,reference_share_up_1200 |
| classic_all_policy_only | 1 | incumbent_proxy_disagreement-024 | 4 | 2 | 2 | 0.0938 | 0.6842 | true | false | selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_all_policy_only | 1 | incumbent_proxy_disagreement-025 | 1 | 1 | 2 | 0.3542 | 0.1133 | true | false | reference_share_up_384 |
| classic_all_policy_only | 1 | incumbent_proxy_disagreement-035 | 4 | 4 | 3 | 0.1146 | 0.1467 | true | false | reference_share_up_384,reference_share_up_1200 |
| classic_all_policy_only | 2 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9688 | 0.9833 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_all_policy_only | 2 | incumbent_proxy_disagreement-014 | 2 | 1 | 4 | 0.2109 | 0.1658 | true | false | reference_share_up_384 |
| classic_all_policy_only | 2 | incumbent_proxy_disagreement-022 | 3 | 5 | 3 | 0.8984 | 0.4242 | true | false | selection_improved_384,reference_share_up_384,reference_share_up_1200 |
| classic_all_policy_only | 2 | incumbent_proxy_disagreement-024 | 2 | 2 | 2 | 0.9219 | 0.9517 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_all_policy_only | 2 | incumbent_proxy_disagreement-025 | 2 | 2 | 2 | 0.9089 | 0.6567 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_all_policy_only | 2 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.9922 | 0.9792 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_only_008 | 1 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9844 | 0.9933 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_only_008 | 1 | incumbent_proxy_disagreement-014 | 2 | 2 | 4 | 0.0026 | 0.0175 | false | false | no_change_vs_current |
| classic_only_008 | 1 | incumbent_proxy_disagreement-022 | 5 | 5 | 3 | 0.0026 | 0.0017 | false | false | no_change_vs_current |
| classic_only_008 | 1 | incumbent_proxy_disagreement-024 | 1 | 2 | 2 | 0.0599 | 0.6925 | true | false | selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_only_008 | 1 | incumbent_proxy_disagreement-025 | 4 | 4 | 2 | 0.0026 | 0.0008 | true | false | reference_share_up_384 |
| classic_only_008 | 1 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.8698 | 0.9275 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_only_008 | 2 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9714 | 0.9833 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_only_008 | 2 | incumbent_proxy_disagreement-014 | 2 | 1 | 4 | 0.0026 | 0.0008 | false | false | no_change_vs_current |
| classic_only_008 | 2 | incumbent_proxy_disagreement-022 | 5 | 2 | 3 | 0.0026 | 0.2292 | true | false | reference_share_up_1200 |
| classic_only_008 | 2 | incumbent_proxy_disagreement-024 | 1 | 1 | 2 | 0.0026 | 0.0008 | false | false | no_change_vs_current |
| classic_only_008 | 2 | incumbent_proxy_disagreement-025 | 2 | 2 | 2 | 0.9453 | 0.9692 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_only_008 | 2 | incumbent_proxy_disagreement-035 | 1 | 1 | 3 | 0.0130 | 0.0067 | true | false | reference_share_up_384,reference_share_up_1200 |
| classic_only_014 | 1 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9427 | 0.9725 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_only_014 | 1 | incumbent_proxy_disagreement-014 | 2 | 2 | 4 | 0.1979 | 0.2250 | true | false | reference_share_up_384 |
| classic_only_014 | 1 | incumbent_proxy_disagreement-022 | 3 | 5 | 3 | 0.5938 | 0.3925 | true | false | selection_improved_384,reference_share_up_384,reference_share_up_1200 |
| classic_only_014 | 1 | incumbent_proxy_disagreement-024 | 1 | 1 | 2 | 0.0026 | 0.0008 | false | false | no_change_vs_current |
| classic_only_014 | 1 | incumbent_proxy_disagreement-025 | 3 | 3 | 2 | 0.0000 | 0.0000 | false | false | no_change_vs_current |
| classic_only_014 | 1 | incumbent_proxy_disagreement-035 | 1 | 1 | 3 | 0.0260 | 0.0433 | true | false | reference_share_up_384,reference_share_up_1200 |
| classic_only_014 | 2 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.6536 | 0.8825 | false | true | no_change_vs_current |
| classic_only_014 | 2 | incumbent_proxy_disagreement-014 | 4 | 4 | 4 | 0.9948 | 0.6692 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_only_014 | 2 | incumbent_proxy_disagreement-022 | 3 | 3 | 3 | 0.8516 | 0.7500 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_only_014 | 2 | incumbent_proxy_disagreement-024 | 1 | 1 | 2 | 0.0026 | 0.0008 | false | false | no_change_vs_current |
| classic_only_014 | 2 | incumbent_proxy_disagreement-025 | 4 | 4 | 2 | 0.0000 | 0.0000 | false | false | no_change_vs_current |
| classic_only_014 | 2 | incumbent_proxy_disagreement-035 | 4 | 4 | 3 | 0.0443 | 0.0667 | true | false | reference_share_up_384,reference_share_up_1200 |
| classic_only_014 | 4 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.8203 | 0.9325 | false | true | no_change_vs_current |
| classic_only_014 | 4 | incumbent_proxy_disagreement-014 | 4 | 4 | 4 | 0.9922 | 0.9967 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_only_014 | 4 | incumbent_proxy_disagreement-022 | 2 | 1 | 3 | 0.2474 | 0.2733 | true | false | reference_share_up_384,reference_share_up_1200 |
| classic_only_014 | 4 | incumbent_proxy_disagreement-024 | 4 | 4 | 2 | 0.0026 | 0.0008 | false | false | no_change_vs_current |
| classic_only_014 | 4 | incumbent_proxy_disagreement-025 | 4 | 3 | 2 | 0.0000 | 0.0000 | false | false | no_change_vs_current |
| classic_only_014 | 4 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.9505 | 0.9425 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_only_022 | 1 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9297 | 0.9658 | false | true | no_change_vs_current |
| classic_only_022 | 1 | incumbent_proxy_disagreement-014 | 2 | 2 | 4 | 0.1042 | 0.0525 | true | false | reference_share_up_384 |
| classic_only_022 | 1 | incumbent_proxy_disagreement-022 | 3 | 3 | 3 | 0.9036 | 0.6525 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_only_022 | 1 | incumbent_proxy_disagreement-024 | 1 | 1 | 2 | 0.0026 | 0.0008 | false | false | no_change_vs_current |
| classic_only_022 | 1 | incumbent_proxy_disagreement-025 | 3 | 3 | 2 | 0.0000 | 0.0000 | false | false | no_change_vs_current |
| classic_only_022 | 1 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.8802 | 0.9233 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_only_022 | 2 | incumbent_proxy_disagreement-008 | 3 | 2 | 2 | 0.0104 | 0.6717 | false | false | regressed_from_prior_pass |
| classic_only_022 | 2 | incumbent_proxy_disagreement-014 | 0 | 0 | 4 | 0.3203 | 0.1942 | true | false | reference_share_up_384 |
| classic_only_022 | 2 | incumbent_proxy_disagreement-022 | 3 | 3 | 3 | 0.9896 | 0.9592 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_only_022 | 2 | incumbent_proxy_disagreement-024 | 3 | 3 | 2 | 0.0026 | 0.0008 | false | false | no_change_vs_current |
| classic_only_022 | 2 | incumbent_proxy_disagreement-025 | 3 | 3 | 2 | 0.0000 | 0.0000 | false | false | no_change_vs_current |
| classic_only_022 | 2 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.9635 | 0.9825 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_only_024 | 1 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9609 | 0.9767 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_only_024 | 1 | incumbent_proxy_disagreement-014 | 5 | 5 | 4 | 0.0026 | 0.0117 | false | false | no_change_vs_current |
| classic_only_024 | 1 | incumbent_proxy_disagreement-022 | 5 | 5 | 3 | 0.0026 | 0.0017 | false | false | no_change_vs_current |
| classic_only_024 | 1 | incumbent_proxy_disagreement-024 | 2 | 2 | 2 | 0.9740 | 0.9800 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_only_024 | 1 | incumbent_proxy_disagreement-025 | 4 | 4 | 2 | 0.0026 | 0.0008 | true | false | reference_share_up_384 |
| classic_only_024 | 1 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.9427 | 0.9700 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_only_024 | 2 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9922 | 0.9933 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_only_024 | 2 | incumbent_proxy_disagreement-014 | 5 | 5 | 4 | 0.0052 | 0.0150 | true | false | reference_share_up_384 |
| classic_only_024 | 2 | incumbent_proxy_disagreement-022 | 5 | 5 | 3 | 0.0026 | 0.0008 | false | false | no_change_vs_current |
| classic_only_024 | 2 | incumbent_proxy_disagreement-024 | 2 | 2 | 2 | 0.9922 | 0.9975 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_only_024 | 2 | incumbent_proxy_disagreement-025 | 4 | 4 | 2 | 0.0026 | 0.0008 | true | false | reference_share_up_384 |
| classic_only_024 | 2 | incumbent_proxy_disagreement-035 | 1 | 1 | 3 | 0.0781 | 0.0467 | true | false | reference_share_up_384,reference_share_up_1200 |
| classic_only_025 | 1 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9661 | 0.9858 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_only_025 | 1 | incumbent_proxy_disagreement-014 | 1 | 1 | 4 | 0.0026 | 0.0008 | false | false | no_change_vs_current |
| classic_only_025 | 1 | incumbent_proxy_disagreement-022 | 5 | 2 | 3 | 0.0026 | 0.0008 | false | false | no_change_vs_current |
| classic_only_025 | 1 | incumbent_proxy_disagreement-024 | 1 | 2 | 2 | 0.3203 | 0.6083 | true | false | selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_only_025 | 1 | incumbent_proxy_disagreement-025 | 2 | 2 | 2 | 0.4219 | 0.8092 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_only_025 | 1 | incumbent_proxy_disagreement-035 | 4 | 4 | 3 | 0.0078 | 0.0033 | false | false | no_change_vs_current |
| classic_only_025 | 2 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9635 | 0.9883 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_only_025 | 2 | incumbent_proxy_disagreement-014 | 1 | 1 | 4 | 0.0026 | 0.0008 | false | false | no_change_vs_current |
| classic_only_025 | 2 | incumbent_proxy_disagreement-022 | 5 | 2 | 3 | 0.0026 | 0.0008 | false | false | no_change_vs_current |
| classic_only_025 | 2 | incumbent_proxy_disagreement-024 | 1 | 1 | 2 | 0.0625 | 0.0242 | true | false | reference_share_up_384,reference_share_up_1200 |
| classic_only_025 | 2 | incumbent_proxy_disagreement-025 | 2 | 2 | 2 | 0.9922 | 0.5192 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_only_025 | 2 | incumbent_proxy_disagreement-035 | 1 | 1 | 3 | 0.0104 | 0.0042 | false | false | no_change_vs_current |
| classic_only_035 | 1 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.8984 | 0.9633 | false | true | no_change_vs_current |
| classic_only_035 | 1 | incumbent_proxy_disagreement-014 | 5 | 5 | 4 | 0.0417 | 0.0183 | true | false | reference_share_up_384 |
| classic_only_035 | 1 | incumbent_proxy_disagreement-022 | 5 | 3 | 3 | 0.0547 | 0.5992 | true | false | selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_only_035 | 1 | incumbent_proxy_disagreement-024 | 3 | 2 | 2 | 0.2448 | 0.6208 | true | false | selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_only_035 | 1 | incumbent_proxy_disagreement-025 | 4 | 4 | 2 | 0.0000 | 0.0000 | false | false | no_change_vs_current |
| classic_only_035 | 1 | incumbent_proxy_disagreement-035 | 4 | 4 | 3 | 0.0573 | 0.1508 | true | false | reference_share_up_384,reference_share_up_1200 |
| classic_only_035 | 2 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9661 | 0.9800 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_only_035 | 2 | incumbent_proxy_disagreement-014 | 5 | 5 | 4 | 0.0547 | 0.0208 | true | false | reference_share_up_384 |
| classic_only_035 | 2 | incumbent_proxy_disagreement-022 | 5 | 5 | 3 | 0.0286 | 0.0608 | true | false | reference_share_up_384,reference_share_up_1200 |
| classic_only_035 | 2 | incumbent_proxy_disagreement-024 | 2 | 2 | 2 | 0.6979 | 0.8308 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_only_035 | 2 | incumbent_proxy_disagreement-025 | 3 | 3 | 2 | 0.0000 | 0.0000 | false | false | no_change_vs_current |
| classic_only_035 | 2 | incumbent_proxy_disagreement-035 | 3 | 4 | 3 | 0.5182 | 0.1658 | true | false | selection_improved_384,reference_share_up_384,reference_share_up_1200 |
| classic_without_008 | 1 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9583 | 0.9817 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_without_008 | 1 | incumbent_proxy_disagreement-014 | 4 | 4 | 4 | 0.8385 | 0.9475 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_without_008 | 1 | incumbent_proxy_disagreement-022 | 2 | 2 | 3 | 0.0286 | 0.1817 | true | false | reference_share_up_384,reference_share_up_1200 |
| classic_without_008 | 1 | incumbent_proxy_disagreement-024 | 2 | 2 | 2 | 0.5469 | 0.6867 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_without_008 | 1 | incumbent_proxy_disagreement-025 | 3 | 4 | 2 | 0.0000 | 0.0000 | false | false | no_change_vs_current |
| classic_without_008 | 1 | incumbent_proxy_disagreement-035 | 3 | 4 | 3 | 0.9557 | 0.3908 | true | false | selection_improved_384,reference_share_up_384,reference_share_up_1200 |
| classic_without_008 | 2 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9557 | 0.9725 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_without_008 | 2 | incumbent_proxy_disagreement-014 | 4 | 4 | 4 | 0.8490 | 0.9217 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_without_008 | 2 | incumbent_proxy_disagreement-022 | 3 | 3 | 3 | 0.9271 | 0.9308 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_without_008 | 2 | incumbent_proxy_disagreement-024 | 2 | 2 | 2 | 0.9740 | 0.9833 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_without_008 | 2 | incumbent_proxy_disagreement-025 | 4 | 4 | 2 | 0.0000 | 0.0000 | false | false | no_change_vs_current |
| classic_without_008 | 2 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.9714 | 0.5908 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_without_014 | 1 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9714 | 0.9883 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_without_014 | 1 | incumbent_proxy_disagreement-014 | 5 | 5 | 4 | 0.0495 | 0.0558 | true | false | reference_share_up_384 |
| classic_without_014 | 1 | incumbent_proxy_disagreement-022 | 5 | 3 | 3 | 0.0443 | 0.5658 | true | false | selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_without_014 | 1 | incumbent_proxy_disagreement-024 | 2 | 2 | 2 | 0.9635 | 0.9675 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_without_014 | 1 | incumbent_proxy_disagreement-025 | 4 | 4 | 2 | 0.0000 | 0.0008 | false | false | no_change_vs_current |
| classic_without_014 | 1 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.7630 | 0.8567 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_without_014 | 2 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9818 | 0.9875 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_without_014 | 2 | incumbent_proxy_disagreement-014 | 2 | 5 | 4 | 0.2214 | 0.1058 | true | false | reference_share_up_384 |
| classic_without_014 | 2 | incumbent_proxy_disagreement-022 | 3 | 3 | 3 | 0.6901 | 0.7042 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_without_014 | 2 | incumbent_proxy_disagreement-024 | 2 | 2 | 2 | 0.9818 | 0.9858 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_without_014 | 2 | incumbent_proxy_disagreement-025 | 4 | 4 | 2 | 0.0026 | 0.0008 | true | false | reference_share_up_384 |
| classic_without_014 | 2 | incumbent_proxy_disagreement-035 | 3 | 4 | 3 | 0.9922 | 0.3500 | true | false | selection_improved_384,reference_share_up_384,reference_share_up_1200 |
| classic_without_022 | 1 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9583 | 0.9825 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_without_022 | 1 | incumbent_proxy_disagreement-014 | 4 | 4 | 4 | 0.8203 | 0.7300 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_without_022 | 1 | incumbent_proxy_disagreement-022 | 2 | 5 | 3 | 0.0312 | 0.0100 | true | false | reference_share_up_384,reference_share_up_1200 |
| classic_without_022 | 1 | incumbent_proxy_disagreement-024 | 2 | 2 | 2 | 0.9609 | 0.9633 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_without_022 | 1 | incumbent_proxy_disagreement-025 | 4 | 4 | 2 | 0.0000 | 0.0000 | false | false | no_change_vs_current |
| classic_without_022 | 1 | incumbent_proxy_disagreement-035 | 3 | 1 | 3 | 0.7031 | 0.2775 | true | false | selection_improved_384,reference_share_up_384,reference_share_up_1200 |
| classic_without_022 | 2 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9922 | 0.9900 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_without_022 | 2 | incumbent_proxy_disagreement-014 | 4 | 4 | 4 | 0.8099 | 0.9242 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_without_022 | 2 | incumbent_proxy_disagreement-022 | 2 | 2 | 3 | 0.0078 | 0.0075 | true | false | reference_share_up_384,reference_share_up_1200 |
| classic_without_022 | 2 | incumbent_proxy_disagreement-024 | 2 | 2 | 2 | 0.9948 | 0.9967 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_without_022 | 2 | incumbent_proxy_disagreement-025 | 4 | 4 | 2 | 0.0000 | 0.0008 | false | false | no_change_vs_current |
| classic_without_022 | 2 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.9844 | 0.9917 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_without_024 | 1 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9271 | 0.9575 | false | true | no_change_vs_current |
| classic_without_024 | 1 | incumbent_proxy_disagreement-014 | 2 | 1 | 4 | 0.3047 | 0.1342 | true | false | reference_share_up_384 |
| classic_without_024 | 1 | incumbent_proxy_disagreement-022 | 3 | 3 | 3 | 0.6172 | 0.6992 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_without_024 | 1 | incumbent_proxy_disagreement-024 | 3 | 3 | 2 | 0.0026 | 0.0033 | false | false | no_change_vs_current |
| classic_without_024 | 1 | incumbent_proxy_disagreement-025 | 3 | 3 | 2 | 0.0000 | 0.0000 | false | false | no_change_vs_current |
| classic_without_024 | 1 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.9740 | 0.9883 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_without_024 | 2 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9557 | 0.9650 | true | true | reference_share_up_384 |
| classic_without_024 | 2 | incumbent_proxy_disagreement-014 | 4 | 2 | 4 | 0.8646 | 0.2883 | true | false | selection_improved_384,reference_share_up_384 |
| classic_without_024 | 2 | incumbent_proxy_disagreement-022 | 2 | 2 | 3 | 0.4297 | 0.2283 | true | false | reference_share_up_384,reference_share_up_1200 |
| classic_without_024 | 2 | incumbent_proxy_disagreement-024 | 3 | 3 | 2 | 0.0599 | 0.0250 | true | false | reference_share_up_384,reference_share_up_1200 |
| classic_without_024 | 2 | incumbent_proxy_disagreement-025 | 3 | 3 | 2 | 0.0000 | 0.0000 | false | false | no_change_vs_current |
| classic_without_024 | 2 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.9922 | 0.9950 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_without_025 | 1 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9271 | 0.9575 | false | true | no_change_vs_current |
| classic_without_025 | 1 | incumbent_proxy_disagreement-014 | 2 | 1 | 4 | 0.3047 | 0.1342 | true | false | reference_share_up_384 |
| classic_without_025 | 1 | incumbent_proxy_disagreement-022 | 3 | 3 | 3 | 0.6172 | 0.6992 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_without_025 | 1 | incumbent_proxy_disagreement-024 | 3 | 3 | 2 | 0.0026 | 0.0033 | false | false | no_change_vs_current |
| classic_without_025 | 1 | incumbent_proxy_disagreement-025 | 3 | 3 | 2 | 0.0000 | 0.0000 | false | false | no_change_vs_current |
| classic_without_025 | 1 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.9740 | 0.9883 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_without_025 | 2 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9557 | 0.9650 | true | true | reference_share_up_384 |
| classic_without_025 | 2 | incumbent_proxy_disagreement-014 | 4 | 2 | 4 | 0.8646 | 0.2883 | true | false | selection_improved_384,reference_share_up_384 |
| classic_without_025 | 2 | incumbent_proxy_disagreement-022 | 2 | 2 | 3 | 0.4297 | 0.2283 | true | false | reference_share_up_384,reference_share_up_1200 |
| classic_without_025 | 2 | incumbent_proxy_disagreement-024 | 3 | 3 | 2 | 0.0599 | 0.0250 | true | false | reference_share_up_384,reference_share_up_1200 |
| classic_without_025 | 2 | incumbent_proxy_disagreement-025 | 3 | 3 | 2 | 0.0000 | 0.0000 | false | false | no_change_vs_current |
| classic_without_025 | 2 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.9922 | 0.9950 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_without_035 | 1 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9635 | 0.9758 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_without_035 | 1 | incumbent_proxy_disagreement-014 | 2 | 2 | 4 | 0.2214 | 0.2842 | true | false | reference_share_up_384 |
| classic_without_035 | 1 | incumbent_proxy_disagreement-022 | 3 | 3 | 3 | 0.6693 | 0.7308 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_without_035 | 1 | incumbent_proxy_disagreement-024 | 2 | 2 | 2 | 0.9375 | 0.9483 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_without_035 | 1 | incumbent_proxy_disagreement-025 | 3 | 3 | 2 | 0.0000 | 0.0000 | false | false | no_change_vs_current |
| classic_without_035 | 1 | incumbent_proxy_disagreement-035 | 1 | 1 | 3 | 0.0521 | 0.2483 | true | false | reference_share_up_384,reference_share_up_1200 |
| classic_without_035 | 2 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9818 | 0.9883 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_without_035 | 2 | incumbent_proxy_disagreement-014 | 4 | 4 | 4 | 0.5495 | 0.8558 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_without_035 | 2 | incumbent_proxy_disagreement-022 | 2 | 3 | 3 | 0.2370 | 0.6175 | true | false | selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_without_035 | 2 | incumbent_proxy_disagreement-024 | 2 | 2 | 2 | 0.9896 | 0.9967 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_without_035 | 2 | incumbent_proxy_disagreement-025 | 4 | 3 | 2 | 0.0000 | 0.0000 | false | false | no_change_vs_current |
| classic_without_035 | 2 | incumbent_proxy_disagreement-035 | 1 | 1 | 3 | 0.2734 | 0.0875 | true | false | reference_share_up_384,reference_share_up_1200 |
| classic_without_035 | 4 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9271 | 0.9758 | true | true | reference_share_up_1200 |
| classic_without_035 | 4 | incumbent_proxy_disagreement-014 | 4 | 4 | 4 | 0.9922 | 0.9967 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_without_035 | 4 | incumbent_proxy_disagreement-022 | 3 | 3 | 3 | 0.9896 | 0.5342 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_without_035 | 4 | incumbent_proxy_disagreement-024 | 2 | 2 | 2 | 0.9688 | 0.9800 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_without_035 | 4 | incumbent_proxy_disagreement-025 | 4 | 4 | 2 | 0.0000 | 0.0000 | false | false | no_change_vs_current |
| classic_without_035 | 4 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.9922 | 0.9725 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| preservation_only | 1 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9844 | 0.9933 | true | true | reference_share_up_384,reference_share_up_1200 |
| preservation_only | 1 | incumbent_proxy_disagreement-014 | 2 | 2 | 4 | 0.0026 | 0.0175 | false | false | no_change_vs_current |
| preservation_only | 1 | incumbent_proxy_disagreement-022 | 5 | 5 | 3 | 0.0026 | 0.0017 | false | false | no_change_vs_current |
| preservation_only | 1 | incumbent_proxy_disagreement-024 | 1 | 2 | 2 | 0.0599 | 0.6925 | true | false | selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| preservation_only | 1 | incumbent_proxy_disagreement-025 | 4 | 4 | 2 | 0.0026 | 0.0008 | true | false | reference_share_up_384 |
| preservation_only | 1 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.8698 | 0.9275 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| preservation_only | 2 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9714 | 0.9833 | true | true | reference_share_up_384,reference_share_up_1200 |
| preservation_only | 2 | incumbent_proxy_disagreement-014 | 2 | 1 | 4 | 0.0026 | 0.0008 | false | false | no_change_vs_current |
| preservation_only | 2 | incumbent_proxy_disagreement-022 | 5 | 2 | 3 | 0.0026 | 0.2292 | true | false | reference_share_up_1200 |
| preservation_only | 2 | incumbent_proxy_disagreement-024 | 1 | 1 | 2 | 0.0026 | 0.0008 | false | false | no_change_vs_current |
| preservation_only | 2 | incumbent_proxy_disagreement-025 | 2 | 2 | 2 | 0.9453 | 0.9692 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| preservation_only | 2 | incumbent_proxy_disagreement-035 | 1 | 1 | 3 | 0.0130 | 0.0067 | true | false | reference_share_up_384,reference_share_up_1200 |
| preservation_only | 4 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9792 | 0.9767 | true | true | reference_share_up_384,reference_share_up_1200 |
| preservation_only | 4 | incumbent_proxy_disagreement-014 | 5 | 5 | 4 | 0.0026 | 0.0008 | false | false | no_change_vs_current |
| preservation_only | 4 | incumbent_proxy_disagreement-022 | 5 | 5 | 3 | 0.0026 | 0.0008 | false | false | no_change_vs_current |
| preservation_only | 4 | incumbent_proxy_disagreement-024 | 0 | 0 | 2 | 0.0026 | 0.4317 | true | false | reference_share_up_1200 |
| preservation_only | 4 | incumbent_proxy_disagreement-025 | 2 | 2 | 2 | 0.9922 | 0.9975 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| preservation_only | 4 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.6016 | 0.6942 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| target_only | 1 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9583 | 0.9817 | true | true | reference_share_up_384,reference_share_up_1200 |
| target_only | 1 | incumbent_proxy_disagreement-014 | 4 | 4 | 4 | 0.8385 | 0.9475 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| target_only | 1 | incumbent_proxy_disagreement-022 | 2 | 2 | 3 | 0.0286 | 0.1817 | true | false | reference_share_up_384,reference_share_up_1200 |
| target_only | 1 | incumbent_proxy_disagreement-024 | 2 | 2 | 2 | 0.5469 | 0.6867 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| target_only | 1 | incumbent_proxy_disagreement-025 | 3 | 4 | 2 | 0.0000 | 0.0000 | false | false | no_change_vs_current |
| target_only | 1 | incumbent_proxy_disagreement-035 | 3 | 4 | 3 | 0.9557 | 0.3908 | true | false | selection_improved_384,reference_share_up_384,reference_share_up_1200 |
| target_only | 2 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9557 | 0.9725 | true | true | reference_share_up_384,reference_share_up_1200 |
| target_only | 2 | incumbent_proxy_disagreement-014 | 4 | 4 | 4 | 0.8490 | 0.9217 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| target_only | 2 | incumbent_proxy_disagreement-022 | 3 | 3 | 3 | 0.9271 | 0.9308 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| target_only | 2 | incumbent_proxy_disagreement-024 | 2 | 2 | 2 | 0.9740 | 0.9833 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| target_only | 2 | incumbent_proxy_disagreement-025 | 4 | 4 | 2 | 0.0000 | 0.0000 | false | false | no_change_vs_current |
| target_only | 2 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.9714 | 0.5908 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| target_only | 4 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.7995 | 0.9267 | false | true | no_change_vs_current |
| target_only | 4 | incumbent_proxy_disagreement-014 | 4 | 4 | 4 | 0.9896 | 0.9883 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| target_only | 4 | incumbent_proxy_disagreement-022 | 3 | 2 | 3 | 0.9219 | 0.4433 | true | false | selection_improved_384,reference_share_up_384,reference_share_up_1200 |
| target_only | 4 | incumbent_proxy_disagreement-024 | 2 | 2 | 2 | 0.9818 | 0.9925 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| target_only | 4 | incumbent_proxy_disagreement-025 | 4 | 4 | 2 | 0.0000 | 0.0000 | false | false | no_change_vs_current |
| target_only | 4 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.9922 | 0.9908 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |

## 8. PUCT interference results

| artifact_name | epoch | row_id | puct_preferred_move | selected_384 | selected_1200 | puct_preferred_visit_share_384 | puct_preferred_visit_share_1200 | selected_equals_puct_preferred_1200 | cross_teacher_regression | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_leave_one_out_low_lr | 1 | incumbent_proxy_disagreement-007 | 3 | 5 | 2 | 0.4115 | 0.1433 | false | true | lost_puct_selection_1200,large_preferred_share_drop_1200 |
| best_leave_one_out_low_lr | 1 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.8438 | 0.9458 | true | false | no_heavy_puct_regression |
| best_leave_one_out_low_lr | 1 | incumbent_proxy_disagreement-021 | 5 | 5 | 3 | 0.8802 | 0.3467 | false | true | lost_puct_selection_1200,large_preferred_share_drop_1200 |
| best_leave_one_out_low_lr | 1 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.9141 | 0.9708 | true | false | no_heavy_puct_regression |
| best_leave_one_out_low_lr | 1 | incumbent_proxy_disagreement-028 | 5 | 5 | 5 | 0.4714 | 0.7750 | true | true | large_preferred_share_drop_384 |
| best_leave_one_out_low_lr | 1 | incumbent_proxy_disagreement-032 | 1 | 1 | 1 | 0.5078 | 0.3967 | true | true | large_preferred_share_drop_1200 |
| best_leave_one_out_low_lr | 1 | incumbent_proxy_disagreement-033 | 0 | 1 | 1 | 0.1615 | 0.0517 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| best_leave_one_out_low_lr | 2 | incumbent_proxy_disagreement-007 | 3 | 2 | 3 | 0.3125 | 0.7700 | true | false | no_heavy_puct_regression |
| best_leave_one_out_low_lr | 2 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.7370 | 0.9158 | true | false | no_heavy_puct_regression |
| best_leave_one_out_low_lr | 2 | incumbent_proxy_disagreement-021 | 5 | 3 | 5 | 0.4271 | 0.7100 | true | true | lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| best_leave_one_out_low_lr | 2 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.9167 | 0.9725 | true | false | no_heavy_puct_regression |
| best_leave_one_out_low_lr | 2 | incumbent_proxy_disagreement-028 | 5 | 5 | 5 | 0.8047 | 0.9142 | true | false | no_heavy_puct_regression |
| best_leave_one_out_low_lr | 2 | incumbent_proxy_disagreement-032 | 1 | 1 | 3 | 0.4479 | 0.1433 | false | true | lost_puct_selection_1200,large_preferred_share_drop_1200 |
| best_leave_one_out_low_lr | 2 | incumbent_proxy_disagreement-033 | 0 | 0 | 0 | 0.4609 | 0.7575 | true | false | no_heavy_puct_regression |
| classic_all | 1 | incumbent_proxy_disagreement-007 | 3 | 0 | 0 | 0.1016 | 0.0325 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_all | 1 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.7396 | 0.8500 | true | false | no_heavy_puct_regression |
| classic_all | 1 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_all | 1 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.8359 | 0.8550 | true | false | no_heavy_puct_regression |
| classic_all | 1 | incumbent_proxy_disagreement-028 | 5 | 5 | 5 | 0.4505 | 0.7875 | true | true | large_preferred_share_drop_384 |
| classic_all | 1 | incumbent_proxy_disagreement-032 | 1 | 3 | 2 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_all | 1 | incumbent_proxy_disagreement-033 | 0 | 4 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_all | 2 | incumbent_proxy_disagreement-007 | 3 | 2 | 2 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_all | 2 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.9922 | 0.7950 | true | false | no_heavy_puct_regression |
| classic_all | 2 | incumbent_proxy_disagreement-021 | 5 | 2 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_all | 2 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.7865 | 0.9308 | true | true | large_preferred_share_drop_384 |
| classic_all | 2 | incumbent_proxy_disagreement-028 | 5 | 4 | 4 | 0.0026 | 0.0500 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_all | 2 | incumbent_proxy_disagreement-032 | 1 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_all | 2 | incumbent_proxy_disagreement-033 | 0 | 3 | 4 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_all | 4 | incumbent_proxy_disagreement-007 | 3 | 2 | 2 | 0.0990 | 0.0333 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_all | 4 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.9922 | 0.9975 | true | false | no_heavy_puct_regression |
| classic_all | 4 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_all | 4 | incumbent_proxy_disagreement-026 | 5 | 5 | 0 | 0.5078 | 0.3708 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_all | 4 | incumbent_proxy_disagreement-028 | 5 | 4 | 4 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_all | 4 | incumbent_proxy_disagreement-032 | 1 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_all | 4 | incumbent_proxy_disagreement-033 | 0 | 3 | 3 | 0.0000 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_all_low_lr | 1 | incumbent_proxy_disagreement-007 | 3 | 5 | 5 | 0.4115 | 0.1517 | false | true | lost_puct_selection_1200,large_preferred_share_drop_1200 |
| classic_all_low_lr | 1 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.8672 | 0.9450 | true | false | no_heavy_puct_regression |
| classic_all_low_lr | 1 | incumbent_proxy_disagreement-021 | 5 | 5 | 3 | 0.7917 | 0.2550 | false | true | lost_puct_selection_1200,large_preferred_share_drop_1200 |
| classic_all_low_lr | 1 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.9219 | 0.9708 | true | false | no_heavy_puct_regression |
| classic_all_low_lr | 1 | incumbent_proxy_disagreement-028 | 5 | 5 | 5 | 0.8724 | 0.9217 | true | false | no_heavy_puct_regression |
| classic_all_low_lr | 1 | incumbent_proxy_disagreement-032 | 1 | 2 | 1 | 0.1510 | 0.5858 | true | false | no_heavy_puct_regression |
| classic_all_low_lr | 1 | incumbent_proxy_disagreement-033 | 0 | 1 | 1 | 0.1953 | 0.0625 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_all_low_lr | 2 | incumbent_proxy_disagreement-007 | 3 | 2 | 3 | 0.2943 | 0.7300 | true | false | no_heavy_puct_regression |
| classic_all_low_lr | 2 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.7969 | 0.9117 | true | false | no_heavy_puct_regression |
| classic_all_low_lr | 2 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.1354 | 0.0442 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_all_low_lr | 2 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.9010 | 0.9650 | true | false | no_heavy_puct_regression |
| classic_all_low_lr | 2 | incumbent_proxy_disagreement-028 | 5 | 5 | 5 | 0.6979 | 0.8783 | true | true | large_preferred_share_drop_384 |
| classic_all_low_lr | 2 | incumbent_proxy_disagreement-032 | 1 | 1 | 0 | 0.3932 | 0.1883 | false | true | lost_puct_selection_1200,large_preferred_share_drop_1200 |
| classic_all_low_lr | 2 | incumbent_proxy_disagreement-033 | 0 | 0 | 0 | 0.4557 | 0.8142 | true | false | no_heavy_puct_regression |
| classic_all_policy_only | 1 | incumbent_proxy_disagreement-007 | 3 | 3 | 3 | 0.8802 | 0.7350 | true | false | no_heavy_puct_regression |
| classic_all_policy_only | 1 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.5000 | 0.8025 | true | true | large_preferred_share_drop_384 |
| classic_all_policy_only | 1 | incumbent_proxy_disagreement-021 | 5 | 5 | 3 | 0.4401 | 0.1575 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_all_policy_only | 1 | incumbent_proxy_disagreement-026 | 5 | 2 | 5 | 0.3568 | 0.7392 | true | true | lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_all_policy_only | 1 | incumbent_proxy_disagreement-028 | 5 | 5 | 5 | 0.6693 | 0.8417 | true | true | large_preferred_share_drop_384 |
| classic_all_policy_only | 1 | incumbent_proxy_disagreement-032 | 1 | 4 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_all_policy_only | 1 | incumbent_proxy_disagreement-033 | 0 | 3 | 3 | 0.0885 | 0.0608 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_all_policy_only | 2 | incumbent_proxy_disagreement-007 | 3 | 0 | 0 | 0.2943 | 0.0942 | false | true | lost_puct_selection_1200,large_preferred_share_drop_1200 |
| classic_all_policy_only | 2 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.9635 | 0.7442 | true | true | large_preferred_share_drop_1200 |
| classic_all_policy_only | 2 | incumbent_proxy_disagreement-021 | 5 | 2 | 1 | 0.2031 | 0.1875 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_all_policy_only | 2 | incumbent_proxy_disagreement-026 | 5 | 2 | 2 | 0.0078 | 0.0042 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_all_policy_only | 2 | incumbent_proxy_disagreement-028 | 5 | 2 | 2 | 0.0234 | 0.0075 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_all_policy_only | 2 | incumbent_proxy_disagreement-032 | 1 | 2 | 4 | 0.0026 | 0.2558 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_all_policy_only | 2 | incumbent_proxy_disagreement-033 | 0 | 2 | 2 | 0.0026 | 0.1250 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_008 | 1 | incumbent_proxy_disagreement-007 | 3 | 0 | 0 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_008 | 1 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.5182 | 0.8408 | true | true | large_preferred_share_drop_384 |
| classic_only_008 | 1 | incumbent_proxy_disagreement-021 | 5 | 5 | 5 | 0.7917 | 0.7825 | true | false | no_heavy_puct_regression |
| classic_only_008 | 1 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.9505 | 0.8633 | true | false | no_heavy_puct_regression |
| classic_only_008 | 1 | incumbent_proxy_disagreement-028 | 5 | 2 | 2 | 0.4661 | 0.1492 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_008 | 1 | incumbent_proxy_disagreement-032 | 1 | 2 | 2 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_008 | 1 | incumbent_proxy_disagreement-033 | 0 | 2 | 2 | 0.0547 | 0.0183 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_008 | 2 | incumbent_proxy_disagreement-007 | 3 | 2 | 0 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_008 | 2 | incumbent_proxy_disagreement-009 | 3 | 5 | 5 | 0.1328 | 0.0433 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_008 | 2 | incumbent_proxy_disagreement-021 | 5 | 5 | 5 | 0.5833 | 0.6333 | true | true | large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_008 | 2 | incumbent_proxy_disagreement-026 | 5 | 5 | 2 | 0.8203 | 0.3333 | false | true | lost_puct_selection_1200,large_preferred_share_drop_1200 |
| classic_only_008 | 2 | incumbent_proxy_disagreement-028 | 5 | 5 | 2 | 0.6380 | 0.2642 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_008 | 2 | incumbent_proxy_disagreement-032 | 1 | 2 | 2 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_008 | 2 | incumbent_proxy_disagreement-033 | 0 | 2 | 2 | 0.0521 | 0.0692 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_014 | 1 | incumbent_proxy_disagreement-007 | 3 | 3 | 0 | 0.6536 | 0.2092 | false | true | lost_puct_selection_1200,large_preferred_share_drop_1200 |
| classic_only_014 | 1 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.8411 | 0.9142 | true | false | no_heavy_puct_regression |
| classic_only_014 | 1 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.1120 | 0.0383 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_014 | 1 | incumbent_proxy_disagreement-026 | 5 | 5 | 2 | 0.6302 | 0.2983 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_014 | 1 | incumbent_proxy_disagreement-028 | 5 | 4 | 3 | 0.2031 | 0.1550 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_014 | 1 | incumbent_proxy_disagreement-032 | 1 | 2 | 2 | 0.0026 | 0.0317 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_014 | 1 | incumbent_proxy_disagreement-033 | 0 | 3 | 1 | 0.1328 | 0.0425 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_014 | 2 | incumbent_proxy_disagreement-007 | 3 | 3 | 2 | 0.8203 | 0.2708 | false | true | lost_puct_selection_1200,large_preferred_share_drop_1200 |
| classic_only_014 | 2 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.9922 | 0.9975 | true | false | no_heavy_puct_regression |
| classic_only_014 | 2 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_014 | 2 | incumbent_proxy_disagreement-026 | 5 | 2 | 2 | 0.0781 | 0.0783 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_014 | 2 | incumbent_proxy_disagreement-028 | 5 | 4 | 4 | 0.0000 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_014 | 2 | incumbent_proxy_disagreement-032 | 1 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_014 | 2 | incumbent_proxy_disagreement-033 | 0 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_014 | 4 | incumbent_proxy_disagreement-007 | 3 | 3 | 3 | 0.8490 | 0.6842 | true | false | no_heavy_puct_regression |
| classic_only_014 | 4 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.9922 | 0.9900 | true | false | no_heavy_puct_regression |
| classic_only_014 | 4 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_014 | 4 | incumbent_proxy_disagreement-026 | 5 | 2 | 5 | 0.0026 | 0.3417 | true | true | lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_014 | 4 | incumbent_proxy_disagreement-028 | 5 | 4 | 4 | 0.0000 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_014 | 4 | incumbent_proxy_disagreement-032 | 1 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_014 | 4 | incumbent_proxy_disagreement-033 | 0 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_022 | 1 | incumbent_proxy_disagreement-007 | 3 | 2 | 2 | 0.0547 | 0.4400 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384 |
| classic_only_022 | 1 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.6745 | 0.8958 | true | false | no_heavy_puct_regression |
| classic_only_022 | 1 | incumbent_proxy_disagreement-021 | 5 | 2 | 3 | 0.1432 | 0.0517 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_022 | 1 | incumbent_proxy_disagreement-026 | 5 | 5 | 2 | 0.5391 | 0.3350 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_022 | 1 | incumbent_proxy_disagreement-028 | 5 | 3 | 5 | 0.2682 | 0.4850 | true | true | lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_022 | 1 | incumbent_proxy_disagreement-032 | 1 | 2 | 2 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_022 | 1 | incumbent_proxy_disagreement-033 | 0 | 3 | 0 | 0.0026 | 0.4142 | true | true | lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_022 | 2 | incumbent_proxy_disagreement-007 | 3 | 3 | 3 | 0.9245 | 0.6092 | true | false | no_heavy_puct_regression |
| classic_only_022 | 2 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.9922 | 0.9975 | true | false | no_heavy_puct_regression |
| classic_only_022 | 2 | incumbent_proxy_disagreement-021 | 5 | 2 | 2 | 0.0000 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_022 | 2 | incumbent_proxy_disagreement-026 | 5 | 3 | 2 | 0.0260 | 0.0092 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_022 | 2 | incumbent_proxy_disagreement-028 | 5 | 4 | 4 | 0.0026 | 0.1067 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_022 | 2 | incumbent_proxy_disagreement-032 | 1 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_022 | 2 | incumbent_proxy_disagreement-033 | 0 | 3 | 3 | 0.0000 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_024 | 1 | incumbent_proxy_disagreement-007 | 3 | 5 | 5 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_024 | 1 | incumbent_proxy_disagreement-009 | 3 | 5 | 5 | 0.0078 | 0.0050 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_024 | 1 | incumbent_proxy_disagreement-021 | 5 | 5 | 5 | 0.9167 | 0.7400 | true | true | large_preferred_share_drop_1200 |
| classic_only_024 | 1 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.9922 | 0.9900 | true | false | no_heavy_puct_regression |
| classic_only_024 | 1 | incumbent_proxy_disagreement-028 | 5 | 5 | 5 | 0.9089 | 0.9542 | true | false | no_heavy_puct_regression |
| classic_only_024 | 1 | incumbent_proxy_disagreement-032 | 1 | 2 | 2 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_024 | 1 | incumbent_proxy_disagreement-033 | 0 | 4 | 4 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_024 | 2 | incumbent_proxy_disagreement-007 | 3 | 2 | 2 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_024 | 2 | incumbent_proxy_disagreement-009 | 3 | 5 | 5 | 0.0130 | 0.0067 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_024 | 2 | incumbent_proxy_disagreement-021 | 5 | 5 | 5 | 0.7500 | 0.8842 | true | false | no_heavy_puct_regression |
| classic_only_024 | 2 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.9688 | 0.9767 | true | false | no_heavy_puct_regression |
| classic_only_024 | 2 | incumbent_proxy_disagreement-028 | 5 | 2 | 2 | 0.3203 | 0.4633 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_024 | 2 | incumbent_proxy_disagreement-032 | 1 | 2 | 4 | 0.0000 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_024 | 2 | incumbent_proxy_disagreement-033 | 0 | 2 | 2 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_025 | 1 | incumbent_proxy_disagreement-007 | 3 | 2 | 2 | 0.0026 | 0.0017 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_025 | 1 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.8229 | 0.8725 | true | false | no_heavy_puct_regression |
| classic_only_025 | 1 | incumbent_proxy_disagreement-021 | 5 | 5 | 5 | 0.7344 | 0.8683 | true | false | no_heavy_puct_regression |
| classic_only_025 | 1 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.9297 | 0.9583 | true | false | no_heavy_puct_regression |
| classic_only_025 | 1 | incumbent_proxy_disagreement-028 | 5 | 5 | 5 | 0.8984 | 0.9508 | true | false | no_heavy_puct_regression |
| classic_only_025 | 1 | incumbent_proxy_disagreement-032 | 1 | 1 | 1 | 0.8411 | 0.9492 | true | false | no_heavy_puct_regression |
| classic_only_025 | 1 | incumbent_proxy_disagreement-033 | 0 | 1 | 1 | 0.0365 | 0.0125 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_025 | 2 | incumbent_proxy_disagreement-007 | 3 | 2 | 2 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_025 | 2 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.6224 | 0.8392 | true | true | large_preferred_share_drop_384 |
| classic_only_025 | 2 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.1198 | 0.0392 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_025 | 2 | incumbent_proxy_disagreement-026 | 5 | 2 | 2 | 0.3229 | 0.1492 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_025 | 2 | incumbent_proxy_disagreement-028 | 5 | 2 | 2 | 0.3099 | 0.0992 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_025 | 2 | incumbent_proxy_disagreement-032 | 1 | 1 | 1 | 0.8411 | 0.6617 | true | false | no_heavy_puct_regression |
| classic_only_025 | 2 | incumbent_proxy_disagreement-033 | 0 | 2 | 2 | 0.0365 | 0.0125 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_035 | 1 | incumbent_proxy_disagreement-007 | 3 | 3 | 3 | 0.7656 | 0.6433 | true | false | no_heavy_puct_regression |
| classic_only_035 | 1 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.6068 | 0.8742 | true | true | large_preferred_share_drop_384 |
| classic_only_035 | 1 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.1562 | 0.0500 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_035 | 1 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.9714 | 0.9875 | true | false | no_heavy_puct_regression |
| classic_only_035 | 1 | incumbent_proxy_disagreement-028 | 5 | 5 | 5 | 0.9167 | 0.9267 | true | false | no_heavy_puct_regression |
| classic_only_035 | 1 | incumbent_proxy_disagreement-032 | 1 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_035 | 1 | incumbent_proxy_disagreement-033 | 0 | 4 | 4 | 0.0026 | 0.0200 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_035 | 2 | incumbent_proxy_disagreement-007 | 3 | 3 | 2 | 0.6562 | 0.3058 | false | true | lost_puct_selection_1200,large_preferred_share_drop_1200 |
| classic_only_035 | 2 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.6849 | 0.8600 | true | false | no_heavy_puct_regression |
| classic_only_035 | 2 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.2396 | 0.1008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_035 | 2 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.9531 | 0.9233 | true | false | no_heavy_puct_regression |
| classic_only_035 | 2 | incumbent_proxy_disagreement-028 | 5 | 5 | 5 | 0.8255 | 0.7650 | true | false | no_heavy_puct_regression |
| classic_only_035 | 2 | incumbent_proxy_disagreement-032 | 1 | 4 | 3 | 0.0000 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_only_035 | 2 | incumbent_proxy_disagreement-033 | 0 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_008 | 1 | incumbent_proxy_disagreement-007 | 3 | 3 | 3 | 0.9167 | 0.9075 | true | false | no_heavy_puct_regression |
| classic_without_008 | 1 | incumbent_proxy_disagreement-009 | 3 | 5 | 3 | 0.2057 | 0.7458 | true | true | lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_008 | 1 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.0026 | 0.0367 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_008 | 1 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.8672 | 0.9492 | true | false | no_heavy_puct_regression |
| classic_without_008 | 1 | incumbent_proxy_disagreement-028 | 5 | 5 | 5 | 0.8307 | 0.8517 | true | false | no_heavy_puct_regression |
| classic_without_008 | 1 | incumbent_proxy_disagreement-032 | 1 | 4 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_008 | 1 | incumbent_proxy_disagreement-033 | 0 | 4 | 4 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_008 | 2 | incumbent_proxy_disagreement-007 | 3 | 2 | 2 | 0.3099 | 0.1117 | false | true | lost_puct_selection_1200,large_preferred_share_drop_1200 |
| classic_without_008 | 2 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.6380 | 0.8842 | true | true | large_preferred_share_drop_384 |
| classic_without_008 | 2 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_008 | 2 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.8229 | 0.8575 | true | false | no_heavy_puct_regression |
| classic_without_008 | 2 | incumbent_proxy_disagreement-028 | 5 | 4 | 5 | 0.0026 | 0.5842 | true | true | lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_008 | 2 | incumbent_proxy_disagreement-032 | 1 | 4 | 2 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_008 | 2 | incumbent_proxy_disagreement-033 | 0 | 4 | 4 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_014 | 1 | incumbent_proxy_disagreement-007 | 3 | 0 | 0 | 0.0234 | 0.0075 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_014 | 1 | incumbent_proxy_disagreement-009 | 3 | 5 | 5 | 0.0703 | 0.3467 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_014 | 1 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.0521 | 0.1425 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_014 | 1 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.9193 | 0.9700 | true | false | no_heavy_puct_regression |
| classic_without_014 | 1 | incumbent_proxy_disagreement-028 | 5 | 3 | 3 | 0.3021 | 0.1033 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_014 | 1 | incumbent_proxy_disagreement-032 | 1 | 2 | 2 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_014 | 1 | incumbent_proxy_disagreement-033 | 0 | 3 | 2 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_014 | 2 | incumbent_proxy_disagreement-007 | 3 | 2 | 2 | 0.0026 | 0.0075 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_014 | 2 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.6641 | 0.6567 | true | true | large_preferred_share_drop_1200 |
| classic_without_014 | 2 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_014 | 2 | incumbent_proxy_disagreement-026 | 5 | 5 | 2 | 0.4844 | 0.1550 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_014 | 2 | incumbent_proxy_disagreement-028 | 5 | 5 | 5 | 0.8333 | 0.8850 | true | false | no_heavy_puct_regression |
| classic_without_014 | 2 | incumbent_proxy_disagreement-032 | 1 | 2 | 4 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_014 | 2 | incumbent_proxy_disagreement-033 | 0 | 3 | 2 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_022 | 1 | incumbent_proxy_disagreement-007 | 3 | 0 | 0 | 0.0208 | 0.0067 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_022 | 1 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.7865 | 0.8758 | true | false | no_heavy_puct_regression |
| classic_without_022 | 1 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.0026 | 0.0683 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_022 | 1 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.9297 | 0.9592 | true | false | no_heavy_puct_regression |
| classic_without_022 | 1 | incumbent_proxy_disagreement-028 | 5 | 5 | 5 | 0.8620 | 0.9142 | true | false | no_heavy_puct_regression |
| classic_without_022 | 1 | incumbent_proxy_disagreement-032 | 1 | 4 | 4 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_022 | 1 | incumbent_proxy_disagreement-033 | 0 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_022 | 2 | incumbent_proxy_disagreement-007 | 3 | 0 | 0 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_022 | 2 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.6484 | 0.7233 | true | true | large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_022 | 2 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_022 | 2 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.6849 | 0.8692 | true | true | large_preferred_share_drop_384 |
| classic_without_022 | 2 | incumbent_proxy_disagreement-028 | 5 | 2 | 5 | 0.0026 | 0.6433 | true | true | lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_022 | 2 | incumbent_proxy_disagreement-032 | 1 | 2 | 4 | 0.0000 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_022 | 2 | incumbent_proxy_disagreement-033 | 0 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_024 | 1 | incumbent_proxy_disagreement-007 | 3 | 2 | 3 | 0.4844 | 0.7817 | true | false | no_heavy_puct_regression |
| classic_without_024 | 1 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.9740 | 0.9467 | true | false | no_heavy_puct_regression |
| classic_without_024 | 1 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_024 | 1 | incumbent_proxy_disagreement-026 | 5 | 5 | 2 | 0.6484 | 0.4233 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_024 | 1 | incumbent_proxy_disagreement-028 | 5 | 3 | 5 | 0.2292 | 0.6200 | true | true | lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_024 | 1 | incumbent_proxy_disagreement-032 | 1 | 2 | 1 | 0.0026 | 0.4408 | true | true | large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_024 | 1 | incumbent_proxy_disagreement-033 | 0 | 3 | 1 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_024 | 2 | incumbent_proxy_disagreement-007 | 3 | 2 | 0 | 0.2292 | 0.0733 | false | true | lost_puct_selection_1200,large_preferred_share_drop_1200 |
| classic_without_024 | 2 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.9922 | 0.9975 | true | false | no_heavy_puct_regression |
| classic_without_024 | 2 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_024 | 2 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.6615 | 0.4708 | true | true | large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_024 | 2 | incumbent_proxy_disagreement-028 | 5 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_024 | 2 | incumbent_proxy_disagreement-032 | 1 | 3 | 2 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_024 | 2 | incumbent_proxy_disagreement-033 | 0 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_025 | 1 | incumbent_proxy_disagreement-007 | 3 | 2 | 3 | 0.4844 | 0.7817 | true | false | no_heavy_puct_regression |
| classic_without_025 | 1 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.9740 | 0.9467 | true | false | no_heavy_puct_regression |
| classic_without_025 | 1 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_025 | 1 | incumbent_proxy_disagreement-026 | 5 | 5 | 2 | 0.6484 | 0.4233 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_025 | 1 | incumbent_proxy_disagreement-028 | 5 | 3 | 5 | 0.2292 | 0.6200 | true | true | lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_025 | 1 | incumbent_proxy_disagreement-032 | 1 | 2 | 1 | 0.0026 | 0.4408 | true | true | large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_025 | 1 | incumbent_proxy_disagreement-033 | 0 | 3 | 1 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_025 | 2 | incumbent_proxy_disagreement-007 | 3 | 2 | 0 | 0.2292 | 0.0733 | false | true | lost_puct_selection_1200,large_preferred_share_drop_1200 |
| classic_without_025 | 2 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.9922 | 0.9975 | true | false | no_heavy_puct_regression |
| classic_without_025 | 2 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_025 | 2 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.6615 | 0.4708 | true | true | large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_025 | 2 | incumbent_proxy_disagreement-028 | 5 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_025 | 2 | incumbent_proxy_disagreement-032 | 1 | 3 | 2 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_025 | 2 | incumbent_proxy_disagreement-033 | 0 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_035 | 1 | incumbent_proxy_disagreement-007 | 3 | 2 | 0 | 0.0677 | 0.0217 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_035 | 1 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.7500 | 0.8667 | true | false | no_heavy_puct_regression |
| classic_without_035 | 1 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_035 | 1 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.4766 | 0.7167 | true | true | large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_035 | 1 | incumbent_proxy_disagreement-028 | 5 | 5 | 5 | 0.7422 | 0.9067 | true | false | no_heavy_puct_regression |
| classic_without_035 | 1 | incumbent_proxy_disagreement-032 | 1 | 4 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_035 | 1 | incumbent_proxy_disagreement-033 | 0 | 4 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_035 | 2 | incumbent_proxy_disagreement-007 | 3 | 2 | 2 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_035 | 2 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.9922 | 0.9642 | true | false | no_heavy_puct_regression |
| classic_without_035 | 2 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_035 | 2 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.6068 | 0.6850 | true | true | large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_035 | 2 | incumbent_proxy_disagreement-028 | 5 | 4 | 4 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_035 | 2 | incumbent_proxy_disagreement-032 | 1 | 3 | 4 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_035 | 2 | incumbent_proxy_disagreement-033 | 0 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_035 | 4 | incumbent_proxy_disagreement-007 | 3 | 2 | 2 | 0.0990 | 0.1900 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_035 | 4 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.9922 | 0.9975 | true | false | no_heavy_puct_regression |
| classic_without_035 | 4 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_035 | 4 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.4818 | 0.8333 | true | true | large_preferred_share_drop_384 |
| classic_without_035 | 4 | incumbent_proxy_disagreement-028 | 5 | 4 | 4 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_035 | 4 | incumbent_proxy_disagreement-032 | 1 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_without_035 | 4 | incumbent_proxy_disagreement-033 | 0 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| preservation_only | 1 | incumbent_proxy_disagreement-007 | 3 | 0 | 0 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| preservation_only | 1 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.5182 | 0.8408 | true | true | large_preferred_share_drop_384 |
| preservation_only | 1 | incumbent_proxy_disagreement-021 | 5 | 5 | 5 | 0.7917 | 0.7825 | true | false | no_heavy_puct_regression |
| preservation_only | 1 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.9505 | 0.8633 | true | false | no_heavy_puct_regression |
| preservation_only | 1 | incumbent_proxy_disagreement-028 | 5 | 2 | 2 | 0.4661 | 0.1492 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| preservation_only | 1 | incumbent_proxy_disagreement-032 | 1 | 2 | 2 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| preservation_only | 1 | incumbent_proxy_disagreement-033 | 0 | 2 | 2 | 0.0547 | 0.0183 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| preservation_only | 2 | incumbent_proxy_disagreement-007 | 3 | 2 | 0 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| preservation_only | 2 | incumbent_proxy_disagreement-009 | 3 | 5 | 5 | 0.1328 | 0.0433 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| preservation_only | 2 | incumbent_proxy_disagreement-021 | 5 | 5 | 5 | 0.5833 | 0.6333 | true | true | large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| preservation_only | 2 | incumbent_proxy_disagreement-026 | 5 | 5 | 2 | 0.8203 | 0.3333 | false | true | lost_puct_selection_1200,large_preferred_share_drop_1200 |
| preservation_only | 2 | incumbent_proxy_disagreement-028 | 5 | 5 | 2 | 0.6380 | 0.2642 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| preservation_only | 2 | incumbent_proxy_disagreement-032 | 1 | 2 | 2 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| preservation_only | 2 | incumbent_proxy_disagreement-033 | 0 | 2 | 2 | 0.0521 | 0.0692 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| preservation_only | 4 | incumbent_proxy_disagreement-007 | 3 | 2 | 2 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| preservation_only | 4 | incumbent_proxy_disagreement-009 | 3 | 5 | 3 | 0.3047 | 0.6100 | true | true | lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| preservation_only | 4 | incumbent_proxy_disagreement-021 | 5 | 5 | 5 | 0.7344 | 0.7600 | true | true | large_preferred_share_drop_1200 |
| preservation_only | 4 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.8385 | 0.6150 | true | true | large_preferred_share_drop_1200 |
| preservation_only | 4 | incumbent_proxy_disagreement-028 | 5 | 2 | 2 | 0.1562 | 0.1000 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| preservation_only | 4 | incumbent_proxy_disagreement-032 | 1 | 2 | 2 | 0.0000 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| preservation_only | 4 | incumbent_proxy_disagreement-033 | 0 | 2 | 2 | 0.1354 | 0.1483 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| target_only | 1 | incumbent_proxy_disagreement-007 | 3 | 3 | 3 | 0.9167 | 0.9075 | true | false | no_heavy_puct_regression |
| target_only | 1 | incumbent_proxy_disagreement-009 | 3 | 5 | 3 | 0.2057 | 0.7458 | true | true | lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| target_only | 1 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.0026 | 0.0367 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| target_only | 1 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.8672 | 0.9492 | true | false | no_heavy_puct_regression |
| target_only | 1 | incumbent_proxy_disagreement-028 | 5 | 5 | 5 | 0.8307 | 0.8517 | true | false | no_heavy_puct_regression |
| target_only | 1 | incumbent_proxy_disagreement-032 | 1 | 4 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| target_only | 1 | incumbent_proxy_disagreement-033 | 0 | 4 | 4 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| target_only | 2 | incumbent_proxy_disagreement-007 | 3 | 2 | 2 | 0.3099 | 0.1117 | false | true | lost_puct_selection_1200,large_preferred_share_drop_1200 |
| target_only | 2 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.6380 | 0.8842 | true | true | large_preferred_share_drop_384 |
| target_only | 2 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| target_only | 2 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.8229 | 0.8575 | true | false | no_heavy_puct_regression |
| target_only | 2 | incumbent_proxy_disagreement-028 | 5 | 4 | 5 | 0.0026 | 0.5842 | true | true | lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| target_only | 2 | incumbent_proxy_disagreement-032 | 1 | 4 | 2 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| target_only | 2 | incumbent_proxy_disagreement-033 | 0 | 4 | 4 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| target_only | 4 | incumbent_proxy_disagreement-007 | 3 | 2 | 2 | 0.1302 | 0.0417 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| target_only | 4 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.9922 | 0.7858 | true | false | no_heavy_puct_regression |
| target_only | 4 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| target_only | 4 | incumbent_proxy_disagreement-026 | 5 | 5 | 2 | 0.6432 | 0.2350 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| target_only | 4 | incumbent_proxy_disagreement-028 | 5 | 4 | 3 | 0.0026 | 0.0417 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| target_only | 4 | incumbent_proxy_disagreement-032 | 1 | 3 | 4 | 0.0026 | 0.0008 | false | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| target_only | 4 | incumbent_proxy_disagreement-033 | 0 | 3 | 4 | 0.0026 | 0.0117 | false | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |

## 9. Attribution analysis

| artifact_name | classic_gain_count | puct_damage_count | excluded_drift_count | interference_score | likely_driver | notes |
| --- | --- | --- | --- | --- | --- | --- |
| best_leave_one_out_low_lr | 4 | 2 | 1 | 0 | - | lr=0.00025, value_loss_weight=0.3, epochs=[1, 2] |
| classic_all | 4 | 6 | 6 | 14 | update_size_sensitive_interference | lr=0.001, value_loss_weight=0.3, epochs=[1, 2, 4] |
| classic_all_low_lr | 5 | 3 | 1 | 1 | - | lr=0.00025, value_loss_weight=0.3, epochs=[1, 2] |
| classic_all_policy_only | 5 | 7 | 0 | 8 | - | lr=0.001, value_loss_weight=0.0, epochs=[1, 2] |
| classic_only_008 | 3 | 7 | 1 | 11 | - | lr=0.001, value_loss_weight=0.3, epochs=[1, 2] |
| classic_only_014 | 3 | 5 | 6 | 12 | update_size_sensitive_interference | lr=0.001, value_loss_weight=0.3, epochs=[1, 2, 4] |
| classic_only_022 | 3 | 5 | 4 | 11 | - | lr=0.001, value_loss_weight=0.3, epochs=[1, 2] |
| classic_only_024 | 4 | 5 | 4 | 10 | - | lr=0.001, value_loss_weight=0.3, epochs=[1, 2] |
| classic_only_025 | 2 | 6 | 0 | 9 | - | lr=0.001, value_loss_weight=0.3, epochs=[1, 2] |
| classic_only_035 | 4 | 4 | 2 | 6 | - | lr=0.001, value_loss_weight=0.3, epochs=[1, 2] |
| classic_without_008 | 4 | 6 | 3 | 9 | - | lr=0.001, value_loss_weight=0.3, epochs=[1, 2] |
| classic_without_014 | 5 | 6 | 2 | 8 | - | lr=0.001, value_loss_weight=0.3, epochs=[1, 2] |
| classic_without_022 | 4 | 7 | 4 | 11 | - | lr=0.001, value_loss_weight=0.3, epochs=[1, 2] |
| classic_without_024 | 4 | 6 | 4 | 11 | - | lr=0.001, value_loss_weight=0.3, epochs=[1, 2] |
| classic_without_025 | 4 | 6 | 4 | 11 | - | lr=0.001, value_loss_weight=0.3, epochs=[1, 2] |
| classic_without_035 | 4 | 6 | 6 | 13 | update_size_sensitive_interference | lr=0.001, value_loss_weight=0.3, epochs=[1, 2, 4] |
| preservation_only | 3 | 7 | 2 | 10 | - | lr=0.001, value_loss_weight=0.3, epochs=[1, 2, 4] |
| target_only | 4 | 6 | 5 | 13 | - | lr=0.001, value_loss_weight=0.3, epochs=[1, 2, 4] |

## 10. Optional gradient attribution

- Strongest negative policy-gradient cosine pairs at initialization:
- `incumbent_proxy_disagreement-014` vs `incumbent_proxy_disagreement-021`: preferred=-0.749159, active_reference=0.291015
- `incumbent_proxy_disagreement-022` vs `incumbent_proxy_disagreement-021`: preferred=-0.735593, active_reference=0.169374
- `incumbent_proxy_disagreement-014` vs `incumbent_proxy_disagreement-028`: preferred=-0.551857, active_reference=-0.551857
- `incumbent_proxy_disagreement-022` vs `incumbent_proxy_disagreement-028`: preferred=-0.487305, active_reference=-0.487305
- `incumbent_proxy_disagreement-035` vs `incumbent_proxy_disagreement-026`: preferred=-0.321776, active_reference=-0.321776

## 11. Interpretation

- Final classification: `update_size_sensitive_interference`.
- Likely driver: `update_size_sensitive_interference`.
- Interference score formula: `puct_lost_selection_1200_count + heavy_regression_count + excluded_drift_count - classic_gain_count`.
- classic_all_low_lr cut heavy PUCT regressions from 6 to 3 at epoch 2 without reducing Classic gains
- Follow-up low-LR lane result: `low_lr_lane_gate_failed` in `docs/alphazero-lite-incumbent-proxy-classic-teacher-low-lr-non-regression-lane-results.md`.
- The best low-LR checkpoint still failed the strict PUCT non-regression gate at epoch 1 due to lost PUCT selections at 384 and 1200 plus heavy regression.

## 12. Exactly one recommended next action

| classification | supporting_evidence | rejected_alternatives | next_action |
| --- | --- | --- | --- |
| update_size_sensitive_interference_followup_failed | classic_all_low_lr reduced damage in attribution, but the dedicated low-LR follow-up lane still failed the strict PUCT non-regression gate at epoch 1 with lost PUCT selections at 384 and 1200 plus heavy regression | rejected advancing the low-LR replay lane because it never cleared the strict gate; rejected single-row removal as the primary next branch because LR reduction was already insufficient | test softer Classic policy targets under the same strict PUCT non-regression gate; do not advance the low-LR replay lane. |
