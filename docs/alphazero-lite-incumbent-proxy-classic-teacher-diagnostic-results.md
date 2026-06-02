# AlphaZero-lite Incumbent Proxy Classic Teacher Diagnostic Results

## 1. Context

- No arena was run.
- No model was promoted.
- Active references stayed read-only at `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- No production lane was trained.
- The experiment used `storage/ai/alphazero_lite/current` only as the initializer.
- Temporary exports were written only under `/tmp/azlite_incumbent_proxy_classic_teacher_diagnostic/exports/`.

## 2. Why PR #47 selected the Classic diagnostic branch

- PR #47 separated incompatible teacher policies into Classic, PUCT, and excluded buckets.
- The Classic bucket offered 5 target candidates plus 1 preservation control without mutating active references.
- The PUCT bucket still requires a separate reference-policy branch and was therefore kept out of training targets here.

## 3. Artifact construction and validation

| row_id | role | active_reference_move | preferred_teacher | preferred_move | policy_target_reference_mass | value_source | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-014 | target_candidate | 4 | classic_mcts | 4 | 0.8500 | generated | ok | downstream child evidence favors classic/reference; classic-child decisive; puct-child near-zero split; child delta 0.1667 favors move 4; child delta -0.0141 remains near zero |
| incumbent_proxy_disagreement-022 | target_candidate | 3 | classic_mcts | 3 | 0.8500 | generated | ok | ClassicMCTS, artifact PUCT, and active reference all agree |
| incumbent_proxy_disagreement-024 | target_candidate | 2 | classic_mcts | 2 | 0.8500 | generated | ok | paired continuations support classic/reference under both policies |
| incumbent_proxy_disagreement-025 | target_candidate | 2 | classic_mcts | 2 | 0.8500 | generated | ok | ClassicMCTS, artifact PUCT, and active reference all agree |
| incumbent_proxy_disagreement-035 | target_candidate | 3 | classic_mcts | 3 | 0.8500 | generated | ok | downstream child evidence favors classic/reference; classic-child decisive; puct-child near-zero split; child delta 0.0356 favors move 3; child delta -0.0197 remains near zero |
| incumbent_proxy_disagreement-008 | preservation_control | 2 | classic_mcts | 2 | 0.8500 | generated | ok | ClassicMCTS, artifact PUCT, and active reference all agree |

Artifact validation status: `ok`.
Row count: `6`. Target candidates: `5`. Preservation controls: `1`.

## 4. Trace definitions

| trace_name | data_files | replay_weights | epochs | status | notes |
| --- | --- | --- | --- | --- | --- |
| classic_artifact_only_w1 | /tmp/azlite_incumbent_proxy_classic_teacher_diagnostic/classic_teacher_diagnostic_artifact.jsonl | 1 | 1, 2, 4 | completed | all 6 Classic rows only, replay weight 1 |
| classic_artifact_only_w2 | /tmp/azlite_incumbent_proxy_classic_teacher_diagnostic/classic_teacher_diagnostic_artifact.jsonl | 2 | 1, 2, 4 | completed | all 6 Classic rows only, replay weight 2 |
| classic_artifact_plus_preservation_boost | /tmp/azlite_incumbent_proxy_classic_teacher_diagnostic/classic_teacher_target_candidates.jsonl, /tmp/azlite_incumbent_proxy_classic_teacher_diagnostic/classic_teacher_preservation_controls.jsonl | 1, 2 | 1, 2, 4 | completed | target candidates weight 1 plus preservation control weight 2 |

## 5. Classic-bucket local results

| trace_name | epoch | row_id | role | active_reference_move | selected_move_384 | selected_move_1200 | reference_visit_share_384 | reference_visit_share_1200 | reference_policy_probability | reference_policy_rank | improved_vs_current | strict_pass | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classic_artifact_only_w1 | 1 | incumbent_proxy_disagreement-008 | preservation_control | 2 | 2 | 2 | 0.9583 | 0.9767 | 0.6839 | 1 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_artifact_only_w1 | 1 | incumbent_proxy_disagreement-014 | target_candidate | 4 | 4 | 4 | 0.4974 | 0.7433 | 0.8170 | 1 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_artifact_only_w1 | 1 | incumbent_proxy_disagreement-022 | target_candidate | 3 | 2 | 2 | 0.1068 | 0.0342 | 0.3324 | 2 | true | false | reference_share_up_384,reference_share_up_1200 |
| classic_artifact_only_w1 | 1 | incumbent_proxy_disagreement-024 | target_candidate | 2 | 2 | 2 | 0.7917 | 0.9100 | 0.4709 | 1 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_artifact_only_w1 | 1 | incumbent_proxy_disagreement-025 | target_candidate | 2 | 4 | 4 | 0.0000 | 0.0000 | 0.0005 | 4 | false | false | no_change_vs_current |
| classic_artifact_only_w1 | 1 | incumbent_proxy_disagreement-035 | target_candidate | 3 | 3 | 3 | 0.9609 | 0.8733 | 0.7766 | 1 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_artifact_only_w1 | 2 | incumbent_proxy_disagreement-008 | preservation_control | 2 | 2 | 2 | 0.9792 | 0.9875 | 0.8603 | 1 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_artifact_only_w1 | 2 | incumbent_proxy_disagreement-014 | target_candidate | 4 | 4 | 4 | 0.8177 | 0.8983 | 0.7482 | 1 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_artifact_only_w1 | 2 | incumbent_proxy_disagreement-022 | target_candidate | 3 | 2 | 2 | 0.0469 | 0.2158 | 0.3372 | 2 | true | false | reference_share_up_384,reference_share_up_1200 |
| classic_artifact_only_w1 | 2 | incumbent_proxy_disagreement-024 | target_candidate | 2 | 2 | 2 | 0.9896 | 0.9867 | 0.9021 | 1 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_artifact_only_w1 | 2 | incumbent_proxy_disagreement-025 | target_candidate | 2 | 3 | 3 | 0.0000 | 0.0000 | 0.0058 | 3 | false | false | no_change_vs_current |
| classic_artifact_only_w1 | 2 | incumbent_proxy_disagreement-035 | target_candidate | 3 | 3 | 3 | 0.9844 | 0.9942 | 0.8976 | 1 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_artifact_only_w1 | 4 | incumbent_proxy_disagreement-008 | preservation_control | 2 | 2 | 2 | 0.9349 | 0.9767 | 0.7917 | 1 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_artifact_only_w1 | 4 | incumbent_proxy_disagreement-014 | target_candidate | 4 | 4 | 4 | 0.9922 | 0.9967 | 0.9364 | 1 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_artifact_only_w1 | 4 | incumbent_proxy_disagreement-022 | target_candidate | 3 | 3 | 3 | 0.9661 | 0.8917 | 0.8590 | 1 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_artifact_only_w1 | 4 | incumbent_proxy_disagreement-024 | target_candidate | 2 | 2 | 2 | 0.9661 | 0.9767 | 0.8256 | 1 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_artifact_only_w1 | 4 | incumbent_proxy_disagreement-025 | target_candidate | 2 | 4 | 4 | 0.0000 | 0.0000 | 0.0014 | 3 | false | false | no_change_vs_current |
| classic_artifact_only_w1 | 4 | incumbent_proxy_disagreement-035 | target_candidate | 3 | 3 | 3 | 0.9922 | 0.9950 | 0.9343 | 1 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_artifact_only_w2 | 1 | incumbent_proxy_disagreement-008 | preservation_control | 2 | 2 | 2 | 0.9583 | 0.9767 | 0.6839 | 1 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_artifact_only_w2 | 1 | incumbent_proxy_disagreement-014 | target_candidate | 4 | 4 | 4 | 0.4974 | 0.7433 | 0.8170 | 1 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_artifact_only_w2 | 1 | incumbent_proxy_disagreement-022 | target_candidate | 3 | 2 | 2 | 0.1068 | 0.0342 | 0.3324 | 2 | true | false | reference_share_up_384,reference_share_up_1200 |
| classic_artifact_only_w2 | 1 | incumbent_proxy_disagreement-024 | target_candidate | 2 | 2 | 2 | 0.7917 | 0.9100 | 0.4709 | 1 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_artifact_only_w2 | 1 | incumbent_proxy_disagreement-025 | target_candidate | 2 | 4 | 4 | 0.0000 | 0.0000 | 0.0005 | 4 | false | false | no_change_vs_current |
| classic_artifact_only_w2 | 1 | incumbent_proxy_disagreement-035 | target_candidate | 3 | 3 | 3 | 0.9609 | 0.8733 | 0.7766 | 1 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_artifact_only_w2 | 2 | incumbent_proxy_disagreement-008 | preservation_control | 2 | 2 | 2 | 0.9792 | 0.9875 | 0.8603 | 1 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_artifact_only_w2 | 2 | incumbent_proxy_disagreement-014 | target_candidate | 4 | 4 | 4 | 0.8177 | 0.8983 | 0.7482 | 1 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_artifact_only_w2 | 2 | incumbent_proxy_disagreement-022 | target_candidate | 3 | 2 | 2 | 0.0469 | 0.2158 | 0.3372 | 2 | true | false | reference_share_up_384,reference_share_up_1200 |
| classic_artifact_only_w2 | 2 | incumbent_proxy_disagreement-024 | target_candidate | 2 | 2 | 2 | 0.9896 | 0.9867 | 0.9021 | 1 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_artifact_only_w2 | 2 | incumbent_proxy_disagreement-025 | target_candidate | 2 | 3 | 3 | 0.0000 | 0.0000 | 0.0058 | 3 | false | false | no_change_vs_current |
| classic_artifact_only_w2 | 2 | incumbent_proxy_disagreement-035 | target_candidate | 3 | 3 | 3 | 0.9844 | 0.9942 | 0.8976 | 1 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_artifact_only_w2 | 4 | incumbent_proxy_disagreement-008 | preservation_control | 2 | 2 | 2 | 0.9349 | 0.9767 | 0.7917 | 1 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_artifact_only_w2 | 4 | incumbent_proxy_disagreement-014 | target_candidate | 4 | 4 | 4 | 0.9922 | 0.9967 | 0.9364 | 1 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_artifact_only_w2 | 4 | incumbent_proxy_disagreement-022 | target_candidate | 3 | 3 | 3 | 0.9661 | 0.8917 | 0.8590 | 1 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_artifact_only_w2 | 4 | incumbent_proxy_disagreement-024 | target_candidate | 2 | 2 | 2 | 0.9661 | 0.9767 | 0.8256 | 1 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_artifact_only_w2 | 4 | incumbent_proxy_disagreement-025 | target_candidate | 2 | 4 | 4 | 0.0000 | 0.0000 | 0.0014 | 3 | false | false | no_change_vs_current |
| classic_artifact_only_w2 | 4 | incumbent_proxy_disagreement-035 | target_candidate | 3 | 3 | 3 | 0.9922 | 0.9950 | 0.9343 | 1 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_artifact_plus_preservation_boost | 1 | incumbent_proxy_disagreement-008 | preservation_control | 2 | 2 | 2 | 0.9635 | 0.9758 | 0.7787 | 1 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_artifact_plus_preservation_boost | 1 | incumbent_proxy_disagreement-014 | target_candidate | 4 | 4 | 5 | 0.7266 | 0.3867 | 0.7085 | 1 | true | false | selection_improved_384,reference_share_up_384,reference_share_up_1200 |
| classic_artifact_plus_preservation_boost | 1 | incumbent_proxy_disagreement-022 | target_candidate | 3 | 2 | 2 | 0.0286 | 0.0392 | 0.2416 | 2 | true | false | reference_share_up_384,reference_share_up_1200 |
| classic_artifact_plus_preservation_boost | 1 | incumbent_proxy_disagreement-024 | target_candidate | 2 | 2 | 2 | 0.9297 | 0.9558 | 0.5434 | 1 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_artifact_plus_preservation_boost | 1 | incumbent_proxy_disagreement-025 | target_candidate | 2 | 4 | 4 | 0.0000 | 0.0008 | 0.0010 | 3 | false | false | no_change_vs_current |
| classic_artifact_plus_preservation_boost | 1 | incumbent_proxy_disagreement-035 | target_candidate | 3 | 3 | 1 | 0.9089 | 0.2908 | 0.7631 | 1 | true | false | selection_improved_384,reference_share_up_384,reference_share_up_1200 |
| classic_artifact_plus_preservation_boost | 2 | incumbent_proxy_disagreement-008 | preservation_control | 2 | 2 | 2 | 0.9922 | 0.9883 | 0.8723 | 1 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_artifact_plus_preservation_boost | 2 | incumbent_proxy_disagreement-014 | target_candidate | 4 | 4 | 4 | 0.6849 | 0.8825 | 0.8505 | 1 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_artifact_plus_preservation_boost | 2 | incumbent_proxy_disagreement-022 | target_candidate | 3 | 2 | 2 | 0.0443 | 0.2358 | 0.3540 | 2 | true | false | reference_share_up_384,reference_share_up_1200 |
| classic_artifact_plus_preservation_boost | 2 | incumbent_proxy_disagreement-024 | target_candidate | 2 | 2 | 2 | 0.9896 | 0.9942 | 0.9224 | 1 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_artifact_plus_preservation_boost | 2 | incumbent_proxy_disagreement-025 | target_candidate | 2 | 3 | 3 | 0.0000 | 0.0000 | 0.0025 | 3 | false | false | no_change_vs_current |
| classic_artifact_plus_preservation_boost | 2 | incumbent_proxy_disagreement-035 | target_candidate | 3 | 3 | 3 | 0.9844 | 0.9942 | 0.9076 | 1 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_artifact_plus_preservation_boost | 4 | incumbent_proxy_disagreement-008 | preservation_control | 2 | 2 | 2 | 0.9557 | 0.9767 | 0.7767 | 1 | true | true | reference_share_up_384,reference_share_up_1200 |
| classic_artifact_plus_preservation_boost | 4 | incumbent_proxy_disagreement-014 | target_candidate | 4 | 4 | 4 | 0.9922 | 0.9967 | 0.9403 | 1 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_artifact_plus_preservation_boost | 4 | incumbent_proxy_disagreement-022 | target_candidate | 3 | 3 | 3 | 0.9896 | 0.9133 | 0.8653 | 1 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_artifact_plus_preservation_boost | 4 | incumbent_proxy_disagreement-024 | target_candidate | 2 | 2 | 2 | 0.9766 | 0.9892 | 0.8114 | 1 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |
| classic_artifact_plus_preservation_boost | 4 | incumbent_proxy_disagreement-025 | target_candidate | 2 | 3 | 3 | 0.0000 | 0.0000 | 0.0008 | 4 | false | false | no_change_vs_current |
| classic_artifact_plus_preservation_boost | 4 | incumbent_proxy_disagreement-035 | target_candidate | 3 | 3 | 3 | 0.9922 | 0.6542 | 0.9224 | 1 | true | true | selection_improved_384,selection_improved_1200,reference_share_up_384,reference_share_up_1200 |

## 6. PUCT-bucket non-regression check

| trace_name | epoch | row_id | puct_preferred_move | selected_move_384 | selected_move_1200 | selected_equals_puct_preferred | puct_preferred_visit_share_384 | puct_preferred_visit_share_1200 | cross_teacher_regression | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classic_artifact_only_w1 | 1 | incumbent_proxy_disagreement-007 | 3 | 0 | 0 | false | 0.1016 | 0.0325 | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_only_w1 | 1 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | true | 0.7396 | 0.8500 | false | no_heavy_puct_regression |
| classic_artifact_only_w1 | 1 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | false | 0.0026 | 0.0008 | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_only_w1 | 1 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | true | 0.8359 | 0.8550 | false | no_heavy_puct_regression |
| classic_artifact_only_w1 | 1 | incumbent_proxy_disagreement-028 | 5 | 5 | 5 | true | 0.4505 | 0.7875 | true | large_preferred_share_drop_384 |
| classic_artifact_only_w1 | 1 | incumbent_proxy_disagreement-032 | 1 | 3 | 2 | false | 0.0026 | 0.0008 | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_only_w1 | 1 | incumbent_proxy_disagreement-033 | 0 | 4 | 3 | false | 0.0026 | 0.0008 | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_only_w1 | 2 | incumbent_proxy_disagreement-007 | 3 | 2 | 2 | false | 0.0026 | 0.0008 | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_only_w1 | 2 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | true | 0.9922 | 0.7950 | false | no_heavy_puct_regression |
| classic_artifact_only_w1 | 2 | incumbent_proxy_disagreement-021 | 5 | 2 | 3 | false | 0.0026 | 0.0008 | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_only_w1 | 2 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | true | 0.7865 | 0.9308 | true | large_preferred_share_drop_384 |
| classic_artifact_only_w1 | 2 | incumbent_proxy_disagreement-028 | 5 | 4 | 4 | false | 0.0026 | 0.0500 | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_only_w1 | 2 | incumbent_proxy_disagreement-032 | 1 | 3 | 3 | false | 0.0026 | 0.0008 | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_only_w1 | 2 | incumbent_proxy_disagreement-033 | 0 | 3 | 4 | false | 0.0026 | 0.0008 | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_only_w1 | 4 | incumbent_proxy_disagreement-007 | 3 | 2 | 2 | false | 0.0990 | 0.0333 | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_only_w1 | 4 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | true | 0.9922 | 0.9975 | false | no_heavy_puct_regression |
| classic_artifact_only_w1 | 4 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | false | 0.0026 | 0.0008 | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_only_w1 | 4 | incumbent_proxy_disagreement-026 | 5 | 5 | 0 | false | 0.5078 | 0.3708 | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_only_w1 | 4 | incumbent_proxy_disagreement-028 | 5 | 4 | 4 | false | 0.0026 | 0.0008 | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_only_w1 | 4 | incumbent_proxy_disagreement-032 | 1 | 3 | 3 | false | 0.0026 | 0.0008 | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_only_w1 | 4 | incumbent_proxy_disagreement-033 | 0 | 3 | 3 | false | 0.0000 | 0.0008 | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_only_w2 | 1 | incumbent_proxy_disagreement-007 | 3 | 0 | 0 | false | 0.1016 | 0.0325 | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_only_w2 | 1 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | true | 0.7396 | 0.8500 | false | no_heavy_puct_regression |
| classic_artifact_only_w2 | 1 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | false | 0.0026 | 0.0008 | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_only_w2 | 1 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | true | 0.8359 | 0.8550 | false | no_heavy_puct_regression |
| classic_artifact_only_w2 | 1 | incumbent_proxy_disagreement-028 | 5 | 5 | 5 | true | 0.4505 | 0.7875 | true | large_preferred_share_drop_384 |
| classic_artifact_only_w2 | 1 | incumbent_proxy_disagreement-032 | 1 | 3 | 2 | false | 0.0026 | 0.0008 | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_only_w2 | 1 | incumbent_proxy_disagreement-033 | 0 | 4 | 3 | false | 0.0026 | 0.0008 | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_only_w2 | 2 | incumbent_proxy_disagreement-007 | 3 | 2 | 2 | false | 0.0026 | 0.0008 | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_only_w2 | 2 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | true | 0.9922 | 0.7950 | false | no_heavy_puct_regression |
| classic_artifact_only_w2 | 2 | incumbent_proxy_disagreement-021 | 5 | 2 | 3 | false | 0.0026 | 0.0008 | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_only_w2 | 2 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | true | 0.7865 | 0.9308 | true | large_preferred_share_drop_384 |
| classic_artifact_only_w2 | 2 | incumbent_proxy_disagreement-028 | 5 | 4 | 4 | false | 0.0026 | 0.0500 | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_only_w2 | 2 | incumbent_proxy_disagreement-032 | 1 | 3 | 3 | false | 0.0026 | 0.0008 | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_only_w2 | 2 | incumbent_proxy_disagreement-033 | 0 | 3 | 4 | false | 0.0026 | 0.0008 | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_only_w2 | 4 | incumbent_proxy_disagreement-007 | 3 | 2 | 2 | false | 0.0990 | 0.0333 | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_only_w2 | 4 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | true | 0.9922 | 0.9975 | false | no_heavy_puct_regression |
| classic_artifact_only_w2 | 4 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | false | 0.0026 | 0.0008 | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_only_w2 | 4 | incumbent_proxy_disagreement-026 | 5 | 5 | 0 | false | 0.5078 | 0.3708 | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_only_w2 | 4 | incumbent_proxy_disagreement-028 | 5 | 4 | 4 | false | 0.0026 | 0.0008 | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_only_w2 | 4 | incumbent_proxy_disagreement-032 | 1 | 3 | 3 | false | 0.0026 | 0.0008 | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_only_w2 | 4 | incumbent_proxy_disagreement-033 | 0 | 3 | 3 | false | 0.0000 | 0.0008 | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_plus_preservation_boost | 1 | incumbent_proxy_disagreement-007 | 3 | 0 | 0 | false | 0.0208 | 0.0075 | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_plus_preservation_boost | 1 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | true | 0.7995 | 0.8833 | false | no_heavy_puct_regression |
| classic_artifact_plus_preservation_boost | 1 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | false | 0.0026 | 0.0725 | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_plus_preservation_boost | 1 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | true | 0.8776 | 0.9042 | false | no_heavy_puct_regression |
| classic_artifact_plus_preservation_boost | 1 | incumbent_proxy_disagreement-028 | 5 | 3 | 3 | false | 0.3438 | 0.1142 | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_plus_preservation_boost | 1 | incumbent_proxy_disagreement-032 | 1 | 4 | 4 | false | 0.0026 | 0.0008 | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_plus_preservation_boost | 1 | incumbent_proxy_disagreement-033 | 0 | 4 | 3 | false | 0.0026 | 0.0008 | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_plus_preservation_boost | 2 | incumbent_proxy_disagreement-007 | 3 | 2 | 2 | false | 0.0026 | 0.0008 | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_plus_preservation_boost | 2 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | true | 0.9922 | 0.6300 | true | large_preferred_share_drop_1200 |
| classic_artifact_plus_preservation_boost | 2 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | false | 0.0026 | 0.0008 | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_plus_preservation_boost | 2 | incumbent_proxy_disagreement-026 | 5 | 5 | 2 | false | 0.6406 | 0.3317 | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_plus_preservation_boost | 2 | incumbent_proxy_disagreement-028 | 5 | 3 | 5 | false | 0.0026 | 0.4233 | true | lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_plus_preservation_boost | 2 | incumbent_proxy_disagreement-032 | 1 | 3 | 4 | false | 0.0026 | 0.0008 | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_plus_preservation_boost | 2 | incumbent_proxy_disagreement-033 | 0 | 3 | 3 | false | 0.0026 | 0.0008 | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_plus_preservation_boost | 4 | incumbent_proxy_disagreement-007 | 3 | 2 | 2 | false | 0.0964 | 0.0333 | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_plus_preservation_boost | 4 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | true | 0.9922 | 0.9975 | false | no_heavy_puct_regression |
| classic_artifact_plus_preservation_boost | 4 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | false | 0.0026 | 0.0008 | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_plus_preservation_boost | 4 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | true | 0.6458 | 0.8700 | true | large_preferred_share_drop_384 |
| classic_artifact_plus_preservation_boost | 4 | incumbent_proxy_disagreement-028 | 5 | 4 | 4 | false | 0.0026 | 0.0008 | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_plus_preservation_boost | 4 | incumbent_proxy_disagreement-032 | 1 | 3 | 3 | false | 0.0026 | 0.0008 | true | lost_puct_selection_1200,large_preferred_share_drop_384,large_preferred_share_drop_1200 |
| classic_artifact_plus_preservation_boost | 4 | incumbent_proxy_disagreement-033 | 0 | 3 | 3 | false | 0.0026 | 0.0008 | true | lost_puct_selection_1200,lost_puct_selection_384,large_preferred_share_drop_384,large_preferred_share_drop_1200 |

## 7. Excluded diagnostic sanity check

| trace_name | epoch | row_id | selected_move_1200 | policy_entropy | unexpected_drift | notes |
| --- | --- | --- | --- | --- | --- | --- |
| classic_artifact_only_w1 | 1 | incumbent_proxy_disagreement-003 | 4 | 0.4975 | true | selection_changed_with_entropy_shift |
| classic_artifact_only_w1 | 1 | incumbent_proxy_disagreement-010 | 4 | 0.8956 | true | selection_changed_with_entropy_shift |
| classic_artifact_only_w1 | 1 | incumbent_proxy_disagreement-012 | 3 | 1.6054 | false | no_large_unexpected_drift |
| classic_artifact_only_w1 | 1 | incumbent_proxy_disagreement-018 | 2 | 1.5918 | false | no_large_unexpected_drift |
| classic_artifact_only_w1 | 1 | incumbent_proxy_disagreement-020 | 3 | 2.0958 | false | no_large_unexpected_drift |
| classic_artifact_only_w1 | 1 | incumbent_proxy_disagreement-023 | 3 | 1.8019 | false | no_large_unexpected_drift |
| classic_artifact_only_w1 | 1 | incumbent_proxy_disagreement-027 | 5 | 1.8443 | false | no_large_unexpected_drift |
| classic_artifact_only_w1 | 1 | incumbent_proxy_disagreement-029 | 4 | 1.2670 | false | no_large_unexpected_drift |
| classic_artifact_only_w1 | 2 | incumbent_proxy_disagreement-003 | 4 | 0.6625 | false | no_large_unexpected_drift |
| classic_artifact_only_w1 | 2 | incumbent_proxy_disagreement-010 | 4 | 1.0008 | false | no_large_unexpected_drift |
| classic_artifact_only_w1 | 2 | incumbent_proxy_disagreement-012 | 3 | 1.3898 | true | selection_changed_with_entropy_shift |
| classic_artifact_only_w1 | 2 | incumbent_proxy_disagreement-018 | 3 | 1.3596 | false | no_large_unexpected_drift |
| classic_artifact_only_w1 | 2 | incumbent_proxy_disagreement-020 | 3 | 1.8378 | false | no_large_unexpected_drift |
| classic_artifact_only_w1 | 2 | incumbent_proxy_disagreement-023 | 2 | 1.0872 | true | selection_changed_with_entropy_shift |
| classic_artifact_only_w1 | 2 | incumbent_proxy_disagreement-027 | 2 | 1.4408 | false | no_large_unexpected_drift |
| classic_artifact_only_w1 | 2 | incumbent_proxy_disagreement-029 | 2 | 1.5879 | false | no_large_unexpected_drift |
| classic_artifact_only_w1 | 4 | incumbent_proxy_disagreement-003 | 4 | 0.3195 | true | selection_changed_with_entropy_shift |
| classic_artifact_only_w1 | 4 | incumbent_proxy_disagreement-010 | 4 | 0.6543 | true | selection_changed_with_entropy_shift |
| classic_artifact_only_w1 | 4 | incumbent_proxy_disagreement-012 | 3 | 0.7811 | true | selection_changed_with_entropy_shift,large_entropy_shift |
| classic_artifact_only_w1 | 4 | incumbent_proxy_disagreement-018 | 2 | 0.9399 | true | selection_changed_with_entropy_shift |
| classic_artifact_only_w1 | 4 | incumbent_proxy_disagreement-020 | 3 | 1.5986 | true | selection_changed_with_entropy_shift |
| classic_artifact_only_w1 | 4 | incumbent_proxy_disagreement-023 | 2 | 1.2793 | true | selection_changed_with_entropy_shift |
| classic_artifact_only_w1 | 4 | incumbent_proxy_disagreement-027 | 2 | 1.4917 | false | no_large_unexpected_drift |
| classic_artifact_only_w1 | 4 | incumbent_proxy_disagreement-029 | 4 | 1.0331 | false | no_large_unexpected_drift |
| classic_artifact_only_w2 | 1 | incumbent_proxy_disagreement-003 | 4 | 0.4975 | true | selection_changed_with_entropy_shift |
| classic_artifact_only_w2 | 1 | incumbent_proxy_disagreement-010 | 4 | 0.8956 | true | selection_changed_with_entropy_shift |
| classic_artifact_only_w2 | 1 | incumbent_proxy_disagreement-012 | 3 | 1.6054 | false | no_large_unexpected_drift |
| classic_artifact_only_w2 | 1 | incumbent_proxy_disagreement-018 | 2 | 1.5918 | false | no_large_unexpected_drift |
| classic_artifact_only_w2 | 1 | incumbent_proxy_disagreement-020 | 3 | 2.0958 | false | no_large_unexpected_drift |
| classic_artifact_only_w2 | 1 | incumbent_proxy_disagreement-023 | 3 | 1.8019 | false | no_large_unexpected_drift |
| classic_artifact_only_w2 | 1 | incumbent_proxy_disagreement-027 | 5 | 1.8443 | false | no_large_unexpected_drift |
| classic_artifact_only_w2 | 1 | incumbent_proxy_disagreement-029 | 4 | 1.2670 | false | no_large_unexpected_drift |
| classic_artifact_only_w2 | 2 | incumbent_proxy_disagreement-003 | 4 | 0.6625 | false | no_large_unexpected_drift |
| classic_artifact_only_w2 | 2 | incumbent_proxy_disagreement-010 | 4 | 1.0008 | false | no_large_unexpected_drift |
| classic_artifact_only_w2 | 2 | incumbent_proxy_disagreement-012 | 3 | 1.3898 | true | selection_changed_with_entropy_shift |
| classic_artifact_only_w2 | 2 | incumbent_proxy_disagreement-018 | 3 | 1.3596 | false | no_large_unexpected_drift |
| classic_artifact_only_w2 | 2 | incumbent_proxy_disagreement-020 | 3 | 1.8378 | false | no_large_unexpected_drift |
| classic_artifact_only_w2 | 2 | incumbent_proxy_disagreement-023 | 2 | 1.0872 | true | selection_changed_with_entropy_shift |
| classic_artifact_only_w2 | 2 | incumbent_proxy_disagreement-027 | 2 | 1.4408 | false | no_large_unexpected_drift |
| classic_artifact_only_w2 | 2 | incumbent_proxy_disagreement-029 | 2 | 1.5879 | false | no_large_unexpected_drift |
| classic_artifact_only_w2 | 4 | incumbent_proxy_disagreement-003 | 4 | 0.3195 | true | selection_changed_with_entropy_shift |
| classic_artifact_only_w2 | 4 | incumbent_proxy_disagreement-010 | 4 | 0.6543 | true | selection_changed_with_entropy_shift |
| classic_artifact_only_w2 | 4 | incumbent_proxy_disagreement-012 | 3 | 0.7811 | true | selection_changed_with_entropy_shift,large_entropy_shift |
| classic_artifact_only_w2 | 4 | incumbent_proxy_disagreement-018 | 2 | 0.9399 | true | selection_changed_with_entropy_shift |
| classic_artifact_only_w2 | 4 | incumbent_proxy_disagreement-020 | 3 | 1.5986 | true | selection_changed_with_entropy_shift |
| classic_artifact_only_w2 | 4 | incumbent_proxy_disagreement-023 | 2 | 1.2793 | true | selection_changed_with_entropy_shift |
| classic_artifact_only_w2 | 4 | incumbent_proxy_disagreement-027 | 2 | 1.4917 | false | no_large_unexpected_drift |
| classic_artifact_only_w2 | 4 | incumbent_proxy_disagreement-029 | 4 | 1.0331 | false | no_large_unexpected_drift |
| classic_artifact_plus_preservation_boost | 1 | incumbent_proxy_disagreement-003 | 4 | 0.7338 | false | no_large_unexpected_drift |
| classic_artifact_plus_preservation_boost | 1 | incumbent_proxy_disagreement-010 | 4 | 1.1084 | false | no_large_unexpected_drift |
| classic_artifact_plus_preservation_boost | 1 | incumbent_proxy_disagreement-012 | 1 | 1.6516 | false | no_large_unexpected_drift |
| classic_artifact_plus_preservation_boost | 1 | incumbent_proxy_disagreement-018 | 5 | 1.5527 | false | no_large_unexpected_drift |
| classic_artifact_plus_preservation_boost | 1 | incumbent_proxy_disagreement-020 | 3 | 2.0208 | false | no_large_unexpected_drift |
| classic_artifact_plus_preservation_boost | 1 | incumbent_proxy_disagreement-023 | 3 | 1.8555 | false | no_large_unexpected_drift |
| classic_artifact_plus_preservation_boost | 1 | incumbent_proxy_disagreement-027 | 5 | 1.7696 | false | no_large_unexpected_drift |
| classic_artifact_plus_preservation_boost | 1 | incumbent_proxy_disagreement-029 | 4 | 1.3980 | false | no_large_unexpected_drift |
| classic_artifact_plus_preservation_boost | 2 | incumbent_proxy_disagreement-003 | 4 | 0.5331 | false | no_large_unexpected_drift |
| classic_artifact_plus_preservation_boost | 2 | incumbent_proxy_disagreement-010 | 4 | 0.8069 | true | selection_changed_with_entropy_shift |
| classic_artifact_plus_preservation_boost | 2 | incumbent_proxy_disagreement-012 | 3 | 1.3436 | true | selection_changed_with_entropy_shift |
| classic_artifact_plus_preservation_boost | 2 | incumbent_proxy_disagreement-018 | 3 | 1.3398 | false | no_large_unexpected_drift |
| classic_artifact_plus_preservation_boost | 2 | incumbent_proxy_disagreement-020 | 3 | 1.8505 | false | no_large_unexpected_drift |
| classic_artifact_plus_preservation_boost | 2 | incumbent_proxy_disagreement-023 | 2 | 1.0020 | true | selection_changed_with_entropy_shift |
| classic_artifact_plus_preservation_boost | 2 | incumbent_proxy_disagreement-027 | 5 | 1.4831 | false | no_large_unexpected_drift |
| classic_artifact_plus_preservation_boost | 2 | incumbent_proxy_disagreement-029 | 4 | 1.5337 | false | no_large_unexpected_drift |
| classic_artifact_plus_preservation_boost | 4 | incumbent_proxy_disagreement-003 | 4 | 0.2981 | true | selection_changed_with_entropy_shift |
| classic_artifact_plus_preservation_boost | 4 | incumbent_proxy_disagreement-010 | 4 | 0.6692 | true | selection_changed_with_entropy_shift |
| classic_artifact_plus_preservation_boost | 4 | incumbent_proxy_disagreement-012 | 3 | 0.7611 | true | selection_changed_with_entropy_shift,large_entropy_shift |
| classic_artifact_plus_preservation_boost | 4 | incumbent_proxy_disagreement-018 | 3 | 0.9182 | false | no_large_unexpected_drift |
| classic_artifact_plus_preservation_boost | 4 | incumbent_proxy_disagreement-020 | 3 | 1.5555 | true | selection_changed_with_entropy_shift |
| classic_artifact_plus_preservation_boost | 4 | incumbent_proxy_disagreement-023 | 2 | 1.3179 | true | selection_changed_with_entropy_shift |
| classic_artifact_plus_preservation_boost | 4 | incumbent_proxy_disagreement-027 | 5 | 1.4793 | false | no_large_unexpected_drift |
| classic_artifact_plus_preservation_boost | 4 | incumbent_proxy_disagreement-029 | 4 | 1.0578 | false | no_large_unexpected_drift |

## 8. Training metrics

| trace_name | epoch | policy_loss | value_loss | total_loss | artifact_cross_entropy | target_candidate_cross_entropy | preservation_control_cross_entropy | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classic_artifact_only_w1 | 1 | 2.0976 | 0.0718 | 2.1192 | 2.0976 | 2.2853 | 1.1591 | post_epoch |
| classic_artifact_only_w1 | 2 | 1.6412 | 0.0514 | 1.6566 | 1.6412 | 1.8088 | 0.8032 | post_epoch |
| classic_artifact_only_w1 | 4 | 1.7080 | 0.0409 | 1.7203 | 1.7080 | 1.8994 | 0.7512 | post_epoch |
| classic_artifact_only_w2 | 1 | 2.0976 | 0.0718 | 2.1192 | 2.0976 | 2.2853 | 1.1591 | post_epoch |
| classic_artifact_only_w2 | 2 | 1.6412 | 0.0514 | 1.6566 | 1.6412 | 1.8088 | 0.8032 | post_epoch |
| classic_artifact_only_w2 | 4 | 1.7080 | 0.0409 | 1.7203 | 1.7080 | 1.8994 | 0.7512 | post_epoch |
| classic_artifact_plus_preservation_boost | 1 | 1.9734 | 0.0599 | 1.9913 | 1.9734 | 2.1820 | 0.9299 | post_epoch |
| classic_artifact_plus_preservation_boost | 2 | 1.7633 | 0.0406 | 1.7755 | 1.7633 | 1.9532 | 0.8140 | post_epoch |
| classic_artifact_plus_preservation_boost | 4 | 1.7694 | 0.0379 | 1.7807 | 1.7694 | 1.9788 | 0.7221 | post_epoch |

## 9. Interpretation

- Final classification: `cross_teacher_interference`.
- Acceptance checks: arena_run=`false`, model_promoted=`false`, active_references_mutated=`false`, puct_rows_trained=`false`, excluded_rows_trained=`false`.
- The local gate used 384 and 1200 search budgets only for per-row diagnostics; no arena or benchmark sweep was run.

## 10. Exactly one recommended next action

Recommendation: **keep Classic and PUCT branches isolated; do not production-train this artifact until interference is understood.**
