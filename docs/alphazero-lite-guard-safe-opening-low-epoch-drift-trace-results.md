# AlphaZero-lite Guard-Safe Opening Low-Epoch Drift Trace Results (Regression-Aware Compatibility)

Compatibility basis: post-init regression relative to the initializer.

## 1. Context

- PR #34 selected a statically guard-safe opening replay artifact for a low-epoch optimization drift trace.
- This run stayed diagnostic-only: no production training, no arena, no promotion, no artifact overwrite.
- Compatibility here means training did not worsen corrected guard behavior relative to the initializer, even if the initializer already had raw-policy mismatch.
- Corrected references: `/home/alex/Mancala/ai/ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- Current initialization artifact: `/home/alex/Mancala/ai/storage/ai/alphazero_lite/current`.

## 2. Selected artifact validation

| artifact_path | row_count | guard_rows_present | duplicate_conflicts | stale_reference_conflicts | status | notes |
| --- | --- | --- | --- | --- | --- | --- |
| `/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl` | 26 | true | 0 | 0 | `ok` | ok |

## 3. Trace variants

| trace_name | data_files | replay_weights | lr | epochs | batch_size | init_checkpoint | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| artifact_only_lr_1e-4 | `["/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl"]` | `[1]` | 0.0001 | `[0, 1, 2, 4, 8]` | 32 | `/tmp/azlite_guard_safe_opening_drift_trace/current_init_checkpoint.npz` | `completed` | artifact only diagnostic trace |
| artifact_only_lr_1e-5 | `["/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl"]` | `[1]` | 1e-05 | `[0, 1, 2, 4, 8]` | 32 | `/tmp/azlite_guard_safe_opening_drift_trace/current_init_checkpoint.npz` | `completed` | artifact only lower learning rate trace |
| artifact_plus_guard_controls_lr_1e-4 | `["/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl", "/tmp/azlite_guard_safe_opening_replay/guard_safe_controls_only.jsonl"]` | `[1, 2]` | 0.0001 | `[0, 1, 2, 4, 8]` | 32 | `/tmp/azlite_guard_safe_opening_drift_trace/current_init_checkpoint.npz` | `completed` | selected artifact plus upweighted guard controls |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | `["/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_edge_move_5_preference.jsonl"]` | `[1]` | 0.0001 | `[0, 1, 2]` | 32 | `/tmp/azlite_guard_safe_opening_drift_trace/current_init_checkpoint.npz` | `completed` | two-epoch leave-one-out microtrace |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | `["/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_missed_extra_turn_continuation.jsonl"]` | `[1]` | 0.0001 | `[0, 1, 2]` | 32 | `/tmp/azlite_guard_safe_opening_drift_trace/current_init_checkpoint.npz` | `completed` | two-epoch leave-one-out microtrace |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | `["/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl"]` | `[1]` | 0.0001 | `[0, 1, 2]` | 32 | `/tmp/azlite_guard_safe_opening_drift_trace/current_init_checkpoint.npz` | `completed` | two-epoch leave-one-out microtrace |

## 4. Guard policy drift results

| trace_name | checkpoint_step | row_id | corrected_reference_move | policy_top_move | reference_policy_probability | reference_policy_rank | policy_entropy | puct_selected_move_384 | puct_selected_move_1200 | selected_is_reference_384 | selected_is_reference_1200 | reference_visit_share_384 | reference_visit_share_1200 | selected_minus_reference_q_margin_384 | selected_minus_reference_q_margin_1200 | drift_classification | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| artifact_only_lr_1e-4 | 0 | capture_available-002 | 2 | 0 | 0.1299 | 3 | 2.0375 | 2 | 2 | true | true | 0.4036 | 0.5308 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 0 | capture_available-003 | 2 | 1 | 0.1050 | 3 | 1.6846 | 2 | 2 | true | true | 0.4740 | 0.6742 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 0 | capture_available-006 | 2 | 1 | 0.1098 | 4 | 2.0740 | 2 | 2 | true | true | 0.5547 | 0.6600 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 0 | capture_available-007 | 2 | 1 | 0.2590 | 2 | 1.9299 | 2 | 2 | true | true | 0.5859 | 0.7400 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 0 | capture_available-008 | 1 | 1 | 0.6554 | 1 | 1.5620 | 1 | 1 | true | true | 0.7682 | 0.8733 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_only_lr_1e-4 | 1 | capture_available-002 | 2 | 0 | 0.1467 | 3 | 2.1211 | 2 | 2 | true | true | 0.4557 | 0.5392 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 1 | capture_available-003 | 2 | 1 | 0.1095 | 4 | 1.7335 | 2 | 2 | true | true | 0.4557 | 0.6433 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 1 | capture_available-006 | 2 | 1 | 0.1208 | 4 | 2.1459 | 2 | 2 | true | true | 0.5130 | 0.6533 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 1 | capture_available-007 | 2 | 1 | 0.2597 | 2 | 1.9501 | 2 | 2 | true | true | 0.5703 | 0.7400 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 1 | capture_available-008 | 1 | 1 | 0.6535 | 1 | 1.5681 | 1 | 1 | true | true | 0.7943 | 0.8692 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_only_lr_1e-4 | 2 | capture_available-002 | 2 | 0 | 0.1644 | 3 | 2.1882 | 2 | 2 | true | true | 0.4401 | 0.5625 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 2 | capture_available-003 | 2 | 1 | 0.1130 | 4 | 1.7888 | 2 | 2 | true | true | 0.4167 | 0.6150 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 2 | capture_available-006 | 2 | 1 | 0.1293 | 4 | 2.2034 | 2 | 2 | true | true | 0.4974 | 0.6658 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 2 | capture_available-007 | 2 | 1 | 0.2588 | 2 | 1.9743 | 2 | 2 | true | true | 0.5391 | 0.7283 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 2 | capture_available-008 | 1 | 1 | 0.6442 | 1 | 1.5912 | 1 | 1 | true | true | 0.7865 | 0.8725 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_only_lr_1e-4 | 4 | capture_available-002 | 2 | 1 | 0.2007 | 3 | 2.2800 | 2 | 2 | true | true | 0.3490 | 0.5633 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 4 | capture_available-003 | 2 | 1 | 0.1238 | 4 | 1.9256 | 2 | 2 | true | true | 0.4193 | 0.6100 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 4 | capture_available-006 | 2 | 1 | 0.1425 | 5 | 2.2747 | 2 | 2 | true | true | 0.5130 | 0.6475 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 4 | capture_available-007 | 2 | 1 | 0.2727 | 2 | 2.0321 | 2 | 2 | true | true | 0.5208 | 0.7083 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 4 | capture_available-008 | 1 | 1 | 0.5967 | 1 | 1.6954 | 1 | 1 | true | true | 0.7812 | 0.8617 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_only_lr_1e-4 | 8 | capture_available-002 | 2 | 2 | 0.2666 | 1 | 2.2984 | 2 | 2 | true | true | 0.3750 | 0.5500 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_only_lr_1e-4 | 8 | capture_available-003 | 2 | 1 | 0.1820 | 4 | 2.0825 | 2 | 2 | true | true | 0.4609 | 0.5892 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 8 | capture_available-006 | 2 | 1 | 0.1781 | 4 | 2.2773 | 2 | 2 | true | true | 0.5365 | 0.6633 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 8 | capture_available-007 | 2 | 2 | 0.3620 | 1 | 2.0228 | 2 | 2 | true | true | 0.6068 | 0.7300 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_only_lr_1e-4 | 8 | capture_available-008 | 1 | 1 | 0.4896 | 1 | 1.8839 | 1 | 1 | true | true | 0.7161 | 0.7808 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_only_lr_1e-5 | 0 | capture_available-002 | 2 | 0 | 0.1299 | 3 | 2.0375 | 2 | 2 | true | true | 0.4036 | 0.5308 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 0 | capture_available-003 | 2 | 1 | 0.1050 | 3 | 1.6846 | 2 | 2 | true | true | 0.4740 | 0.6742 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 0 | capture_available-006 | 2 | 1 | 0.1098 | 4 | 2.0740 | 2 | 2 | true | true | 0.5547 | 0.6600 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 0 | capture_available-007 | 2 | 1 | 0.2590 | 2 | 1.9299 | 2 | 2 | true | true | 0.5859 | 0.7400 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 0 | capture_available-008 | 1 | 1 | 0.6554 | 1 | 1.5620 | 1 | 1 | true | true | 0.7682 | 0.8733 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_only_lr_1e-5 | 1 | capture_available-002 | 2 | 0 | 0.1315 | 3 | 2.0468 | 2 | 2 | true | true | 0.4323 | 0.5400 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 1 | capture_available-003 | 2 | 1 | 0.1056 | 3 | 1.6901 | 2 | 2 | true | true | 0.4818 | 0.6758 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 1 | capture_available-006 | 2 | 1 | 0.1109 | 4 | 2.0819 | 2 | 2 | true | true | 0.5391 | 0.6675 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 1 | capture_available-007 | 2 | 1 | 0.2593 | 2 | 1.9318 | 2 | 2 | true | true | 0.5833 | 0.7375 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 1 | capture_available-008 | 1 | 1 | 0.6553 | 1 | 1.5626 | 1 | 1 | true | true | 0.7708 | 0.8725 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_only_lr_1e-5 | 2 | capture_available-002 | 2 | 0 | 0.1332 | 3 | 2.0560 | 2 | 2 | true | true | 0.4531 | 0.5417 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 2 | capture_available-003 | 2 | 1 | 0.1061 | 4 | 1.6954 | 2 | 2 | true | true | 0.4557 | 0.6733 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 2 | capture_available-006 | 2 | 1 | 0.1120 | 4 | 2.0897 | 2 | 2 | true | true | 0.5417 | 0.6575 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 2 | capture_available-007 | 2 | 1 | 0.2595 | 2 | 1.9339 | 2 | 2 | true | true | 0.5755 | 0.7433 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 2 | capture_available-008 | 1 | 1 | 0.6550 | 1 | 1.5633 | 1 | 1 | true | true | 0.7734 | 0.8683 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_only_lr_1e-5 | 4 | capture_available-002 | 2 | 0 | 0.1366 | 3 | 2.0741 | 2 | 2 | true | true | 0.4297 | 0.5250 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 4 | capture_available-003 | 2 | 1 | 0.1070 | 4 | 1.7060 | 2 | 2 | true | true | 0.4479 | 0.6667 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 4 | capture_available-006 | 2 | 1 | 0.1142 | 4 | 2.1050 | 2 | 2 | true | true | 0.5156 | 0.6633 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 4 | capture_available-007 | 2 | 1 | 0.2601 | 2 | 1.9383 | 2 | 2 | true | true | 0.5938 | 0.7400 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 4 | capture_available-008 | 1 | 1 | 0.6545 | 1 | 1.5651 | 1 | 1 | true | true | 0.7812 | 0.8700 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_only_lr_1e-5 | 8 | capture_available-002 | 2 | 0 | 0.1435 | 3 | 2.1070 | 2 | 2 | true | true | 0.4453 | 0.5492 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 8 | capture_available-003 | 2 | 1 | 0.1087 | 4 | 1.7269 | 2 | 2 | true | true | 0.4245 | 0.6592 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 8 | capture_available-006 | 2 | 1 | 0.1186 | 4 | 2.1331 | 2 | 2 | true | true | 0.5182 | 0.6575 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 8 | capture_available-007 | 2 | 1 | 0.2603 | 2 | 1.9476 | 2 | 2 | true | true | 0.5833 | 0.7383 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 8 | capture_available-008 | 1 | 1 | 0.6530 | 1 | 1.5695 | 1 | 1 | true | true | 0.7969 | 0.8675 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_plus_guard_controls_lr_1e-4 | 0 | capture_available-002 | 2 | 0 | 0.1299 | 3 | 2.0375 | 2 | 2 | true | true | 0.4036 | 0.5308 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_plus_guard_controls_lr_1e-4 | 0 | capture_available-003 | 2 | 1 | 0.1050 | 3 | 1.6846 | 2 | 2 | true | true | 0.4740 | 0.6742 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_plus_guard_controls_lr_1e-4 | 0 | capture_available-006 | 2 | 1 | 0.1098 | 4 | 2.0740 | 2 | 2 | true | true | 0.5547 | 0.6600 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_plus_guard_controls_lr_1e-4 | 0 | capture_available-007 | 2 | 1 | 0.2590 | 2 | 1.9299 | 2 | 2 | true | true | 0.5859 | 0.7400 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_plus_guard_controls_lr_1e-4 | 0 | capture_available-008 | 1 | 1 | 0.6554 | 1 | 1.5620 | 1 | 1 | true | true | 0.7682 | 0.8733 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_plus_guard_controls_lr_1e-4 | 1 | capture_available-002 | 2 | 0 | 0.1735 | 3 | 2.1494 | 2 | 2 | true | true | 0.4531 | 0.5608 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_plus_guard_controls_lr_1e-4 | 1 | capture_available-003 | 2 | 1 | 0.1638 | 2 | 1.9123 | 2 | 2 | true | true | 0.5156 | 0.6992 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_plus_guard_controls_lr_1e-4 | 1 | capture_available-006 | 2 | 1 | 0.1404 | 4 | 2.1802 | 2 | 2 | true | true | 0.5391 | 0.6633 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_plus_guard_controls_lr_1e-4 | 1 | capture_available-007 | 2 | 2 | 0.3536 | 1 | 1.9867 | 2 | 2 | true | true | 0.6302 | 0.7700 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_plus_guard_controls_lr_1e-4 | 1 | capture_available-008 | 1 | 1 | 0.5712 | 1 | 1.7390 | 1 | 1 | true | true | 0.7604 | 0.8558 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_plus_guard_controls_lr_1e-4 | 2 | capture_available-002 | 2 | 0 | 0.2285 | 3 | 2.2256 | 2 | 2 | true | true | 0.4531 | 0.5725 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_plus_guard_controls_lr_1e-4 | 2 | capture_available-003 | 2 | 1 | 0.2296 | 2 | 2.0406 | 2 | 2 | true | true | 0.5521 | 0.7108 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_plus_guard_controls_lr_1e-4 | 2 | capture_available-006 | 2 | 1 | 0.1724 | 3 | 2.2539 | 2 | 2 | true | true | 0.5156 | 0.6908 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_plus_guard_controls_lr_1e-4 | 2 | capture_available-007 | 2 | 2 | 0.4399 | 1 | 1.9379 | 2 | 2 | true | true | 0.6562 | 0.7917 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_plus_guard_controls_lr_1e-4 | 2 | capture_available-008 | 1 | 1 | 0.4904 | 1 | 1.8514 | 1 | 1 | true | true | 0.7344 | 0.8367 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_plus_guard_controls_lr_1e-4 | 4 | capture_available-002 | 2 | 2 | 0.3324 | 1 | 2.2353 | 2 | 2 | true | true | 0.4505 | 0.5567 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_plus_guard_controls_lr_1e-4 | 4 | capture_available-003 | 2 | 2 | 0.3389 | 1 | 2.0536 | 2 | 2 | true | true | 0.5833 | 0.7458 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_plus_guard_controls_lr_1e-4 | 4 | capture_available-006 | 2 | 1 | 0.2420 | 2 | 2.2911 | 2 | 2 | true | true | 0.5443 | 0.6942 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_plus_guard_controls_lr_1e-4 | 4 | capture_available-007 | 2 | 2 | 0.5435 | 1 | 1.7765 | 2 | 2 | true | true | 0.6641 | 0.8250 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_plus_guard_controls_lr_1e-4 | 4 | capture_available-008 | 1 | 1 | 0.3983 | 1 | 1.8917 | 1 | 1 | true | true | 0.7266 | 0.8192 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_plus_guard_controls_lr_1e-4 | 8 | capture_available-002 | 2 | 2 | 0.4763 | 1 | 2.0280 | 2 | 2 | true | true | 0.5391 | 0.6367 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_plus_guard_controls_lr_1e-4 | 8 | capture_available-003 | 2 | 2 | 0.4435 | 1 | 1.9137 | 2 | 2 | true | true | 0.6328 | 0.7108 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_plus_guard_controls_lr_1e-4 | 8 | capture_available-006 | 2 | 2 | 0.3370 | 1 | 2.1882 | 2 | 2 | true | true | 0.5703 | 0.7117 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_plus_guard_controls_lr_1e-4 | 8 | capture_available-007 | 2 | 2 | 0.6220 | 1 | 1.5760 | 2 | 2 | true | true | 0.7109 | 0.8567 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_plus_guard_controls_lr_1e-4 | 8 | capture_available-008 | 1 | 1 | 0.4105 | 1 | 1.7818 | 1 | 1 | true | true | 0.6901 | 0.7900 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 0 | capture_available-002 | 2 | 0 | 0.1299 | 3 | 2.0375 | 2 | 2 | true | true | 0.4036 | 0.5308 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 0 | capture_available-003 | 2 | 1 | 0.1050 | 3 | 1.6846 | 2 | 2 | true | true | 0.4740 | 0.6742 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 0 | capture_available-006 | 2 | 1 | 0.1098 | 4 | 2.0740 | 2 | 2 | true | true | 0.5547 | 0.6600 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 0 | capture_available-007 | 2 | 1 | 0.2590 | 2 | 1.9299 | 2 | 2 | true | true | 0.5859 | 0.7400 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 0 | capture_available-008 | 1 | 1 | 0.6554 | 1 | 1.5620 | 1 | 1 | true | true | 0.7682 | 0.8733 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 1 | capture_available-002 | 2 | 0 | 0.1300 | 3 | 2.0703 | 2 | 2 | true | true | 0.4557 | 0.5500 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 1 | capture_available-003 | 2 | 1 | 0.1051 | 4 | 1.8305 | 2 | 2 | true | true | 0.4583 | 0.6758 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 1 | capture_available-006 | 2 | 1 | 0.1043 | 4 | 2.1199 | 2 | 2 | true | true | 0.5312 | 0.6408 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 1 | capture_available-007 | 2 | 1 | 0.2564 | 2 | 2.0087 | 2 | 2 | true | true | 0.5859 | 0.7358 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 1 | capture_available-008 | 1 | 1 | 0.6106 | 1 | 1.6941 | 1 | 1 | true | true | 0.7552 | 0.8525 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 2 | capture_available-002 | 2 | 0 | 0.1289 | 4 | 2.1069 | 2 | 2 | true | true | 0.4062 | 0.5550 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 2 | capture_available-003 | 2 | 1 | 0.1023 | 4 | 1.9209 | 2 | 2 | true | true | 0.4479 | 0.6708 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 2 | capture_available-006 | 2 | 1 | 0.0998 | 5 | 2.1580 | 2 | 2 | true | true | 0.5182 | 0.6258 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 2 | capture_available-007 | 2 | 1 | 0.2441 | 2 | 2.0614 | 2 | 2 | true | true | 0.5521 | 0.7175 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 2 | capture_available-008 | 1 | 1 | 0.5788 | 1 | 1.7823 | 1 | 1 | true | true | 0.7344 | 0.8350 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 0 | capture_available-002 | 2 | 0 | 0.1299 | 3 | 2.0375 | 2 | 2 | true | true | 0.4036 | 0.5308 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 0 | capture_available-003 | 2 | 1 | 0.1050 | 3 | 1.6846 | 2 | 2 | true | true | 0.4740 | 0.6742 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 0 | capture_available-006 | 2 | 1 | 0.1098 | 4 | 2.0740 | 2 | 2 | true | true | 0.5547 | 0.6600 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 0 | capture_available-007 | 2 | 1 | 0.2590 | 2 | 1.9299 | 2 | 2 | true | true | 0.5859 | 0.7400 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 0 | capture_available-008 | 1 | 1 | 0.6554 | 1 | 1.5620 | 1 | 1 | true | true | 0.7682 | 0.8733 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 1 | capture_available-002 | 2 | 0 | 0.1464 | 3 | 2.1178 | 2 | 2 | true | true | 0.4766 | 0.5067 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 1 | capture_available-003 | 2 | 1 | 0.1249 | 4 | 1.9295 | 2 | 2 | true | true | 0.4479 | 0.7017 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 1 | capture_available-006 | 2 | 1 | 0.1168 | 4 | 2.1767 | 2 | 2 | true | true | 0.5495 | 0.6442 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 1 | capture_available-007 | 2 | 1 | 0.2930 | 2 | 2.0498 | 2 | 2 | true | true | 0.5911 | 0.7417 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 1 | capture_available-008 | 1 | 1 | 0.5697 | 1 | 1.7791 | 1 | 1 | true | true | 0.7396 | 0.8525 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 2 | capture_available-002 | 2 | 0 | 0.1590 | 3 | 2.1947 | 2 | 2 | true | true | 0.4323 | 0.4525 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 2 | capture_available-003 | 2 | 1 | 0.1396 | 4 | 2.0863 | 2 | 2 | true | true | 0.4688 | 0.7217 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 2 | capture_available-006 | 2 | 1 | 0.1206 | 5 | 2.2473 | 2 | 2 | true | true | 0.5000 | 0.6183 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 2 | capture_available-007 | 2 | 2 | 0.3148 | 1 | 2.1122 | 2 | 2 | true | true | 0.5807 | 0.7608 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 2 | capture_available-008 | 1 | 1 | 0.4831 | 1 | 1.9510 | 1 | 1 | true | true | 0.7135 | 0.8175 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 0 | capture_available-002 | 2 | 0 | 0.1299 | 3 | 2.0375 | 2 | 2 | true | true | 0.4036 | 0.5308 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 0 | capture_available-003 | 2 | 1 | 0.1050 | 3 | 1.6846 | 2 | 2 | true | true | 0.4740 | 0.6742 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 0 | capture_available-006 | 2 | 1 | 0.1098 | 4 | 2.0740 | 2 | 2 | true | true | 0.5547 | 0.6600 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 0 | capture_available-007 | 2 | 1 | 0.2590 | 2 | 1.9299 | 2 | 2 | true | true | 0.5859 | 0.7400 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 0 | capture_available-008 | 1 | 1 | 0.6554 | 1 | 1.5620 | 1 | 1 | true | true | 0.7682 | 0.8733 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 1 | capture_available-002 | 2 | 0 | 0.1467 | 3 | 2.1211 | 2 | 2 | true | true | 0.4557 | 0.5392 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 1 | capture_available-003 | 2 | 1 | 0.1095 | 4 | 1.7335 | 2 | 2 | true | true | 0.4557 | 0.6433 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 1 | capture_available-006 | 2 | 1 | 0.1208 | 4 | 2.1459 | 2 | 2 | true | true | 0.5130 | 0.6533 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 1 | capture_available-007 | 2 | 1 | 0.2597 | 2 | 1.9501 | 2 | 2 | true | true | 0.5703 | 0.7400 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 1 | capture_available-008 | 1 | 1 | 0.6535 | 1 | 1.5681 | 1 | 1 | true | true | 0.7943 | 0.8692 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 2 | capture_available-002 | 2 | 0 | 0.1644 | 3 | 2.1882 | 2 | 2 | true | true | 0.4401 | 0.5625 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 2 | capture_available-003 | 2 | 1 | 0.1130 | 4 | 1.7888 | 2 | 2 | true | true | 0.4167 | 0.6150 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 2 | capture_available-006 | 2 | 1 | 0.1293 | 4 | 2.2034 | 2 | 2 | true | true | 0.4974 | 0.6658 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 2 | capture_available-007 | 2 | 1 | 0.2588 | 2 | 1.9743 | 2 | 2 | true | true | 0.5391 | 0.7283 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 2 | capture_available-008 | 1 | 1 | 0.6442 | 1 | 1.5912 | 1 | 1 | true | true | 0.7865 | 0.8725 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |

## 5. Guard search drift results

- Evaluated rows: `["capture_available-002", "capture_available-003", "capture_available-006", "capture_available-007", "capture_available-008"]`.
- Search settings matched the prior corrected-reference control baseline via `build_eval_search_options()` with no new root-prior intervention.

## 6. Training metric trace

| trace_name | checkpoint_step | policy_loss | value_loss | total_loss | guard_cross_entropy | non_guard_opening_cross_entropy | policy_head_delta_norm | value_head_delta_norm | trunk_delta_norm | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| artifact_only_lr_1e-4 | 0 | 2.2755 | 0.0764 | 2.2984 | 1.8396 | 2.3793 | 0.0000 | 0.0000 | 0.0000 | initial_checkpoint |
| artifact_only_lr_1e-4 | 1 | 2.1540 | 0.0727 | 2.1758 | 1.7668 | 2.2462 | 0.0087 | 0.0055 | 0.0205 | post_epoch |
| artifact_only_lr_1e-4 | 2 | 2.0529 | 0.0687 | 2.0735 | 1.7098 | 2.1345 | 0.0169 | 0.0111 | 0.0399 | post_epoch |
| artifact_only_lr_1e-4 | 4 | 1.9008 | 0.0600 | 1.9188 | 1.6136 | 1.9692 | 0.0316 | 0.0221 | 0.0751 | post_epoch |
| artifact_only_lr_1e-4 | 8 | 1.7086 | 0.0444 | 1.7219 | 1.4522 | 1.7696 | 0.0548 | 0.0442 | 0.1319 | post_epoch |
| artifact_only_lr_1e-5 | 0 | 2.2755 | 0.0764 | 2.2984 | 1.8396 | 2.3793 | 0.0000 | 0.0000 | 0.0000 | initial_checkpoint |
| artifact_only_lr_1e-5 | 1 | 2.2624 | 0.0760 | 2.2852 | 1.8315 | 2.3650 | 0.0009 | 0.0006 | 0.0021 | post_epoch |
| artifact_only_lr_1e-5 | 2 | 2.2495 | 0.0756 | 2.2722 | 1.8237 | 2.3509 | 0.0017 | 0.0011 | 0.0041 | post_epoch |
| artifact_only_lr_1e-5 | 4 | 2.2241 | 0.0749 | 2.2466 | 1.8084 | 2.3231 | 0.0035 | 0.0022 | 0.0082 | post_epoch |
| artifact_only_lr_1e-5 | 8 | 2.1760 | 0.0733 | 2.1980 | 1.7806 | 2.2701 | 0.0069 | 0.0044 | 0.0162 | post_epoch |
| artifact_plus_guard_controls_lr_1e-4 | 0 | 2.2052 | 0.0847 | 2.2306 | 1.8396 | 2.3793 | 0.0000 | 0.0000 | 0.0000 | initial_checkpoint |
| artifact_plus_guard_controls_lr_1e-4 | 1 | 2.0950 | 0.0779 | 2.1183 | 1.6295 | 2.3166 | 0.0087 | 0.0055 | 0.0205 | post_epoch |
| artifact_plus_guard_controls_lr_1e-4 | 2 | 2.0047 | 0.0716 | 2.0262 | 1.4689 | 2.2599 | 0.0169 | 0.0111 | 0.0398 | post_epoch |
| artifact_plus_guard_controls_lr_1e-4 | 4 | 1.8710 | 0.0609 | 1.8893 | 1.2723 | 2.1561 | 0.0319 | 0.0221 | 0.0756 | post_epoch |
| artifact_plus_guard_controls_lr_1e-4 | 8 | 1.6739 | 0.0433 | 1.6869 | 1.1085 | 1.9431 | 0.0552 | 0.0442 | 0.1352 | post_epoch |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 0 | 2.4012 | 0.0849 | 2.4267 | 1.8396 | 2.5233 | 0.0000 | 0.0000 | 0.0000 | initial_checkpoint |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 1 | 2.2548 | 0.0787 | 2.2784 | 1.8419 | 2.3446 | 0.0088 | 0.0056 | 0.0205 | post_epoch |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 2 | 2.1291 | 0.0736 | 2.1511 | 1.8517 | 2.1893 | 0.0171 | 0.0111 | 0.0397 | post_epoch |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 0 | 2.0776 | 0.0723 | 2.0993 | 1.8396 | 2.1216 | 0.0000 | 0.0000 | 0.0000 | initial_checkpoint |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 1 | 1.9695 | 0.0658 | 1.9892 | 1.7529 | 2.0096 | 0.0088 | 0.0056 | 0.0206 | post_epoch |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 2 | 1.8805 | 0.0600 | 1.8985 | 1.7021 | 1.9136 | 0.0173 | 0.0113 | 0.0401 | post_epoch |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 0 | 2.2755 | 0.0764 | 2.2984 | 1.8396 | 2.3793 | 0.0000 | 0.0000 | 0.0000 | initial_checkpoint |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 1 | 2.1540 | 0.0727 | 2.1758 | 1.7668 | 2.2462 | 0.0087 | 0.0055 | 0.0205 | post_epoch |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 2 | 2.0529 | 0.0687 | 2.0735 | 1.7098 | 2.1345 | 0.0169 | 0.0111 | 0.0399 | post_epoch |

## 7. Family/Subset Sensitivity

| trace_name | row_outcomes | status | notes |
| --- | --- | --- | --- |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | `{"capture_available-002": {"baseline_classification": "policy_drift_only", "baseline_step": 0, "classification": "policy_drift_only", "first_drift_step": 0, "first_regression_step": null, "has_regression": false, "regression_classification": null, "regression_timing": null, "timing": "immediate_drift"}, "capture_available-003": {"baseline_classification": "policy_drift_only", "baseline_step": 0, "classification": "policy_drift_only", "first_drift_step": 0, "first_regression_step": null, "has_regression": false, "regression_classification": null, "regression_timing": null, "timing": "immediate_drift"}, "capture_available-006": {"baseline_classification": "policy_drift_only", "baseline_step": 0, "classification": "policy_drift_only", "first_drift_step": 0, "first_regression_step": null, "has_regression": false, "regression_classification": null, "regression_timing": null, "timing": "immediate_drift"}, "capture_available-007": {"baseline_classification": "policy_drift_only", "baseline_step": 0, "classification": "policy_drift_only", "first_drift_step": 0, "first_regression_step": null, "has_regression": false, "regression_classification": null, "regression_timing": null, "timing": "immediate_drift"}, "capture_available-008": {"baseline_classification": "stable_guard", "baseline_step": 0, "classification": "stable_guard", "first_drift_step": null, "first_regression_step": null, "has_regression": false, "regression_classification": null, "regression_timing": null, "timing": null}}` | `completed` | two-epoch leave-one-out microtrace |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | `{"capture_available-002": {"baseline_classification": "policy_drift_only", "baseline_step": 0, "classification": "policy_drift_only", "first_drift_step": 0, "first_regression_step": null, "has_regression": false, "regression_classification": null, "regression_timing": null, "timing": "immediate_drift"}, "capture_available-003": {"baseline_classification": "policy_drift_only", "baseline_step": 0, "classification": "policy_drift_only", "first_drift_step": 0, "first_regression_step": null, "has_regression": false, "regression_classification": null, "regression_timing": null, "timing": "immediate_drift"}, "capture_available-006": {"baseline_classification": "policy_drift_only", "baseline_step": 0, "classification": "policy_drift_only", "first_drift_step": 0, "first_regression_step": null, "has_regression": false, "regression_classification": null, "regression_timing": null, "timing": "immediate_drift"}, "capture_available-007": {"baseline_classification": "policy_drift_only", "baseline_step": 0, "classification": "policy_drift_only", "first_drift_step": 0, "first_regression_step": null, "has_regression": false, "regression_classification": null, "regression_timing": null, "timing": "immediate_drift"}, "capture_available-008": {"baseline_classification": "stable_guard", "baseline_step": 0, "classification": "stable_guard", "first_drift_step": null, "first_regression_step": null, "has_regression": false, "regression_classification": null, "regression_timing": null, "timing": null}}` | `completed` | two-epoch leave-one-out microtrace |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | `{"capture_available-002": {"baseline_classification": "policy_drift_only", "baseline_step": 0, "classification": "policy_drift_only", "first_drift_step": 0, "first_regression_step": null, "has_regression": false, "regression_classification": null, "regression_timing": null, "timing": "immediate_drift"}, "capture_available-003": {"baseline_classification": "policy_drift_only", "baseline_step": 0, "classification": "policy_drift_only", "first_drift_step": 0, "first_regression_step": null, "has_regression": false, "regression_classification": null, "regression_timing": null, "timing": "immediate_drift"}, "capture_available-006": {"baseline_classification": "policy_drift_only", "baseline_step": 0, "classification": "policy_drift_only", "first_drift_step": 0, "first_regression_step": null, "has_regression": false, "regression_classification": null, "regression_timing": null, "timing": "immediate_drift"}, "capture_available-007": {"baseline_classification": "policy_drift_only", "baseline_step": 0, "classification": "policy_drift_only", "first_drift_step": 0, "first_regression_step": null, "has_regression": false, "regression_classification": null, "regression_timing": null, "timing": "immediate_drift"}, "capture_available-008": {"baseline_classification": "stable_guard", "baseline_step": 0, "classification": "stable_guard", "first_drift_step": null, "first_regression_step": null, "has_regression": false, "regression_classification": null, "regression_timing": null, "timing": null}}` | `completed` | two-epoch leave-one-out microtrace |

## 8. Interpretation

- Classification: `compatible_with_inherited_policy_drift`.
- Interpretation: No post-init guard regression was observed; remaining policy-only mismatch was inherited from the initializer.
- Primary question is answered only if artifact validation passed and drift rows were collected.

## 9. Exactly one recommended next action

Recommendation: **use the selected artifact lane only with the existing corrected guard search gate; training did not worsen inherited guard policy mismatch.**.
