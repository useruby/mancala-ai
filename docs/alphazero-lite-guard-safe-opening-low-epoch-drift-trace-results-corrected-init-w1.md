# AlphaZero-lite Guard-Safe Opening Low-Epoch Drift Trace Results (Regression-Aware Compatibility)

Compatibility basis: post-init regression relative to the initializer.

## 1. Context

- PR #34 selected a statically guard-safe opening replay artifact for a low-epoch optimization drift trace.
- This run stayed diagnostic-only: no production training, no arena, no promotion, no artifact overwrite.
- Compatibility here means training did not worsen corrected guard behavior relative to the initializer, even if the initializer already had raw-policy mismatch.
- Corrected references: `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- Current initialization artifact: `/tmp/azlite_corrected_reference_targeted_hard_state_replay/corrected-reference-targeted-hard-state-replay/variants/w1/versions/aggressive-v3-targeted-hard-state-replay-hard-state-replay-w1-iter1`.

## 2. Selected artifact validation

| artifact_path | row_count | guard_rows_present | duplicate_conflicts | stale_reference_conflicts | status | notes |
| --- | --- | --- | --- | --- | --- | --- |
| `/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl` | 26 | true | 0 | 0 | `ok` | ok |

## 3. Trace variants

| trace_name | data_files | replay_weights | lr | epochs | batch_size | init_checkpoint | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| artifact_only_lr_1e-4 | `["/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl"]` | `[1]` | 0.0001 | `[0, 1, 2, 4, 8]` | 32 | `/tmp/azlite_corrected_reference_targeted_hard_state_replay/corrected-reference-targeted-hard-state-replay/variants/w1/versions/aggressive-v3-targeted-hard-state-replay-hard-state-replay-w1-iter1/checkpoint.npz` | `completed` | artifact only diagnostic trace |
| artifact_only_lr_1e-5 | `["/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl"]` | `[1]` | 1e-05 | `[0, 1, 2, 4, 8]` | 32 | `/tmp/azlite_corrected_reference_targeted_hard_state_replay/corrected-reference-targeted-hard-state-replay/variants/w1/versions/aggressive-v3-targeted-hard-state-replay-hard-state-replay-w1-iter1/checkpoint.npz` | `completed` | artifact only lower learning rate trace |
| artifact_plus_guard_controls_lr_1e-4 | `["/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl", "/tmp/azlite_guard_safe_opening_replay/guard_safe_controls_only.jsonl"]` | `[1, 2]` | 0.0001 | `[0, 1, 2, 4, 8]` | 32 | `/tmp/azlite_corrected_reference_targeted_hard_state_replay/corrected-reference-targeted-hard-state-replay/variants/w1/versions/aggressive-v3-targeted-hard-state-replay-hard-state-replay-w1-iter1/checkpoint.npz` | `completed` | selected artifact plus upweighted guard controls |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | `["/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_edge_move_5_preference.jsonl"]` | `[1]` | 0.0001 | `[0, 1, 2]` | 32 | `/tmp/azlite_corrected_reference_targeted_hard_state_replay/corrected-reference-targeted-hard-state-replay/variants/w1/versions/aggressive-v3-targeted-hard-state-replay-hard-state-replay-w1-iter1/checkpoint.npz` | `completed` | two-epoch leave-one-out microtrace |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | `["/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_missed_extra_turn_continuation.jsonl"]` | `[1]` | 0.0001 | `[0, 1, 2]` | 32 | `/tmp/azlite_corrected_reference_targeted_hard_state_replay/corrected-reference-targeted-hard-state-replay/variants/w1/versions/aggressive-v3-targeted-hard-state-replay-hard-state-replay-w1-iter1/checkpoint.npz` | `completed` | two-epoch leave-one-out microtrace |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | `["/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl"]` | `[1]` | 0.0001 | `[0, 1, 2]` | 32 | `/tmp/azlite_corrected_reference_targeted_hard_state_replay/corrected-reference-targeted-hard-state-replay/variants/w1/versions/aggressive-v3-targeted-hard-state-replay-hard-state-replay-w1-iter1/checkpoint.npz` | `completed` | two-epoch leave-one-out microtrace |

## 4. Guard policy drift results

| trace_name | checkpoint_step | row_id | corrected_reference_move | policy_top_move | reference_policy_probability | reference_policy_rank | policy_entropy | puct_selected_move_384 | puct_selected_move_1200 | selected_is_reference_384 | selected_is_reference_1200 | reference_visit_share_384 | reference_visit_share_1200 | selected_minus_reference_q_margin_384 | selected_minus_reference_q_margin_1200 | drift_classification | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| artifact_only_lr_1e-4 | 0 | capture_available-002 | 2 | 0 | 0.1579 | 3 | 2.1321 | 2 | 2 | true | true | 0.7188 | 0.8592 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 0 | capture_available-003 | 2 | 1 | 0.1307 | 4 | 2.1916 | 2 | 2 | true | true | 0.8099 | 0.9092 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 0 | capture_available-006 | 2 | 0 | 0.1792 | 3 | 2.2120 | 2 | 2 | true | true | 0.7214 | 0.8883 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 0 | capture_available-007 | 2 | 1 | 0.1806 | 3 | 2.2296 | 2 | 2 | true | true | 0.6589 | 0.8692 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 0 | capture_available-008 | 1 | 1 | 0.3403 | 1 | 2.1637 | 1 | 1 | true | true | 0.7865 | 0.9025 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_only_lr_1e-4 | 1 | capture_available-002 | 2 | 0 | 0.1871 | 3 | 2.1958 | 2 | 2 | true | true | 0.7188 | 0.8625 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 1 | capture_available-003 | 2 | 1 | 0.1478 | 4 | 2.2003 | 2 | 2 | true | true | 0.8021 | 0.8975 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 1 | capture_available-006 | 2 | 1 | 0.2080 | 3 | 2.2535 | 2 | 2 | true | true | 0.6953 | 0.8917 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 1 | capture_available-007 | 2 | 1 | 0.2016 | 2 | 2.2372 | 2 | 2 | true | true | 0.6615 | 0.8700 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 1 | capture_available-008 | 1 | 1 | 0.3653 | 1 | 2.1764 | 1 | 1 | true | true | 0.8073 | 0.8992 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_only_lr_1e-4 | 2 | capture_available-002 | 2 | 1 | 0.2191 | 3 | 2.2329 | 2 | 2 | true | true | 0.7500 | 0.8700 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 2 | capture_available-003 | 2 | 1 | 0.1651 | 2 | 2.2022 | 2 | 2 | true | true | 0.8073 | 0.9025 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 2 | capture_available-006 | 2 | 1 | 0.2365 | 2 | 2.2716 | 2 | 2 | true | true | 0.7031 | 0.8808 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 2 | capture_available-007 | 2 | 1 | 0.2254 | 2 | 2.2362 | 2 | 2 | true | true | 0.6172 | 0.8667 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 2 | capture_available-008 | 1 | 1 | 0.3823 | 1 | 2.1747 | 1 | 1 | true | true | 0.8307 | 0.8975 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_only_lr_1e-4 | 4 | capture_available-002 | 2 | 2 | 0.2830 | 1 | 2.2383 | 2 | 2 | true | true | 0.7500 | 0.8725 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_only_lr_1e-4 | 4 | capture_available-003 | 2 | 1 | 0.2014 | 2 | 2.1856 | 2 | 2 | true | true | 0.8021 | 0.9100 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 4 | capture_available-006 | 2 | 2 | 0.2950 | 1 | 2.2540 | 2 | 2 | true | true | 0.6823 | 0.8800 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_only_lr_1e-4 | 4 | capture_available-007 | 2 | 1 | 0.2754 | 2 | 2.2023 | 2 | 2 | true | true | 0.8281 | 0.8858 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 4 | capture_available-008 | 1 | 1 | 0.4013 | 1 | 2.1457 | 1 | 1 | true | true | 0.8594 | 0.8942 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_only_lr_1e-4 | 8 | capture_available-002 | 2 | 2 | 0.3890 | 1 | 2.1239 | 2 | 2 | true | true | 0.7708 | 0.8942 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_only_lr_1e-4 | 8 | capture_available-003 | 2 | 1 | 0.2863 | 2 | 2.1235 | 2 | 2 | true | true | 0.8177 | 0.9125 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-4 | 8 | capture_available-006 | 2 | 2 | 0.3836 | 1 | 2.1404 | 2 | 2 | true | true | 0.6901 | 0.8742 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_only_lr_1e-4 | 8 | capture_available-007 | 2 | 2 | 0.3711 | 1 | 2.0805 | 2 | 2 | true | true | 0.8490 | 0.9267 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_only_lr_1e-4 | 8 | capture_available-008 | 1 | 1 | 0.4025 | 1 | 2.0672 | 1 | 1 | true | true | 0.8359 | 0.8975 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_only_lr_1e-5 | 0 | capture_available-002 | 2 | 0 | 0.1579 | 3 | 2.1321 | 2 | 2 | true | true | 0.7188 | 0.8592 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 0 | capture_available-003 | 2 | 1 | 0.1307 | 4 | 2.1916 | 2 | 2 | true | true | 0.8099 | 0.9092 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 0 | capture_available-006 | 2 | 0 | 0.1792 | 3 | 2.2120 | 2 | 2 | true | true | 0.7214 | 0.8883 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 0 | capture_available-007 | 2 | 1 | 0.1806 | 3 | 2.2296 | 2 | 2 | true | true | 0.6589 | 0.8692 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 0 | capture_available-008 | 1 | 1 | 0.3403 | 1 | 2.1637 | 1 | 1 | true | true | 0.7865 | 0.9025 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_only_lr_1e-5 | 1 | capture_available-002 | 2 | 0 | 0.1607 | 3 | 2.1397 | 2 | 2 | true | true | 0.7188 | 0.8642 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 1 | capture_available-003 | 2 | 1 | 0.1325 | 4 | 2.1932 | 2 | 2 | true | true | 0.8073 | 0.9092 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 1 | capture_available-006 | 2 | 0 | 0.1822 | 3 | 2.2175 | 2 | 2 | true | true | 0.7188 | 0.8833 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 1 | capture_available-007 | 2 | 1 | 0.1828 | 3 | 2.2309 | 2 | 2 | true | true | 0.6562 | 0.8658 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 1 | capture_available-008 | 1 | 1 | 0.3430 | 1 | 2.1659 | 1 | 1 | true | true | 0.7943 | 0.9033 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_only_lr_1e-5 | 2 | capture_available-002 | 2 | 0 | 0.1635 | 3 | 2.1470 | 2 | 2 | true | true | 0.7057 | 0.8617 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 2 | capture_available-003 | 2 | 1 | 0.1342 | 4 | 2.1946 | 2 | 2 | true | true | 0.8099 | 0.9058 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 2 | capture_available-006 | 2 | 0 | 0.1852 | 3 | 2.2228 | 2 | 2 | true | true | 0.7005 | 0.8892 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 2 | capture_available-007 | 2 | 1 | 0.1850 | 3 | 2.2320 | 2 | 2 | true | true | 0.6536 | 0.8692 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 2 | capture_available-008 | 1 | 1 | 0.3455 | 1 | 2.1680 | 1 | 1 | true | true | 0.7969 | 0.9008 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_only_lr_1e-5 | 4 | capture_available-002 | 2 | 0 | 0.1693 | 3 | 2.1608 | 2 | 2 | true | true | 0.7083 | 0.8683 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 4 | capture_available-003 | 2 | 1 | 0.1379 | 4 | 2.1973 | 2 | 2 | true | true | 0.7995 | 0.9058 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 4 | capture_available-006 | 2 | 0 | 0.1913 | 3 | 2.2322 | 2 | 2 | true | true | 0.7057 | 0.8933 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 4 | capture_available-007 | 2 | 1 | 0.1891 | 3 | 2.2340 | 2 | 2 | true | true | 0.6615 | 0.8742 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 4 | capture_available-008 | 1 | 1 | 0.3505 | 1 | 2.1715 | 1 | 1 | true | true | 0.7995 | 0.9000 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_only_lr_1e-5 | 8 | capture_available-002 | 2 | 0 | 0.1813 | 3 | 2.1851 | 2 | 2 | true | true | 0.7109 | 0.8717 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 8 | capture_available-003 | 2 | 1 | 0.1450 | 4 | 2.2007 | 2 | 2 | true | true | 0.8021 | 0.9000 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 8 | capture_available-006 | 2 | 0 | 0.2027 | 3 | 2.2473 | 2 | 2 | true | true | 0.7135 | 0.8900 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 8 | capture_available-007 | 2 | 1 | 0.1976 | 2 | 2.2369 | 2 | 2 | true | true | 0.6771 | 0.8750 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_only_lr_1e-5 | 8 | capture_available-008 | 1 | 1 | 0.3599 | 1 | 2.1761 | 1 | 1 | true | true | 0.7969 | 0.9008 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_plus_guard_controls_lr_1e-4 | 0 | capture_available-002 | 2 | 0 | 0.1579 | 3 | 2.1321 | 2 | 2 | true | true | 0.7188 | 0.8592 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_plus_guard_controls_lr_1e-4 | 0 | capture_available-003 | 2 | 1 | 0.1307 | 4 | 2.1916 | 2 | 2 | true | true | 0.8099 | 0.9092 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_plus_guard_controls_lr_1e-4 | 0 | capture_available-006 | 2 | 0 | 0.1792 | 3 | 2.2120 | 2 | 2 | true | true | 0.7214 | 0.8883 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_plus_guard_controls_lr_1e-4 | 0 | capture_available-007 | 2 | 1 | 0.1806 | 3 | 2.2296 | 2 | 2 | true | true | 0.6589 | 0.8692 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_plus_guard_controls_lr_1e-4 | 0 | capture_available-008 | 1 | 1 | 0.3403 | 1 | 2.1637 | 1 | 1 | true | true | 0.7865 | 0.9025 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_plus_guard_controls_lr_1e-4 | 1 | capture_available-002 | 2 | 0 | 0.1916 | 3 | 2.1967 | 2 | 2 | true | true | 0.7057 | 0.8683 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_plus_guard_controls_lr_1e-4 | 1 | capture_available-003 | 2 | 1 | 0.1546 | 4 | 2.2103 | 2 | 2 | true | true | 0.8047 | 0.8975 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_plus_guard_controls_lr_1e-4 | 1 | capture_available-006 | 2 | 0 | 0.2132 | 3 | 2.2546 | 2 | 2 | true | true | 0.6849 | 0.8908 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_plus_guard_controls_lr_1e-4 | 1 | capture_available-007 | 2 | 1 | 0.2085 | 2 | 2.2433 | 2 | 2 | true | true | 0.6641 | 0.8667 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_plus_guard_controls_lr_1e-4 | 1 | capture_available-008 | 1 | 1 | 0.3582 | 1 | 2.1834 | 1 | 1 | true | true | 0.8073 | 0.8983 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_plus_guard_controls_lr_1e-4 | 2 | capture_available-002 | 2 | 0 | 0.2286 | 3 | 2.2342 | 2 | 2 | true | true | 0.7292 | 0.8675 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_plus_guard_controls_lr_1e-4 | 2 | capture_available-003 | 2 | 1 | 0.1789 | 2 | 2.2215 | 2 | 2 | true | true | 0.8125 | 0.9042 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_plus_guard_controls_lr_1e-4 | 2 | capture_available-006 | 2 | 1 | 0.2477 | 2 | 2.2725 | 2 | 2 | true | true | 0.7161 | 0.8858 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_plus_guard_controls_lr_1e-4 | 2 | capture_available-007 | 2 | 1 | 0.2402 | 2 | 2.2456 | 2 | 2 | true | true | 0.6380 | 0.8675 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_plus_guard_controls_lr_1e-4 | 2 | capture_available-008 | 1 | 1 | 0.3685 | 1 | 2.1891 | 1 | 1 | true | true | 0.8307 | 0.8975 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_plus_guard_controls_lr_1e-4 | 4 | capture_available-002 | 2 | 2 | 0.3069 | 1 | 2.2356 | 2 | 2 | true | true | 0.7839 | 0.8842 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_plus_guard_controls_lr_1e-4 | 4 | capture_available-003 | 2 | 1 | 0.2378 | 2 | 2.2215 | 2 | 2 | true | true | 0.7995 | 0.9083 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| artifact_plus_guard_controls_lr_1e-4 | 4 | capture_available-006 | 2 | 2 | 0.3195 | 1 | 2.2480 | 2 | 2 | true | true | 0.7292 | 0.8967 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_plus_guard_controls_lr_1e-4 | 4 | capture_available-007 | 2 | 2 | 0.3134 | 1 | 2.2034 | 2 | 2 | true | true | 0.8490 | 0.8808 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_plus_guard_controls_lr_1e-4 | 4 | capture_available-008 | 1 | 1 | 0.3663 | 1 | 2.1781 | 1 | 1 | true | true | 0.8177 | 0.8933 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_plus_guard_controls_lr_1e-4 | 8 | capture_available-002 | 2 | 2 | 0.4443 | 1 | 2.0636 | 2 | 2 | true | true | 0.7448 | 0.8967 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_plus_guard_controls_lr_1e-4 | 8 | capture_available-003 | 2 | 2 | 0.3548 | 1 | 2.1105 | 2 | 2 | true | true | 0.8229 | 0.9217 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_plus_guard_controls_lr_1e-4 | 8 | capture_available-006 | 2 | 2 | 0.4320 | 1 | 2.0862 | 2 | 2 | true | true | 0.7031 | 0.8850 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_plus_guard_controls_lr_1e-4 | 8 | capture_available-007 | 2 | 2 | 0.4463 | 1 | 2.0064 | 2 | 2 | true | true | 0.8646 | 0.9358 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| artifact_plus_guard_controls_lr_1e-4 | 8 | capture_available-008 | 1 | 1 | 0.3478 | 1 | 2.0848 | 1 | 1 | true | true | 0.7917 | 0.8958 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 0 | capture_available-002 | 2 | 0 | 0.1579 | 3 | 2.1321 | 2 | 2 | true | true | 0.7188 | 0.8592 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 0 | capture_available-003 | 2 | 1 | 0.1307 | 4 | 2.1916 | 2 | 2 | true | true | 0.8099 | 0.9092 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 0 | capture_available-006 | 2 | 0 | 0.1792 | 3 | 2.2120 | 2 | 2 | true | true | 0.7214 | 0.8883 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 0 | capture_available-007 | 2 | 1 | 0.1806 | 3 | 2.2296 | 2 | 2 | true | true | 0.6589 | 0.8692 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 0 | capture_available-008 | 1 | 1 | 0.3403 | 1 | 2.1637 | 1 | 1 | true | true | 0.7865 | 0.9025 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 1 | capture_available-002 | 2 | 0 | 0.1885 | 3 | 2.1981 | 2 | 2 | true | true | 0.7031 | 0.8658 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 1 | capture_available-003 | 2 | 1 | 0.1508 | 4 | 2.2148 | 2 | 2 | true | true | 0.8125 | 0.9025 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 1 | capture_available-006 | 2 | 0 | 0.2085 | 3 | 2.2575 | 2 | 2 | true | true | 0.6771 | 0.8900 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 1 | capture_available-007 | 2 | 1 | 0.2039 | 2 | 2.2470 | 2 | 2 | true | true | 0.6536 | 0.8692 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 1 | capture_available-008 | 1 | 1 | 0.3549 | 1 | 2.1889 | 1 | 1 | true | true | 0.8099 | 0.9008 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 2 | capture_available-002 | 2 | 0 | 0.2217 | 3 | 2.2397 | 2 | 2 | true | true | 0.6771 | 0.8650 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 2 | capture_available-003 | 2 | 1 | 0.1704 | 2 | 2.2292 | 2 | 2 | true | true | 0.8073 | 0.8967 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 2 | capture_available-006 | 2 | 1 | 0.2374 | 2 | 2.2811 | 2 | 2 | true | true | 0.7109 | 0.8992 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 2 | capture_available-007 | 2 | 1 | 0.2302 | 2 | 2.2538 | 2 | 2 | true | true | 0.8021 | 0.8692 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 2 | capture_available-008 | 1 | 1 | 0.3593 | 1 | 2.2040 | 1 | 1 | true | true | 0.8255 | 0.9000 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 0 | capture_available-002 | 2 | 0 | 0.1579 | 3 | 2.1321 | 2 | 2 | true | true | 0.7188 | 0.8592 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 0 | capture_available-003 | 2 | 1 | 0.1307 | 4 | 2.1916 | 2 | 2 | true | true | 0.8099 | 0.9092 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 0 | capture_available-006 | 2 | 0 | 0.1792 | 3 | 2.2120 | 2 | 2 | true | true | 0.7214 | 0.8883 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 0 | capture_available-007 | 2 | 1 | 0.1806 | 3 | 2.2296 | 2 | 2 | true | true | 0.6589 | 0.8692 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 0 | capture_available-008 | 1 | 1 | 0.3403 | 1 | 2.1637 | 1 | 1 | true | true | 0.7865 | 0.9025 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 1 | capture_available-002 | 2 | 0 | 0.1891 | 3 | 2.1989 | 2 | 2 | true | true | 0.7161 | 0.8642 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 1 | capture_available-003 | 2 | 1 | 0.1529 | 4 | 2.2273 | 2 | 2 | true | true | 0.8047 | 0.9058 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 1 | capture_available-006 | 2 | 0 | 0.2116 | 3 | 2.2611 | 2 | 2 | true | true | 0.6927 | 0.8892 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 1 | capture_available-007 | 2 | 1 | 0.2067 | 2 | 2.2560 | 2 | 2 | true | true | 0.7865 | 0.8708 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 1 | capture_available-008 | 1 | 1 | 0.3435 | 1 | 2.1990 | 1 | 1 | true | true | 0.8047 | 0.9008 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 2 | capture_available-002 | 2 | 0 | 0.2246 | 3 | 2.2444 | 2 | 2 | true | true | 0.7240 | 0.8708 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 2 | capture_available-003 | 2 | 1 | 0.1758 | 2 | 2.2532 | 2 | 2 | true | true | 0.7865 | 0.8992 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 2 | capture_available-006 | 2 | 2 | 0.2447 | 1 | 2.2867 | 2 | 2 | true | true | 0.7135 | 0.8708 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 2 | capture_available-007 | 2 | 1 | 0.2370 | 2 | 2.2680 | 2 | 2 | true | true | 0.7656 | 0.8642 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 2 | capture_available-008 | 1 | 1 | 0.3372 | 1 | 2.2269 | 1 | 1 | true | true | 0.8359 | 0.8975 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 0 | capture_available-002 | 2 | 0 | 0.1579 | 3 | 2.1321 | 2 | 2 | true | true | 0.7188 | 0.8592 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 0 | capture_available-003 | 2 | 1 | 0.1307 | 4 | 2.1916 | 2 | 2 | true | true | 0.8099 | 0.9092 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 0 | capture_available-006 | 2 | 0 | 0.1792 | 3 | 2.2120 | 2 | 2 | true | true | 0.7214 | 0.8883 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 0 | capture_available-007 | 2 | 1 | 0.1806 | 3 | 2.2296 | 2 | 2 | true | true | 0.6589 | 0.8692 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 0 | capture_available-008 | 1 | 1 | 0.3403 | 1 | 2.1637 | 1 | 1 | true | true | 0.7865 | 0.9025 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 1 | capture_available-002 | 2 | 0 | 0.1871 | 3 | 2.1958 | 2 | 2 | true | true | 0.7188 | 0.8625 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 1 | capture_available-003 | 2 | 1 | 0.1478 | 4 | 2.2003 | 2 | 2 | true | true | 0.8021 | 0.8975 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 1 | capture_available-006 | 2 | 1 | 0.2080 | 3 | 2.2535 | 2 | 2 | true | true | 0.6953 | 0.8917 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 1 | capture_available-007 | 2 | 1 | 0.2016 | 2 | 2.2372 | 2 | 2 | true | true | 0.6615 | 0.8700 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 1 | capture_available-008 | 1 | 1 | 0.3653 | 1 | 2.1764 | 1 | 1 | true | true | 0.8073 | 0.8992 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 2 | capture_available-002 | 2 | 1 | 0.2191 | 3 | 2.2329 | 2 | 2 | true | true | 0.7500 | 0.8700 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 2 | capture_available-003 | 2 | 1 | 0.1651 | 2 | 2.2022 | 2 | 2 | true | true | 0.8073 | 0.9025 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 2 | capture_available-006 | 2 | 1 | 0.2365 | 2 | 2.2716 | 2 | 2 | true | true | 0.7031 | 0.8808 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 2 | capture_available-007 | 2 | 1 | 0.2254 | 2 | 2.2362 | 2 | 2 | true | true | 0.6172 | 0.8667 | 0.0000 | 0.0000 | `policy_drift_only` | policy_top_not_reference |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 2 | capture_available-008 | 1 | 1 | 0.3823 | 1 | 2.1747 | 1 | 1 | true | true | 0.8307 | 0.8975 | 0.0000 | 0.0000 | `stable_guard` | pass_reference_selected |

## 5. Guard search drift results

- Evaluated rows: `["capture_available-002", "capture_available-003", "capture_available-006", "capture_available-007", "capture_available-008"]`.
- Search settings matched the prior corrected-reference control baseline via `build_eval_search_options()` with no new root-prior intervention.

## 6. Training metric trace

| trace_name | checkpoint_step | policy_loss | value_loss | total_loss | guard_cross_entropy | non_guard_opening_cross_entropy | policy_head_delta_norm | value_head_delta_norm | trunk_delta_norm | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| artifact_only_lr_1e-4 | 0 | 1.7287 | 0.0429 | 1.7415 | 1.8111 | 1.7090 | 0.0000 | 0.0000 | 0.0000 | initial_checkpoint |
| artifact_only_lr_1e-4 | 1 | 1.6871 | 0.0379 | 1.6985 | 1.7082 | 1.6820 | 0.0085 | 0.0058 | 0.0200 | post_epoch |
| artifact_only_lr_1e-4 | 2 | 1.6526 | 0.0335 | 1.6627 | 1.6180 | 1.6609 | 0.0165 | 0.0114 | 0.0389 | post_epoch |
| artifact_only_lr_1e-4 | 4 | 1.6013 | 0.0258 | 1.6090 | 1.4718 | 1.6321 | 0.0321 | 0.0224 | 0.0751 | post_epoch |
| artifact_only_lr_1e-4 | 8 | 1.5392 | 0.0213 | 1.5456 | 1.2883 | 1.5990 | 0.0601 | 0.0428 | 0.1400 | post_epoch |
| artifact_only_lr_1e-5 | 0 | 1.7287 | 0.0429 | 1.7415 | 1.8111 | 1.7090 | 0.0000 | 0.0000 | 0.0000 | initial_checkpoint |
| artifact_only_lr_1e-5 | 1 | 1.7242 | 0.0424 | 1.7369 | 1.8001 | 1.7061 | 0.0008 | 0.0006 | 0.0020 | post_epoch |
| artifact_only_lr_1e-5 | 2 | 1.7198 | 0.0419 | 1.7323 | 1.7891 | 1.7033 | 0.0017 | 0.0012 | 0.0040 | post_epoch |
| artifact_only_lr_1e-5 | 4 | 1.7111 | 0.0409 | 1.7234 | 1.7677 | 1.6976 | 0.0034 | 0.0023 | 0.0079 | post_epoch |
| artifact_only_lr_1e-5 | 8 | 1.6948 | 0.0388 | 1.7064 | 1.7269 | 1.6871 | 0.0067 | 0.0046 | 0.0158 | post_epoch |
| artifact_plus_guard_controls_lr_1e-4 | 0 | 1.7420 | 0.0430 | 1.7549 | 1.8111 | 1.7090 | 0.0000 | 0.0000 | 0.0000 | initial_checkpoint |
| artifact_plus_guard_controls_lr_1e-4 | 1 | 1.6890 | 0.0376 | 1.7003 | 1.6930 | 1.6871 | 0.0085 | 0.0058 | 0.0200 | post_epoch |
| artifact_plus_guard_controls_lr_1e-4 | 2 | 1.6446 | 0.0329 | 1.6545 | 1.5900 | 1.6707 | 0.0166 | 0.0114 | 0.0392 | post_epoch |
| artifact_plus_guard_controls_lr_1e-4 | 4 | 1.5766 | 0.0250 | 1.5841 | 1.4181 | 1.6521 | 0.0321 | 0.0225 | 0.0757 | post_epoch |
| artifact_plus_guard_controls_lr_1e-4 | 8 | 1.5005 | 0.0200 | 1.5065 | 1.2129 | 1.6375 | 0.0604 | 0.0437 | 0.1416 | post_epoch |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 0 | 1.7560 | 0.0413 | 1.7683 | 1.8111 | 1.7440 | 0.0000 | 0.0000 | 0.0000 | initial_checkpoint |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 1 | 1.7177 | 0.0337 | 1.7278 | 1.7067 | 1.7200 | 0.0085 | 0.0058 | 0.0200 | post_epoch |
| family_leave_one_out_microtrace_without_opening_edge_move_5_preference | 2 | 1.6856 | 0.0283 | 1.6942 | 1.6173 | 1.7005 | 0.0166 | 0.0115 | 0.0390 | post_epoch |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 0 | 1.7570 | 0.0392 | 1.7687 | 1.8111 | 1.7470 | 0.0000 | 0.0000 | 0.0000 | initial_checkpoint |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 1 | 1.7068 | 0.0325 | 1.7165 | 1.7032 | 1.7075 | 0.0085 | 0.0058 | 0.0202 | post_epoch |
| family_leave_one_out_microtrace_without_opening_missed_extra_turn_continuation | 2 | 1.6682 | 0.0276 | 1.6765 | 1.6080 | 1.6794 | 0.0167 | 0.0115 | 0.0394 | post_epoch |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 0 | 1.7287 | 0.0429 | 1.7415 | 1.8111 | 1.7090 | 0.0000 | 0.0000 | 0.0000 | initial_checkpoint |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 1 | 1.6871 | 0.0379 | 1.6985 | 1.7082 | 1.6820 | 0.0085 | 0.0058 | 0.0200 | post_epoch |
| family_leave_one_out_microtrace_without_opening_extra_turn_overbias | 2 | 1.6526 | 0.0335 | 1.6627 | 1.6180 | 1.6609 | 0.0165 | 0.0114 | 0.0389 | post_epoch |

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

Recommendation: **use the selected artifact lane only with the existing corrected guard search gate; training did not worsen inherited guard policy mismatch.**
