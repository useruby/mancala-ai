# AlphaZero-lite Guard-Safe Opening Replay Artifact Audit Results

## 1. Context

- PR #33 classified corrected opening replay as `replay_induced_guard_regression`.
- This audit avoids another broad replay sweep and instead isolates static artifact rows that conflict with corrected guard behavior.
- Corrected tracked references remained the default source of truth throughout the audit.

## 2. Why PR #33 paused replay

- Current already passes corrected `capture_available-002/003` at baseline under the corrected tracked references.
- Corrected hard-state replay stayed locally acceptable on the 1200-budget guard rows.
- The replay regression was isolated to the corrected opening replay branch, especially `opening_extra_turn_overbias_corrected_w1/w2`.
- The right next step was therefore artifact surgery, not more replay weight search, training, arena, or promotion.

## 3. Source artifact summary

- Source artifact: `/tmp/azlite_failure_family_diag/opening_plies_family_full_guarded_artifact.jsonl`.
- Source row count audited after adding missing corrected guards 006/007/008: `38`.
- The source mix contains opening replay rows plus explicit corrected guard rows.

## 4. Row-level conflict audit

| source_family | rows | conflict_rows | target_extra_turn_rate | no_extra_turn_capture_available_rate | high_similarity_to_002_count | high_similarity_to_003_count | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| corrected_capture_guard | 5 | 2 | 0.8000 | 0.4000 | 1 | 1 | guard_rows |
| opening_edge_move_5_preference | 10 | 0 | 0.0000 | 0.0000 | 0 | 0 | clean |
| opening_extra_turn_overbias | 12 | 0 | 0.0000 | 0.0000 | 0 | 0 | clean |
| opening_missed_extra_turn_continuation | 6 | 0 | 1.0000 | 0.0000 | 0 | 0 | clean |
| opening_other_mismatch | 5 | 0 | 0.2000 | 0.0000 | 0 | 0 | clean |

## 5. Filtered artifact variants

| artifact_name | path | row_count | excluded_count | included_families | excluded_families | guard_rows_present | conflict_flag_count | classification | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| guard_safe_strict | `/tmp/azlite_guard_safe_opening_replay/guard_safe_strict.jsonl` | 38 | 0 | `["corrected_capture_guard", "opening_edge_move_5_preference", "opening_extra_turn_overbias", "opening_missed_extra_turn_continuation", "opening_other_mismatch"]` | `[]` | true | 5 | `guard_safe` | guard-safe static artifact |
| guard_safe_no_extra_turn_overbias | `/tmp/azlite_guard_safe_opening_replay/guard_safe_no_extra_turn_overbias.jsonl` | 38 | 0 | `["corrected_capture_guard", "opening_edge_move_5_preference", "opening_extra_turn_overbias", "opening_missed_extra_turn_continuation", "opening_other_mismatch"]` | `[]` | true | 5 | `guard_safe` | guard-safe static artifact |
| guard_safe_controls_only | `/tmp/azlite_guard_safe_opening_replay/guard_safe_controls_only.jsonl` | 5 | 33 | `["corrected_capture_guard"]` | `["opening_edge_move_5_preference", "opening_extra_turn_overbias", "opening_missed_extra_turn_continuation", "opening_other_mismatch"]` | true | 5 | `guard_safe` | diagnostic-only guard artifact |
| family_leave_one_out_without_opening_extra_turn_overbias | `/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl` | 26 | 12 | `["corrected_capture_guard", "opening_edge_move_5_preference", "opening_missed_extra_turn_continuation", "opening_other_mismatch"]` | `["opening_extra_turn_overbias"]` | true | 5 | `guard_safe` | guard-safe static artifact |
| family_leave_one_out_without_opening_edge_move_5_preference | `/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_edge_move_5_preference.jsonl` | 28 | 10 | `["corrected_capture_guard", "opening_extra_turn_overbias", "opening_missed_extra_turn_continuation", "opening_other_mismatch"]` | `["opening_edge_move_5_preference"]` | true | 5 | `guard_safe` | guard-safe static artifact |
| family_leave_one_out_without_opening_missed_extra_turn_continuation | `/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_missed_extra_turn_continuation.jsonl` | 32 | 6 | `["corrected_capture_guard", "opening_edge_move_5_preference", "opening_extra_turn_overbias", "opening_other_mismatch"]` | `["opening_missed_extra_turn_continuation"]` | true | 5 | `guard_safe` | guard-safe static artifact |

## 6. Offline guard-safety validation

| artifact_name | row_id | corrected_reference_move | guard_row_present | conflicting_duplicate_count | near_duplicate_conflict_count | classification | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| family_leave_one_out_without_opening_edge_move_5_preference | capture_available-002 | 2 | true | 0 | 0 | `guard_safe` | ok |
| family_leave_one_out_without_opening_edge_move_5_preference | capture_available-003 | 2 | true | 0 | 0 | `guard_safe` | ok |
| family_leave_one_out_without_opening_edge_move_5_preference | capture_available-006 | 2 | true | 0 | 0 | `guard_safe` | ok |
| family_leave_one_out_without_opening_edge_move_5_preference | capture_available-007 | 2 | true | 0 | 0 | `guard_safe` | ok |
| family_leave_one_out_without_opening_edge_move_5_preference | capture_available-008 | 1 | true | 0 | 0 | `guard_safe` | ok |
| family_leave_one_out_without_opening_extra_turn_overbias | capture_available-002 | 2 | true | 0 | 0 | `guard_safe` | ok |
| family_leave_one_out_without_opening_extra_turn_overbias | capture_available-003 | 2 | true | 0 | 0 | `guard_safe` | ok |
| family_leave_one_out_without_opening_extra_turn_overbias | capture_available-006 | 2 | true | 0 | 0 | `guard_safe` | ok |
| family_leave_one_out_without_opening_extra_turn_overbias | capture_available-007 | 2 | true | 0 | 0 | `guard_safe` | ok |
| family_leave_one_out_without_opening_extra_turn_overbias | capture_available-008 | 1 | true | 0 | 0 | `guard_safe` | ok |
| family_leave_one_out_without_opening_missed_extra_turn_continuation | capture_available-002 | 2 | true | 0 | 0 | `guard_safe` | ok |
| family_leave_one_out_without_opening_missed_extra_turn_continuation | capture_available-003 | 2 | true | 0 | 0 | `guard_safe` | ok |
| family_leave_one_out_without_opening_missed_extra_turn_continuation | capture_available-006 | 2 | true | 0 | 0 | `guard_safe` | ok |
| family_leave_one_out_without_opening_missed_extra_turn_continuation | capture_available-007 | 2 | true | 0 | 0 | `guard_safe` | ok |
| family_leave_one_out_without_opening_missed_extra_turn_continuation | capture_available-008 | 1 | true | 0 | 0 | `guard_safe` | ok |
| guard_safe_controls_only | capture_available-002 | 2 | true | 0 | 0 | `guard_safe` | ok |
| guard_safe_controls_only | capture_available-003 | 2 | true | 0 | 0 | `guard_safe` | ok |
| guard_safe_controls_only | capture_available-006 | 2 | true | 0 | 0 | `guard_safe` | ok |
| guard_safe_controls_only | capture_available-007 | 2 | true | 0 | 0 | `guard_safe` | ok |
| guard_safe_controls_only | capture_available-008 | 1 | true | 0 | 0 | `guard_safe` | ok |
| guard_safe_no_extra_turn_overbias | capture_available-002 | 2 | true | 0 | 0 | `guard_safe` | ok |
| guard_safe_no_extra_turn_overbias | capture_available-003 | 2 | true | 0 | 0 | `guard_safe` | ok |
| guard_safe_no_extra_turn_overbias | capture_available-006 | 2 | true | 0 | 0 | `guard_safe` | ok |
| guard_safe_no_extra_turn_overbias | capture_available-007 | 2 | true | 0 | 0 | `guard_safe` | ok |
| guard_safe_no_extra_turn_overbias | capture_available-008 | 1 | true | 0 | 0 | `guard_safe` | ok |
| guard_safe_strict | capture_available-002 | 2 | true | 0 | 0 | `guard_safe` | ok |
| guard_safe_strict | capture_available-003 | 2 | true | 0 | 0 | `guard_safe` | ok |
| guard_safe_strict | capture_available-006 | 2 | true | 0 | 0 | `guard_safe` | ok |
| guard_safe_strict | capture_available-007 | 2 | true | 0 | 0 | `guard_safe` | ok |
| guard_safe_strict | capture_available-008 | 1 | true | 0 | 0 | `guard_safe` | ok |

## 7. Optional overfit probe

- Status: `skipped`.
- Notes: `tiny_probe_requires_weight_json_to_train_init_bridge`.
- The probe was intentionally not replaced with production training, arena, or export logic.

## 8. Selected artifact for next experiment

- Selected artifact: `family_leave_one_out_without_opening_extra_turn_overbias`.
- Path: `/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl`.
- Why: smallest guard-safe artifact that still retains meaningful opening coverage (`21` opening rows) while preserving corrected 002/003/006/007/008 guards.

## 9. Exactly one recommended next action

Classification: `optimization_interaction_not_static_artifact_conflict`.

Recommendation: **run a tiny overfit probe or low-epoch training trace on `family_leave_one_out_without_opening_extra_turn_overbias` to identify when corrected 002/003/006/007/008 guard policy drifts.**.
