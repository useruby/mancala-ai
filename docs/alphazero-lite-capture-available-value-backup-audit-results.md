# AlphaZero-lite Capture Available Value/Backup Audit Results

## 1. Context

- This audit evaluates child-afterstate values and backup behavior for the `capture_available` corrected non-opening failure family.
- No training was run.
- No arena was run.
- No model was promoted.
- No replay artifacts were created.
- Corrected references were not mutated.
- Corrected references: `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- Current artifact: `storage/ai/alphazero_lite/current`.
- Selected family rows: `/tmp/azlite_corrected_non_opening_failure_mining_v4/selected_non_opening_family_rows_v4.jsonl`.

## 2. Why capture_available was selected

- PR #57 selected `capture_available` as the next corrected non-opening family after excluding opening replay, guard rows as training targets, and noisy reference families.
- Family stats: `{"avg_reference_visit_share_1200": 0.2082, "avg_reference_visit_share_384": 0.2473, "avg_selected_minus_reference_q_margin_1200": 0.0651, "avg_selected_minus_reference_q_margin_384": 0.0709, "classification": "value_or_backup_issue", "conflicting_target_count": 0, "dominant_failure_mode": "value_q", "duplicate_canonical_state_count": 0, "fail_rows": 14, "failure_rate": 0.7, "family": "capture_available", "high_severity_count": 14, "medium_severity_count": 0, "notes": "persistent failures remain Q/value-dominant", "pass_rows": 6, "persistent_1200_failures": 14, "rank_score": 261.4018, "recovered_at_1200": 0, "stable_corrected_reference_count": 20, "total_rows": 20}`.

## 3. Row validation

- Corrected guard rows kept out of target candidates: `['capture_available-002', 'capture_available-003', 'capture_available-006', 'capture_available-007', 'capture_available-008']`.
- Perspective conversion: `+1 when child current_player == root current_player, else -1`.
- Implementation rule: `PUCT _search negates the returned child value exactly when child.game.current_player != parent.game.current_player`.

| row_id | role | corrected_reference_move | legal | reference_unstable | status | notes |
| --- | --- | --- | --- | --- | --- | --- |
| capture_available-024 | target_candidate | 3 | true | false | ok | validated, required target row present |
| capture_available-017 | target_candidate | 3 | true | false | ok | validated, required target row present |
| capture_available-019 | target_candidate | 3 | true | false | ok | validated, required target row present |
| capture_available-018 | target_candidate | 3 | true | false | ok | validated, required target row present |
| capture_available-013 | target_candidate | 4 | true | false | ok | validated, required target row present |
| capture_available-023 | target_candidate | 4 | true | false | ok | validated, required target row present |
| capture_available-022 | target_candidate | 3 | true | false | ok | validated, required target row present |
| capture_available-016 | target_candidate | 4 | true | false | ok | validated, required target row present |
| capture_available-025 | target_candidate | 4 | true | false | ok | validated, required target row present |
| capture_available-009 | target_candidate | 4 | true | false | ok | validated, required target row present |
| capture_available-012 | target_candidate | 5 | true | false | ok | validated |
| capture_available-001 | target_candidate | 3 | true | false | ok | validated |
| capture_available-015 | holdout_candidate | 3 | true | false | ok | validated |
| capture_available-027 | holdout_candidate | 4 | true | false | ok | validated |
| capture_available-021 | preservation_control | 1 | true | false | ok | validated, required control row present |
| capture_available-010 | preservation_control | 2 | true | false | ok | validated, required control row present |
| capture_available-020 | preservation_control | 3 | true | false | ok | validated, required control row present |
| capture_available-011 | preservation_control | 2 | true | false | ok | validated, required control row present |

## 4. Root PUCT baseline

| row_id | role | budget | corrected_reference_move | selected_move | selected_is_reference | reference_visit_share | selected_visit_share | reference_q | selected_q | selected_minus_reference_q_margin | reference_policy_probability | selected_policy_probability | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| capture_available-001 | target_candidate | 384 | 3 | 2 | false | 0.0026 | 0.8880 | -0.0936 | 0.0957 | 0.1892 | 0.0647 | 0.3566 | target_candidate, selected away from reference |
| capture_available-001 | target_candidate | 1200 | 3 | 2 | false | 0.0183 | 0.9408 | 0.1016 | 0.1276 | 0.0259 | 0.0647 | 0.3566 | target_candidate, selected away from reference |
| capture_available-001 | target_candidate | 2400 | 3 | 2 | false | 0.0108 | 0.9679 | 0.0773 | 0.1058 | 0.0286 | 0.0647 | 0.3566 | target_candidate, selected away from reference |
| capture_available-009 | target_candidate | 384 | 4 | 1 | false | 0.0156 | 0.9714 | -0.0254 | 0.0040 | 0.0294 | 0.0522 | 0.7687 | target_candidate, selected away from reference |
| capture_available-009 | target_candidate | 1200 | 4 | 1 | false | 0.0050 | 0.9825 | -0.0254 | 0.0101 | 0.0355 | 0.0522 | 0.7687 | target_candidate, selected away from reference |
| capture_available-009 | target_candidate | 2400 | 4 | 1 | false | 0.0029 | 0.9879 | -0.0436 | 0.0305 | 0.0741 | 0.0522 | 0.7687 | target_candidate, selected away from reference |
| capture_available-010 | preservation_control | 384 | 2 | 2 | true | 0.9740 | 0.9740 | 0.1974 | 0.1974 | 0.0000 | 0.7783 | 0.7783 | preservation_control, selected reference |
| capture_available-010 | preservation_control | 1200 | 2 | 2 | true | 0.9767 | 0.9767 | 0.1937 | 0.1937 | 0.0000 | 0.7783 | 0.7783 | preservation_control, selected reference |
| capture_available-010 | preservation_control | 2400 | 2 | 2 | true | 0.9875 | 0.9875 | 0.2021 | 0.2021 | 0.0000 | 0.7783 | 0.7783 | preservation_control, selected reference |
| capture_available-011 | preservation_control | 384 | 2 | 2 | true | 0.8099 | 0.8099 | 0.1137 | 0.1137 | 0.0000 | 0.5385 | 0.5385 | preservation_control, selected reference |
| capture_available-011 | preservation_control | 1200 | 2 | 2 | true | 0.5225 | 0.5225 | 0.0775 | 0.0775 | 0.0000 | 0.5385 | 0.5385 | preservation_control, selected reference |
| capture_available-011 | preservation_control | 2400 | 2 | 3 | false | 0.3333 | 0.3700 | 0.0727 | 0.1081 | 0.0354 | 0.5385 | 0.0862 | preservation_control, selected away from reference |
| capture_available-012 | target_candidate | 384 | 5 | 3 | false | 0.0911 | 0.7995 | 0.1135 | 0.1218 | 0.0083 | 0.2039 | 0.2643 | target_candidate, selected away from reference |
| capture_available-012 | target_candidate | 1200 | 5 | 3 | false | 0.0325 | 0.8858 | 0.1045 | 0.1382 | 0.0336 | 0.2039 | 0.2643 | target_candidate, selected away from reference |
| capture_available-012 | target_candidate | 2400 | 5 | 3 | false | 0.0163 | 0.9417 | 0.1045 | 0.1549 | 0.0504 | 0.2039 | 0.2643 | target_candidate, selected away from reference |
| capture_available-013 | target_candidate | 384 | 4 | 2 | false | 0.0078 | 0.6484 | 0.0193 | 0.0962 | 0.0769 | 0.0546 | 0.5475 | target_candidate, selected away from reference |
| capture_available-013 | target_candidate | 1200 | 4 | 2 | false | 0.0025 | 0.7608 | 0.0193 | 0.0842 | 0.0649 | 0.0546 | 0.5475 | target_candidate, selected away from reference |
| capture_available-013 | target_candidate | 2400 | 4 | 2 | false | 0.0029 | 0.7854 | 0.0380 | 0.0698 | 0.0318 | 0.0546 | 0.5475 | target_candidate, selected away from reference |
| capture_available-015 | holdout_candidate | 384 | 3 | 2 | false | 0.1797 | 0.6979 | 0.1336 | 0.1364 | 0.0028 | 0.3242 | 0.3915 | holdout_candidate, selected away from reference |
| capture_available-015 | holdout_candidate | 1200 | 3 | 4 | false | 0.0608 | 0.7025 | 0.1268 | 0.1333 | 0.0065 | 0.3242 | 0.0987 | holdout_candidate, selected away from reference |
| capture_available-015 | holdout_candidate | 2400 | 3 | 4 | false | 0.0571 | 0.7612 | 0.1144 | 0.1222 | 0.0078 | 0.3242 | 0.0987 | holdout_candidate, selected away from reference |
| capture_available-016 | target_candidate | 384 | 4 | 0 | false | 0.0026 | 0.5234 | 0.0615 | 0.1320 | 0.0705 | 0.0294 | 0.7053 | target_candidate, selected away from reference |
| capture_available-016 | target_candidate | 1200 | 4 | 0 | false | 0.0017 | 0.7025 | 0.0559 | 0.1007 | 0.0447 | 0.0294 | 0.7053 | target_candidate, selected away from reference |
| capture_available-016 | target_candidate | 2400 | 4 | 0 | false | 0.0537 | 0.5129 | 0.1343 | 0.1013 | -0.0330 | 0.0294 | 0.7053 | target_candidate, selected away from reference |
| capture_available-017 | target_candidate | 384 | 3 | 4 | false | 0.0026 | 0.8620 | 0.0151 | 0.1103 | 0.0952 | 0.0195 | 0.0439 | target_candidate, selected away from reference |
| capture_available-017 | target_candidate | 1200 | 3 | 2 | false | 0.0008 | 0.3317 | 0.0151 | 0.1693 | 0.1542 | 0.0195 | 0.0592 | target_candidate, selected away from reference |
| capture_available-017 | target_candidate | 2400 | 3 | 2 | false | 0.0004 | 0.6658 | 0.0151 | 0.1411 | 0.1260 | 0.0195 | 0.0592 | target_candidate, selected away from reference |
| capture_available-018 | target_candidate | 384 | 3 | 2 | false | 0.0026 | 0.5443 | 0.0219 | 0.1380 | 0.1160 | 0.0213 | 0.0830 | target_candidate, selected away from reference |
| capture_available-018 | target_candidate | 1200 | 3 | 0 | false | 0.0008 | 0.6650 | 0.0219 | 0.1136 | 0.0916 | 0.0213 | 0.5779 | target_candidate, selected away from reference |
| capture_available-018 | target_candidate | 2400 | 3 | 0 | false | 0.0004 | 0.7242 | 0.0219 | 0.1115 | 0.0895 | 0.0213 | 0.5779 | target_candidate, selected away from reference |
| capture_available-019 | target_candidate | 384 | 3 | 0 | false | 0.0026 | 0.7786 | 0.0292 | 0.1409 | 0.1117 | 0.0093 | 0.7613 | target_candidate, selected away from reference |
| capture_available-019 | target_candidate | 1200 | 3 | 4 | false | 0.0008 | 0.5217 | 0.0292 | 0.1441 | 0.1149 | 0.0093 | 0.0232 | target_candidate, selected away from reference |
| capture_available-019 | target_candidate | 2400 | 3 | 4 | false | 0.0004 | 0.7508 | 0.0292 | 0.1600 | 0.1308 | 0.0093 | 0.0232 | target_candidate, selected away from reference |
| capture_available-020 | preservation_control | 384 | 3 | 3 | true | 0.7448 | 0.7448 | 0.1013 | 0.1013 | 0.0000 | 0.2920 | 0.2920 | preservation_control, selected reference |
| capture_available-020 | preservation_control | 1200 | 3 | 3 | true | 0.9142 | 0.9142 | 0.1405 | 0.1405 | 0.0000 | 0.2920 | 0.2920 | preservation_control, selected reference |
| capture_available-020 | preservation_control | 2400 | 3 | 3 | true | 0.9550 | 0.9550 | 0.1261 | 0.1261 | 0.0000 | 0.2920 | 0.2920 | preservation_control, selected reference |
| capture_available-021 | preservation_control | 384 | 1 | 1 | true | 0.9479 | 0.9479 | 0.1073 | 0.1073 | 0.0000 | 0.7903 | 0.7903 | preservation_control, selected reference |
| capture_available-021 | preservation_control | 1200 | 1 | 1 | true | 0.9817 | 0.9817 | 0.1362 | 0.1362 | 0.0000 | 0.7903 | 0.7903 | preservation_control, selected reference |
| capture_available-021 | preservation_control | 2400 | 1 | 1 | true | 0.9904 | 0.9904 | 0.1913 | 0.1913 | 0.0000 | 0.7903 | 0.7903 | preservation_control, selected reference |
| capture_available-022 | target_candidate | 384 | 3 | 1 | false | 0.0391 | 0.9375 | 0.0730 | 0.0926 | 0.0196 | 0.0647 | 0.6921 | target_candidate, selected away from reference |
| capture_available-022 | target_candidate | 1200 | 3 | 1 | false | 0.0125 | 0.9708 | 0.0730 | 0.1183 | 0.0453 | 0.0647 | 0.6921 | target_candidate, selected away from reference |
| capture_available-022 | target_candidate | 2400 | 3 | 1 | false | 0.0063 | 0.9833 | 0.0730 | 0.1447 | 0.0717 | 0.0647 | 0.6921 | target_candidate, selected away from reference |
| capture_available-023 | target_candidate | 384 | 4 | 1 | false | 0.0260 | 0.9219 | 0.0757 | 0.1119 | 0.0361 | 0.0934 | 0.6104 | target_candidate, selected away from reference |
| capture_available-023 | target_candidate | 1200 | 4 | 1 | false | 0.0083 | 0.9700 | 0.0757 | 0.1211 | 0.0453 | 0.0934 | 0.6104 | target_candidate, selected away from reference |
| capture_available-023 | target_candidate | 2400 | 4 | 1 | false | 0.0046 | 0.9829 | 0.0749 | 0.1226 | 0.0477 | 0.0934 | 0.6104 | target_candidate, selected away from reference |
| capture_available-024 | target_candidate | 384 | 3 | 1 | false | 0.1224 | 0.7812 | -0.0293 | 0.1037 | 0.1330 | 0.0315 | 0.7589 | target_candidate, selected away from reference |
| capture_available-024 | target_candidate | 1200 | 3 | 1 | false | 0.0392 | 0.9283 | -0.0293 | 0.1776 | 0.2069 | 0.0315 | 0.7589 | target_candidate, selected away from reference |
| capture_available-024 | target_candidate | 2400 | 3 | 1 | false | 0.0196 | 0.9629 | -0.0293 | 0.2149 | 0.2442 | 0.0315 | 0.7589 | target_candidate, selected away from reference |
| capture_available-025 | target_candidate | 384 | 4 | 0 | false | 0.2786 | 0.6120 | -0.0330 | -0.0217 | 0.0113 | 0.1710 | 0.4855 | target_candidate, selected away from reference |
| capture_available-025 | target_candidate | 1200 | 4 | 0 | false | 0.0892 | 0.8758 | -0.0330 | 0.0110 | 0.0440 | 0.1710 | 0.4855 | target_candidate, selected away from reference |
| capture_available-025 | target_candidate | 2400 | 4 | 0 | false | 0.0446 | 0.9379 | -0.0330 | -0.0141 | 0.0189 | 0.1710 | 0.4855 | target_candidate, selected away from reference |
| capture_available-027 | holdout_candidate | 384 | 4 | 0 | false | 0.0234 | 0.9219 | -0.1684 | -0.0755 | 0.0929 | 0.2299 | 0.1352 | holdout_candidate, selected away from reference |
| capture_available-027 | holdout_candidate | 1200 | 4 | 0 | false | 0.0708 | 0.8267 | -0.0867 | -0.0897 | -0.0030 | 0.2299 | 0.1352 | holdout_candidate, selected away from reference |
| capture_available-027 | holdout_candidate | 2400 | 4 | 4 | true | 0.5317 | 0.5317 | -0.0616 | -0.0616 | 0.0000 | 0.2299 | 0.2299 | holdout_candidate, selected reference |

- ran 2400 root budget: projected ~3.3s across 12 target rows

## 5. Move consequence audit

| row_id | move | is_corrected_reference | is_selected | gives_extra_turn | produces_capture | capture_count | immediate_store_delta | side_to_move_after | game_over_after_move | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| capture_available-024 | 0 | false | false | false | true | 2 | 2 | 0 | false |  |
| capture_available-024 | 1 | false | true | true | false | 0 | 1 | 1 | false | selected move gains extra turn; selected move reduces next-side seed supply |
| capture_available-024 | 2 | false | false | false | false | 0 | 0 | 0 | false |  |
| capture_available-024 | 3 | true | false | false | false | 0 | 1 | 0 | false |  |
| capture_available-024 | 4 | false | false | false | false | 0 | 1 | 0 | false |  |
| capture_available-017 | 0 | false | false | false | true | 2 | 2 | 1 | false |  |
| capture_available-017 | 1 | false | false | false | false | 0 | 0 | 1 | false |  |
| capture_available-017 | 2 | false | true | false | false | 0 | 1 | 1 | false | selected move presents a safer-looking handoff |
| capture_available-017 | 3 | true | false | false | false | 0 | 1 | 1 | false |  |
| capture_available-017 | 4 | false | false | false | false | 0 | 1 | 1 | false |  |
| capture_available-019 | 0 | false | false | false | true | 2 | 2 | 1 | false |  |
| capture_available-019 | 1 | false | false | false | false | 0 | 0 | 1 | false |  |
| capture_available-019 | 2 | false | false | false | false | 0 | 1 | 1 | false |  |
| capture_available-019 | 3 | true | false | false | false | 0 | 1 | 1 | false |  |
| capture_available-019 | 4 | false | true | false | false | 0 | 1 | 1 | false | selected move presents a safer-looking handoff |
| capture_available-018 | 0 | false | true | false | true | 2 | 2 | 1 | false | selected move captures more immediately; selected move has larger immediate store gain; selected move leaves fewer seeds in pits; selected move creates stronger starvation pressure; selected move presents a safer-looking handoff; selected move reduces next-side seed supply |
| capture_available-018 | 1 | false | false | false | false | 0 | 0 | 1 | false |  |
| capture_available-018 | 2 | false | false | false | false | 0 | 1 | 1 | false |  |
| capture_available-018 | 3 | true | false | false | false | 0 | 1 | 1 | false |  |
| capture_available-018 | 4 | false | false | false | false | 0 | 1 | 1 | false |  |
| capture_available-013 | 0 | false | false | false | true | 2 | 2 | 1 | false |  |
| capture_available-013 | 2 | false | true | false | false | 0 | 1 | 1 | false | selected move presents a safer-looking handoff; selected move reduces next-side seed supply |
| capture_available-013 | 3 | false | false | false | false | 0 | 1 | 1 | false |  |
| capture_available-013 | 4 | true | false | false | false | 0 | 1 | 1 | false |  |
| capture_available-013 | 5 | false | false | false | false | 0 | 1 | 1 | false |  |
| capture_available-023 | 0 | false | false | false | true | 2 | 2 | 0 | false |  |
| capture_available-023 | 1 | false | true | true | false | 0 | 1 | 1 | false | selected move gains extra turn; selected move reduces next-side seed supply |
| capture_available-023 | 2 | false | false | false | false | 0 | 0 | 0 | false |  |
| capture_available-023 | 3 | false | false | false | false | 0 | 1 | 0 | false |  |
| capture_available-023 | 4 | true | false | false | false | 0 | 1 | 0 | false |  |
| capture_available-022 | 0 | false | false | false | true | 2 | 2 | 0 | false |  |
| capture_available-022 | 1 | false | true | true | false | 0 | 1 | 1 | false | selected move gains extra turn; selected move reduces next-side seed supply |
| capture_available-022 | 2 | false | false | false | false | 0 | 0 | 0 | false |  |
| capture_available-022 | 3 | true | false | false | false | 0 | 1 | 0 | false |  |
| capture_available-022 | 4 | false | false | false | false | 0 | 1 | 0 | false |  |
| capture_available-016 | 0 | false | true | false | true | 2 | 2 | 1 | false | selected move captures more immediately; selected move has larger immediate store gain; selected move leaves fewer seeds in pits; selected move creates stronger starvation pressure; selected move presents a safer-looking handoff; selected move reduces next-side seed supply |
| capture_available-016 | 1 | false | false | false | false | 0 | 0 | 1 | false |  |
| capture_available-016 | 2 | false | false | false | false | 0 | 1 | 1 | false |  |
| capture_available-016 | 3 | false | false | false | false | 0 | 1 | 1 | false |  |
| capture_available-016 | 4 | true | false | false | false | 0 | 1 | 1 | false |  |
| capture_available-025 | 0 | false | true | false | true | 2 | 2 | 0 | false | selected move captures more immediately; selected move has larger immediate store gain; selected move leaves fewer seeds in pits; selected move creates stronger starvation pressure; selected move presents a safer-looking handoff; selected move reduces next-side seed supply |
| capture_available-025 | 2 | false | false | false | false | 0 | 1 | 0 | false |  |
| capture_available-025 | 3 | false | false | false | false | 0 | 1 | 0 | false |  |
| capture_available-025 | 4 | true | false | false | false | 0 | 1 | 0 | false |  |
| capture_available-009 | 0 | false | false | false | true | 2 | 2 | 0 | false |  |
| capture_available-009 | 1 | false | true | true | false | 0 | 1 | 1 | false | selected move gains extra turn; selected move reduces next-side seed supply |
| capture_available-009 | 2 | false | false | false | false | 0 | 1 | 0 | false |  |
| capture_available-009 | 3 | false | false | false | false | 0 | 1 | 0 | false |  |
| capture_available-009 | 4 | true | false | false | false | 0 | 1 | 0 | false |  |
| capture_available-012 | 0 | false | false | false | true | 3 | 3 | 0 | false |  |
| capture_available-012 | 1 | false | false | false | false | 0 | 0 | 0 | false |  |
| capture_available-012 | 3 | false | true | false | false | 0 | 1 | 0 | false | selected move presents a safer-looking handoff; selected move reduces next-side seed supply |
| capture_available-012 | 5 | true | false | false | false | 0 | 1 | 0 | false |  |
| capture_available-001 | 0 | false | false | false | true | 3 | 3 | 0 | false |  |
| capture_available-001 | 1 | false | false | false | false | 0 | 0 | 0 | false |  |
| capture_available-001 | 2 | false | true | true | false | 0 | 1 | 1 | false | selected move gains extra turn; selected move reduces next-side seed supply |
| capture_available-001 | 3 | true | false | false | false | 0 | 1 | 0 | false |  |
| capture_available-001 | 5 | false | false | false | false | 0 | 1 | 0 | false |  |
| capture_available-015 | 0 | false | false | false | true | 2 | 2 | 1 | false |  |
| capture_available-015 | 2 | false | false | false | false | 0 | 1 | 1 | false |  |
| capture_available-015 | 3 | true | false | false | false | 0 | 1 | 1 | false |  |
| capture_available-015 | 4 | false | true | false | false | 0 | 1 | 1 | false | selected move presents a safer-looking handoff |
| capture_available-027 | 0 | false | true | false | true | 2 | 2 | 0 | false | selected move captures more immediately; selected move has larger immediate store gain; selected move leaves fewer seeds in pits; selected move creates stronger starvation pressure; selected move presents a safer-looking handoff; selected move reduces next-side seed supply |
| capture_available-027 | 1 | false | false | false | false | 0 | 1 | 0 | false |  |
| capture_available-027 | 3 | false | false | false | false | 0 | 1 | 0 | false |  |
| capture_available-027 | 4 | true | false | false | false | 0 | 1 | 0 | false |  |
| capture_available-027 | 5 | false | false | false | false | 0 | 1 | 0 | false |  |
| capture_available-021 | 0 | false | false | false | true | 2 | 2 | 0 | false |  |
| capture_available-021 | 1 | true | true | true | false | 0 | 1 | 1 | false |  |
| capture_available-021 | 3 | false | false | false | false | 0 | 1 | 0 | false |  |
| capture_available-021 | 4 | false | false | false | false | 0 | 1 | 0 | false |  |
| capture_available-010 | 0 | false | false | false | true | 2 | 2 | 1 | false |  |
| capture_available-010 | 1 | false | false | false | false | 0 | 0 | 1 | false |  |
| capture_available-010 | 2 | true | true | true | false | 0 | 1 | 0 | false |  |
| capture_available-010 | 3 | false | false | false | false | 0 | 1 | 1 | false |  |
| capture_available-010 | 5 | false | false | false | false | 0 | 1 | 1 | false |  |
| capture_available-020 | 0 | false | false | false | true | 2 | 2 | 0 | false |  |
| capture_available-020 | 1 | false | false | false | true | 2 | 2 | 0 | false |  |
| capture_available-020 | 3 | true | true | false | false | 0 | 1 | 0 | false |  |
| capture_available-020 | 4 | false | false | false | false | 0 | 1 | 0 | false |  |
| capture_available-011 | 0 | false | false | false | true | 3 | 3 | 1 | false |  |
| capture_available-011 | 2 | true | true | false | false | 0 | 1 | 1 | false |  |
| capture_available-011 | 3 | false | false | false | false | 0 | 1 | 1 | false |  |
| capture_available-011 | 5 | false | false | false | false | 0 | 1 | 1 | false |  |

## 6. Neural child-afterstate value audit

| row_id | corrected_reference_move | selected_move | child | raw_value | root_perspective_value | child_ref_minus_child_selected | neural_prefers_reference | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| capture_available-024 | 3 | 1 | corrected_reference_child | -0.1097 | 0.1097 | -0.2412 | false | sign flip to root perspective |
| capture_available-024 | 3 | 1 | selected_child | 0.3509 | 0.3509 | -0.2412 | false | identity conversion |
| capture_available-017 | 3 | 2 | corrected_reference_child | -0.0151 | 0.0151 | 0.0140 | true | sign flip to root perspective |
| capture_available-017 | 3 | 2 | selected_child | -0.0011 | 0.0011 | 0.0140 | true | sign flip to root perspective |
| capture_available-019 | 3 | 4 | corrected_reference_child | -0.0292 | 0.0292 | -0.0747 | false | sign flip to root perspective |
| capture_available-019 | 3 | 4 | selected_child | -0.1039 | 0.1039 | -0.0747 | false | sign flip to root perspective |
| capture_available-018 | 3 | 0 | corrected_reference_child | -0.0219 | 0.0219 | -0.0246 | false | sign flip to root perspective |
| capture_available-018 | 3 | 0 | selected_child | -0.0466 | 0.0466 | -0.0246 | false | sign flip to root perspective |
| capture_available-013 | 4 | 2 | corrected_reference_child | -0.0173 | 0.0173 | 0.0283 | true | sign flip to root perspective |
| capture_available-013 | 4 | 2 | selected_child | 0.0110 | -0.0110 | 0.0283 | true | sign flip to root perspective |
| capture_available-023 | 4 | 1 | corrected_reference_child | 0.0391 | -0.0391 | -0.2840 | false | sign flip to root perspective |
| capture_available-023 | 4 | 1 | selected_child | 0.2448 | 0.2448 | -0.2840 | false | identity conversion |
| capture_available-022 | 3 | 1 | corrected_reference_child | 0.0220 | -0.0220 | -0.3070 | false | sign flip to root perspective |
| capture_available-022 | 3 | 1 | selected_child | 0.2850 | 0.2850 | -0.3070 | false | identity conversion |
| capture_available-016 | 4 | 0 | corrected_reference_child | -0.0615 | 0.0615 | 0.1513 | true | sign flip to root perspective |
| capture_available-016 | 4 | 0 | selected_child | 0.0898 | -0.0898 | 0.1513 | true | sign flip to root perspective |
| capture_available-025 | 4 | 0 | corrected_reference_child | -0.0436 | 0.0436 | 0.0245 | true | sign flip to root perspective |
| capture_available-025 | 4 | 0 | selected_child | -0.0191 | 0.0191 | 0.0245 | true | sign flip to root perspective |
| capture_available-009 | 4 | 1 | corrected_reference_child | -0.0152 | 0.0152 | -0.1103 | false | sign flip to root perspective |
| capture_available-009 | 4 | 1 | selected_child | 0.1255 | 0.1255 | -0.1103 | false | identity conversion |
| capture_available-012 | 5 | 3 | corrected_reference_child | -0.0751 | 0.0751 | 0.0221 | true | sign flip to root perspective |
| capture_available-012 | 5 | 3 | selected_child | -0.0530 | 0.0530 | 0.0221 | true | sign flip to root perspective |
| capture_available-001 | 3 | 2 | corrected_reference_child | 0.0936 | -0.0936 | -0.2741 | false | sign flip to root perspective |
| capture_available-001 | 3 | 2 | selected_child | 0.1806 | 0.1806 | -0.2741 | false | identity conversion |

## 7. ClassicMCTS child-afterstate audit

| row_id | budget | seeds | child_ref_value_root_mean | child_selected_value_root_mean | child_ref_minus_child_selected | teacher_prefers_reference | stable | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| capture_available-024 | 1200 | [11, 23, 37, 42, 101] | 0.0559 | 0.8221 | -0.7662 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-024 | 2400 | [11, 23, 37, 42, 101] | -0.1468 | 0.7990 | -0.9458 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-024 | 5000 | [11, 23, 37, 42, 101] | -0.3341 | 0.7910 | -1.1251 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-017 | 1200 | [11, 23, 37, 42, 101] | -0.2493 | -0.4951 | 0.2459 | true | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-017 | 2400 | [11, 23, 37, 42, 101] | -0.2927 | -0.6371 | 0.3444 | true | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-017 | 5000 | [11, 23, 37, 42, 101] | -0.3310 | -0.7488 | 0.4178 | true | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-019 | 1200 | [11, 23, 37, 42, 101] | -0.1630 | -0.2223 | 0.0594 | true | false | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-019 | 2400 | [11, 23, 37, 42, 101] | -0.3480 | -0.4123 | 0.0644 | true | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-019 | 5000 | [11, 23, 37, 42, 101] | -0.4598 | -0.5415 | 0.0817 | true | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-018 | 1200 | [11, 23, 37, 42, 101] | -0.2245 | -0.1332 | -0.0913 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-018 | 2400 | [11, 23, 37, 42, 101] | -0.3181 | -0.2862 | -0.0319 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-018 | 5000 | [11, 23, 37, 42, 101] | -0.4932 | -0.4676 | -0.0256 | false | false | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-013 | 1200 | [11, 23, 37, 42, 101] | -0.0578 | -0.3034 | 0.2456 | true | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-013 | 2400 | [11, 23, 37, 42, 101] | -0.2641 | -0.3760 | 0.1120 | true | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-013 | 5000 | [11, 23, 37, 42, 101] | -0.5831 | -0.4346 | -0.1485 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-023 | 1200 | [11, 23, 37, 42, 101] | -0.0851 | 0.7900 | -0.8750 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-023 | 2400 | [11, 23, 37, 42, 101] | -0.3269 | 0.7285 | -1.0554 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-023 | 5000 | [11, 23, 37, 42, 101] | -0.6077 | 0.7268 | -1.3345 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-022 | 1200 | [11, 23, 37, 42, 101] | -0.0687 | 0.7925 | -0.8612 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-022 | 2400 | [11, 23, 37, 42, 101] | -0.1673 | 0.7599 | -0.9273 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-022 | 5000 | [11, 23, 37, 42, 101] | -0.4149 | 0.7229 | -1.1378 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-016 | 1200 | [11, 23, 37, 42, 101] | -0.3840 | -0.0969 | -0.2872 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-016 | 2400 | [11, 23, 37, 42, 101] | -0.5086 | -0.2305 | -0.2780 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-016 | 5000 | [11, 23, 37, 42, 101] | -0.6365 | -0.4192 | -0.2173 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-025 | 1200 | [11, 23, 37, 42, 101] | -0.4745 | -0.1041 | -0.3704 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-025 | 2400 | [11, 23, 37, 42, 101] | -0.5959 | -0.3099 | -0.2860 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-025 | 5000 | [11, 23, 37, 42, 101] | -0.6758 | -0.5898 | -0.0859 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-009 | 1200 | [11, 23, 37, 42, 101] | -0.1957 | 0.7079 | -0.9037 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-009 | 2400 | [11, 23, 37, 42, 101] | -0.3818 | 0.6867 | -1.0685 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-009 | 5000 | [11, 23, 37, 42, 101] | -0.5507 | 0.7014 | -1.2521 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-012 | 1200 | [11, 23, 37, 42, 101] | -0.1981 | -0.2437 | 0.0456 | true | false | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-012 | 2400 | [11, 23, 37, 42, 101] | -0.3954 | -0.5055 | 0.1101 | true | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-012 | 5000 | [11, 23, 37, 42, 101] | -0.6122 | -0.6316 | 0.0194 | true | false | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-001 | 1200 | [11, 23, 37, 42, 101] | -0.2181 | 0.6937 | -0.9118 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-001 | 2400 | [11, 23, 37, 42, 101] | -0.4801 | 0.6675 | -1.1476 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| capture_available-001 | 5000 | [11, 23, 37, 42, 101] | -0.6380 | 0.6657 | -1.3037 | false | true | ClassicMCTS child-afterstate teacher aggregate |

## 8. PUCT child-afterstate audit

| row_id | budget | child_ref_value_root | child_selected_value_root | child_ref_minus_child_selected | puct_prefers_reference | notes |
| --- | --- | --- | --- | --- | --- | --- |
| capture_available-024 | 384 | 0.0048 | 0.1108 | -0.1061 | false | artifact PUCT child-afterstate audit; agrees_with_classic_teacher=true, agrees_with_neural_child_value=true |
| capture_available-024 | 1200 | 0.0410 | 0.1839 | -0.1429 | false | artifact PUCT child-afterstate audit; agrees_with_classic_teacher=true, agrees_with_neural_child_value=true |
| capture_available-017 | 384 | 0.1470 | 0.1707 | -0.0238 | false | artifact PUCT child-afterstate audit; agrees_with_classic_teacher=false, agrees_with_neural_child_value=false |
| capture_available-017 | 1200 | 0.1011 | 0.1324 | -0.0313 | false | artifact PUCT child-afterstate audit; agrees_with_classic_teacher=false, agrees_with_neural_child_value=false |
| capture_available-019 | 384 | 0.1376 | 0.1601 | -0.0225 | false | artifact PUCT child-afterstate audit; agrees_with_classic_teacher=false, agrees_with_neural_child_value=true |
| capture_available-019 | 1200 | 0.2267 | 0.1716 | 0.0551 | true | artifact PUCT child-afterstate audit; agrees_with_classic_teacher=true, agrees_with_neural_child_value=false |
| capture_available-018 | 384 | 0.1180 | 0.1265 | -0.0086 | false | artifact PUCT child-afterstate audit; agrees_with_classic_teacher=true, agrees_with_neural_child_value=true |
| capture_available-018 | 1200 | 0.0788 | 0.1038 | -0.0250 | false | artifact PUCT child-afterstate audit; agrees_with_classic_teacher=true, agrees_with_neural_child_value=true |
| capture_available-013 | 384 | 0.0718 | 0.1051 | -0.0334 | false | artifact PUCT child-afterstate audit; agrees_with_classic_teacher=true, agrees_with_neural_child_value=false |
| capture_available-013 | 1200 | 0.0524 | 0.0822 | -0.0298 | false | artifact PUCT child-afterstate audit; agrees_with_classic_teacher=true, agrees_with_neural_child_value=false |
| capture_available-023 | 384 | 0.0278 | 0.1145 | -0.0867 | false | artifact PUCT child-afterstate audit; agrees_with_classic_teacher=true, agrees_with_neural_child_value=true |
| capture_available-023 | 1200 | 0.0436 | 0.1234 | -0.0798 | false | artifact PUCT child-afterstate audit; agrees_with_classic_teacher=true, agrees_with_neural_child_value=true |
| capture_available-022 | 384 | -0.0138 | 0.0922 | -0.1061 | false | artifact PUCT child-afterstate audit; agrees_with_classic_teacher=true, agrees_with_neural_child_value=true |
| capture_available-022 | 1200 | 0.0006 | 0.1191 | -0.1185 | false | artifact PUCT child-afterstate audit; agrees_with_classic_teacher=true, agrees_with_neural_child_value=true |
| capture_available-016 | 384 | 0.1172 | 0.1077 | 0.0094 | true | artifact PUCT child-afterstate audit; agrees_with_classic_teacher=false, agrees_with_neural_child_value=true |
| capture_available-016 | 1200 | 0.1329 | 0.1044 | 0.0285 | true | artifact PUCT child-afterstate audit; agrees_with_classic_teacher=false, agrees_with_neural_child_value=true |
| capture_available-025 | 384 | -0.0453 | -0.0037 | -0.0416 | false | artifact PUCT child-afterstate audit; agrees_with_classic_teacher=true, agrees_with_neural_child_value=false |
| capture_available-025 | 1200 | -0.0217 | 0.0099 | -0.0316 | false | artifact PUCT child-afterstate audit; agrees_with_classic_teacher=true, agrees_with_neural_child_value=false |
| capture_available-009 | 384 | -0.0757 | 0.0033 | -0.0790 | false | artifact PUCT child-afterstate audit; agrees_with_classic_teacher=true, agrees_with_neural_child_value=true |
| capture_available-009 | 1200 | -0.0891 | 0.0096 | -0.0986 | false | artifact PUCT child-afterstate audit; agrees_with_classic_teacher=true, agrees_with_neural_child_value=true |
| capture_available-012 | 384 | 0.0393 | 0.1128 | -0.0736 | false | artifact PUCT child-afterstate audit; agrees_with_classic_teacher=false, agrees_with_neural_child_value=false |
| capture_available-012 | 1200 | 0.0872 | 0.1487 | -0.0614 | false | artifact PUCT child-afterstate audit; agrees_with_classic_teacher=false, agrees_with_neural_child_value=false |
| capture_available-001 | 384 | 0.0653 | 0.0967 | -0.0314 | false | artifact PUCT child-afterstate audit; agrees_with_classic_teacher=true, agrees_with_neural_child_value=true |
| capture_available-001 | 1200 | 0.0862 | 0.1292 | -0.0429 | false | artifact PUCT child-afterstate audit; agrees_with_classic_teacher=true, agrees_with_neural_child_value=true |

## 9. Root counterfactual diagnostics

| row_id | intervention | budget | selected_move | selected_is_reference | reference_visit_share | selected_visit_share | selected_minus_reference_q_margin | flipped | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| capture_available-024 | original | 384 | 1 | false | 0.1224 | 0.7812 | 0.1330 | false | no intervention |
| capture_available-024 | original | 1200 | 1 | false | 0.0392 | 0.9283 | 0.2069 | false | no intervention |
| capture_available-024 | equalize_root_priors | 384 | 1 | false | 0.1250 | 0.7812 | 0.1387 | false | equalize corrected reference and selected root priors |
| capture_available-024 | equalize_root_priors | 1200 | 1 | false | 0.0400 | 0.9283 | 0.2125 | false | equalize corrected reference and selected root priors |
| capture_available-024 | uniform_legal_prior | 384 | 1 | false | 0.1250 | 0.7656 | 0.1347 | false | uniform legal prior positive control |
| capture_available-024 | uniform_legal_prior | 1200 | 1 | false | 0.0400 | 0.9217 | 0.2130 | false | uniform legal prior positive control |
| capture_available-024 | teacher_child_value_override | 384 | 1 | false | 0.0026 | 0.9219 | 0.4340 | false | override first child expansion values with teacher child values |
| capture_available-024 | teacher_child_value_override | 1200 | 1 | false | 0.0008 | 0.9725 | 0.5125 | false | override first child expansion values with teacher child values |
| capture_available-024 | neural_child_value_swap | 384 | 3 | true | 0.6302 | 0.6302 | 0.0000 | true | swap child neural values as diagnostic sanity check |
| capture_available-024 | neural_child_value_swap | 1200 | 1 | false | 0.2017 | 0.7658 | 0.1740 | false | swap child neural values as diagnostic sanity check |
| capture_available-017 | original | 384 | 4 | false | 0.0026 | 0.8620 | 0.0952 | false | no intervention |
| capture_available-017 | original | 1200 | 2 | false | 0.0008 | 0.3317 | 0.1542 | false | no intervention |
| capture_available-017 | equalize_root_priors | 384 | 4 | false | 0.0026 | 0.8620 | 0.0952 | false | equalize corrected reference and selected root priors |
| capture_available-017 | equalize_root_priors | 1200 | 2 | false | 0.0283 | 0.3008 | 0.0689 | false | equalize corrected reference and selected root priors |
| capture_available-017 | uniform_legal_prior | 384 | 2 | false | 0.0755 | 0.7135 | 0.0315 | false | uniform legal prior positive control |
| capture_available-017 | uniform_legal_prior | 1200 | 2 | false | 0.0267 | 0.8033 | 0.0319 | false | uniform legal prior positive control |
| capture_available-017 | teacher_child_value_override | 384 | 0 | false | 0.0026 | 0.4661 | 0.4097 | false | override first child expansion values with teacher child values |
| capture_available-017 | teacher_child_value_override | 1200 | 0 | false | 0.0008 | 0.4967 | 0.4249 | false | override first child expansion values with teacher child values |
| capture_available-017 | neural_child_value_swap | 384 | 4 | false | 0.0026 | 0.8620 | 0.1092 | false | swap child neural values as diagnostic sanity check |
| capture_available-017 | neural_child_value_swap | 1200 | 2 | false | 0.0008 | 0.4275 | 0.1603 | false | swap child neural values as diagnostic sanity check |
| capture_available-019 | original | 384 | 0 | false | 0.0026 | 0.7786 | 0.1117 | false | no intervention |
| capture_available-019 | original | 1200 | 4 | false | 0.0008 | 0.5217 | 0.1149 | false | no intervention |
| capture_available-019 | equalize_root_priors | 384 | 0 | false | 0.0026 | 0.7708 | 0.1138 | false | equalize corrected reference and selected root priors |
| capture_available-019 | equalize_root_priors | 1200 | 4 | false | 0.0008 | 0.5192 | 0.1155 | false | equalize corrected reference and selected root priors |
| capture_available-019 | uniform_legal_prior | 384 | 4 | false | 0.0547 | 0.7526 | -0.0243 | false | uniform legal prior positive control |
| capture_available-019 | uniform_legal_prior | 1200 | 4 | false | 0.3192 | 0.6125 | 0.0006 | false | uniform legal prior positive control |
| capture_available-019 | teacher_child_value_override | 384 | 0 | false | 0.0026 | 0.7344 | 0.6050 | false | override first child expansion values with teacher child values |
| capture_available-019 | teacher_child_value_override | 1200 | 0 | false | 0.0008 | 0.8850 | 0.5959 | false | override first child expansion values with teacher child values |
| capture_available-019 | neural_child_value_swap | 384 | 0 | false | 0.0026 | 0.7786 | 0.0371 | false | swap child neural values as diagnostic sanity check |
| capture_available-019 | neural_child_value_swap | 1200 | 0 | false | 0.0008 | 0.9142 | 0.0333 | false | swap child neural values as diagnostic sanity check |
| capture_available-018 | original | 384 | 2 | false | 0.0026 | 0.5443 | 0.1160 | false | no intervention |
| capture_available-018 | original | 1200 | 0 | false | 0.0008 | 0.6650 | 0.0916 | false | no intervention |
| capture_available-018 | equalize_root_priors | 384 | 3 | true | 0.8880 | 0.8880 | 0.0000 | true | equalize corrected reference and selected root priors |
| capture_available-018 | equalize_root_priors | 1200 | 0 | false | 0.3308 | 0.6425 | 0.0045 | false | equalize corrected reference and selected root priors |
| capture_available-018 | uniform_legal_prior | 384 | 2 | false | 0.0964 | 0.6953 | 0.0132 | false | uniform legal prior positive control |
| capture_available-018 | uniform_legal_prior | 1200 | 3 | true | 0.4225 | 0.4225 | 0.0000 | true | uniform legal prior positive control |
| capture_available-018 | teacher_child_value_override | 384 | 0 | false | 0.0026 | 0.5286 | 0.6154 | false | override first child expansion values with teacher child values |
| capture_available-018 | teacher_child_value_override | 1200 | 0 | false | 0.0008 | 0.6225 | 0.6129 | false | override first child expansion values with teacher child values |
| capture_available-018 | neural_child_value_swap | 384 | 2 | false | 0.0026 | 0.5781 | 0.0834 | false | swap child neural values as diagnostic sanity check |
| capture_available-018 | neural_child_value_swap | 1200 | 0 | false | 0.0008 | 0.6975 | 0.0661 | false | swap child neural values as diagnostic sanity check |
| capture_available-013 | original | 384 | 2 | false | 0.0078 | 0.6484 | 0.0769 | false | no intervention |
| capture_available-013 | original | 1200 | 2 | false | 0.0025 | 0.7608 | 0.0649 | false | no intervention |
| capture_available-013 | equalize_root_priors | 384 | 5 | false | 0.0964 | 0.5938 | 0.0392 | false | equalize corrected reference and selected root priors |
| capture_available-013 | equalize_root_priors | 1200 | 2 | false | 0.0350 | 0.7450 | 0.0430 | false | equalize corrected reference and selected root priors |
| capture_available-013 | uniform_legal_prior | 384 | 5 | false | 0.0208 | 0.9141 | 0.0536 | false | uniform legal prior positive control |
| capture_available-013 | uniform_legal_prior | 1200 | 3 | false | 0.0192 | 0.5900 | 0.0542 | false | uniform legal prior positive control |
| capture_available-013 | teacher_child_value_override | 384 | 2 | false | 0.0026 | 0.6849 | 0.6797 | false | override first child expansion values with teacher child values |
| capture_available-013 | teacher_child_value_override | 1200 | 3 | false | 0.0017 | 0.5833 | 0.3661 | false | override first child expansion values with teacher child values |
| capture_available-013 | neural_child_value_swap | 384 | 2 | false | 0.0026 | 0.7031 | 0.1102 | false | swap child neural values as diagnostic sanity check |
| capture_available-013 | neural_child_value_swap | 1200 | 2 | false | 0.0025 | 0.7608 | 0.0743 | false | swap child neural values as diagnostic sanity check |
| capture_available-023 | original | 384 | 1 | false | 0.0260 | 0.9219 | 0.0361 | false | no intervention |
| capture_available-023 | original | 1200 | 1 | false | 0.0083 | 0.9700 | 0.0453 | false | no intervention |
| capture_available-023 | equalize_root_priors | 384 | 1 | false | 0.0443 | 0.9453 | 0.0805 | false | equalize corrected reference and selected root priors |
| capture_available-023 | equalize_root_priors | 1200 | 1 | false | 0.0192 | 0.9592 | 0.1029 | false | equalize corrected reference and selected root priors |
| capture_available-023 | uniform_legal_prior | 384 | 1 | false | 0.0339 | 0.8464 | 0.0512 | false | uniform legal prior positive control |
| capture_available-023 | uniform_legal_prior | 1200 | 1 | false | 0.0117 | 0.9492 | 0.0767 | false | uniform legal prior positive control |
| capture_available-023 | teacher_child_value_override | 384 | 1 | false | 0.0052 | 0.8646 | 0.3594 | false | override first child expansion values with teacher child values |
| capture_available-023 | teacher_child_value_override | 1200 | 1 | false | 0.0025 | 0.9542 | 0.2847 | false | override first child expansion values with teacher child values |
| capture_available-023 | neural_child_value_swap | 384 | 1 | false | 0.0365 | 0.9115 | 0.0459 | false | swap child neural values as diagnostic sanity check |
| capture_available-023 | neural_child_value_swap | 1200 | 1 | false | 0.0117 | 0.9667 | 0.0576 | false | swap child neural values as diagnostic sanity check |
| capture_available-022 | original | 384 | 1 | false | 0.0391 | 0.9375 | 0.0196 | false | no intervention |
| capture_available-022 | original | 1200 | 1 | false | 0.0125 | 0.9708 | 0.0453 | false | no intervention |
| capture_available-022 | equalize_root_priors | 384 | 1 | false | 0.0729 | 0.9167 | 0.0568 | false | equalize corrected reference and selected root priors |
| capture_available-022 | equalize_root_priors | 1200 | 1 | false | 0.0400 | 0.9458 | 0.0747 | false | equalize corrected reference and selected root priors |
| capture_available-022 | uniform_legal_prior | 384 | 1 | false | 0.0417 | 0.8698 | 0.0343 | false | uniform legal prior positive control |
| capture_available-022 | uniform_legal_prior | 1200 | 1 | false | 0.0133 | 0.9567 | 0.0619 | false | uniform legal prior positive control |
| capture_available-022 | teacher_child_value_override | 384 | 1 | false | 0.0026 | 0.8750 | 0.5072 | false | override first child expansion values with teacher child values |
| capture_available-022 | teacher_child_value_override | 1200 | 1 | false | 0.0017 | 0.9475 | 0.1638 | false | override first child expansion values with teacher child values |
| capture_available-022 | neural_child_value_swap | 384 | 1 | false | 0.0443 | 0.9323 | 0.0224 | false | swap child neural values as diagnostic sanity check |
| capture_available-022 | neural_child_value_swap | 1200 | 1 | false | 0.0142 | 0.9692 | 0.0497 | false | swap child neural values as diagnostic sanity check |
| capture_available-016 | original | 384 | 0 | false | 0.0026 | 0.5234 | 0.0705 | false | no intervention |
| capture_available-016 | original | 1200 | 0 | false | 0.0017 | 0.7025 | 0.0447 | false | no intervention |
| capture_available-016 | equalize_root_priors | 384 | 4 | true | 0.8776 | 0.8776 | 0.0000 | true | equalize corrected reference and selected root priors |
| capture_available-016 | equalize_root_priors | 1200 | 4 | true | 0.6533 | 0.6533 | 0.0000 | true | equalize corrected reference and selected root priors |
| capture_available-016 | uniform_legal_prior | 384 | 4 | true | 0.5495 | 0.5495 | 0.0000 | true | uniform legal prior positive control |
| capture_available-016 | uniform_legal_prior | 1200 | 2 | false | 0.1758 | 0.7667 | 0.0338 | false | uniform legal prior positive control |
| capture_available-016 | teacher_child_value_override | 384 | 0 | false | 0.0026 | 0.7891 | 0.7630 | false | override first child expansion values with teacher child values |
| capture_available-016 | teacher_child_value_override | 1200 | 0 | false | 0.0008 | 0.6983 | 0.7372 | false | override first child expansion values with teacher child values |
| capture_available-016 | neural_child_value_swap | 384 | 0 | false | 0.0026 | 0.8984 | 0.2080 | false | swap child neural values as diagnostic sanity check |
| capture_available-016 | neural_child_value_swap | 1200 | 0 | false | 0.0008 | 0.6867 | 0.1938 | false | swap child neural values as diagnostic sanity check |
| capture_available-025 | original | 384 | 0 | false | 0.2786 | 0.6120 | 0.0113 | false | no intervention |
| capture_available-025 | original | 1200 | 0 | false | 0.0892 | 0.8758 | 0.0440 | false | no intervention |
| capture_available-025 | equalize_root_priors | 384 | 0 | false | 0.2891 | 0.6406 | 0.0194 | false | equalize corrected reference and selected root priors |
| capture_available-025 | equalize_root_priors | 1200 | 0 | false | 0.0925 | 0.8750 | 0.0493 | false | equalize corrected reference and selected root priors |
| capture_available-025 | uniform_legal_prior | 384 | 0 | false | 0.2891 | 0.6042 | 0.0143 | false | uniform legal prior positive control |
| capture_available-025 | uniform_legal_prior | 1200 | 0 | false | 0.0925 | 0.8733 | 0.0494 | false | uniform legal prior positive control |
| capture_available-025 | teacher_child_value_override | 384 | 0 | false | 0.0104 | 0.8672 | 0.1260 | false | override first child expansion values with teacher child values |
| capture_available-025 | teacher_child_value_override | 1200 | 0 | false | 0.0058 | 0.9550 | 0.0702 | false | override first child expansion values with teacher child values |
| capture_available-025 | neural_child_value_swap | 384 | 0 | false | 0.2786 | 0.6120 | 0.0116 | false | swap child neural values as diagnostic sanity check |
| capture_available-025 | neural_child_value_swap | 1200 | 0 | false | 0.0892 | 0.8758 | 0.0442 | false | swap child neural values as diagnostic sanity check |
| capture_available-009 | original | 384 | 1 | false | 0.0156 | 0.9714 | 0.0294 | false | no intervention |
| capture_available-009 | original | 1200 | 1 | false | 0.0050 | 0.9825 | 0.0355 | false | no intervention |
| capture_available-009 | equalize_root_priors | 384 | 1 | false | 0.0260 | 0.9661 | 0.0577 | false | equalize corrected reference and selected root priors |
| capture_available-009 | equalize_root_priors | 1200 | 1 | false | 0.0158 | 0.9792 | 0.1033 | false | equalize corrected reference and selected root priors |
| capture_available-009 | uniform_legal_prior | 384 | 1 | false | 0.0182 | 0.9089 | 0.0500 | false | uniform legal prior positive control |
| capture_available-009 | uniform_legal_prior | 1200 | 1 | false | 0.0092 | 0.9600 | 0.0855 | false | uniform legal prior positive control |
| capture_available-009 | teacher_child_value_override | 384 | 1 | false | 0.0026 | 0.9141 | 0.5590 | false | override first child expansion values with teacher child values |
| capture_available-009 | teacher_child_value_override | 1200 | 1 | false | 0.0017 | 0.9633 | 0.2860 | false | override first child expansion values with teacher child values |
| capture_available-009 | neural_child_value_swap | 384 | 1 | false | 0.0286 | 0.9583 | 0.0369 | false | swap child neural values as diagnostic sanity check |
| capture_available-009 | neural_child_value_swap | 1200 | 1 | false | 0.0092 | 0.9783 | 0.0421 | false | swap child neural values as diagnostic sanity check |
| capture_available-012 | original | 384 | 3 | false | 0.0911 | 0.7995 | 0.0083 | false | no intervention |
| capture_available-012 | original | 1200 | 3 | false | 0.0325 | 0.8858 | 0.0336 | false | no intervention |
| capture_available-012 | equalize_root_priors | 384 | 3 | false | 0.0938 | 0.7240 | 0.0099 | false | equalize corrected reference and selected root priors |
| capture_available-012 | equalize_root_priors | 1200 | 3 | false | 0.0550 | 0.8633 | 0.0339 | false | equalize corrected reference and selected root priors |
| capture_available-012 | uniform_legal_prior | 384 | 3 | false | 0.0182 | 0.9505 | 0.0049 | false | uniform legal prior positive control |
| capture_available-012 | uniform_legal_prior | 1200 | 3 | false | 0.0325 | 0.9442 | 0.0361 | false | uniform legal prior positive control |
| capture_available-012 | teacher_child_value_override | 384 | 3 | false | 0.0104 | 0.6484 | 0.1696 | false | override first child expansion values with teacher child values |
| capture_available-012 | teacher_child_value_override | 1200 | 3 | false | 0.0067 | 0.8267 | 0.1154 | false | override first child expansion values with teacher child values |
| capture_available-012 | neural_child_value_swap | 384 | 3 | false | 0.0911 | 0.7995 | 0.0090 | false | swap child neural values as diagnostic sanity check |
| capture_available-012 | neural_child_value_swap | 1200 | 3 | false | 0.0317 | 0.8867 | 0.0329 | false | swap child neural values as diagnostic sanity check |
| capture_available-001 | original | 384 | 2 | false | 0.0026 | 0.8880 | 0.1892 | false | no intervention |
| capture_available-001 | original | 1200 | 2 | false | 0.0183 | 0.9408 | 0.0259 | false | no intervention |
| capture_available-001 | equalize_root_priors | 384 | 2 | false | 0.1198 | 0.8177 | 0.0292 | false | equalize corrected reference and selected root priors |
| capture_available-001 | equalize_root_priors | 1200 | 2 | false | 0.0383 | 0.9333 | 0.0641 | false | equalize corrected reference and selected root priors |
| capture_available-001 | uniform_legal_prior | 384 | 2 | false | 0.1198 | 0.7812 | 0.0274 | false | uniform legal prior positive control |
| capture_available-001 | uniform_legal_prior | 1200 | 2 | false | 0.0383 | 0.9300 | 0.0649 | false | uniform legal prior positive control |
| capture_available-001 | teacher_child_value_override | 384 | 2 | false | 0.0026 | 0.7031 | 0.7270 | false | override first child expansion values with teacher child values |
| capture_available-001 | teacher_child_value_override | 1200 | 2 | false | 0.0017 | 0.8883 | 0.3772 | false | override first child expansion values with teacher child values |
| capture_available-001 | neural_child_value_swap | 384 | 2 | false | 0.1615 | 0.7656 | 0.0555 | false | swap child neural values as diagnostic sanity check |
| capture_available-001 | neural_child_value_swap | 1200 | 2 | false | 0.0517 | 0.9158 | 0.0952 | false | swap child neural values as diagnostic sanity check |

## 10. Row classifications

| row_id | role | row_classification | supporting_evidence | recommended_use | notes |
| --- | --- | --- | --- | --- | --- |
| capture_available-024 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| capture_available-017 | target_candidate | puct_child_search_value_mismatch | ClassicMCTS child teacher and neural child values prefer corrected reference child but child PUCT does not | target_candidate | root still fails |
| capture_available-019 | target_candidate | value_head_miscalibration | ClassicMCTS child teacher prefers corrected reference child but raw neural child values prefer selected child | target_candidate | root still fails |
| capture_available-018 | target_candidate | inconclusive | ClassicMCTS child teacher is unstable across seeds | target_candidate | root still fails |
| capture_available-013 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| capture_available-023 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| capture_available-022 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| capture_available-016 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| capture_available-025 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| capture_available-009 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| capture_available-012 | target_candidate | inconclusive | ClassicMCTS child teacher is unstable across seeds | target_candidate | root still fails |
| capture_available-001 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| capture_available-015 | holdout_candidate | inconclusive | holdout row reserved for out-of-sample follow-up rather than mechanism counting | holdout | root still fails |
| capture_available-027 | holdout_candidate | inconclusive | holdout row reserved for out-of-sample follow-up rather than mechanism counting | holdout | root still fails |
| capture_available-021 | preservation_control | inconclusive | preservation control retained to guard against regression | preserve_control | baseline pass |
| capture_available-010 | preservation_control | inconclusive | preservation control retained to guard against regression | preserve_control | baseline pass |
| capture_available-020 | preservation_control | inconclusive | preservation control retained to guard against regression | preserve_control | baseline pass |
| capture_available-011 | preservation_control | inconclusive | preservation control retained to guard against regression | preserve_control | baseline pass |

## 11. Family-level interpretation

| family_classification | value_head_miscalibration_count | puct_child_search_mismatch_count | root_selection_pressure_count | corrected_reference_suspicious_count | inconclusive_count | next_action |
| --- | --- | --- | --- | --- | --- | --- |
| reference_family_uncertain | 1 | 1 | 0 | 8 | 2 | adjudicate capture_available references before training. |

## 12. Exactly one recommended next action

Recommendation: **adjudicate capture_available references before training.**
