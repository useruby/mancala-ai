# AlphaZero-lite Incumbent Proxy Disagreement Value/Backup Audit Results

## 1. Context

- No training was run.
- No arena was run.
- No model was promoted.
- Corrected references: `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- Current artifact: `storage/ai/alphazero_lite/current`.
- Selected family rows: `/tmp/azlite_corrected_non_opening_failure_mining/selected_non_opening_family_rows.jsonl`.

## 2. Why incumbent_proxy_disagreement was selected

- PR #42 selected `incumbent_proxy_disagreement` as the highest-ranked corrected non-opening family still classified as a value/backup issue.
- Family stats: ``{"avg_reference_visit_share": 0.3726, "avg_selected_minus_reference_q_margin": 0.0624, "classification": "value_or_backup_issue", "conflicting_target_count": 0, "duplicate_canonical_state_count": 0, "fail_rows": 23, "failure_mechanism": "value_q", "failure_rate": 0.7188, "family": "incumbent_proxy_disagreement", "high_severity_count": 0, "medium_severity_count": 23, "notes": "persistent failures remain Q/value-dominant", "pass_rows": 9, "rank_score": 302.261, "stable_reference_rows": 32, "total_rows": 32}``.

## 3. Row selection and validation

| row_id | role | corrected_reference_move | current_selected_384 | current_selected_1200 | reference_stable | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-009 | target_candidate | 0 | 3 | 3 | true | ok | reference and suite canonical state validated |
| incumbent_proxy_disagreement-035 | target_candidate | 3 | 4 | 4 | true | ok | reference and suite canonical state validated |
| incumbent_proxy_disagreement-022 | target_candidate | 3 | 5 | 5 | true | ok | reference and suite canonical state validated |
| incumbent_proxy_disagreement-024 | target_candidate | 2 | 1 | 1 | true | ok | reference and suite canonical state validated |
| incumbent_proxy_disagreement-011 | target_candidate | 4 | 5 | 5 | true | ok | reference and suite canonical state validated |
| incumbent_proxy_disagreement-014 | target_candidate | 4 | 5 | 5 | true | ok | reference and suite canonical state validated |
| incumbent_proxy_disagreement-007 | target_candidate | 0 | 5 | 3 | true | ok | reference and suite canonical state validated |
| incumbent_proxy_disagreement-025 | target_candidate | 2 | 4 | 4 | true | ok | reference and suite canonical state validated |
| incumbent_proxy_disagreement-023 | target_candidate | 2 | 1 | 1 | true | ok | reference and suite canonical state validated |
| incumbent_proxy_disagreement-021 | target_candidate | 3 | 5 | 5 | true | ok | reference and suite canonical state validated |
| incumbent_proxy_disagreement-017 | target_candidate | 4 | 5 | 4 | true | ok | reference and suite canonical state validated |
| incumbent_proxy_disagreement-020 | target_candidate | 3 | 5 | 5 | true | ok | reference and suite canonical state validated |
| incumbent_proxy_disagreement-018 | target_candidate | 2 | 5 | 3 | true | ok | reference and suite canonical state validated |
| incumbent_proxy_disagreement-010 | target_candidate | 4 | 5 | 5 | true | ok | reference and suite canonical state validated |
| incumbent_proxy_disagreement-012 | target_candidate | 3 | 5 | 5 | true | ok | reference and suite canonical state validated |
| incumbent_proxy_disagreement-033 | target_candidate | 4 | 0 | 0 | true | ok | reference and suite canonical state validated |
| incumbent_proxy_disagreement-015 | target_candidate | 4 | 5 | 4 | true | ok | reference and suite canonical state validated |
| incumbent_proxy_disagreement-027 | target_candidate | 3 | 5 | 5 | true | ok | reference and suite canonical state validated |
| incumbent_proxy_disagreement-002 | target_candidate | 3 | 3 | 3 | true | ok | reference and suite canonical state validated |
| incumbent_proxy_disagreement-003 | target_candidate | 4 | 4 | 5 | true | ok | reference and suite canonical state validated |
| incumbent_proxy_disagreement-001 | target_candidate | 3 | 3 | 3 | true | ok | reference and suite canonical state validated |
| incumbent_proxy_disagreement-019 | holdout_candidate | 3 | 3 | 3 | true | ok | reference and suite canonical state validated |
| incumbent_proxy_disagreement-030 | holdout_candidate | 3 | 3 | 3 | true | ok | reference and suite canonical state validated |
| incumbent_proxy_disagreement-026 | preservation_control | 5 | 5 | 5 | true | ok | reference and suite canonical state validated |
| incumbent_proxy_disagreement-028 | preservation_control | 5 | 5 | 5 | true | ok | reference and suite canonical state validated |
| incumbent_proxy_disagreement-029 | preservation_control | 4 | 4 | 4 | true | ok | reference and suite canonical state validated |
| incumbent_proxy_disagreement-008 | preservation_control | 2 | 2 | 2 | true | ok | reference and suite canonical state validated |

- Audited target rows: `['incumbent_proxy_disagreement-007', 'incumbent_proxy_disagreement-009', 'incumbent_proxy_disagreement-011', 'incumbent_proxy_disagreement-014', 'incumbent_proxy_disagreement-021', 'incumbent_proxy_disagreement-022', 'incumbent_proxy_disagreement-023', 'incumbent_proxy_disagreement-024', 'incumbent_proxy_disagreement-025', 'incumbent_proxy_disagreement-035']`.
- Audited holdout rows: `['incumbent_proxy_disagreement-019', 'incumbent_proxy_disagreement-030']`.
- Audited control rows: `['incumbent_proxy_disagreement-008', 'incumbent_proxy_disagreement-026', 'incumbent_proxy_disagreement-028', 'incumbent_proxy_disagreement-029']`.
- Perspective conversion: `+1 when child current_player == root current_player, else -1`.

## 4. Root PUCT baseline

| row_id | corrected_reference_move | selected_move | budget | reference_visit_share | selected_visit_share | reference_q | selected_q | selected_minus_reference_q_margin | reference_prior | selected_prior | pass | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-007 | 0 | 5 | 384 | 0.0052 | 0.4219 | 0.0434 | 0.1452 | 0.1018 | 0.0909 | 0.3676 | false | deterministic PUCT baseline |
| incumbent_proxy_disagreement-007 | 0 | 3 | 1200 | 0.0025 | 0.5167 | 0.0774 | 0.1507 | 0.0733 | 0.0909 | 0.1087 | false | deterministic PUCT baseline |
| incumbent_proxy_disagreement-008 | 2 | 2 | 384 | 0.9297 | 0.9297 | 0.3892 | 0.3892 | 0.0000 | 0.4721 | 0.4721 | true | deterministic PUCT baseline |
| incumbent_proxy_disagreement-008 | 2 | 2 | 1200 | 0.9708 | 0.9708 | 0.3624 | 0.3624 | 0.0000 | 0.4721 | 0.4721 | true | deterministic PUCT baseline |
| incumbent_proxy_disagreement-009 | 0 | 3 | 384 | 0.0026 | 0.8516 | -0.0295 | 0.1793 | 0.2088 | 0.0197 | 0.1513 | false | deterministic PUCT baseline |
| incumbent_proxy_disagreement-009 | 0 | 3 | 1200 | 0.0008 | 0.9500 | -0.0295 | 0.1844 | 0.2139 | 0.0197 | 0.1513 | false | deterministic PUCT baseline |
| incumbent_proxy_disagreement-011 | 4 | 5 | 384 | 0.0339 | 0.9531 | 0.0640 | 0.1836 | 0.1195 | 0.3903 | 0.3616 | false | deterministic PUCT baseline |
| incumbent_proxy_disagreement-011 | 4 | 5 | 1200 | 0.0475 | 0.9342 | 0.1225 | 0.1939 | 0.0714 | 0.3903 | 0.3616 | false | deterministic PUCT baseline |
| incumbent_proxy_disagreement-014 | 4 | 5 | 384 | 0.0026 | 0.9870 | -0.0561 | 0.0469 | 0.1030 | 0.0817 | 0.8185 | false | deterministic PUCT baseline |
| incumbent_proxy_disagreement-014 | 4 | 5 | 1200 | 0.3392 | 0.5592 | 0.0345 | 0.0312 | -0.0033 | 0.0817 | 0.8185 | false | deterministic PUCT baseline |
| incumbent_proxy_disagreement-019 | 3 | 3 | 384 | 0.7292 | 0.7292 | 0.1962 | 0.1962 | 0.0000 | 0.0971 | 0.0971 | true | deterministic PUCT baseline |
| incumbent_proxy_disagreement-019 | 3 | 3 | 1200 | 0.8925 | 0.8925 | 0.2049 | 0.2049 | 0.0000 | 0.0971 | 0.0971 | true | deterministic PUCT baseline |
| incumbent_proxy_disagreement-021 | 3 | 5 | 384 | 0.0651 | 0.9167 | 0.0741 | 0.1474 | 0.0733 | 0.2072 | 0.4779 | false | deterministic PUCT baseline |
| incumbent_proxy_disagreement-021 | 3 | 5 | 1200 | 0.0208 | 0.9692 | 0.0741 | 0.2013 | 0.1272 | 0.2072 | 0.4779 | false | deterministic PUCT baseline |
| incumbent_proxy_disagreement-022 | 3 | 5 | 384 | 0.0052 | 0.6328 | 0.0001 | 0.1458 | 0.1457 | 0.1046 | 0.4716 | false | deterministic PUCT baseline |
| incumbent_proxy_disagreement-022 | 3 | 5 | 1200 | 0.0033 | 0.6000 | 0.0784 | 0.1324 | 0.0540 | 0.1046 | 0.4716 | false | deterministic PUCT baseline |
| incumbent_proxy_disagreement-023 | 2 | 1 | 384 | 0.0130 | 0.9557 | -0.2115 | -0.1197 | 0.0918 | 0.1087 | 0.5339 | false | deterministic PUCT baseline |
| incumbent_proxy_disagreement-023 | 2 | 1 | 1200 | 0.0050 | 0.9792 | -0.2531 | -0.1566 | 0.0965 | 0.1087 | 0.5339 | false | deterministic PUCT baseline |
| incumbent_proxy_disagreement-024 | 2 | 1 | 384 | 0.0104 | 0.9479 | -0.2742 | -0.1485 | 0.1256 | 0.0928 | 0.3747 | false | deterministic PUCT baseline |
| incumbent_proxy_disagreement-024 | 2 | 1 | 1200 | 0.0067 | 0.9725 | -0.2492 | -0.1484 | 0.1009 | 0.0928 | 0.3747 | false | deterministic PUCT baseline |
| incumbent_proxy_disagreement-025 | 2 | 4 | 384 | 0.0000 | 0.8750 | 0.0000 | 0.1011 | 0.1011 | 0.0012 | 0.9454 | false | deterministic PUCT baseline |
| incumbent_proxy_disagreement-025 | 2 | 4 | 1200 | 0.2650 | 0.3717 | 0.1212 | 0.0620 | -0.0592 | 0.0012 | 0.9454 | false | deterministic PUCT baseline |
| incumbent_proxy_disagreement-026 | 5 | 5 | 384 | 0.9922 | 0.9922 | 0.1303 | 0.1303 | 0.0000 | 0.9168 | 0.9168 | true | deterministic PUCT baseline |
| incumbent_proxy_disagreement-026 | 5 | 5 | 1200 | 0.9650 | 0.9650 | 0.1098 | 0.1098 | 0.0000 | 0.9168 | 0.9168 | true | deterministic PUCT baseline |
| incumbent_proxy_disagreement-028 | 5 | 5 | 384 | 0.9323 | 0.9323 | 0.1054 | 0.1054 | 0.0000 | 0.6276 | 0.6276 | true | deterministic PUCT baseline |
| incumbent_proxy_disagreement-028 | 5 | 5 | 1200 | 0.9250 | 0.9250 | 0.1254 | 0.1254 | 0.0000 | 0.6276 | 0.6276 | true | deterministic PUCT baseline |
| incumbent_proxy_disagreement-029 | 4 | 4 | 384 | 0.9323 | 0.9323 | 0.2153 | 0.2153 | 0.0000 | 0.7822 | 0.7822 | true | deterministic PUCT baseline |
| incumbent_proxy_disagreement-029 | 4 | 4 | 1200 | 0.9767 | 0.9767 | 0.2549 | 0.2549 | 0.0000 | 0.7822 | 0.7822 | true | deterministic PUCT baseline |
| incumbent_proxy_disagreement-030 | 3 | 3 | 384 | 0.7318 | 0.7318 | 0.0997 | 0.0997 | 0.0000 | 0.1124 | 0.1124 | true | deterministic PUCT baseline |
| incumbent_proxy_disagreement-030 | 3 | 3 | 1200 | 0.8875 | 0.8875 | 0.1067 | 0.1067 | 0.0000 | 0.1124 | 0.1124 | true | deterministic PUCT baseline |
| incumbent_proxy_disagreement-035 | 3 | 4 | 384 | 0.0104 | 0.9479 | -0.2530 | -0.1016 | 0.1515 | 0.1615 | 0.2475 | false | deterministic PUCT baseline |
| incumbent_proxy_disagreement-035 | 3 | 4 | 1200 | 0.0058 | 0.6350 | -0.2649 | -0.1247 | 0.1401 | 0.1615 | 0.2475 | false | deterministic PUCT baseline |

- `incumbent_proxy_disagreement-019` legal moves and consequences: `[{"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 0, "produces_capture": false, "remaining_seed_count": 43, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 1, "produces_capture": false, "remaining_seed_count": 43, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 2, "produces_capture": false, "remaining_seed_count": 43, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 3, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 5, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}]`.
- `incumbent_proxy_disagreement-030` legal moves and consequences: `[{"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 0, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 1, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 2, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 3, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 4, "produces_capture": false, "remaining_seed_count": 41, "side_to_move_after": 1}]`.
- `incumbent_proxy_disagreement-008` legal moves and consequences: `[{"capture_count": 8, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 8, "move": 0, "produces_capture": true, "remaining_seed_count": 36, "side_to_move_after": 1}, {"capture_count": 10, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 10, "move": 2, "produces_capture": true, "remaining_seed_count": 34, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 3, "produces_capture": false, "remaining_seed_count": 43, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 5, "produces_capture": false, "remaining_seed_count": 43, "side_to_move_after": 1}]`.
- `incumbent_proxy_disagreement-026` legal moves and consequences: `[{"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 0, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 1, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 2, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 3, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 5, "produces_capture": false, "remaining_seed_count": 41, "side_to_move_after": 1}]`.
- `incumbent_proxy_disagreement-028` legal moves and consequences: `[{"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 0, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 1, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 2, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 3, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 4, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 5, "produces_capture": false, "remaining_seed_count": 41, "side_to_move_after": 1}]`.
- `incumbent_proxy_disagreement-029` legal moves and consequences: `[{"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 0, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 1, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 2, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 3, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": true, "immediate_store_delta": 1, "move": 4, "produces_capture": false, "remaining_seed_count": 41, "side_to_move_after": 0}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 5, "produces_capture": false, "remaining_seed_count": 41, "side_to_move_after": 1}]`.
- `incumbent_proxy_disagreement-007` legal moves and consequences: `[{"capture_count": 8, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 8, "move": 0, "produces_capture": true, "remaining_seed_count": 36, "side_to_move_after": 1}, {"capture_count": 2, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 2, "move": 2, "produces_capture": true, "remaining_seed_count": 42, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 3, "produces_capture": false, "remaining_seed_count": 43, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 5, "produces_capture": false, "remaining_seed_count": 43, "side_to_move_after": 1}]`.
- `incumbent_proxy_disagreement-009` legal moves and consequences: `[{"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 0, "produces_capture": false, "remaining_seed_count": 44, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 1, "produces_capture": false, "remaining_seed_count": 44, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 3, "produces_capture": false, "remaining_seed_count": 43, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 5, "produces_capture": false, "remaining_seed_count": 43, "side_to_move_after": 1}]`.
- `incumbent_proxy_disagreement-011` legal moves and consequences: `[{"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 0, "produces_capture": false, "remaining_seed_count": 43, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 1, "produces_capture": false, "remaining_seed_count": 43, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 2, "produces_capture": false, "remaining_seed_count": 43, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 4, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 5, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}]`.
- `incumbent_proxy_disagreement-014` legal moves and consequences: `[{"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 0, "produces_capture": false, "remaining_seed_count": 43, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 1, "produces_capture": false, "remaining_seed_count": 43, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 2, "produces_capture": false, "remaining_seed_count": 43, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 4, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 5, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}]`.
- `incumbent_proxy_disagreement-021` legal moves and consequences: `[{"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 0, "produces_capture": false, "remaining_seed_count": 43, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 1, "produces_capture": false, "remaining_seed_count": 43, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 2, "produces_capture": false, "remaining_seed_count": 43, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 3, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 5, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}]`.
- `incumbent_proxy_disagreement-022` legal moves and consequences: `[{"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 0, "produces_capture": false, "remaining_seed_count": 43, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 1, "produces_capture": false, "remaining_seed_count": 43, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 2, "produces_capture": false, "remaining_seed_count": 43, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 3, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 5, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}]`.
- `incumbent_proxy_disagreement-023` legal moves and consequences: `[{"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 0, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 0}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 1, "produces_capture": false, "remaining_seed_count": 41, "side_to_move_after": 0}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 2, "produces_capture": false, "remaining_seed_count": 41, "side_to_move_after": 0}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 3, "produces_capture": false, "remaining_seed_count": 41, "side_to_move_after": 0}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 4, "produces_capture": false, "remaining_seed_count": 41, "side_to_move_after": 0}]`.
- `incumbent_proxy_disagreement-024` legal moves and consequences: `[{"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 0, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 0}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 1, "produces_capture": false, "remaining_seed_count": 41, "side_to_move_after": 0}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 2, "produces_capture": false, "remaining_seed_count": 41, "side_to_move_after": 0}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 3, "produces_capture": false, "remaining_seed_count": 41, "side_to_move_after": 0}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 4, "produces_capture": false, "remaining_seed_count": 41, "side_to_move_after": 0}]`.
- `incumbent_proxy_disagreement-025` legal moves and consequences: `[{"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 0, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 1, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 2, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 0, "move": 3, "produces_capture": false, "remaining_seed_count": 42, "side_to_move_after": 1}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 4, "produces_capture": false, "remaining_seed_count": 41, "side_to_move_after": 1}]`.
- `incumbent_proxy_disagreement-035` legal moves and consequences: `[{"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 1, "produces_capture": false, "remaining_seed_count": 43, "side_to_move_after": 0}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 3, "produces_capture": false, "remaining_seed_count": 43, "side_to_move_after": 0}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 4, "produces_capture": false, "remaining_seed_count": 43, "side_to_move_after": 0}, {"capture_count": 0, "game_over_after_move": false, "gives_extra_turn": false, "immediate_store_delta": 1, "move": 5, "produces_capture": false, "remaining_seed_count": 43, "side_to_move_after": 0}]`.

## 5. Child-afterstate neural value audit

| row_id | corrected_reference_move | selected_move | child | raw_value | root_perspective_value | child_ref_minus_child_selected | agrees_with_corrected_reference | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-019 | 3 | 3 | corrected_reference_child | -0.0584 | 0.0584 | 0.0000 | false | sign flip to root perspective |
| incumbent_proxy_disagreement-019 | 3 | 3 | selected_child | -0.0584 | 0.0584 | 0.0000 | false | sign flip to root perspective |
| incumbent_proxy_disagreement-030 | 3 | 3 | corrected_reference_child | -0.0116 | 0.0116 | 0.0000 | false | sign flip to root perspective |
| incumbent_proxy_disagreement-030 | 3 | 3 | selected_child | -0.0116 | 0.0116 | 0.0000 | false | sign flip to root perspective |
| incumbent_proxy_disagreement-007 | 0 | 3 | corrected_reference_child | -0.0756 | 0.0756 | -0.0173 | false | sign flip to root perspective |
| incumbent_proxy_disagreement-007 | 0 | 3 | selected_child | -0.0929 | 0.0929 | -0.0173 | false | sign flip to root perspective |
| incumbent_proxy_disagreement-009 | 0 | 3 | corrected_reference_child | 0.0295 | -0.0295 | -0.1350 | false | sign flip to root perspective |
| incumbent_proxy_disagreement-009 | 0 | 3 | selected_child | -0.1055 | 0.1055 | -0.1350 | false | sign flip to root perspective |
| incumbent_proxy_disagreement-011 | 4 | 5 | corrected_reference_child | -0.0857 | 0.0857 | 0.0225 | true | sign flip to root perspective |
| incumbent_proxy_disagreement-011 | 4 | 5 | selected_child | -0.0632 | 0.0632 | 0.0225 | true | sign flip to root perspective |
| incumbent_proxy_disagreement-014 | 4 | 5 | corrected_reference_child | 0.0561 | -0.0561 | 0.0255 | true | sign flip to root perspective |
| incumbent_proxy_disagreement-014 | 4 | 5 | selected_child | 0.0816 | -0.0816 | 0.0255 | true | sign flip to root perspective |
| incumbent_proxy_disagreement-021 | 3 | 5 | corrected_reference_child | -0.1280 | 0.1280 | 0.1150 | true | sign flip to root perspective |
| incumbent_proxy_disagreement-021 | 3 | 5 | selected_child | -0.0130 | 0.0130 | 0.1150 | true | sign flip to root perspective |
| incumbent_proxy_disagreement-022 | 3 | 5 | corrected_reference_child | -0.0274 | 0.0274 | -0.0745 | false | sign flip to root perspective |
| incumbent_proxy_disagreement-022 | 3 | 5 | selected_child | -0.1019 | 0.1019 | -0.0745 | false | sign flip to root perspective |
| incumbent_proxy_disagreement-023 | 2 | 1 | corrected_reference_child | 0.1763 | -0.1763 | 0.0847 | true | sign flip to root perspective |
| incumbent_proxy_disagreement-023 | 2 | 1 | selected_child | 0.2610 | -0.2610 | 0.0847 | true | sign flip to root perspective |
| incumbent_proxy_disagreement-024 | 2 | 1 | corrected_reference_child | 0.3977 | -0.3977 | 0.0414 | true | sign flip to root perspective |
| incumbent_proxy_disagreement-024 | 2 | 1 | selected_child | 0.4392 | -0.4392 | 0.0414 | true | sign flip to root perspective |
| incumbent_proxy_disagreement-025 | 2 | 4 | corrected_reference_child | -0.0957 | 0.0957 | -0.0565 | false | sign flip to root perspective |
| incumbent_proxy_disagreement-025 | 2 | 4 | selected_child | -0.1522 | 0.1522 | -0.0565 | false | sign flip to root perspective |
| incumbent_proxy_disagreement-035 | 3 | 4 | corrected_reference_child | 0.1014 | -0.1014 | -0.0197 | false | sign flip to root perspective |
| incumbent_proxy_disagreement-035 | 3 | 4 | selected_child | 0.0817 | -0.0817 | -0.0197 | false | sign flip to root perspective |

## 6. Child-afterstate teacher audit

| row_id | budget | seed | child_ref_value_root | child_selected_value_root | child_ref_minus_child_selected | teacher_prefers_corrected_reference | stable | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-019 | 1200 | 11 | -0.0496 | -0.0496 | 0.0000 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-019 | 1200 | 23 | -0.0762 | -0.0762 | 0.0000 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-019 | 1200 | 37 | -0.1149 | -0.1149 | 0.0000 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-019 | 1200 | 42 | -0.0853 | -0.0853 | 0.0000 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-019 | 1200 | 101 | -0.0374 | -0.0374 | 0.0000 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-019 | 2400 | 11 | -0.2842 | -0.2842 | 0.0000 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-019 | 2400 | 23 | -0.2737 | -0.2737 | 0.0000 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-019 | 2400 | 37 | -0.3171 | -0.3171 | 0.0000 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-019 | 2400 | 42 | -0.3241 | -0.3241 | 0.0000 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-019 | 2400 | 101 | -0.3162 | -0.3162 | 0.0000 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-019 | 5000 | 11 | -0.5873 | -0.5873 | 0.0000 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-019 | 5000 | 23 | -0.6245 | -0.6245 | 0.0000 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-019 | 5000 | 37 | -0.5964 | -0.5964 | 0.0000 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-019 | 5000 | 42 | -0.5938 | -0.5938 | 0.0000 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-019 | 5000 | 101 | -0.5911 | -0.5911 | 0.0000 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-030 | 1200 | 11 | -0.4193 | -0.4193 | 0.0000 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-030 | 1200 | 23 | -0.4200 | -0.4200 | 0.0000 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-030 | 1200 | 37 | -0.4244 | -0.4244 | 0.0000 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-030 | 1200 | 42 | -0.4371 | -0.4371 | 0.0000 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-030 | 1200 | 101 | -0.4470 | -0.4470 | 0.0000 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-030 | 2400 | 11 | -0.5043 | -0.5043 | 0.0000 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-030 | 2400 | 23 | -0.5064 | -0.5064 | 0.0000 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-030 | 2400 | 37 | -0.4888 | -0.4888 | 0.0000 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-030 | 2400 | 42 | -0.4951 | -0.4951 | 0.0000 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-030 | 2400 | 101 | -0.5148 | -0.5148 | 0.0000 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-030 | 5000 | 11 | -0.5791 | -0.5791 | 0.0000 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-030 | 5000 | 23 | -0.5765 | -0.5765 | 0.0000 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-030 | 5000 | 37 | -0.5734 | -0.5734 | 0.0000 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-030 | 5000 | 42 | -0.5699 | -0.5699 | 0.0000 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-030 | 5000 | 101 | -0.5642 | -0.5642 | 0.0000 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-007 | 1200 | 11 | -0.0497 | 0.1894 | -0.2391 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-007 | 1200 | 23 | -0.0941 | 0.3259 | -0.4201 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-007 | 1200 | 37 | -0.0359 | 0.4127 | -0.4487 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-007 | 1200 | 42 | 0.2915 | 0.3219 | -0.0304 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-007 | 1200 | 101 | -0.0437 | 0.3047 | -0.3484 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-007 | 2400 | 11 | -0.5151 | -0.2805 | -0.2345 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-007 | 2400 | 23 | -0.5190 | -0.1154 | -0.4036 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-007 | 2400 | 37 | -0.5017 | 0.1095 | -0.6112 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-007 | 2400 | 42 | -0.3562 | -0.2849 | -0.0713 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-007 | 2400 | 101 | -0.5482 | -0.2247 | -0.3235 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-007 | 5000 | 11 | -0.7349 | -0.5630 | -0.1719 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-007 | 5000 | 23 | -0.7346 | -0.5321 | -0.2026 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-007 | 5000 | 37 | -0.7236 | -0.1487 | -0.5749 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-007 | 5000 | 42 | -0.6722 | -0.5786 | -0.0936 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-007 | 5000 | 101 | -0.7841 | -0.5562 | -0.2279 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-009 | 1200 | 11 | -0.3448 | -0.0954 | -0.2494 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-009 | 1200 | 23 | -0.3613 | -0.1615 | -0.1997 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-009 | 1200 | 37 | -0.2871 | 0.1751 | -0.4622 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-009 | 1200 | 42 | -0.3121 | -0.1351 | -0.1770 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-009 | 1200 | 101 | -0.3598 | -0.1098 | -0.2499 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-009 | 2400 | 11 | -0.5823 | -0.4683 | -0.1141 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-009 | 2400 | 23 | -0.5884 | -0.4017 | -0.1867 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-009 | 2400 | 37 | -0.5313 | -0.3588 | -0.1724 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-009 | 2400 | 42 | -0.5438 | -0.4098 | -0.1340 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-009 | 2400 | 101 | -0.5948 | -0.4692 | -0.1257 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-009 | 5000 | 11 | -0.7597 | -0.6503 | -0.1095 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-009 | 5000 | 23 | -0.7470 | -0.6008 | -0.1462 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-009 | 5000 | 37 | -0.7315 | -0.6137 | -0.1178 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-009 | 5000 | 42 | -0.7411 | -0.6051 | -0.1361 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-009 | 5000 | 101 | -0.7677 | -0.6417 | -0.1261 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-011 | 1200 | 11 | 0.1272 | 0.0341 | 0.0931 | true | false | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-011 | 1200 | 23 | 0.1585 | 0.0724 | 0.0861 | true | false | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-011 | 1200 | 37 | -0.1387 | 0.1862 | -0.3250 | false | false | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-011 | 1200 | 42 | 0.2720 | 0.1496 | 0.1225 | true | false | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-011 | 1200 | 101 | 0.0344 | 0.2456 | -0.2112 | false | false | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-011 | 2400 | 11 | -0.0831 | -0.1347 | 0.0516 | true | false | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-011 | 2400 | 23 | -0.3507 | -0.3011 | -0.0496 | false | false | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-011 | 2400 | 37 | -0.4896 | -0.0082 | -0.4815 | false | false | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-011 | 2400 | 42 | -0.0404 | -0.0390 | -0.0014 | false | false | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-011 | 2400 | 101 | -0.0510 | -0.1140 | 0.0630 | true | false | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-011 | 5000 | 11 | -0.2825 | -0.5429 | 0.2604 | true | false | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-011 | 5000 | 23 | -0.5636 | -0.6214 | 0.0578 | true | false | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-011 | 5000 | 37 | -0.6064 | -0.4678 | -0.1387 | false | false | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-011 | 5000 | 42 | -0.1820 | -0.4998 | 0.3178 | true | false | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-011 | 5000 | 101 | -0.2996 | -0.5635 | 0.2639 | true | false | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-014 | 1200 | 11 | 0.0729 | -0.3616 | 0.4345 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-014 | 1200 | 23 | -0.1854 | -0.2726 | 0.0872 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-014 | 1200 | 37 | -0.0191 | -0.3157 | 0.2966 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-014 | 1200 | 42 | 0.1138 | -0.2138 | 0.3275 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-014 | 1200 | 101 | 0.0786 | -0.1369 | 0.2154 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-014 | 2400 | 11 | -0.4206 | -0.5947 | 0.1741 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-014 | 2400 | 23 | -0.5442 | -0.5876 | 0.0433 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-014 | 2400 | 37 | -0.4026 | -0.6285 | 0.2259 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-014 | 2400 | 42 | -0.4356 | -0.5862 | 0.1506 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-014 | 2400 | 101 | -0.4315 | -0.5554 | 0.1239 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-014 | 5000 | 11 | -0.6787 | -0.7443 | 0.0656 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-014 | 5000 | 23 | -0.7374 | -0.7499 | 0.0125 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-014 | 5000 | 37 | -0.6847 | -0.7768 | 0.0921 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-014 | 5000 | 42 | -0.6855 | -0.7652 | 0.0797 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-014 | 5000 | 101 | -0.6778 | -0.7308 | 0.0530 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-021 | 1200 | 11 | -0.6450 | -0.6607 | 0.0157 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-021 | 1200 | 23 | -0.6527 | -0.6684 | 0.0157 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-021 | 1200 | 37 | -0.6322 | -0.6794 | 0.0472 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-021 | 1200 | 42 | -0.6450 | -0.6469 | 0.0020 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-021 | 1200 | 101 | -0.6217 | -0.6283 | 0.0066 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-021 | 2400 | 11 | -0.7337 | -0.7275 | -0.0062 | false | false | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-021 | 2400 | 23 | -0.7306 | -0.7362 | 0.0056 | true | false | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-021 | 2400 | 37 | -0.7316 | -0.7314 | -0.0002 | false | false | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-021 | 2400 | 42 | -0.7377 | -0.7174 | -0.0202 | false | false | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-021 | 2400 | 101 | -0.7281 | -0.7231 | -0.0050 | false | false | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-021 | 5000 | 11 | -0.8000 | -0.7804 | -0.0195 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-021 | 5000 | 23 | -0.7976 | -0.7887 | -0.0089 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-021 | 5000 | 37 | -0.8086 | -0.7944 | -0.0142 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-021 | 5000 | 42 | -0.8018 | -0.7832 | -0.0186 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-021 | 5000 | 101 | -0.8005 | -0.7844 | -0.0162 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-022 | 1200 | 11 | 0.3659 | -0.6390 | 1.0049 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-022 | 1200 | 23 | 0.4169 | -0.6752 | 1.0922 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-022 | 1200 | 37 | 0.2780 | -0.6821 | 0.9600 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-022 | 1200 | 42 | 0.4038 | -0.6639 | 1.0677 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-022 | 1200 | 101 | 0.3524 | -0.6660 | 1.0185 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-022 | 2400 | 11 | 0.0830 | -0.7518 | 0.8347 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-022 | 2400 | 23 | 0.1766 | -0.7775 | 0.9541 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-022 | 2400 | 37 | -0.0524 | -0.7773 | 0.7249 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-022 | 2400 | 42 | 0.0763 | -0.7602 | 0.8365 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-022 | 2400 | 101 | 0.0135 | -0.7657 | 0.7792 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-022 | 5000 | 11 | -0.2329 | -0.8153 | 0.5824 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-022 | 5000 | 23 | -0.3474 | -0.8325 | 0.4851 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-022 | 5000 | 37 | -0.3224 | -0.8321 | 0.5097 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-022 | 5000 | 42 | -0.3149 | -0.8195 | 0.5046 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-022 | 5000 | 101 | -0.3005 | -0.8166 | 0.5160 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-023 | 1200 | 11 | -0.9641 | -0.5906 | -0.3735 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-023 | 1200 | 23 | -0.9630 | -0.6124 | -0.3505 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-023 | 1200 | 37 | -0.9716 | -0.5578 | -0.4138 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-023 | 1200 | 42 | -0.9695 | -0.6433 | -0.3262 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-023 | 1200 | 101 | -0.9564 | -0.5877 | -0.3687 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-023 | 2400 | 11 | -0.9444 | -0.6208 | -0.3236 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-023 | 2400 | 23 | -0.9469 | -0.6012 | -0.3457 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-023 | 2400 | 37 | -0.9455 | -0.6049 | -0.3406 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-023 | 2400 | 42 | -0.9460 | -0.6235 | -0.3225 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-023 | 2400 | 101 | -0.9430 | -0.6224 | -0.3206 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-023 | 5000 | 11 | -0.9274 | -0.7330 | -0.1944 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-023 | 5000 | 23 | -0.9301 | -0.7076 | -0.2225 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-023 | 5000 | 37 | -0.9276 | -0.7448 | -0.1828 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-023 | 5000 | 42 | -0.9284 | -0.6911 | -0.2374 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-023 | 5000 | 101 | -0.9238 | -0.7149 | -0.2090 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-024 | 1200 | 11 | -0.6597 | -0.5743 | -0.0854 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-024 | 1200 | 23 | -0.6904 | -0.5714 | -0.1190 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-024 | 1200 | 37 | -0.6716 | -0.5430 | -0.1286 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-024 | 1200 | 42 | -0.6882 | -0.5483 | -0.1399 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-024 | 1200 | 101 | -0.6697 | -0.5695 | -0.1002 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-024 | 2400 | 11 | -0.7575 | -0.5405 | -0.2170 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-024 | 2400 | 23 | -0.7586 | -0.5302 | -0.2284 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-024 | 2400 | 37 | -0.7591 | -0.5383 | -0.2207 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-024 | 2400 | 42 | -0.7542 | -0.5177 | -0.2365 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-024 | 2400 | 101 | -0.7510 | -0.5304 | -0.2206 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-024 | 5000 | 11 | -0.8342 | -0.6434 | -0.1907 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-024 | 5000 | 23 | -0.8359 | -0.6770 | -0.1589 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-024 | 5000 | 37 | -0.8390 | -0.7122 | -0.1268 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-024 | 5000 | 42 | -0.8360 | -0.6792 | -0.1569 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-024 | 5000 | 101 | -0.8333 | -0.6109 | -0.2223 | false | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-025 | 1200 | 11 | -0.3570 | -0.3810 | 0.0239 | true | false | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-025 | 1200 | 23 | -0.3690 | -0.3347 | -0.0343 | false | false | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-025 | 1200 | 37 | -0.3758 | -0.1757 | -0.2001 | false | false | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-025 | 1200 | 42 | -0.3575 | -0.2694 | -0.0880 | false | false | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-025 | 1200 | 101 | -0.3613 | -0.3593 | -0.0020 | false | false | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-025 | 2400 | 11 | -0.4333 | -0.5845 | 0.1512 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-025 | 2400 | 23 | -0.4638 | -0.5676 | 0.1038 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-025 | 2400 | 37 | -0.4544 | -0.5615 | 0.1071 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-025 | 2400 | 42 | -0.4717 | -0.5537 | 0.0821 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-025 | 2400 | 101 | -0.4433 | -0.5767 | 0.1334 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-025 | 5000 | 11 | -0.5845 | -0.6832 | 0.0988 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-025 | 5000 | 23 | -0.5755 | -0.6781 | 0.1026 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-025 | 5000 | 37 | -0.6318 | -0.7012 | 0.0694 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-025 | 5000 | 42 | -0.6493 | -0.6902 | 0.0408 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-025 | 5000 | 101 | -0.6354 | -0.6843 | 0.0488 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-035 | 1200 | 11 | -0.4943 | -0.6156 | 0.1214 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-035 | 1200 | 23 | -0.5574 | -0.6159 | 0.0585 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-035 | 1200 | 37 | -0.5382 | -0.5997 | 0.0615 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-035 | 1200 | 42 | -0.5131 | -0.6213 | 0.1082 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-035 | 1200 | 101 | -0.5506 | -0.5974 | 0.0468 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-035 | 2400 | 11 | -0.5926 | -0.6327 | 0.0401 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-035 | 2400 | 23 | -0.6092 | -0.6490 | 0.0399 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-035 | 2400 | 37 | -0.6062 | -0.6508 | 0.0446 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-035 | 2400 | 42 | -0.6086 | -0.6551 | 0.0465 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-035 | 2400 | 101 | -0.6140 | -0.6467 | 0.0327 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-035 | 5000 | 11 | -0.7087 | -0.7421 | 0.0334 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-035 | 5000 | 23 | -0.7126 | -0.7478 | 0.0352 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-035 | 5000 | 37 | -0.7212 | -0.7400 | 0.0188 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-035 | 5000 | 42 | -0.7103 | -0.7506 | 0.0403 | true | true | classic MCTS child-afterstate teacher audit |
| incumbent_proxy_disagreement-035 | 5000 | 101 | -0.7179 | -0.7420 | 0.0240 | true | true | classic MCTS child-afterstate teacher audit |

## 7. Child-afterstate PUCT audit

| row_id | budget | child_ref_value_root | child_selected_value_root | child_ref_minus_child_selected | puct_prefers_corrected_reference | notes |
| --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-019 | 384 | 0.2088 | 0.2088 | 0.0000 | false | artifact PUCT child-afterstate audit |
| incumbent_proxy_disagreement-019 | 1200 | 0.2018 | 0.2018 | 0.0000 | false | artifact PUCT child-afterstate audit |
| incumbent_proxy_disagreement-030 | 384 | 0.0859 | 0.0859 | 0.0000 | false | artifact PUCT child-afterstate audit |
| incumbent_proxy_disagreement-030 | 1200 | 0.1077 | 0.1077 | 0.0000 | false | artifact PUCT child-afterstate audit |
| incumbent_proxy_disagreement-007 | 384 | 0.2786 | 0.1644 | 0.1142 | true | artifact PUCT child-afterstate audit |
| incumbent_proxy_disagreement-007 | 1200 | 0.2698 | 0.1484 | 0.1214 | true | artifact PUCT child-afterstate audit |
| incumbent_proxy_disagreement-009 | 384 | 0.0641 | 0.1821 | -0.1180 | false | artifact PUCT child-afterstate audit |
| incumbent_proxy_disagreement-009 | 1200 | 0.0960 | 0.1812 | -0.0851 | false | artifact PUCT child-afterstate audit |
| incumbent_proxy_disagreement-011 | 384 | 0.1540 | 0.1832 | -0.0292 | false | artifact PUCT child-afterstate audit |
| incumbent_proxy_disagreement-011 | 1200 | 0.1364 | 0.1901 | -0.0537 | false | artifact PUCT child-afterstate audit |
| incumbent_proxy_disagreement-014 | 384 | 0.0006 | 0.0471 | -0.0465 | false | artifact PUCT child-afterstate audit |
| incumbent_proxy_disagreement-014 | 1200 | 0.0650 | 0.0145 | 0.0504 | true | artifact PUCT child-afterstate audit |
| incumbent_proxy_disagreement-021 | 384 | 0.1675 | 0.1489 | 0.0186 | true | artifact PUCT child-afterstate audit |
| incumbent_proxy_disagreement-021 | 1200 | 0.1965 | 0.2053 | -0.0088 | false | artifact PUCT child-afterstate audit |
| incumbent_proxy_disagreement-022 | 384 | 0.1760 | 0.1359 | 0.0400 | true | artifact PUCT child-afterstate audit |
| incumbent_proxy_disagreement-022 | 1200 | 0.1313 | 0.1265 | 0.0048 | true | artifact PUCT child-afterstate audit |
| incumbent_proxy_disagreement-023 | 384 | -0.3822 | -0.1178 | -0.2644 | false | artifact PUCT child-afterstate audit |
| incumbent_proxy_disagreement-023 | 1200 | -0.4041 | -0.1573 | -0.2468 | false | artifact PUCT child-afterstate audit |
| incumbent_proxy_disagreement-024 | 384 | -0.1425 | -0.1447 | 0.0022 | true | artifact PUCT child-afterstate audit |
| incumbent_proxy_disagreement-024 | 1200 | -0.1728 | -0.1519 | -0.0209 | false | artifact PUCT child-afterstate audit |
| incumbent_proxy_disagreement-025 | 384 | 0.0733 | -0.0080 | 0.0813 | true | artifact PUCT child-afterstate audit |
| incumbent_proxy_disagreement-025 | 1200 | 0.0841 | -0.0495 | 0.1336 | true | artifact PUCT child-afterstate audit |
| incumbent_proxy_disagreement-035 | 384 | -0.1085 | -0.1121 | 0.0037 | true | artifact PUCT child-afterstate audit |
| incumbent_proxy_disagreement-035 | 1200 | -0.1334 | -0.1321 | -0.0013 | false | artifact PUCT child-afterstate audit |

## 8. Root counterfactual diagnostics

| row_id | intervention | budget | selected_move | selected_is_corrected_reference | reference_visit_share | selected_visit_share | selected_minus_reference_q_margin | flipped | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-019 | original | 384 | 3 | true | 0.7292 | 0.7292 | 0.0000 | true | no intervention |
| incumbent_proxy_disagreement-019 | original | 1200 | 3 | true | 0.8925 | 0.8925 | 0.0000 | true | no intervention |
| incumbent_proxy_disagreement-019 | equalize_root_priors | 384 | 3 | true | 0.7292 | 0.7292 | 0.0000 | true | equalize corrected reference and selected root priors |
| incumbent_proxy_disagreement-019 | equalize_root_priors | 1200 | 3 | true | 0.8925 | 0.8925 | 0.0000 | true | equalize corrected reference and selected root priors |
| incumbent_proxy_disagreement-019 | uniform_legal_prior | 384 | 3 | true | 0.7708 | 0.7708 | 0.0000 | true | uniform legal prior positive control |
| incumbent_proxy_disagreement-019 | uniform_legal_prior | 1200 | 3 | true | 0.9167 | 0.9167 | 0.0000 | true | uniform legal prior positive control |
| incumbent_proxy_disagreement-019 | teacher_child_value_override | 384 | 2 | false | 0.0052 | 0.7135 | 0.3152 | false | override first child expansion values with teacher child values |
| incumbent_proxy_disagreement-019 | teacher_child_value_override | 1200 | 2 | false | 0.0033 | 0.8733 | 0.1951 | false | override first child expansion values with teacher child values |
| incumbent_proxy_disagreement-019 | neural_child_value_swap | 384 | 3 | true | 0.7292 | 0.7292 | 0.0000 | true | swap child neural values as diagnostic sanity check |
| incumbent_proxy_disagreement-019 | neural_child_value_swap | 1200 | 3 | true | 0.8925 | 0.8925 | 0.0000 | true | swap child neural values as diagnostic sanity check |
| incumbent_proxy_disagreement-030 | original | 384 | 3 | true | 0.7318 | 0.7318 | 0.0000 | true | no intervention |
| incumbent_proxy_disagreement-030 | original | 1200 | 3 | true | 0.8875 | 0.8875 | 0.0000 | true | no intervention |
| incumbent_proxy_disagreement-030 | equalize_root_priors | 384 | 3 | true | 0.7318 | 0.7318 | 0.0000 | true | equalize corrected reference and selected root priors |
| incumbent_proxy_disagreement-030 | equalize_root_priors | 1200 | 3 | true | 0.8875 | 0.8875 | 0.0000 | true | equalize corrected reference and selected root priors |
| incumbent_proxy_disagreement-030 | uniform_legal_prior | 384 | 3 | true | 0.3490 | 0.3490 | 0.0000 | true | uniform legal prior positive control |
| incumbent_proxy_disagreement-030 | uniform_legal_prior | 1200 | 0 | false | 0.1892 | 0.5342 | -0.0003 | false | uniform legal prior positive control |
| incumbent_proxy_disagreement-030 | teacher_child_value_override | 384 | 1 | false | 0.0052 | 0.5026 | 0.2532 | false | override first child expansion values with teacher child values |
| incumbent_proxy_disagreement-030 | teacher_child_value_override | 1200 | 3 | true | 0.4483 | 0.4483 | 0.0000 | true | override first child expansion values with teacher child values |
| incumbent_proxy_disagreement-030 | neural_child_value_swap | 384 | 3 | true | 0.7318 | 0.7318 | 0.0000 | true | swap child neural values as diagnostic sanity check |
| incumbent_proxy_disagreement-030 | neural_child_value_swap | 1200 | 3 | true | 0.8875 | 0.8875 | 0.0000 | true | swap child neural values as diagnostic sanity check |
| incumbent_proxy_disagreement-007 | original | 384 | 5 | false | 0.0052 | 0.4219 | 0.1018 | false | no intervention |
| incumbent_proxy_disagreement-007 | original | 1200 | 3 | false | 0.0025 | 0.5167 | 0.0733 | false | no intervention |
| incumbent_proxy_disagreement-007 | equalize_root_priors | 384 | 5 | false | 0.0052 | 0.4219 | 0.1018 | false | equalize corrected reference and selected root priors |
| incumbent_proxy_disagreement-007 | equalize_root_priors | 1200 | 3 | false | 0.0033 | 0.5183 | 0.0810 | false | equalize corrected reference and selected root priors |
| incumbent_proxy_disagreement-007 | uniform_legal_prior | 384 | 3 | false | 0.0286 | 0.8828 | 0.0660 | false | uniform legal prior positive control |
| incumbent_proxy_disagreement-007 | uniform_legal_prior | 1200 | 3 | false | 0.3258 | 0.5308 | -0.1254 | false | uniform legal prior positive control |
| incumbent_proxy_disagreement-007 | teacher_child_value_override | 384 | 2 | false | 0.0052 | 0.5130 | 0.5283 | false | override first child expansion values with teacher child values |
| incumbent_proxy_disagreement-007 | teacher_child_value_override | 1200 | 2 | false | 0.0025 | 0.8167 | 0.3523 | false | override first child expansion values with teacher child values |
| incumbent_proxy_disagreement-007 | neural_child_value_swap | 384 | 5 | false | 0.0052 | 0.6328 | 0.1091 | false | swap child neural values as diagnostic sanity check |
| incumbent_proxy_disagreement-007 | neural_child_value_swap | 1200 | 5 | false | 0.0025 | 0.4858 | 0.0814 | false | swap child neural values as diagnostic sanity check |
| incumbent_proxy_disagreement-009 | original | 384 | 3 | false | 0.0026 | 0.8516 | 0.2088 | false | no intervention |
| incumbent_proxy_disagreement-009 | original | 1200 | 3 | false | 0.0008 | 0.9500 | 0.2139 | false | no intervention |
| incumbent_proxy_disagreement-009 | equalize_root_priors | 384 | 3 | false | 0.0078 | 0.8516 | 0.1107 | false | equalize corrected reference and selected root priors |
| incumbent_proxy_disagreement-009 | equalize_root_priors | 1200 | 3 | false | 0.0050 | 0.9500 | 0.1140 | false | equalize corrected reference and selected root priors |
| incumbent_proxy_disagreement-009 | uniform_legal_prior | 384 | 3 | false | 0.0156 | 0.8385 | 0.1088 | false | uniform legal prior positive control |
| incumbent_proxy_disagreement-009 | uniform_legal_prior | 1200 | 3 | false | 0.0458 | 0.8542 | 0.0446 | false | uniform legal prior positive control |
| incumbent_proxy_disagreement-009 | teacher_child_value_override | 384 | 3 | false | 0.0026 | 0.4505 | 0.9045 | false | override first child expansion values with teacher child values |
| incumbent_proxy_disagreement-009 | teacher_child_value_override | 1200 | 3 | false | 0.0008 | 0.7150 | 0.9231 | false | override first child expansion values with teacher child values |
| incumbent_proxy_disagreement-009 | neural_child_value_swap | 384 | 0 | true | 0.8464 | 0.8464 | 0.0000 | true | swap child neural values as diagnostic sanity check |
| incumbent_proxy_disagreement-009 | neural_child_value_swap | 1200 | 3 | false | 0.3508 | 0.6017 | 0.1032 | false | swap child neural values as diagnostic sanity check |
| incumbent_proxy_disagreement-011 | original | 384 | 5 | false | 0.0339 | 0.9531 | 0.1195 | false | no intervention |
| incumbent_proxy_disagreement-011 | original | 1200 | 5 | false | 0.0475 | 0.9342 | 0.0714 | false | no intervention |
| incumbent_proxy_disagreement-011 | equalize_root_priors | 384 | 5 | false | 0.0339 | 0.9531 | 0.1195 | false | equalize corrected reference and selected root priors |
| incumbent_proxy_disagreement-011 | equalize_root_priors | 1200 | 5 | false | 0.0467 | 0.9358 | 0.0676 | false | equalize corrected reference and selected root priors |
| incumbent_proxy_disagreement-011 | uniform_legal_prior | 384 | 5 | false | 0.0286 | 0.5599 | 0.0457 | false | uniform legal prior positive control |
| incumbent_proxy_disagreement-011 | uniform_legal_prior | 1200 | 5 | false | 0.0100 | 0.8467 | 0.1164 | false | uniform legal prior positive control |
| incumbent_proxy_disagreement-011 | teacher_child_value_override | 384 | 5 | false | 0.1406 | 0.8438 | 0.0663 | false | override first child expansion values with teacher child values |
| incumbent_proxy_disagreement-011 | teacher_child_value_override | 1200 | 5 | false | 0.0458 | 0.9325 | 0.0721 | false | override first child expansion values with teacher child values |
| incumbent_proxy_disagreement-011 | neural_child_value_swap | 384 | 5 | false | 0.0339 | 0.9531 | 0.1213 | false | swap child neural values as diagnostic sanity check |
| incumbent_proxy_disagreement-011 | neural_child_value_swap | 1200 | 5 | false | 0.0475 | 0.9342 | 0.0718 | false | swap child neural values as diagnostic sanity check |
| incumbent_proxy_disagreement-014 | original | 384 | 5 | false | 0.0026 | 0.9870 | 0.1030 | false | no intervention |
| incumbent_proxy_disagreement-014 | original | 1200 | 5 | false | 0.3392 | 0.5592 | -0.0033 | false | no intervention |
| incumbent_proxy_disagreement-014 | equalize_root_priors | 384 | 4 | true | 0.6562 | 0.6562 | 0.0000 | true | equalize corrected reference and selected root priors |
| incumbent_proxy_disagreement-014 | equalize_root_priors | 1200 | 5 | false | 0.3567 | 0.6400 | 0.0017 | false | equalize corrected reference and selected root priors |
| incumbent_proxy_disagreement-014 | uniform_legal_prior | 384 | 2 | false | 0.0443 | 0.6589 | 0.0329 | false | uniform legal prior positive control |
| incumbent_proxy_disagreement-014 | uniform_legal_prior | 1200 | 2 | false | 0.0158 | 0.8883 | 0.0443 | false | uniform legal prior positive control |
| incumbent_proxy_disagreement-014 | teacher_child_value_override | 384 | 5 | false | 0.0026 | 0.8958 | 0.7433 | false | override first child expansion values with teacher child values |
| incumbent_proxy_disagreement-014 | teacher_child_value_override | 1200 | 2 | false | 0.0025 | 0.6933 | 0.2631 | false | override first child expansion values with teacher child values |
| incumbent_proxy_disagreement-014 | neural_child_value_swap | 384 | 5 | false | 0.0026 | 0.9870 | 0.1286 | false | swap child neural values as diagnostic sanity check |
| incumbent_proxy_disagreement-014 | neural_child_value_swap | 1200 | 5 | false | 0.3392 | 0.5600 | -0.0032 | false | swap child neural values as diagnostic sanity check |
| incumbent_proxy_disagreement-021 | original | 384 | 5 | false | 0.0651 | 0.9167 | 0.0733 | false | no intervention |
| incumbent_proxy_disagreement-021 | original | 1200 | 5 | false | 0.0208 | 0.9692 | 0.1272 | false | no intervention |
| incumbent_proxy_disagreement-021 | equalize_root_priors | 384 | 5 | false | 0.0964 | 0.8880 | 0.0739 | false | equalize corrected reference and selected root priors |
| incumbent_proxy_disagreement-021 | equalize_root_priors | 1200 | 5 | false | 0.0308 | 0.9600 | 0.1312 | false | equalize corrected reference and selected root priors |
| incumbent_proxy_disagreement-021 | uniform_legal_prior | 384 | 5 | false | 0.0651 | 0.8984 | 0.0699 | false | uniform legal prior positive control |
| incumbent_proxy_disagreement-021 | uniform_legal_prior | 1200 | 5 | false | 0.0208 | 0.9550 | 0.1252 | false | uniform legal prior positive control |
| incumbent_proxy_disagreement-021 | teacher_child_value_override | 384 | 5 | false | 0.0104 | 0.9583 | 0.2122 | false | override first child expansion values with teacher child values |
| incumbent_proxy_disagreement-021 | teacher_child_value_override | 1200 | 5 | false | 0.0133 | 0.9767 | 0.1320 | false | override first child expansion values with teacher child values |
| incumbent_proxy_disagreement-021 | neural_child_value_swap | 384 | 5 | false | 0.0625 | 0.9193 | 0.0723 | false | swap child neural values as diagnostic sanity check |
| incumbent_proxy_disagreement-021 | neural_child_value_swap | 1200 | 5 | false | 0.0200 | 0.9700 | 0.1252 | false | swap child neural values as diagnostic sanity check |
| incumbent_proxy_disagreement-022 | original | 384 | 5 | false | 0.0052 | 0.6328 | 0.1457 | false | no intervention |
| incumbent_proxy_disagreement-022 | original | 1200 | 5 | false | 0.0033 | 0.6000 | 0.0540 | false | no intervention |
| incumbent_proxy_disagreement-022 | equalize_root_priors | 384 | 3 | true | 0.6120 | 0.6120 | 0.0000 | true | equalize corrected reference and selected root priors |
| incumbent_proxy_disagreement-022 | equalize_root_priors | 1200 | 3 | true | 0.7575 | 0.7575 | 0.0000 | true | equalize corrected reference and selected root priors |
| incumbent_proxy_disagreement-022 | uniform_legal_prior | 384 | 1 | false | 0.0104 | 0.4870 | 0.0627 | false | uniform legal prior positive control |
| incumbent_proxy_disagreement-022 | uniform_legal_prior | 1200 | 3 | true | 0.3758 | 0.3758 | 0.0000 | true | uniform legal prior positive control |
| incumbent_proxy_disagreement-022 | teacher_child_value_override | 384 | 5 | false | 0.0052 | 0.5469 | 0.3078 | false | override first child expansion values with teacher child values |
| incumbent_proxy_disagreement-022 | teacher_child_value_override | 1200 | 5 | false | 0.0033 | 0.3692 | 0.1256 | false | override first child expansion values with teacher child values |
| incumbent_proxy_disagreement-022 | neural_child_value_swap | 384 | 5 | false | 0.0052 | 0.5833 | 0.1112 | false | swap child neural values as diagnostic sanity check |
| incumbent_proxy_disagreement-022 | neural_child_value_swap | 1200 | 5 | false | 0.0033 | 0.5983 | 0.0346 | false | swap child neural values as diagnostic sanity check |
| incumbent_proxy_disagreement-023 | original | 384 | 1 | false | 0.0130 | 0.9557 | 0.0918 | false | no intervention |
| incumbent_proxy_disagreement-023 | original | 1200 | 1 | false | 0.0050 | 0.9792 | 0.0965 | false | no intervention |
| incumbent_proxy_disagreement-023 | equalize_root_priors | 384 | 1 | false | 0.2917 | 0.6875 | -0.0097 | false | equalize corrected reference and selected root priors |
| incumbent_proxy_disagreement-023 | equalize_root_priors | 1200 | 1 | false | 0.2308 | 0.7575 | 0.0260 | false | equalize corrected reference and selected root priors |
| incumbent_proxy_disagreement-023 | uniform_legal_prior | 384 | 1 | false | 0.0156 | 0.9401 | 0.1325 | false | uniform legal prior positive control |
| incumbent_proxy_disagreement-023 | uniform_legal_prior | 1200 | 1 | false | 0.2292 | 0.5908 | 0.0187 | false | uniform legal prior positive control |
| incumbent_proxy_disagreement-023 | teacher_child_value_override | 384 | 1 | false | 0.0052 | 0.5443 | 0.4281 | false | override first child expansion values with teacher child values |
| incumbent_proxy_disagreement-023 | teacher_child_value_override | 1200 | 1 | false | 0.0033 | 0.8217 | 0.2344 | false | override first child expansion values with teacher child values |
| incumbent_proxy_disagreement-023 | neural_child_value_swap | 384 | 1 | false | 0.0104 | 0.9583 | 0.1041 | false | swap child neural values as diagnostic sanity check |
| incumbent_proxy_disagreement-023 | neural_child_value_swap | 1200 | 1 | false | 0.0050 | 0.9792 | 0.1107 | false | swap child neural values as diagnostic sanity check |
| incumbent_proxy_disagreement-024 | original | 384 | 1 | false | 0.0104 | 0.9479 | 0.1256 | false | no intervention |
| incumbent_proxy_disagreement-024 | original | 1200 | 1 | false | 0.0067 | 0.9725 | 0.1009 | false | no intervention |
| incumbent_proxy_disagreement-024 | equalize_root_priors | 384 | 2 | true | 0.9427 | 0.9427 | 0.0000 | true | equalize corrected reference and selected root priors |
| incumbent_proxy_disagreement-024 | equalize_root_priors | 1200 | 2 | true | 0.7692 | 0.7692 | 0.0000 | true | equalize corrected reference and selected root priors |
| incumbent_proxy_disagreement-024 | uniform_legal_prior | 384 | 2 | true | 0.9349 | 0.9349 | 0.0000 | true | uniform legal prior positive control |
| incumbent_proxy_disagreement-024 | uniform_legal_prior | 1200 | 2 | true | 0.8342 | 0.8342 | 0.0000 | true | uniform legal prior positive control |
| incumbent_proxy_disagreement-024 | teacher_child_value_override | 384 | 1 | false | 0.0052 | 0.6927 | 0.3216 | false | override first child expansion values with teacher child values |
| incumbent_proxy_disagreement-024 | teacher_child_value_override | 1200 | 1 | false | 0.0025 | 0.9000 | 0.2368 | false | override first child expansion values with teacher child values |
| incumbent_proxy_disagreement-024 | neural_child_value_swap | 384 | 1 | false | 0.0104 | 0.9479 | 0.1361 | false | swap child neural values as diagnostic sanity check |
| incumbent_proxy_disagreement-024 | neural_child_value_swap | 1200 | 1 | false | 0.0067 | 0.9725 | 0.1061 | false | swap child neural values as diagnostic sanity check |
| incumbent_proxy_disagreement-025 | original | 384 | 4 | false | 0.0000 | 0.8750 | 0.1011 | false | no intervention |
| incumbent_proxy_disagreement-025 | original | 1200 | 4 | false | 0.2650 | 0.3717 | -0.0592 | false | no intervention |
| incumbent_proxy_disagreement-025 | equalize_root_priors | 384 | 2 | true | 0.6823 | 0.6823 | 0.0000 | true | equalize corrected reference and selected root priors |
| incumbent_proxy_disagreement-025 | equalize_root_priors | 1200 | 2 | true | 0.6417 | 0.6417 | 0.0000 | true | equalize corrected reference and selected root priors |
| incumbent_proxy_disagreement-025 | uniform_legal_prior | 384 | 2 | true | 0.5729 | 0.5729 | 0.0000 | true | uniform legal prior positive control |
| incumbent_proxy_disagreement-025 | uniform_legal_prior | 1200 | 2 | true | 0.4183 | 0.4183 | 0.0000 | true | uniform legal prior positive control |
| incumbent_proxy_disagreement-025 | teacher_child_value_override | 384 | 4 | false | 0.0000 | 0.8646 | 0.1002 | false | override first child expansion values with teacher child values |
| incumbent_proxy_disagreement-025 | teacher_child_value_override | 1200 | 4 | false | 0.0008 | 0.4717 | 0.6647 | false | override first child expansion values with teacher child values |
| incumbent_proxy_disagreement-025 | neural_child_value_swap | 384 | 4 | false | 0.0000 | 0.8724 | 0.1010 | false | swap child neural values as diagnostic sanity check |
| incumbent_proxy_disagreement-025 | neural_child_value_swap | 1200 | 4 | false | 0.2633 | 0.3725 | -0.0592 | false | swap child neural values as diagnostic sanity check |
| incumbent_proxy_disagreement-035 | original | 384 | 4 | false | 0.0104 | 0.9479 | 0.1515 | false | no intervention |
| incumbent_proxy_disagreement-035 | original | 1200 | 4 | false | 0.0058 | 0.6350 | 0.1401 | false | no intervention |
| incumbent_proxy_disagreement-035 | equalize_root_priors | 384 | 4 | false | 0.0156 | 0.9453 | 0.1342 | false | equalize corrected reference and selected root priors |
| incumbent_proxy_disagreement-035 | equalize_root_priors | 1200 | 4 | false | 0.0075 | 0.6042 | 0.1408 | false | equalize corrected reference and selected root priors |
| incumbent_proxy_disagreement-035 | uniform_legal_prior | 384 | 4 | false | 0.0182 | 0.9453 | 0.1633 | false | uniform legal prior positive control |
| incumbent_proxy_disagreement-035 | uniform_legal_prior | 1200 | 4 | false | 0.0217 | 0.9567 | 0.0220 | false | uniform legal prior positive control |
| incumbent_proxy_disagreement-035 | teacher_child_value_override | 384 | 1 | false | 0.0078 | 0.9557 | 0.3325 | false | override first child expansion values with teacher child values |
| incumbent_proxy_disagreement-035 | teacher_child_value_override | 1200 | 1 | false | 0.0050 | 0.9742 | 0.2159 | false | override first child expansion values with teacher child values |
| incumbent_proxy_disagreement-035 | neural_child_value_swap | 384 | 4 | false | 0.0182 | 0.9245 | 0.1621 | false | swap child neural values as diagnostic sanity check |
| incumbent_proxy_disagreement-035 | neural_child_value_swap | 1200 | 4 | false | 0.0058 | 0.5383 | 0.1428 | false | swap child neural values as diagnostic sanity check |

## 9. Row classifications

| row_id | role | row_classification | supporting_evidence | next_use | notes |
| --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-019 | holdout_candidate | inconclusive | holdout row currently passes at root; retained for out-of-sample monitoring | holdout | baseline root pass |
| incumbent_proxy_disagreement-030 | holdout_candidate | inconclusive | holdout row currently passes at root; retained for out-of-sample monitoring | holdout | baseline root pass |
| incumbent_proxy_disagreement-008 | preservation_control | inconclusive | passing preservation control under baseline | preserve_control | control row retained to guard against regressions |
| incumbent_proxy_disagreement-026 | preservation_control | inconclusive | passing preservation control under baseline | preserve_control | control row retained to guard against regressions |
| incumbent_proxy_disagreement-028 | preservation_control | inconclusive | passing preservation control under baseline | preserve_control | control row retained to guard against regressions |
| incumbent_proxy_disagreement-029 | preservation_control | inconclusive | passing preservation control under baseline | preserve_control | control row retained to guard against regressions |
| incumbent_proxy_disagreement-007 | target_candidate | corrected_reference_suspicious | deepest child teacher does not prefer corrected reference | target_candidate | persistent root disagreement |
| incumbent_proxy_disagreement-009 | target_candidate | corrected_reference_suspicious | deepest child teacher does not prefer corrected reference | target_candidate | persistent root disagreement |
| incumbent_proxy_disagreement-011 | target_candidate | puct_child_search_value_mismatch | teacher and raw neural child values prefer corrected reference while child PUCT does not | target_candidate | persistent root disagreement |
| incumbent_proxy_disagreement-014 | target_candidate | root_prior_or_selection_pressure | child teacher, neural, and child PUCT all support corrected reference but root still selects away | target_candidate | persistent root disagreement |
| incumbent_proxy_disagreement-021 | target_candidate | corrected_reference_suspicious | deepest child teacher does not prefer corrected reference | target_candidate | persistent root disagreement |
| incumbent_proxy_disagreement-022 | target_candidate | value_head_miscalibration | teacher child values prefer corrected reference while raw neural child values prefer selected move | target_candidate | persistent root disagreement |
| incumbent_proxy_disagreement-023 | target_candidate | corrected_reference_suspicious | deepest child teacher does not prefer corrected reference | target_candidate | persistent root disagreement |
| incumbent_proxy_disagreement-024 | target_candidate | corrected_reference_suspicious | deepest child teacher does not prefer corrected reference | target_candidate | persistent root disagreement |
| incumbent_proxy_disagreement-025 | target_candidate | value_head_miscalibration | teacher child values prefer corrected reference while raw neural child values prefer selected move | target_candidate | persistent root disagreement |
| incumbent_proxy_disagreement-035 | target_candidate | value_head_miscalibration | teacher child values prefer corrected reference while raw neural child values prefer selected move | target_candidate | persistent root disagreement |

## 10. Family-level interpretation

- Family classification: `reference_family_uncertain`.
- Mechanism counts: ``{"corrected_reference_suspicious": 5, "puct_child_search_value_mismatch": 1, "root_prior_or_selection_pressure": 1, "value_head_miscalibration": 3}``.
- Interpretation: corrected references need adjudication

## 11. Exactly one recommended next action

Recommendation: **adjudicate incumbent_proxy_disagreement references with deeper multi-seed teacher search before training.**
