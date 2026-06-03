# AlphaZero-lite Starvation Pressure Value/Backup Audit Results

## 1. Context

- This audit evaluates child-afterstate values and backup behavior for the `starvation_pressure` corrected non-opening failure family.
- No training was run.
- No arena was run.
- No model was promoted.
- No replay artifacts were created.
- Corrected references were not mutated.
- Corrected references: `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- Current artifact: `storage/ai/alphazero_lite/current`.
- Selected family rows: `/tmp/azlite_corrected_non_opening_failure_mining_v5/selected_non_opening_family_rows_v5.jsonl`.

## 2. Why starvation_pressure was selected

- PR #60 selected `starvation_pressure` as the next corrected non-opening failure family.
- Family stats: `{"avg_reference_visit_share_1200": 0.291, "avg_reference_visit_share_384": 0.2281, "avg_selected_minus_reference_q_margin_1200": 0.0998, "avg_selected_minus_reference_q_margin_384": 0.0843, "classification": "value_or_backup_issue", "conflicting_target_count": 0, "dominant_failure_mode": "value_q", "duplicate_canonical_state_count": 0, "fail_rows": 11, "failure_rate": 0.4583, "family": "starvation_pressure", "high_severity_count": 0, "medium_severity_count": 11, "notes": "persistent failures remain Q/value-dominant", "pass_rows": 13, "persistent_1200_failures": 9, "rank_score": 229.252, "recovered_at_1200": 1, "stable_corrected_reference_count": 24, "total_rows": 24}`.

## 3. Row validation

| row_id | role | corrected_reference_move | legal | reference_unstable | status | notes |
| --- | --- | --- | --- | --- | --- | --- |
| starvation_pressure-025 | target_candidate | 0 | true | false | ok | validated, representative target row present |
| starvation_pressure-026 | target_candidate | 0 | true | false | ok | validated, representative target row present |
| starvation_pressure-024 | target_candidate | 1 | true | false | ok | validated, representative target row present |
| starvation_pressure-022 | target_candidate | 2 | true | false | ok | validated, representative target row present |
| starvation_pressure-012 | target_candidate | 3 | true | false | ok | validated, representative target row present |
| starvation_pressure-023 | target_candidate | 2 | true | false | ok | validated, representative target row present |
| starvation_pressure-027 | target_candidate | 0 | true | false | ok | validated, representative target row present |
| starvation_pressure-015 | holdout_candidate | 4 | true | false | ok | validated, representative target row present |
| starvation_pressure-001 | holdout_candidate | 1 | true | false | ok | validated, representative target row present |
| starvation_pressure-013 | preservation_control | 4 | true | false | ok | validated, representative control row present |
| starvation_pressure-003 | preservation_control | 1 | true | false | ok | validated, representative control row present |
| starvation_pressure-021 | preservation_control | 0 | true | false | ok | validated, representative control row present |
| starvation_pressure-014 | preservation_control | 2 | true | false | ok | validated, representative control row present |

- Perspective conversion: `+1 when child current_player == root current_player, else -1`.
- Implementation rule: `PUCT _search negates the returned child value exactly when child.game.current_player != parent.game.current_player`.

## 4. Root PUCT baseline

| row_id | role | budget | corrected_reference_move | selected_move | selected_is_reference | reference_visit_share | selected_visit_share | reference_q | selected_q | selected_minus_reference_q_margin | reference_policy_probability | selected_policy_probability | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| starvation_pressure-001 | holdout_candidate | 384 | 1 | 4 | false | 0.0156 | 0.8568 | -0.2812 | -0.1537 | 0.1275 | 0.2612 | 0.1831 | holdout_candidate, selected away from reference |
| starvation_pressure-001 | holdout_candidate | 1200 | 1 | 4 | false | 0.3558 | 0.4308 | -0.1864 | -0.1835 | 0.0030 | 0.2612 | 0.1831 | holdout_candidate, selected away from reference |
| starvation_pressure-001 | holdout_candidate | 2400 | 1 | 4 | false | 0.1787 | 0.7096 | -0.1889 | -0.1823 | 0.0066 | 0.2612 | 0.1831 | holdout_candidate, selected away from reference |
| starvation_pressure-003 | preservation_control | 384 | 1 | 1 | true | 0.8958 | 0.8958 | -0.1540 | -0.1540 | 0.0000 | 0.3693 | 0.3693 | preservation_control, selected reference |
| starvation_pressure-003 | preservation_control | 1200 | 1 | 1 | true | 0.9642 | 0.9642 | -0.1206 | -0.1206 | 0.0000 | 0.3693 | 0.3693 | preservation_control, selected reference |
| starvation_pressure-003 | preservation_control | 2400 | 1 | 1 | true | 0.9800 | 0.9800 | -0.1106 | -0.1106 | 0.0000 | 0.3693 | 0.3693 | preservation_control, selected reference |
| starvation_pressure-012 | target_candidate | 384 | 3 | 5 | false | 0.0130 | 0.9870 | 0.1316 | 0.2068 | 0.0752 | 0.2382 | 0.7618 | target_candidate, selected away from reference |
| starvation_pressure-012 | target_candidate | 1200 | 3 | 5 | false | 0.0083 | 0.9917 | 0.1269 | 0.2255 | 0.0986 | 0.2382 | 0.7618 | target_candidate, selected away from reference |
| starvation_pressure-012 | target_candidate | 2400 | 3 | 5 | false | 0.0058 | 0.9942 | 0.1666 | 0.2647 | 0.0981 | 0.2382 | 0.7618 | target_candidate, selected away from reference |
| starvation_pressure-013 | preservation_control | 384 | 4 | 4 | true | 0.9974 | 0.9974 | 0.2151 | 0.2151 | 0.0000 | 0.9827 | 0.9827 | preservation_control, selected reference |
| starvation_pressure-013 | preservation_control | 1200 | 4 | 4 | true | 0.9992 | 0.9992 | 0.2036 | 0.2036 | 0.0000 | 0.9827 | 0.9827 | preservation_control, selected reference |
| starvation_pressure-013 | preservation_control | 2400 | 4 | 4 | true | 0.9996 | 0.9996 | 0.2093 | 0.2093 | 0.0000 | 0.9827 | 0.9827 | preservation_control, selected reference |
| starvation_pressure-014 | preservation_control | 384 | 2 | 2 | true | 0.9844 | 0.9844 | 0.2580 | 0.2580 | 0.0000 | 0.7140 | 0.7140 | preservation_control, selected reference |
| starvation_pressure-014 | preservation_control | 1200 | 2 | 2 | true | 0.9467 | 0.9467 | 0.2135 | 0.2135 | 0.0000 | 0.7140 | 0.7140 | preservation_control, selected reference |
| starvation_pressure-014 | preservation_control | 2400 | 2 | 2 | true | 0.9692 | 0.9692 | 0.1819 | 0.1819 | 0.0000 | 0.7140 | 0.7140 | preservation_control, selected reference |
| starvation_pressure-015 | holdout_candidate | 384 | 4 | 2 | false | 0.1979 | 0.8021 | 0.2685 | 0.2058 | -0.0627 | 0.1852 | 0.8148 | holdout_candidate, selected away from reference |
| starvation_pressure-015 | holdout_candidate | 1200 | 4 | 2 | false | 0.4800 | 0.5200 | 0.1658 | 0.1854 | 0.0196 | 0.1852 | 0.8148 | holdout_candidate, selected away from reference |
| starvation_pressure-015 | holdout_candidate | 2400 | 4 | 2 | false | 0.2400 | 0.7600 | 0.1658 | 0.1825 | 0.0167 | 0.1852 | 0.8148 | holdout_candidate, selected away from reference |
| starvation_pressure-021 | preservation_control | 384 | 0 | 0 | true | 0.8984 | 0.8984 | 0.8200 | 0.8200 | 0.0000 | 0.0479 | 0.0479 | preservation_control, selected reference |
| starvation_pressure-021 | preservation_control | 1200 | 0 | 0 | true | 0.9608 | 0.9608 | 0.9069 | 0.9069 | 0.0000 | 0.0479 | 0.0479 | preservation_control, selected reference |
| starvation_pressure-021 | preservation_control | 2400 | 0 | 0 | true | 0.9758 | 0.9758 | 0.9312 | 0.9312 | 0.0000 | 0.0479 | 0.0479 | preservation_control, selected reference |
| starvation_pressure-022 | target_candidate | 384 | 2 | 1 | false | 0.0182 | 0.9427 | -0.3117 | -0.2004 | 0.1114 | 0.1113 | 0.5368 | target_candidate, selected away from reference |
| starvation_pressure-022 | target_candidate | 1200 | 2 | 1 | false | 0.0067 | 0.9750 | -0.3352 | -0.2234 | 0.1118 | 0.1113 | 0.5368 | target_candidate, selected away from reference |
| starvation_pressure-022 | target_candidate | 2400 | 2 | 1 | false | 0.0033 | 0.9692 | -0.3352 | -0.2645 | 0.0707 | 0.1113 | 0.5368 | target_candidate, selected away from reference |
| starvation_pressure-023 | target_candidate | 384 | 2 | 1 | false | 0.0417 | 0.7630 | -0.2738 | -0.1999 | 0.0739 | 0.1626 | 0.1300 | target_candidate, selected away from reference |
| starvation_pressure-023 | target_candidate | 1200 | 2 | 1 | false | 0.0225 | 0.9092 | -0.2617 | -0.1964 | 0.0653 | 0.1626 | 0.1300 | target_candidate, selected away from reference |
| starvation_pressure-023 | target_candidate | 2400 | 2 | 1 | false | 0.0183 | 0.6350 | -0.2655 | -0.2245 | 0.0411 | 0.1626 | 0.1300 | target_candidate, selected away from reference |
| starvation_pressure-024 | target_candidate | 384 | 1 | 3 | false | 0.0234 | 0.5807 | -0.2361 | -0.1004 | 0.1356 | 0.1281 | 0.1387 | target_candidate, selected away from reference |
| starvation_pressure-024 | target_candidate | 1200 | 1 | 3 | false | 0.0075 | 0.6167 | -0.2361 | -0.0707 | 0.1653 | 0.1281 | 0.1387 | target_candidate, selected away from reference |
| starvation_pressure-024 | target_candidate | 2400 | 1 | 3 | false | 0.0042 | 0.8008 | -0.2505 | -0.0465 | 0.2040 | 0.1281 | 0.1387 | target_candidate, selected away from reference |
| starvation_pressure-025 | target_candidate | 384 | 0 | 4 | false | 0.0026 | 0.8125 | -0.2387 | -0.0183 | 0.2204 | 0.0183 | 0.0912 | target_candidate, selected away from reference |
| starvation_pressure-025 | target_candidate | 1200 | 0 | 4 | false | 0.0008 | 0.8292 | -0.2387 | -0.0324 | 0.2063 | 0.0183 | 0.0912 | target_candidate, selected away from reference |
| starvation_pressure-025 | target_candidate | 2400 | 0 | 4 | false | 0.0004 | 0.9025 | -0.2387 | -0.0355 | 0.2032 | 0.0183 | 0.0912 | target_candidate, selected away from reference |
| starvation_pressure-026 | target_candidate | 384 | 0 | 4 | false | 0.0156 | 0.8698 | -0.2176 | -0.0028 | 0.2148 | 0.0380 | 0.0867 | target_candidate, selected away from reference |
| starvation_pressure-026 | target_candidate | 1200 | 0 | 4 | false | 0.0050 | 0.9500 | -0.2176 | -0.0308 | 0.1868 | 0.0380 | 0.0867 | target_candidate, selected away from reference |
| starvation_pressure-026 | target_candidate | 2400 | 0 | 4 | false | 0.0025 | 0.9404 | -0.2176 | -0.0994 | 0.1182 | 0.0380 | 0.0867 | target_candidate, selected away from reference |
| starvation_pressure-027 | target_candidate | 384 | 0 | 1 | false | 0.0026 | 0.9505 | -0.1717 | -0.1224 | 0.0493 | 0.0290 | 0.4928 | target_candidate, selected away from reference |
| starvation_pressure-027 | target_candidate | 1200 | 0 | 1 | false | 0.0033 | 0.9650 | -0.2037 | -0.1620 | 0.0417 | 0.0290 | 0.4928 | target_candidate, selected away from reference |
| starvation_pressure-027 | target_candidate | 2400 | 0 | 1 | false | 0.4100 | 0.5696 | -0.0579 | -0.1816 | -0.1237 | 0.0290 | 0.4928 | target_candidate, selected away from reference |

- ran 2400 root budget: projected ~1.8s across 7 target rows

## 5. Move consequence audit

| row_id | move | is_corrected_reference | is_selected | gives_extra_turn | produces_capture | capture_count | immediate_store_delta | side_to_move_after | game_over_after_move | starvation_pressure_feature_if_available | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| starvation_pressure-025 | 0 | true | false | false | false | 0 | 0 | 0 | false | unchanged; self_total=34; opp_total=9; empty_self=1; empty_opp=4; pressure=true |  |
| starvation_pressure-025 | 1 | false | false | false | false | 0 | 1 | 0 | false | reduces; self_total=29; opp_total=13; empty_self=1; empty_opp=1; pressure=false |  |
| starvation_pressure-025 | 2 | false | false | false | false | 0 | 0 | 0 | false | unchanged; self_total=34; opp_total=9; empty_self=1; empty_opp=4; pressure=true |  |
| starvation_pressure-025 | 3 | false | false | false | false | 0 | 1 | 0 | false | reduces; self_total=28; opp_total=14; empty_self=1; empty_opp=0; pressure=false |  |
| starvation_pressure-025 | 4 | false | true | false | false | 0 | 1 | 0 | false | reduces; self_total=28; opp_total=14; empty_self=1; empty_opp=0; pressure=false | selected move has larger immediate store gain; selected move reduces starvation pressure |
| starvation_pressure-025 | 5 | false | false | false | false | 0 | 1 | 0 | false | reduces; self_total=28; opp_total=14; empty_self=1; empty_opp=0; pressure=false |  |
| starvation_pressure-026 | 0 | true | false | false | false | 0 | 0 | 0 | false | unchanged; self_total=34; opp_total=9; empty_self=1; empty_opp=4; pressure=true |  |
| starvation_pressure-026 | 1 | false | false | false | false | 0 | 0 | 0 | false | unchanged; self_total=34; opp_total=9; empty_self=1; empty_opp=4; pressure=true |  |
| starvation_pressure-026 | 2 | false | false | false | false | 0 | 1 | 0 | false | reduces; self_total=28; opp_total=14; empty_self=1; empty_opp=0; pressure=false |  |
| starvation_pressure-026 | 3 | false | false | false | false | 0 | 1 | 0 | false | reduces; self_total=28; opp_total=14; empty_self=1; empty_opp=0; pressure=false |  |
| starvation_pressure-026 | 4 | false | true | false | false | 0 | 1 | 0 | false | reduces; self_total=28; opp_total=14; empty_self=1; empty_opp=0; pressure=false | selected move has larger immediate store gain; selected move reduces starvation pressure |
| starvation_pressure-026 | 5 | false | false | false | false | 0 | 1 | 0 | false | reduces; self_total=28; opp_total=14; empty_self=1; empty_opp=0; pressure=false |  |
| starvation_pressure-024 | 0 | false | false | false | false | 0 | 0 | 0 | false | unchanged; self_total=34; opp_total=9; empty_self=1; empty_opp=4; pressure=true |  |
| starvation_pressure-024 | 1 | true | false | false | false | 0 | 0 | 0 | false | unchanged; self_total=34; opp_total=9; empty_self=1; empty_opp=4; pressure=true |  |
| starvation_pressure-024 | 2 | false | false | false | false | 0 | 1 | 0 | false | reduces; self_total=28; opp_total=14; empty_self=1; empty_opp=0; pressure=false |  |
| starvation_pressure-024 | 3 | false | true | false | false | 0 | 1 | 0 | false | reduces; self_total=28; opp_total=14; empty_self=1; empty_opp=0; pressure=false | selected move has larger immediate store gain; selected move reduces starvation pressure |
| starvation_pressure-024 | 4 | false | false | false | false | 0 | 1 | 0 | false | reduces; self_total=28; opp_total=14; empty_self=1; empty_opp=0; pressure=false |  |
| starvation_pressure-024 | 5 | false | false | false | false | 0 | 1 | 0 | false | reduces; self_total=28; opp_total=14; empty_self=1; empty_opp=0; pressure=false |  |
| starvation_pressure-022 | 0 | false | false | false | false | 0 | 0 | 0 | false | unchanged; self_total=37; opp_total=7; empty_self=1; empty_opp=5; pressure=true |  |
| starvation_pressure-022 | 1 | false | true | false | false | 0 | 1 | 0 | false | reduces; self_total=31; opp_total=12; empty_self=1; empty_opp=1; pressure=false |  |
| starvation_pressure-022 | 2 | true | false | false | false | 0 | 1 | 0 | false | reduces; self_total=33; opp_total=10; empty_self=1; empty_opp=2; pressure=false |  |
| starvation_pressure-022 | 3 | false | false | false | false | 0 | 1 | 0 | false | reduces; self_total=32; opp_total=11; empty_self=1; empty_opp=2; pressure=false |  |
| starvation_pressure-022 | 4 | false | false | false | false | 0 | 1 | 0 | false | reduces; self_total=31; opp_total=12; empty_self=1; empty_opp=1; pressure=false |  |
| starvation_pressure-022 | 5 | false | false | false | false | 0 | 1 | 0 | false | reduces; self_total=32; opp_total=11; empty_self=1; empty_opp=2; pressure=false |  |
| starvation_pressure-012 | 3 | true | false | false | false | 0 | 1 | 1 | false | unchanged; self_total=9; opp_total=35; empty_self=4; empty_opp=0; pressure=true |  |
| starvation_pressure-012 | 5 | false | true | false | false | 0 | 1 | 1 | false | increases; self_total=7; opp_total=37; empty_self=5; empty_opp=0; pressure=true | selected move increases starvation pressure |
| starvation_pressure-023 | 0 | false | false | false | false | 0 | 0 | 0 | false | unchanged; self_total=36; opp_total=8; empty_self=1; empty_opp=4; pressure=true |  |
| starvation_pressure-023 | 1 | false | true | false | false | 0 | 1 | 0 | false | reduces; self_total=30; opp_total=13; empty_self=1; empty_opp=0; pressure=false |  |
| starvation_pressure-023 | 2 | true | false | false | false | 0 | 1 | 0 | false | reduces; self_total=32; opp_total=11; empty_self=1; empty_opp=1; pressure=false |  |
| starvation_pressure-023 | 3 | false | false | false | false | 0 | 1 | 0 | false | reduces; self_total=31; opp_total=12; empty_self=1; empty_opp=1; pressure=false |  |
| starvation_pressure-023 | 4 | false | false | false | false | 0 | 1 | 0 | false | reduces; self_total=30; opp_total=13; empty_self=1; empty_opp=0; pressure=false |  |
| starvation_pressure-023 | 5 | false | false | false | false | 0 | 1 | 0 | false | reduces; self_total=32; opp_total=11; empty_self=1; empty_opp=1; pressure=false |  |
| starvation_pressure-027 | 0 | true | false | false | false | 0 | 0 | 0 | false | unchanged; self_total=34; opp_total=9; empty_self=1; empty_opp=4; pressure=true |  |
| starvation_pressure-027 | 1 | false | true | false | false | 0 | 1 | 0 | false | reduces; self_total=30; opp_total=12; empty_self=1; empty_opp=2; pressure=false | selected move has larger immediate store gain; selected move reduces starvation pressure |
| starvation_pressure-027 | 2 | false | false | false | false | 0 | 0 | 0 | false | unchanged; self_total=34; opp_total=9; empty_self=1; empty_opp=4; pressure=true |  |
| starvation_pressure-027 | 3 | false | false | false | false | 0 | 1 | 0 | false | reduces; self_total=27; opp_total=15; empty_self=1; empty_opp=0; pressure=false |  |
| starvation_pressure-027 | 4 | false | false | false | false | 0 | 1 | 0 | false | reduces; self_total=28; opp_total=14; empty_self=1; empty_opp=0; pressure=false |  |
| starvation_pressure-027 | 5 | false | false | false | false | 0 | 1 | 0 | false | reduces; self_total=28; opp_total=14; empty_self=1; empty_opp=0; pressure=false |  |
| starvation_pressure-015 | 2 | false | true | false | false | 0 | 1 | 1 | false | reduces; self_total=10; opp_total=34; empty_self=3; empty_opp=0; pressure=false | selected move reduces starvation pressure |
| starvation_pressure-015 | 4 | true | false | false | false | 0 | 1 | 1 | false | unchanged; self_total=7; opp_total=37; empty_self=4; empty_opp=0; pressure=true |  |
| starvation_pressure-001 | 0 | false | false | false | false | 0 | 0 | 0 | false | unchanged; self_total=30; opp_total=15; empty_self=1; empty_opp=4; pressure=true |  |
| starvation_pressure-001 | 1 | true | false | false | false | 0 | 1 | 0 | false | reduces; self_total=26; opp_total=18; empty_self=1; empty_opp=1; pressure=false |  |
| starvation_pressure-001 | 2 | false | false | false | false | 0 | 1 | 0 | false | reduces; self_total=27; opp_total=17; empty_self=1; empty_opp=2; pressure=false |  |
| starvation_pressure-001 | 3 | false | false | false | false | 0 | 1 | 0 | false | reduces; self_total=26; opp_total=18; empty_self=1; empty_opp=1; pressure=false |  |
| starvation_pressure-001 | 4 | false | true | false | false | 0 | 1 | 0 | false | reduces; self_total=26; opp_total=18; empty_self=1; empty_opp=1; pressure=false |  |
| starvation_pressure-001 | 5 | false | false | false | false | 0 | 1 | 0 | false | reduces; self_total=26; opp_total=18; empty_self=1; empty_opp=1; pressure=false |  |
| starvation_pressure-013 | 3 | false | false | false | false | 0 | 1 | 1 | false | unchanged; self_total=9; opp_total=35; empty_self=4; empty_opp=0; pressure=true |  |
| starvation_pressure-013 | 4 | true | true | false | false | 0 | 1 | 1 | false | unchanged; self_total=8; opp_total=36; empty_self=4; empty_opp=0; pressure=true |  |
| starvation_pressure-003 | 0 | false | false | false | false | 0 | 0 | 0 | false | unchanged; self_total=31; opp_total=14; empty_self=1; empty_opp=4; pressure=true |  |
| starvation_pressure-003 | 1 | true | true | false | false | 0 | 1 | 0 | false | reduces; self_total=27; opp_total=17; empty_self=1; empty_opp=1; pressure=false |  |
| starvation_pressure-003 | 2 | false | false | false | false | 0 | 1 | 0 | false | reduces; self_total=28; opp_total=16; empty_self=1; empty_opp=2; pressure=false |  |
| starvation_pressure-003 | 3 | false | false | false | false | 0 | 1 | 0 | false | reduces; self_total=27; opp_total=17; empty_self=1; empty_opp=1; pressure=false |  |
| starvation_pressure-003 | 4 | false | false | false | false | 0 | 1 | 0 | false | reduces; self_total=26; opp_total=18; empty_self=1; empty_opp=1; pressure=false |  |
| starvation_pressure-003 | 5 | false | false | false | false | 0 | 1 | 0 | false | reduces; self_total=27; opp_total=17; empty_self=1; empty_opp=1; pressure=false |  |
| starvation_pressure-021 | 0 | true | true | false | false | 0 | 0 | 0 | false | unchanged; self_total=35; opp_total=9; empty_self=1; empty_opp=4; pressure=true |  |
| starvation_pressure-021 | 1 | false | false | false | false | 0 | 1 | 0 | false | reduces; self_total=29; opp_total=14; empty_self=1; empty_opp=0; pressure=false |  |
| starvation_pressure-021 | 2 | false | false | false | false | 0 | 1 | 0 | false | reduces; self_total=31; opp_total=12; empty_self=1; empty_opp=1; pressure=false |  |
| starvation_pressure-021 | 3 | false | false | false | false | 0 | 1 | 0 | false | reduces; self_total=30; opp_total=13; empty_self=1; empty_opp=0; pressure=false |  |
| starvation_pressure-021 | 4 | false | false | false | false | 0 | 1 | 0 | false | reduces; self_total=30; opp_total=13; empty_self=1; empty_opp=0; pressure=false |  |
| starvation_pressure-021 | 5 | false | false | false | false | 0 | 1 | 0 | false | reduces; self_total=31; opp_total=12; empty_self=1; empty_opp=1; pressure=false |  |
| starvation_pressure-014 | 2 | true | true | false | false | 0 | 1 | 1 | false | reduces; self_total=10; opp_total=34; empty_self=3; empty_opp=0; pressure=false |  |
| starvation_pressure-014 | 5 | false | false | false | false | 0 | 1 | 1 | false | increases; self_total=6; opp_total=38; empty_self=5; empty_opp=0; pressure=true |  |

## 6. Neural child-afterstate value audit

| row_id | corrected_reference_move | selected_move | child | raw_value | root_perspective_value | child_ref_minus_child_selected | neural_prefers_reference | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| starvation_pressure-025 | 0 | 4 | corrected_reference_child | 0.2387 | -0.2387 | -0.1628 | false | sign flip to root perspective |
| starvation_pressure-025 | 0 | 4 | selected_child | 0.0759 | -0.0759 | -0.1628 | false | sign flip to root perspective |
| starvation_pressure-026 | 0 | 4 | corrected_reference_child | 0.1746 | -0.1746 | -0.1772 | false | sign flip to root perspective |
| starvation_pressure-026 | 0 | 4 | selected_child | -0.0027 | 0.0027 | -0.1772 | false | sign flip to root perspective |
| starvation_pressure-024 | 1 | 3 | corrected_reference_child | 0.1343 | -0.1343 | -0.1743 | false | sign flip to root perspective |
| starvation_pressure-024 | 1 | 3 | selected_child | -0.0400 | 0.0400 | -0.1743 | false | sign flip to root perspective |
| starvation_pressure-022 | 2 | 1 | corrected_reference_child | 0.2677 | -0.2677 | 0.1329 | true | sign flip to root perspective |
| starvation_pressure-022 | 2 | 1 | selected_child | 0.4007 | -0.4007 | 0.1329 | true | sign flip to root perspective |
| starvation_pressure-012 | 3 | 5 | corrected_reference_child | -0.0575 | 0.0575 | 0.0861 | true | sign flip to root perspective |
| starvation_pressure-012 | 3 | 5 | selected_child | 0.0286 | -0.0286 | 0.0861 | true | sign flip to root perspective |
| starvation_pressure-023 | 2 | 1 | corrected_reference_child | 0.4140 | -0.4140 | -0.0951 | false | sign flip to root perspective |
| starvation_pressure-023 | 2 | 1 | selected_child | 0.3189 | -0.3189 | -0.0951 | false | sign flip to root perspective |
| starvation_pressure-027 | 0 | 1 | corrected_reference_child | 0.1717 | -0.1717 | 0.0045 | true | sign flip to root perspective |
| starvation_pressure-027 | 0 | 1 | selected_child | 0.1762 | -0.1762 | 0.0045 | true | sign flip to root perspective |

## 7. ClassicMCTS child-afterstate audit

| row_id | budget | seeds | child_ref_value_root_mean | child_selected_value_root_mean | child_ref_minus_child_selected | teacher_prefers_reference | stable | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| starvation_pressure-025 | 1200 | [11, 23, 37, 42, 101] | -0.7624 | -0.5767 | -0.1857 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| starvation_pressure-025 | 2400 | [11, 23, 37, 42, 101] | -0.8426 | -0.6828 | -0.1598 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| starvation_pressure-025 | 5000 | [11, 23, 37, 42, 101] | -0.8885 | -0.8069 | -0.0816 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| starvation_pressure-026 | 1200 | [11, 23, 37, 42, 101] | -0.8366 | -0.4408 | -0.3957 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| starvation_pressure-026 | 2400 | [11, 23, 37, 42, 101] | -0.8813 | -0.5517 | -0.3296 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| starvation_pressure-026 | 5000 | [11, 23, 37, 42, 101] | -0.8951 | -0.7828 | -0.1123 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| starvation_pressure-024 | 1200 | [11, 23, 37, 42, 101] | -0.8868 | -0.3861 | -0.5008 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| starvation_pressure-024 | 2400 | [11, 23, 37, 42, 101] | -0.9019 | -0.6422 | -0.2597 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| starvation_pressure-024 | 5000 | [11, 23, 37, 42, 101] | -0.8983 | -0.7936 | -0.1047 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| starvation_pressure-022 | 1200 | [11, 23, 37, 42, 101] | -0.6747 | -0.6628 | -0.0119 | false | false | ClassicMCTS child-afterstate teacher aggregate |
| starvation_pressure-022 | 2400 | [11, 23, 37, 42, 101] | -0.7642 | -0.6486 | -0.1156 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| starvation_pressure-022 | 5000 | [11, 23, 37, 42, 101] | -0.8440 | -0.7415 | -0.1025 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| starvation_pressure-012 | 1200 | [11, 23, 37, 42, 101] | 0.6210 | 0.3911 | 0.2299 | true | true | ClassicMCTS child-afterstate teacher aggregate |
| starvation_pressure-012 | 2400 | [11, 23, 37, 42, 101] | 0.4046 | 0.0070 | 0.3976 | true | true | ClassicMCTS child-afterstate teacher aggregate |
| starvation_pressure-012 | 5000 | [11, 23, 37, 42, 101] | -0.2169 | -0.3586 | 0.1417 | true | false | ClassicMCTS child-afterstate teacher aggregate |
| starvation_pressure-023 | 1200 | [11, 23, 37, 42, 101] | -0.8274 | -0.8135 | -0.0139 | false | false | ClassicMCTS child-afterstate teacher aggregate |
| starvation_pressure-023 | 2400 | [11, 23, 37, 42, 101] | -0.7658 | -0.7691 | 0.0033 | true | false | ClassicMCTS child-afterstate teacher aggregate |
| starvation_pressure-023 | 5000 | [11, 23, 37, 42, 101] | -0.8213 | -0.7061 | -0.1151 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| starvation_pressure-027 | 1200 | [11, 23, 37, 42, 101] | -0.8436 | -0.7912 | -0.0525 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| starvation_pressure-027 | 2400 | [11, 23, 37, 42, 101] | -0.8784 | -0.8444 | -0.0339 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| starvation_pressure-027 | 5000 | [11, 23, 37, 42, 101] | -0.9122 | -0.8903 | -0.0220 | false | true | ClassicMCTS child-afterstate teacher aggregate |

## 8. PUCT child-afterstate audit

| row_id | budget | child_ref_value_root | child_selected_value_root | child_ref_minus_child_selected | puct_prefers_reference | notes |
| --- | --- | --- | --- | --- | --- | --- |
| starvation_pressure-025 | 384 | -0.2818 | -0.0153 | -0.2665 | false | agrees with teacher; agrees with neural |
| starvation_pressure-025 | 1200 | -0.3231 | -0.0331 | -0.2900 | false | agrees with teacher; agrees with neural |
| starvation_pressure-026 | 384 | -0.1827 | -0.0312 | -0.1515 | false | agrees with teacher; agrees with neural |
| starvation_pressure-026 | 1200 | -0.1637 | -0.0298 | -0.1339 | false | agrees with teacher; agrees with neural |
| starvation_pressure-024 | 384 | -0.2591 | -0.1042 | -0.1549 | false | agrees with teacher; agrees with neural |
| starvation_pressure-024 | 1200 | -0.2419 | -0.0513 | -0.1906 | false | agrees with teacher; agrees with neural |
| starvation_pressure-022 | 384 | -0.1660 | -0.2039 | 0.0379 | true | disagrees with teacher; agrees with neural |
| starvation_pressure-022 | 1200 | -0.1793 | -0.2315 | 0.0522 | true | disagrees with teacher; agrees with neural |
| starvation_pressure-012 | 384 | 0.2054 | 0.2077 | -0.0023 | false | disagrees with teacher; disagrees with neural |
| starvation_pressure-012 | 1200 | 0.1515 | 0.2264 | -0.0749 | false | disagrees with teacher; disagrees with neural |
| starvation_pressure-023 | 384 | -0.2364 | -0.2116 | -0.0248 | false | agrees with teacher; agrees with neural |
| starvation_pressure-023 | 1200 | -0.2427 | -0.2026 | -0.0401 | false | agrees with teacher; agrees with neural |
| starvation_pressure-027 | 384 | -0.0722 | -0.1303 | 0.0581 | true | disagrees with teacher; agrees with neural |
| starvation_pressure-027 | 1200 | -0.0787 | -0.1481 | 0.0694 | true | disagrees with teacher; agrees with neural |

## 9. Root counterfactual diagnostics

| row_id | intervention | budget | selected_move | selected_is_reference | reference_visit_share | selected_visit_share | selected_minus_reference_q_margin | flipped | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| starvation_pressure-025 | original | 384 | 4 | false | 0.0026 | 0.8125 | 0.2204 | false | no intervention |
| starvation_pressure-025 | original | 1200 | 4 | false | 0.0008 | 0.8292 | 0.2063 | false | no intervention |
| starvation_pressure-025 | equalize_root_priors | 384 | 4 | false | 0.0078 | 0.8099 | 0.1449 | false | equalize corrected reference and selected root priors |
| starvation_pressure-025 | equalize_root_priors | 1200 | 4 | false | 0.0033 | 0.8267 | 0.2199 | false | equalize corrected reference and selected root priors |
| starvation_pressure-025 | uniform_legal_prior | 384 | 4 | false | 0.0104 | 0.9167 | 0.2358 | false | uniform legal prior positive control |
| starvation_pressure-025 | uniform_legal_prior | 1200 | 4 | false | 0.0058 | 0.9567 | 0.2094 | false | uniform legal prior positive control |
| starvation_pressure-025 | teacher_child_value_override | 384 | 1 | false | 0.0026 | 0.7578 | 0.7732 | false | override first child expansion values with teacher child values |
| starvation_pressure-025 | teacher_child_value_override | 1200 | 1 | false | 0.0008 | 0.8408 | 0.8014 | false | override first child expansion values with teacher child values |
| starvation_pressure-025 | neural_child_value_swap | 384 | 3 | false | 0.0156 | 0.6302 | 0.0330 | false | swap child neural values as diagnostic sanity check |
| starvation_pressure-025 | neural_child_value_swap | 1200 | 3 | false | 0.0050 | 0.7775 | 0.0645 | false | swap child neural values as diagnostic sanity check |
| starvation_pressure-026 | original | 384 | 4 | false | 0.0156 | 0.8698 | 0.2148 | false | no intervention |
| starvation_pressure-026 | original | 1200 | 4 | false | 0.0050 | 0.9500 | 0.1868 | false | no intervention |
| starvation_pressure-026 | equalize_root_priors | 384 | 4 | false | 0.0156 | 0.8620 | 0.2142 | false | equalize corrected reference and selected root priors |
| starvation_pressure-026 | equalize_root_priors | 1200 | 4 | false | 0.0050 | 0.9492 | 0.1867 | false | equalize corrected reference and selected root priors |
| starvation_pressure-026 | uniform_legal_prior | 384 | 4 | false | 0.0156 | 0.9167 | 0.2143 | false | uniform legal prior positive control |
| starvation_pressure-026 | uniform_legal_prior | 1200 | 4 | false | 0.0092 | 0.9583 | 0.1619 | false | uniform legal prior positive control |
| starvation_pressure-026 | teacher_child_value_override | 384 | 2 | false | 0.0026 | 0.6953 | 0.7567 | false | override first child expansion values with teacher child values |
| starvation_pressure-026 | teacher_child_value_override | 1200 | 4 | false | 0.0008 | 0.5875 | 0.8845 | false | override first child expansion values with teacher child values |
| starvation_pressure-026 | neural_child_value_swap | 384 | 2 | false | 0.0807 | 0.8203 | 0.0834 | false | swap child neural values as diagnostic sanity check |
| starvation_pressure-026 | neural_child_value_swap | 1200 | 2 | false | 0.0258 | 0.9400 | 0.1297 | false | swap child neural values as diagnostic sanity check |
| starvation_pressure-024 | original | 384 | 3 | false | 0.0234 | 0.5807 | 0.1356 | false | no intervention |
| starvation_pressure-024 | original | 1200 | 3 | false | 0.0075 | 0.6167 | 0.1653 | false | no intervention |
| starvation_pressure-024 | equalize_root_priors | 384 | 3 | false | 0.0234 | 0.5911 | 0.1327 | false | equalize corrected reference and selected root priors |
| starvation_pressure-024 | equalize_root_priors | 1200 | 3 | false | 0.0075 | 0.6167 | 0.1653 | false | equalize corrected reference and selected root priors |
| starvation_pressure-024 | uniform_legal_prior | 384 | 3 | false | 0.0234 | 0.8880 | 0.1332 | false | uniform legal prior positive control |
| starvation_pressure-024 | uniform_legal_prior | 1200 | 3 | false | 0.0075 | 0.4958 | 0.1453 | false | uniform legal prior positive control |
| starvation_pressure-024 | teacher_child_value_override | 384 | 2 | false | 0.0078 | 0.8646 | 0.2469 | false | override first child expansion values with teacher child values |
| starvation_pressure-024 | teacher_child_value_override | 1200 | 4 | false | 0.0058 | 0.6100 | 0.1978 | false | override first child expansion values with teacher child values |
| starvation_pressure-024 | neural_child_value_swap | 384 | 4 | false | 0.0755 | 0.6016 | 0.1494 | false | swap child neural values as diagnostic sanity check |
| starvation_pressure-024 | neural_child_value_swap | 1200 | 4 | false | 0.0242 | 0.8542 | 0.1605 | false | swap child neural values as diagnostic sanity check |
| starvation_pressure-022 | original | 384 | 1 | false | 0.0182 | 0.9427 | 0.1114 | false | no intervention |
| starvation_pressure-022 | original | 1200 | 1 | false | 0.0067 | 0.9750 | 0.1118 | false | no intervention |
| starvation_pressure-022 | equalize_root_priors | 384 | 1 | false | 0.0885 | 0.8750 | 0.0638 | false | equalize corrected reference and selected root priors |
| starvation_pressure-022 | equalize_root_priors | 1200 | 1 | false | 0.0608 | 0.9225 | 0.0477 | false | equalize corrected reference and selected root priors |
| starvation_pressure-022 | uniform_legal_prior | 384 | 4 | false | 0.0208 | 0.8385 | 0.1470 | false | uniform legal prior positive control |
| starvation_pressure-022 | uniform_legal_prior | 1200 | 4 | false | 0.0117 | 0.5575 | 0.0959 | false | uniform legal prior positive control |
| starvation_pressure-022 | teacher_child_value_override | 384 | 1 | false | 0.0052 | 0.8906 | 0.2562 | false | override first child expansion values with teacher child values |
| starvation_pressure-022 | teacher_child_value_override | 1200 | 1 | false | 0.0033 | 0.9625 | 0.1601 | false | override first child expansion values with teacher child values |
| starvation_pressure-022 | neural_child_value_swap | 384 | 1 | false | 0.0130 | 0.9479 | 0.1073 | false | swap child neural values as diagnostic sanity check |
| starvation_pressure-022 | neural_child_value_swap | 1200 | 1 | false | 0.0058 | 0.9758 | 0.1076 | false | swap child neural values as diagnostic sanity check |
| starvation_pressure-012 | original | 384 | 5 | false | 0.0130 | 0.9870 | 0.0752 | false | no intervention |
| starvation_pressure-012 | original | 1200 | 5 | false | 0.0083 | 0.9917 | 0.0986 | false | no intervention |
| starvation_pressure-012 | equalize_root_priors | 384 | 5 | false | 0.0339 | 0.9661 | 0.0606 | false | equalize corrected reference and selected root priors |
| starvation_pressure-012 | equalize_root_priors | 1200 | 5 | false | 0.0175 | 0.9825 | 0.0030 | false | equalize corrected reference and selected root priors |
| starvation_pressure-012 | uniform_legal_prior | 384 | 5 | false | 0.0339 | 0.9661 | 0.0606 | false | uniform legal prior positive control |
| starvation_pressure-012 | uniform_legal_prior | 1200 | 5 | false | 0.0175 | 0.9825 | 0.0030 | false | uniform legal prior positive control |
| starvation_pressure-012 | teacher_child_value_override | 384 | 5 | false | 0.0130 | 0.9870 | 0.1292 | false | override first child expansion values with teacher child values |
| starvation_pressure-012 | teacher_child_value_override | 1200 | 5 | false | 0.0083 | 0.9917 | 0.1258 | false | override first child expansion values with teacher child values |
| starvation_pressure-012 | neural_child_value_swap | 384 | 5 | false | 0.0130 | 0.9870 | 0.0927 | false | swap child neural values as diagnostic sanity check |
| starvation_pressure-012 | neural_child_value_swap | 1200 | 5 | false | 0.0083 | 0.9917 | 0.1073 | false | swap child neural values as diagnostic sanity check |
| starvation_pressure-023 | original | 384 | 1 | false | 0.0417 | 0.7630 | 0.0739 | false | no intervention |
| starvation_pressure-023 | original | 1200 | 1 | false | 0.0225 | 0.9092 | 0.0653 | false | no intervention |
| starvation_pressure-023 | equalize_root_priors | 384 | 1 | false | 0.0417 | 0.7734 | 0.0746 | false | equalize corrected reference and selected root priors |
| starvation_pressure-023 | equalize_root_priors | 1200 | 1 | false | 0.0225 | 0.9100 | 0.0656 | false | equalize corrected reference and selected root priors |
| starvation_pressure-023 | uniform_legal_prior | 384 | 1 | false | 0.0312 | 0.8698 | 0.0740 | false | uniform legal prior positive control |
| starvation_pressure-023 | uniform_legal_prior | 1200 | 1 | false | 0.0225 | 0.9108 | 0.0656 | false | uniform legal prior positive control |
| starvation_pressure-023 | teacher_child_value_override | 384 | 3 | false | 0.0312 | 0.8516 | 0.1565 | false | override first child expansion values with teacher child values |
| starvation_pressure-023 | teacher_child_value_override | 1200 | 3 | false | 0.0100 | 0.9392 | 0.1278 | false | override first child expansion values with teacher child values |
| starvation_pressure-023 | neural_child_value_swap | 384 | 1 | false | 0.0755 | 0.7214 | 0.0629 | false | swap child neural values as diagnostic sanity check |
| starvation_pressure-023 | neural_child_value_swap | 1200 | 1 | false | 0.0242 | 0.9067 | 0.0604 | false | swap child neural values as diagnostic sanity check |
| starvation_pressure-027 | original | 384 | 1 | false | 0.0026 | 0.9505 | 0.0493 | false | no intervention |
| starvation_pressure-027 | original | 1200 | 1 | false | 0.0033 | 0.9650 | 0.0417 | false | no intervention |
| starvation_pressure-027 | equalize_root_priors | 384 | 0 | true | 0.6615 | 0.6615 | 0.0000 | true | equalize corrected reference and selected root priors |
| starvation_pressure-027 | equalize_root_priors | 1200 | 0 | true | 0.8850 | 0.8850 | 0.0000 | true | equalize corrected reference and selected root priors |
| starvation_pressure-027 | uniform_legal_prior | 384 | 1 | false | 0.3594 | 0.5911 | -0.0391 | false | uniform legal prior positive control |
| starvation_pressure-027 | uniform_legal_prior | 1200 | 0 | true | 0.7808 | 0.7808 | 0.0000 | true | uniform legal prior positive control |
| starvation_pressure-027 | teacher_child_value_override | 384 | 1 | false | 0.0026 | 0.7318 | 0.7827 | false | override first child expansion values with teacher child values |
| starvation_pressure-027 | teacher_child_value_override | 1200 | 1 | false | 0.0008 | 0.5667 | 0.7833 | false | override first child expansion values with teacher child values |
| starvation_pressure-027 | neural_child_value_swap | 384 | 1 | false | 0.0026 | 0.9505 | 0.0538 | false | swap child neural values as diagnostic sanity check |
| starvation_pressure-027 | neural_child_value_swap | 1200 | 1 | false | 0.0033 | 0.9650 | 0.0428 | false | swap child neural values as diagnostic sanity check |

## 10. Row classifications

| row_id | role | row_classification | supporting_evidence | recommended_use | notes |
| --- | --- | --- | --- | --- | --- |
| starvation_pressure-025 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| starvation_pressure-026 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| starvation_pressure-024 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| starvation_pressure-022 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| starvation_pressure-012 | target_candidate | inconclusive | ClassicMCTS child teacher is unstable across seeds | target_candidate | root still fails |
| starvation_pressure-023 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| starvation_pressure-027 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| starvation_pressure-015 | holdout_candidate | inconclusive | holdout row reserved for out-of-sample follow-up rather than mechanism counting | holdout | root still fails |
| starvation_pressure-001 | holdout_candidate | inconclusive | holdout row reserved for out-of-sample follow-up rather than mechanism counting | holdout | root still fails |
| starvation_pressure-013 | preservation_control | inconclusive | preservation control retained to guard against regression | preserve_control | baseline pass |
| starvation_pressure-003 | preservation_control | inconclusive | preservation control retained to guard against regression | preserve_control | baseline pass |
| starvation_pressure-021 | preservation_control | inconclusive | preservation control retained to guard against regression | preserve_control | baseline pass |
| starvation_pressure-014 | preservation_control | inconclusive | preservation control retained to guard against regression | preserve_control | baseline pass |

## 11. Family-level interpretation

| family_classification | value_head_miscalibration_count | puct_child_search_mismatch_count | root_selection_pressure_count | corrected_reference_suspicious_count | inconclusive_count | next_action |
| --- | --- | --- | --- | --- | --- | --- |
| reference_family_uncertain | 0 | 0 | 0 | 6 | 1 | adjudicate starvation_pressure references before training. |

## 12. Exactly one recommended next action

Recommendation: **adjudicate starvation_pressure references before training.**
