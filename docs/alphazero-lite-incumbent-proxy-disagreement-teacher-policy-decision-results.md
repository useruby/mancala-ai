# AlphaZero-lite Incumbent Proxy Disagreement Teacher Policy Decision Results

## 1. Context

- No training was run.
- No arena was run.
- No model was promoted.
- No replay artifacts were created.
- Active reference fixtures stayed read-only at `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- Current artifact: `storage/ai/alphazero_lite/current`.

## 2. Why PR #45 blocked training

- PR #45 concluded that this family still mixed root-selection pressure with value-head failures, while ClassicMCTS and deterministic artifact PUCT disagreed on several active references.
- The exact follow-up required before any training was to decide which teacher should define future references and training targets.
- This audit answers that question without mutating fixtures or generating any training artifacts.

## 3. Teacher definitions

- ClassicMCTS teacher: budgets `1200, 2400, 5000, 10000` across seeds `11,23,37,42,101,202,303`.
- Deterministic artifact PUCT teacher: budgets `384, 1200, 2400, 5000` with `fpu_mode=parent_q`, `reuse_subtree=true`, `normalize_values=true`, `root_policy_mode=deterministic`, `tactical_root_bias=0.1`, `c_puct=1.25`.
- High-sim artifact PUCT teacher: same deterministic settings with highest budget `10000`.
- Exact/tablebase teacher: only used where the repo tablebase can solve the root or compared child states.
- Paired continuations: `8` paired rollouts per unresolved row and per continuation policy.
- The rollout count was runtime-bounded below the nominal 16-continuation default to keep this pre-training audit tractable.

## 4. Row set and validation

| row_id | row_group | active_reference_move | reference_move_legal | canonical_state_match | reference_unstable | excluded_unstable | remaining_seed_count | tablebase_root_available | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-003 | value_head_candidate | 4 | true | true | false | false | 44 | false | validated against active references |
| incumbent_proxy_disagreement-007 | root_selection_pressure | 0 | true | true | false | false | 44 | false | validated against active references |
| incumbent_proxy_disagreement-008 | control | 2 | true | true | false | false | 44 | false | validated against active references |
| incumbent_proxy_disagreement-009 | root_selection_pressure | 0 | true | true | false | false | 44 | false | validated against active references |
| incumbent_proxy_disagreement-010 | excluded_unstable | 4 | true | true | false | true | 43 | false | row 010 kept excluded/unstable by audit policy |
| incumbent_proxy_disagreement-012 | value_head_candidate | 3 | true | true | false | false | 43 | false | validated against active references |
| incumbent_proxy_disagreement-014 | root_selection_pressure | 4 | true | true | false | false | 43 | false | validated against active references |
| incumbent_proxy_disagreement-018 | root_selection_pressure | 2 | true | true | false | false | 43 | false | validated against active references |
| incumbent_proxy_disagreement-020 | value_head_candidate | 3 | true | true | false | false | 43 | false | validated against active references |
| incumbent_proxy_disagreement-021 | root_selection_pressure | 2 | true | true | false | false | 43 | false | validated against active references |
| incumbent_proxy_disagreement-022 | value_head_candidate | 3 | true | true | false | false | 43 | false | validated against active references |
| incumbent_proxy_disagreement-023 | root_selection_pressure | 2 | true | true | false | false | 42 | false | validated against active references |
| incumbent_proxy_disagreement-024 | root_selection_pressure | 2 | true | true | false | false | 42 | false | validated against active references |
| incumbent_proxy_disagreement-025 | value_head_candidate | 2 | true | true | false | false | 42 | false | validated against active references |
| incumbent_proxy_disagreement-026 | control | 5 | true | true | false | false | 42 | false | validated against active references |
| incumbent_proxy_disagreement-027 | value_head_candidate | 3 | true | true | false | false | 42 | false | validated against active references |
| incumbent_proxy_disagreement-028 | control | 5 | true | true | false | false | 42 | false | validated against active references |
| incumbent_proxy_disagreement-029 | control | 4 | true | true | false | false | 42 | false | validated against active references |
| incumbent_proxy_disagreement-032 | root_selection_pressure | 4 | true | true | false | false | 42 | false | validated against active references |
| incumbent_proxy_disagreement-033 | root_selection_pressure | 4 | true | true | false | false | 42 | false | validated against active references |
| incumbent_proxy_disagreement-035 | value_head_candidate | 3 | true | true | false | false | 44 | false | validated against active references |

## 5. Teacher agreement matrix

| row_id | active_reference_move | classic_selected_move | puct_selected_move | high_sim_puct_selected_move | tablebase_selected_move | classic_reference_agreement | puct_reference_agreement | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-003 | 4 | 4 | 5 | 5 | - | true | false | high budgets agree across 2400/5000/10000; deterministic selection changes across budgets; no PR45 teacher metadata parsed |
| incumbent_proxy_disagreement-007 | 0 | 0 | 3 | 3 | - | true | false | high budgets agree across 2400/5000/10000; deterministic selection changes across budgets; PR45 classic=0 puct=3 |
| incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 2 | - | true | true | high budgets agree across 2400/5000/10000; deterministic selection unchanged across budgets; no PR45 teacher metadata parsed |
| incumbent_proxy_disagreement-009 | 0 | 0 | 3 | 3 | - | true | false | high budgets agree across 2400/5000/10000; deterministic selection unchanged across budgets; PR45 classic=0 puct=3 |
| incumbent_proxy_disagreement-010 | 4 | 0 | 4 | 4 | - | false | true | high-budget seed majorities remain mixed; deterministic selection changes across budgets; PR45 classic=0 puct=4 |
| incumbent_proxy_disagreement-012 | 3 | 1 | 5 | 5 | - | false | false | high-budget seed majorities remain mixed; deterministic selection unchanged across budgets; no PR45 teacher metadata parsed |
| incumbent_proxy_disagreement-014 | 4 | 4 | 1 | 1 | - | true | false | high budgets agree across 2400/5000/10000; deterministic selection changes across budgets; no PR45 teacher metadata parsed |
| incumbent_proxy_disagreement-018 | 2 | 2 | 3 | 3 | - | true | false | high budgets agree across 2400/5000/10000; deterministic selection changes across budgets; PR45 classic=2 puct=3 |
| incumbent_proxy_disagreement-020 | 3 | 0 | 5 | 5 | - | false | false | high-budget seed majorities remain mixed; deterministic selection unchanged across budgets; no PR45 teacher metadata parsed |
| incumbent_proxy_disagreement-021 | 2 | 2 | 5 | 3 | - | true | false | high-budget seed majorities remain mixed; deterministic selection unchanged across budgets; PR45 classic=2 puct=5 |
| incumbent_proxy_disagreement-022 | 3 | 3 | 3 | 3 | - | true | true | high budgets agree across 2400/5000/10000; deterministic selection changes across budgets; no PR45 teacher metadata parsed |
| incumbent_proxy_disagreement-023 | 2 | 2 | 1 | 1 | - | true | false | high budgets agree across 2400/5000/10000; deterministic selection unchanged across budgets; PR45 classic=2 puct=1 |
| incumbent_proxy_disagreement-024 | 2 | 2 | 1 | 1 | - | true | false | high budgets agree across 2400/5000/10000; deterministic selection unchanged across budgets; PR45 classic=2 puct=1 |
| incumbent_proxy_disagreement-025 | 2 | 2 | 2 | 2 | - | true | true | high budgets agree across 2400/5000/10000; deterministic selection changes across budgets; no PR45 teacher metadata parsed |
| incumbent_proxy_disagreement-026 | 5 | 0 | 5 | 5 | - | false | true | high budgets agree across 2400/5000/10000; deterministic selection unchanged across budgets; no PR45 teacher metadata parsed |
| incumbent_proxy_disagreement-027 | 3 | 3 | 5 | 5 | - | true | false | high budgets agree across 2400/5000/10000; deterministic selection unchanged across budgets; no PR45 teacher metadata parsed |
| incumbent_proxy_disagreement-028 | 5 | 4 | 5 | 5 | - | false | true | high budgets agree across 2400/5000/10000; deterministic selection unchanged across budgets; no PR45 teacher metadata parsed |
| incumbent_proxy_disagreement-029 | 4 | 3 | 4 | 4 | - | false | true | high-budget seed majorities remain mixed; deterministic selection unchanged across budgets; no PR45 teacher metadata parsed |
| incumbent_proxy_disagreement-032 | 4 | 4 | 1 | 0 | - | true | false | high budgets agree across 2400/5000/10000; deterministic selection changes across budgets; PR45 classic=4 puct=1 |
| incumbent_proxy_disagreement-033 | 4 | 4 | 0 | 0 | - | true | false | high budgets agree across 2400/5000/10000; deterministic selection unchanged across budgets; PR45 classic=4 puct=0 |
| incumbent_proxy_disagreement-035 | 3 | 3 | 4 | 4 | - | true | false | high budgets agree across 2400/5000/10000; deterministic selection unchanged across budgets; no PR45 teacher metadata parsed |

## 6. Child-afterstate outcome adjudication

| row_id | classic_or_reference_move | puct_selected_move | classic_child_value_root | puct_child_value_root | classic_minus_puct_child_value | tablebase_value_if_available | decision | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-003 | 4 | 5 | -0.4052 | -0.7518 | 0.3466 | - | split | classic-child: child delta 0.3466 favors move 4; puct-child: child delta -0.0517 favors move 5 |
| incumbent_proxy_disagreement-007 | 0 | 3 | -0.6089 | -0.3175 | -0.2914 | - | puct | classic-child decisive; puct-child near-zero split; child delta -0.2914 favors move 3; child delta -0.0173 remains near zero |
| incumbent_proxy_disagreement-009 | 0 | 3 | -0.6587 | -0.5219 | -0.1368 | - | puct | classic-child: child delta -0.1368 favors move 3; puct-child: child delta -0.1350 favors move 3 |
| incumbent_proxy_disagreement-010 | 4 | 4 | -0.6551 | -0.6551 | 0.0000 | - | split | classic-child: child delta 0.0000 remains near zero; puct-child: child delta 0.0000 remains near zero |
| incumbent_proxy_disagreement-012 | 3 | 5 | -0.3885 | -0.6366 | 0.2481 | - | split | classic-child: child delta 0.2481 favors move 3; puct-child: child delta -0.1530 favors move 5 |
| incumbent_proxy_disagreement-014 | 4 | 1 | -0.5698 | -0.7365 | 0.1667 | - | classic | classic-child decisive; puct-child near-zero split; child delta 0.1667 favors move 4; child delta -0.0141 remains near zero |
| incumbent_proxy_disagreement-018 | 2 | 3 | -0.5830 | -0.5196 | -0.0634 | - | split | classic-child: child delta -0.0634 favors move 3; puct-child: child delta 0.0350 favors move 2 |
| incumbent_proxy_disagreement-020 | 3 | 5 | -0.2833 | -0.7400 | 0.4567 | - | split | classic-child: child delta 0.4567 favors move 3; puct-child: child delta -0.1048 favors move 5 |
| incumbent_proxy_disagreement-021 | 2 | 5 | -0.8107 | -0.7567 | -0.0540 | - | puct | classic-child: child delta -0.0540 favors move 5; puct-child: child delta -0.0851 favors move 5 |
| incumbent_proxy_disagreement-023 | 2 | 1 | -0.9364 | -0.6664 | -0.2700 | - | split | classic-child: child delta -0.2700 favors move 1; puct-child: child delta 0.0847 favors move 2 |
| incumbent_proxy_disagreement-024 | 2 | 1 | -0.7959 | -0.5979 | -0.1980 | - | split | classic-child: child delta -0.1980 favors move 1; puct-child: child delta 0.0415 favors move 2 |
| incumbent_proxy_disagreement-026 | 0 | 5 | -0.6832 | -0.4931 | -0.1901 | - | puct | classic-child: child delta -0.1901 favors move 5; puct-child: child delta -0.1858 favors move 5 |
| incumbent_proxy_disagreement-027 | 3 | 5 | -0.2979 | -0.4154 | 0.1175 | - | split | classic-child: child delta 0.1175 favors move 3; puct-child: child delta -0.0300 favors move 5 |
| incumbent_proxy_disagreement-028 | 4 | 5 | -0.3249 | -0.4750 | 0.1501 | - | split | classic-child: child delta 0.1501 favors move 4; puct-child: child delta -0.1228 favors move 5 |
| incumbent_proxy_disagreement-029 | 4 | 4 | 0.7414 | 0.7414 | 0.0000 | - | split | classic-child: child delta 0.0000 remains near zero; puct-child: child delta 0.0000 remains near zero |
| incumbent_proxy_disagreement-032 | 4 | 1 | -0.4268 | -0.4163 | -0.0105 | - | puct | puct-child decisive; classic-child near-zero split; child delta -0.0105 remains near zero; child delta -0.0220 favors move 1 |
| incumbent_proxy_disagreement-033 | 4 | 0 | -0.2782 | -0.2602 | -0.0180 | - | puct | puct-child decisive; classic-child near-zero split; child delta -0.0180 remains near zero; child delta -0.1262 favors move 0 |
| incumbent_proxy_disagreement-035 | 3 | 4 | -0.6601 | -0.6957 | 0.0356 | - | classic | classic-child decisive; puct-child near-zero split; child delta 0.0356 favors move 3; child delta -0.0197 remains near zero |

## 7. Paired continuation rollouts

| row_id | continuation_policy | branch_classic_or_reference_outcome_mean | branch_puct_outcome_mean | outcome_delta_classic_minus_puct | continuations | ci_low | ci_high | decision | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-003 | classic | 1.0000 | 1.0000 | 0.0000 | 8 | 0.0000 | 0.0000 | neutral_or_split | continuation budget=1200 |
| incumbent_proxy_disagreement-003 | puct | -1.0000 | 1.0000 | -2.0000 | 8 | -2.0000 | -2.0000 | puct_branch_better | continuation budget=1200 |
| incumbent_proxy_disagreement-010 | classic | 0.8750 | 0.8750 | 0.0000 | 8 | 0.0000 | 0.0000 | neutral_or_split | continuation budget=1200 |
| incumbent_proxy_disagreement-010 | puct | 1.0000 | 1.0000 | 0.0000 | 8 | 0.0000 | 0.0000 | neutral_or_split | continuation budget=1200 |
| incumbent_proxy_disagreement-012 | classic | -0.5000 | 0.7500 | -1.2500 | 8 | -1.9673 | -0.5327 | puct_branch_better | continuation budget=1200 |
| incumbent_proxy_disagreement-012 | puct | 1.0000 | 1.0000 | 0.0000 | 8 | 0.0000 | 0.0000 | neutral_or_split | continuation budget=1200 |
| incumbent_proxy_disagreement-018 | classic | 0.6250 | 0.7500 | -0.1250 | 8 | -0.9053 | 0.6553 | puct_branch_better | continuation budget=1200 |
| incumbent_proxy_disagreement-018 | puct | 1.0000 | 1.0000 | 0.0000 | 8 | 0.0000 | 0.0000 | neutral_or_split | continuation budget=1200 |
| incumbent_proxy_disagreement-020 | classic | -0.2500 | 0.7500 | -1.0000 | 8 | -2.0477 | 0.0477 | puct_branch_better | continuation budget=1200 |
| incumbent_proxy_disagreement-020 | puct | 1.0000 | 0.0000 | 1.0000 | 8 | 1.0000 | 1.0000 | classic_branch_better | continuation budget=1200 |
| incumbent_proxy_disagreement-023 | classic | -0.7500 | -0.6250 | -0.1250 | 8 | -0.9053 | 0.6553 | puct_branch_better | continuation budget=1200 |
| incumbent_proxy_disagreement-023 | puct | -1.0000 | -1.0000 | 0.0000 | 8 | 0.0000 | 0.0000 | neutral_or_split | continuation budget=1200 |
| incumbent_proxy_disagreement-024 | classic | 0.2500 | -0.7500 | 1.0000 | 8 | 0.2592 | 1.7408 | classic_branch_better | continuation budget=1200 |
| incumbent_proxy_disagreement-024 | puct | 0.0000 | -1.0000 | 1.0000 | 8 | 1.0000 | 1.0000 | classic_branch_better | continuation budget=1200 |
| incumbent_proxy_disagreement-027 | classic | 0.5000 | 0.5000 | 0.0000 | 8 | -0.7408 | 0.7408 | neutral_or_split | continuation budget=1200 |
| incumbent_proxy_disagreement-027 | puct | 1.0000 | 1.0000 | 0.0000 | 8 | 0.0000 | 0.0000 | neutral_or_split | continuation budget=1200 |
| incumbent_proxy_disagreement-028 | classic | -0.7500 | -0.2500 | -0.5000 | 8 | -1.1416 | 0.1416 | puct_branch_better | continuation budget=1200 |
| incumbent_proxy_disagreement-028 | puct | 0.0000 | 1.0000 | -1.0000 | 8 | -1.0000 | -1.0000 | puct_branch_better | continuation budget=1200 |
| incumbent_proxy_disagreement-029 | classic | 0.5000 | 0.5000 | 0.0000 | 8 | 0.0000 | 0.0000 | neutral_or_split | continuation budget=1200 |
| incumbent_proxy_disagreement-029 | puct | 1.0000 | 1.0000 | 0.0000 | 8 | 0.0000 | 0.0000 | neutral_or_split | continuation budget=1200 |

## 8. Row-level teacher decisions

| row_id | row_decision | preferred_teacher | preferred_move | active_reference_move | recommended_use | evidence_summary |
| --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-003 | unstable_or_inconclusive | none | 4 | 4 | exclude_from_training_and_keep_diagnostic_only | teacher evidence remains mixed; classic=4 puct=5 high_sim=5; classic-child: child delta 0.3466 favors move 4; puct-child: child delta -0.0517 favors move 5 |
| incumbent_proxy_disagreement-007 | puct_reference_preferred | artifact_puct | 3 | 0 | do_not_train_until_reference_policy_changes | downstream child evidence favors PUCT; classic-child decisive; puct-child near-zero split; child delta -0.2914 favors move 3; child delta -0.0173 remains near zero |
| incumbent_proxy_disagreement-008 | classic_reference_confirmed | classic_mcts | 2 | 2 | safe_control_or_reference_row | ClassicMCTS, artifact PUCT, and active reference all agree |
| incumbent_proxy_disagreement-009 | puct_reference_preferred | artifact_puct | 3 | 0 | do_not_train_until_reference_policy_changes | downstream child evidence favors PUCT; classic-child: child delta -0.1368 favors move 3; puct-child: child delta -0.1350 favors move 3 |
| incumbent_proxy_disagreement-010 | unstable_or_inconclusive | excluded | 4 | 4 | exclude_from_targets_and_keep_diagnostic_only | row 010 remains excluded/unstable regardless of downstream comparisons |
| incumbent_proxy_disagreement-012 | unstable_or_inconclusive | none | 3 | 3 | exclude_from_training_and_keep_diagnostic_only | teacher evidence remains mixed; classic=1 puct=5 high_sim=5; classic-child: child delta 0.2481 favors move 3; puct-child: child delta -0.1530 favors move 5 |
| incumbent_proxy_disagreement-014 | classic_reference_confirmed | classic_mcts | 4 | 4 | confirmed_root_selection_pressure_diagnostic | downstream child evidence favors classic/reference; classic-child decisive; puct-child near-zero split; child delta 0.1667 favors move 4; child delta -0.0141 remains near zero |
| incumbent_proxy_disagreement-018 | unstable_or_inconclusive | none | 2 | 2 | exclude_from_training_and_keep_diagnostic_only | teacher evidence remains mixed; classic=2 puct=3 high_sim=3; classic-child: child delta -0.0634 favors move 3; puct-child: child delta 0.0350 favors move 2 |
| incumbent_proxy_disagreement-020 | unstable_or_inconclusive | none | 3 | 3 | exclude_from_training_and_keep_diagnostic_only | teacher evidence remains mixed; classic=0 puct=5 high_sim=5; classic-child: child delta 0.4567 favors move 3; puct-child: child delta -0.1048 favors move 5 |
| incumbent_proxy_disagreement-021 | puct_reference_preferred | artifact_puct | 5 | 2 | do_not_train_until_reference_policy_changes | downstream child evidence favors PUCT; classic-child: child delta -0.0540 favors move 5; puct-child: child delta -0.0851 favors move 5 |
| incumbent_proxy_disagreement-022 | classic_reference_confirmed | classic_mcts | 3 | 3 | safe_control_or_reference_row | ClassicMCTS, artifact PUCT, and active reference all agree |
| incumbent_proxy_disagreement-023 | unstable_or_inconclusive | none | 2 | 2 | exclude_from_training_and_keep_diagnostic_only | teacher evidence remains mixed; classic=2 puct=1 high_sim=1; classic-child: child delta -0.2700 favors move 1; puct-child: child delta 0.0847 favors move 2 |
| incumbent_proxy_disagreement-024 | classic_reference_confirmed | classic_mcts | 2 | 2 | confirmed_root_selection_pressure_diagnostic | paired continuations support classic/reference under both policies |
| incumbent_proxy_disagreement-025 | classic_reference_confirmed | classic_mcts | 2 | 2 | safe_control_or_reference_row | ClassicMCTS, artifact PUCT, and active reference all agree |
| incumbent_proxy_disagreement-026 | puct_reference_preferred | artifact_puct | 5 | 5 | do_not_train_until_reference_policy_changes | downstream child evidence favors PUCT; classic-child: child delta -0.1901 favors move 5; puct-child: child delta -0.1858 favors move 5 |
| incumbent_proxy_disagreement-027 | unstable_or_inconclusive | none | 3 | 3 | exclude_from_training_and_keep_diagnostic_only | teacher evidence remains mixed; classic=3 puct=5 high_sim=5; classic-child: child delta 0.1175 favors move 3; puct-child: child delta -0.0300 favors move 5 |
| incumbent_proxy_disagreement-028 | puct_reference_preferred | artifact_puct | 5 | 5 | do_not_train_until_reference_policy_changes | paired continuations support the PUCT-selected branch under both policies |
| incumbent_proxy_disagreement-029 | unstable_or_inconclusive | none | 4 | 4 | exclude_from_training_and_keep_diagnostic_only | teacher evidence remains mixed; classic=3 puct=4 high_sim=4; classic-child: child delta 0.0000 remains near zero; puct-child: child delta 0.0000 remains near zero |
| incumbent_proxy_disagreement-032 | puct_reference_preferred | artifact_puct | 1 | 4 | do_not_train_until_reference_policy_changes | downstream child evidence favors PUCT; puct-child decisive; classic-child near-zero split; child delta -0.0105 remains near zero; child delta -0.0220 favors move 1 |
| incumbent_proxy_disagreement-033 | puct_reference_preferred | artifact_puct | 0 | 4 | do_not_train_until_reference_policy_changes | downstream child evidence favors PUCT; puct-child decisive; classic-child near-zero split; child delta -0.0180 remains near zero; child delta -0.1262 favors move 0 |
| incumbent_proxy_disagreement-035 | classic_reference_confirmed | classic_mcts | 3 | 3 | classic_target_or_preservation_row | downstream child evidence favors classic/reference; classic-child decisive; puct-child near-zero split; child delta 0.0356 favors move 3; child delta -0.0197 remains near zero |

## 9. Family-level teacher policy decision

| family_decision | classic_reference_confirmed_count | puct_reference_preferred_count | teacher_policy_dependent_count | exact_or_tablebase_resolved_count | unstable_or_inconclusive_count | next_action |
| --- | --- | --- | --- | --- | --- | --- |
| teacher_policy_split_required | 6 | 7 | 0 | 0 | 8 | split rows into Classic-teacher, PUCT-teacher, and excluded diagnostic buckets before training. |

## 10. Exactly one recommended next action

Recommendation: **split rows into Classic-teacher, PUCT-teacher, and excluded diagnostic buckets before training.**
