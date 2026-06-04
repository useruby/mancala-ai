# Harder Fresh Endgame Tablebase Mining — Results

**Date:** 2026-06-04
**Family:** `harder_fresh_endgame_tablebase`
**Script:** `ml/alphazero_lite/run_harder_fresh_endgame_tablebase_mining.py`

## 1. Context

- Run classification: harder_tablebase_exact_target_family_ready
- Selected family: harder_fresh_endgame_tablebase
- Current artifact: storage/ai/alphazero_lite/current
- Active references: ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json
- No training was run.
- No arena was run.
- No model was promoted.
- Active references were not mutated.

## 2. Why PR #72 was too easy

PR #72 ran diagnostics on 14 fresh endgame-tablebase-unique rows mined from current-model self-play. All 14 rows passed PUCT at every budget from 128 to 5000. No rows exhibited value rank errors, sign errors, or child PUCT failures. The decision was `tablebase_unique_too_small` because zero target candidates were found.

The root cause: the self-play positions were too shallow (ply 29–57) and the tablebase-solvable endgames were trivially solved by the existing network. This run generates harder positions using multiple strategies: deeper self-play, low-budget PUCT failure prescreening, adversarial near-threshold sampling, and direct PUCT-vs-tablebase disagreement sampling.

## 3. Candidate generation strategy

| source | raw_candidates | tablebase_solvable | unique_optimal | puct_disagreements | neural_rank_errors | kept_candidates | notes |
|---|---|---|---|---|---|---|---|
| deeper_self_play | 270 | 118 | 100 | 0 | 0 | 0 | late-game positions from current-model PUCT |
| puct_low_budget_failure | 0 | 118 | 100 | 0 | 0 | 0 | PUCT 128 fails vs tablebase optimal |
| adversarial_near_threshold | 100 | 118 | 100 | 0 | 32 | 100 | random endgame states with unique optimal and meaningful gap |
| puct_tablebase_disagreement | 0 | 118 | 100 | 0 | 0 | 0 | PUCT 384 disagrees with tablebase unique optimal |

## 4. Deduplication and exclusions

| exclusion_reason | count | notes |
|---|---|---|
| Raw candidates generated | 370 | |
| Exact duplicates removed | 252 | |
| Known fixture overlaps | 0 | |
| Existing selected overlap (PR #72) | 0 | |
| Excluded exhausted overlaps | 0 | |
| Tablebase-solvable | 118 | |
| Unique optimal move | 100 | |
| Rejected ties/all-equivalent | 18 | |
| Rejected exhausted family patterns | 0 | |
| Remaining clean candidates | 100 | |

## 5. Exact tablebase enumeration

| candidate_hash | ply | remaining_seed_count | legal_moves | root_value | optimal_move | second_best_value | best_minus_second_best | forced_win_or_loss | exact_signal_class | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| {"current_player... | None | 11 | 2 | 1.0 | 2 | -1.0 | 2.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 9 | 2 | 0.0 | 5 | -1.0 | 1.0 | False | exact_unique_clear_margin | ok |
| {"current_player... | None | 12 | 2 | 0.0 | 3 | -1.0 | 1.0 | False | exact_unique_clear_margin | ok |
| {"current_player... | None | 8 | 2 | 1.0 | 0 | 0.0 | 1.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 12 | 2 | 1.0 | 5 | -1.0 | 2.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 9 | 2 | 1.0 | 5 | 0.0 | 1.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 12 | 2 | 1.0 | 5 | 0.0 | 1.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 7 | 2 | 1.0 | 3 | -1.0 | 2.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 9 | 2 | 1.0 | 2 | -1.0 | 2.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 9 | 2 | 1.0 | 2 | 0.0 | 1.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 12 | 2 | 1.0 | 0 | -1.0 | 2.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 11 | 2 | 1.0 | 3 | -1.0 | 2.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 11 | 2 | 1.0 | 0 | 0.0 | 1.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 6 | 2 | 1.0 | 1 | -1.0 | 2.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 6 | 2 | 1.0 | 3 | -1.0 | 2.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 6 | 2 | 1.0 | 4 | 0.0 | 1.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 8 | 2 | 1.0 | 0 | -1.0 | 2.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 4 | 2 | 1.0 | 4 | 0.0 | 1.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 12 | 2 | 0.0 | 4 | -1.0 | 1.0 | False | exact_unique_clear_margin | ok |
| {"current_player... | None | 3 | 2 | 1.0 | 5 | -1.0 | 2.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 12 | 2 | 1.0 | 4 | 0.0 | 1.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 7 | 2 | 1.0 | 1 | -1.0 | 2.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 9 | 2 | 1.0 | 0 | -1.0 | 2.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 12 | 2 | 1.0 | 4 | -1.0 | 2.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 6 | 2 | 1.0 | 4 | 0.0 | 1.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 8 | 2 | 1.0 | 0 | -1.0 | 2.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 10 | 2 | 1.0 | 3 | -1.0 | 2.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 9 | 2 | 1.0 | 0 | -1.0 | 2.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 11 | 2 | 1.0 | 4 | -1.0 | 2.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 12 | 2 | 0.0 | 5 | -1.0 | 1.0 | False | exact_unique_clear_margin | ok |
| {"current_player... | None | 9 | 2 | 1.0 | 3 | 0.0 | 1.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 11 | 4 | 0.0 | 0 | -1.0 | 1.0 | False | exact_unique_clear_margin | ok |
| {"current_player... | None | 7 | 2 | 1.0 | 0 | 0.0 | 1.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 11 | 2 | 1.0 | 4 | -1.0 | 2.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 8 | 2 | 1.0 | 3 | 0.0 | 1.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 10 | 3 | 1.0 | 5 | 0.0 | 1.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 11 | 2 | 1.0 | 4 | -1.0 | 2.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 8 | 2 | 1.0 | 3 | -1.0 | 2.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 8 | 2 | 1.0 | 2 | -1.0 | 2.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 5 | 2 | 1.0 | 3 | 0.0 | 1.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 12 | 2 | 1.0 | 1 | -1.0 | 2.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 6 | 2 | 1.0 | 4 | -1.0 | 2.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 7 | 2 | 1.0 | 3 | 0.0 | 1.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 10 | 2 | 1.0 | 0 | -1.0 | 2.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 12 | 2 | 1.0 | 0 | -1.0 | 2.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 12 | 2 | 1.0 | 0 | -1.0 | 2.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 8 | 2 | 1.0 | 3 | -1.0 | 2.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 12 | 2 | 1.0 | 1 | -1.0 | 2.0 | True | exact_unique_clear_margin | ok |
| {"current_player... | None | 5 | 2 | 0.0 | 1 | -1.0 | 1.0 | False | exact_unique_clear_margin | ok |
| {"current_player... | None | 7 | 2 | 1.0 | 0 | 0.0 | 1.0 | True | exact_unique_clear_margin | ok |
| ... and 50 more rows | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

## 6. PUCT budget scan

- Total rows evaluated: 100
- Rows with persistent failures (384+1200): 7
- Rows with medium-budget failures (384 only): 5
- Rows with low-budget failures (64/128/256 only): 9
- Clean controls (all budgets pass): 79

| candidate_hash | budget | optimal_move | selected_move | selected_is_optimal | optimal_visit_share | selected_visit_share | selected_minus_optimal_q_margin | optimal_policy_rank | failure_class | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| {"current_player... | 64 | 2 | 2 | True | 0.984375 | 0.984375 | 0.0 | 0 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 128 | 2 | 2 | True | 0.984375 | 0.984375 | 0.0 | 0 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 256 | 2 | 2 | True | 0.988281 | 0.988281 | 0.0 | 0 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 384 | 2 | 2 | True | 0.992188 | 0.992188 | 0.0 | 0 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 1200 | 2 | 2 | True | 0.995 | 0.995 | 0.0 | 0 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 64 | 5 | 5 | True | 0.984375 | 0.984375 | 0.0 | 0 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 128 | 5 | 5 | True | 0.992188 | 0.992188 | 0.0 | 0 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 256 | 5 | 5 | True | 0.984375 | 0.984375 | 0.0 | 0 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 384 | 5 | 5 | True | 0.989583 | 0.989583 | 0.0 | 0 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 1200 | 5 | 5 | True | 0.995 | 0.995 | 0.0 | 0 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 64 | 3 | 3 | True | 0.75 | 0.75 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 128 | 3 | 3 | True | 0.875 | 0.875 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 256 | 3 | 3 | True | 0.9375 | 0.9375 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 384 | 3 | 3 | True | 0.958333 | 0.958333 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 1200 | 3 | 3 | True | 0.981667 | 0.981667 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 64 | 0 | 2 | False | 0.046875 | 0.953125 | 0.071227 | 1 | low_budget_exact_failure | PUCT FAILURE |
| {"current_player... | 128 | 0 | 2 | False | 0.039062 | 0.960938 | 0.04053 | 1 | low_budget_exact_failure | PUCT FAILURE |
| {"current_player... | 256 | 0 | 2 | False | 0.4375 | 0.5625 | -0.151479 | 1 | low_budget_exact_failure | PUCT FAILURE |
| {"current_player... | 384 | 0 | 0 | True | 0.625 | 0.625 | 0.0 | 1 | low_budget_exact_failure | deterministic PUCT baseline |
| {"current_player... | 1200 | 0 | 0 | True | 0.88 | 0.88 | 0.0 | 1 | low_budget_exact_failure | deterministic PUCT baseline |
| {"current_player... | 64 | 5 | 5 | True | 0.875 | 0.875 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 128 | 5 | 5 | True | 0.90625 | 0.90625 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 256 | 5 | 5 | True | 0.933594 | 0.933594 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 384 | 5 | 5 | True | 0.945312 | 0.945312 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 1200 | 5 | 5 | True | 0.969167 | 0.969167 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 64 | 5 | 5 | True | 0.921875 | 0.921875 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 128 | 5 | 5 | True | 0.9375 | 0.9375 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 256 | 5 | 5 | True | 0.957031 | 0.957031 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 384 | 5 | 5 | True | 0.963542 | 0.963542 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 1200 | 5 | 5 | True | 0.979167 | 0.979167 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 64 | 5 | 2 | False | 0.015625 | 0.984375 | 0.196703 | 1 | low_budget_exact_failure | PUCT FAILURE |
| {"current_player... | 128 | 5 | 2 | False | 0.109375 | 0.890625 | -0.206832 | 1 | low_budget_exact_failure | PUCT FAILURE |
| {"current_player... | 256 | 5 | 5 | True | 0.554688 | 0.554688 | 0.0 | 1 | low_budget_exact_failure | deterministic PUCT baseline |
| {"current_player... | 384 | 5 | 5 | True | 0.703125 | 0.703125 | 0.0 | 1 | low_budget_exact_failure | deterministic PUCT baseline |
| {"current_player... | 1200 | 5 | 5 | True | 0.905 | 0.905 | 0.0 | 1 | low_budget_exact_failure | deterministic PUCT baseline |
| {"current_player... | 64 | 3 | 3 | True | 0.640625 | 0.640625 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 128 | 3 | 3 | True | 0.820312 | 0.820312 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 256 | 3 | 3 | True | 0.910156 | 0.910156 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 384 | 3 | 3 | True | 0.940104 | 0.940104 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 1200 | 3 | 3 | True | 0.974167 | 0.974167 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 64 | 2 | 2 | True | 0.90625 | 0.90625 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 128 | 2 | 2 | True | 0.929688 | 0.929688 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 256 | 2 | 2 | True | 0.949219 | 0.949219 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 384 | 2 | 2 | True | 0.958333 | 0.958333 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 1200 | 2 | 2 | True | 0.976667 | 0.976667 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 64 | 2 | 2 | True | 0.59375 | 0.59375 | 0.0 | 0 | persistent_exact_failure | deterministic PUCT baseline |
| {"current_player... | 128 | 2 | 5 | False | 0.296875 | 0.703125 | 0.355541 | 0 | persistent_exact_failure | PUCT FAILURE |
| {"current_player... | 256 | 2 | 5 | False | 0.148438 | 0.851562 | 0.298687 | 0 | persistent_exact_failure | PUCT FAILURE |
| {"current_player... | 384 | 2 | 5 | False | 0.098958 | 0.901042 | 0.282644 | 0 | persistent_exact_failure | PUCT FAILURE |
| {"current_player... | 1200 | 2 | 5 | False | 0.031667 | 0.968333 | 0.263773 | 0 | persistent_exact_failure | PUCT FAILURE |
| {"current_player... | 2400 | 2 | 5 | False | 0.0175 | 0.9825 | 0.249142 | 0 | persistent_exact_failure | PUCT FAILURE |
| {"current_player... | 5000 | 2 | 5 | False | 0.0122 | 0.9878 | 0.190863 | 0 | persistent_exact_failure | PUCT FAILURE |
| {"current_player... | 64 | 0 | 0 | True | 0.984375 | 0.984375 | 0.0 | 0 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 128 | 0 | 0 | True | 0.992188 | 0.992188 | 0.0 | 0 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 256 | 0 | 0 | True | 0.992188 | 0.992188 | 0.0 | 0 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 384 | 0 | 0 | True | 0.992188 | 0.992188 | 0.0 | 0 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 1200 | 0 | 0 | True | 0.995 | 0.995 | 0.0 | 0 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 64 | 3 | 3 | True | 0.875 | 0.875 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 128 | 3 | 3 | True | 0.90625 | 0.90625 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 256 | 3 | 3 | True | 0.933594 | 0.933594 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 384 | 3 | 3 | True | 0.945312 | 0.945312 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 1200 | 3 | 3 | True | 0.968333 | 0.968333 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 64 | 0 | 0 | True | 0.984375 | 0.984375 | 0.0 | 0 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 128 | 0 | 0 | True | 0.992188 | 0.992188 | 0.0 | 0 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 256 | 0 | 0 | True | 0.996094 | 0.996094 | 0.0 | 0 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 384 | 0 | 0 | True | 0.997396 | 0.997396 | 0.0 | 0 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 1200 | 0 | 0 | True | 0.998333 | 0.998333 | 0.0 | 0 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 64 | 1 | 1 | True | 0.9375 | 0.9375 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 128 | 1 | 1 | True | 0.953125 | 0.953125 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 256 | 1 | 1 | True | 0.964844 | 0.964844 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 384 | 1 | 1 | True | 0.96875 | 0.96875 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 1200 | 1 | 1 | True | 0.9825 | 0.9825 | 0.0 | 1 | exact_clean_control | deterministic PUCT baseline |
| {"current_player... | 64 | 3 | 1 | False | 0.015625 | 0.984375 | 0.184726 | 1 | low_budget_exact_failure | PUCT FAILURE |
| {"current_player... | 128 | 3 | 1 | False | 0.351562 | 0.648438 | -1.021642 | 1 | low_budget_exact_failure | PUCT FAILURE |
| {"current_player... | 256 | 3 | 3 | True | 0.675781 | 0.675781 | 0.0 | 1 | low_budget_exact_failure | deterministic PUCT baseline |
| {"current_player... | 384 | 3 | 3 | True | 0.783854 | 0.783854 | 0.0 | 1 | low_budget_exact_failure | deterministic PUCT baseline |
| {"current_player... | 1200 | 3 | 3 | True | 0.930833 | 0.930833 | 0.0 | 1 | low_budget_exact_failure | deterministic PUCT baseline |
| {"current_player... | 64 | 4 | 1 | False | 0.0 | 1.0 | -0.01555 | 1 | persistent_exact_failure | PUCT FAILURE |
| {"current_player... | 128 | 4 | 1 | False | 0.0 | 1.0 | -0.005468 | 1 | persistent_exact_failure | PUCT FAILURE |
| {"current_player... | 256 | 4 | 1 | False | 0.0 | 1.0 | -0.002734 | 1 | persistent_exact_failure | PUCT FAILURE |
| ... and 444 more rows | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

## 7. Neural value rank scan

| candidate_hash | exact_optimal_move | neural_best_move | neural_best_is_exact_optimal | exact_optimal_child_neural_value | neural_best_child_exact_value | value_rank_error | sign_error | notes |
|---|---|---|---|---|---|---|---|---|
| {"current_player... | 2 | 4 | False | -0.383965 | -1.0 | True | True | neural prefers wrong child |
| {"current_player... | 5 | 5 | True | 0.146777 | 0.0 | False | False | neural agrees with exact |
| {"current_player... | 3 | 5 | False | -0.389206 | -1.0 | True | False | neural prefers wrong child |
| {"current_player... | 0 | 2 | False | -0.295268 | 0.0 | True | False | neural prefers wrong child |
| {"current_player... | 5 | 5 | True | -0.006068 | 1.0 | False | False | neural agrees with exact |
| {"current_player... | 5 | 5 | True | 0.390392 | 1.0 | False | False | neural agrees with exact |
| {"current_player... | 5 | 5 | True | -0.172594 | 1.0 | False | False | neural agrees with exact |
| {"current_player... | 3 | 1 | False | -0.341648 | -1.0 | True | False | neural prefers wrong child |
| {"current_player... | 2 | 2 | True | -0.117187 | 1.0 | False | False | neural agrees with exact |
| {"current_player... | 2 | 2 | True | 0.125029 | 1.0 | False | False | neural agrees with exact |
| {"current_player... | 0 | 0 | True | 0.017276 | 1.0 | False | False | neural agrees with exact |
| {"current_player... | 3 | 3 | True | -0.219676 | 1.0 | False | False | neural agrees with exact |
| {"current_player... | 0 | 0 | True | -0.175728 | 1.0 | False | False | neural agrees with exact |
| {"current_player... | 1 | 1 | True | -0.005144 | 1.0 | False | False | neural agrees with exact |
| {"current_player... | 3 | 1 | False | -0.100153 | -1.0 | True | False | neural prefers wrong child |
| {"current_player... | 4 | 4 | True | -0.315468 | 1.0 | False | False | neural agrees with exact |
| {"current_player... | 0 | 0 | True | 0.102191 | 1.0 | False | False | neural agrees with exact |
| {"current_player... | 4 | 4 | True | 0.201506 | 1.0 | False | False | neural agrees with exact |
| {"current_player... | 4 | 4 | True | 0.284909 | 0.0 | False | False | neural agrees with exact |
| {"current_player... | 5 | 5 | True | 0.131153 | 1.0 | False | True | neural agrees with exact |
| {"current_player... | 4 | 2 | False | 0.023404 | 0.0 | True | False | neural prefers wrong child |
| {"current_player... | 1 | 1 | True | -0.073993 | 1.0 | False | False | neural agrees with exact |
| {"current_player... | 0 | 0 | True | -0.212941 | 1.0 | False | False | neural agrees with exact |
| {"current_player... | 4 | 3 | False | -0.021845 | -1.0 | True | False | neural prefers wrong child |
| {"current_player... | 4 | 2 | False | -0.087578 | 0.0 | True | False | neural prefers wrong child |
| {"current_player... | 0 | 0 | True | 0.027798 | 1.0 | False | False | neural agrees with exact |
| {"current_player... | 3 | 3 | True | 0.248172 | 1.0 | False | False | neural agrees with exact |
| {"current_player... | 0 | 1 | False | -0.117187 | -1.0 | True | False | neural prefers wrong child |
| {"current_player... | 4 | 4 | True | -0.100892 | 1.0 | False | False | neural agrees with exact |
| {"current_player... | 5 | 5 | True | -0.079353 | 0.0 | False | False | neural agrees with exact |
| {"current_player... | 3 | 0 | False | -0.259929 | 0.0 | True | False | neural prefers wrong child |
| {"current_player... | 0 | 2 | False | -0.302865 | -1.0 | True | False | neural prefers wrong child |
| {"current_player... | 0 | 0 | True | -0.247356 | 1.0 | False | False | neural agrees with exact |
| {"current_player... | 4 | 4 | True | -0.015209 | 1.0 | False | False | neural agrees with exact |
| {"current_player... | 3 | 3 | True | -0.145048 | 1.0 | False | False | neural agrees with exact |
| {"current_player... | 5 | 5 | True | 0.022819 | 1.0 | False | False | neural agrees with exact |
| {"current_player... | 4 | 4 | True | 0.08806 | 1.0 | False | False | neural agrees with exact |
| {"current_player... | 3 | 3 | True | 0.032611 | 1.0 | False | False | neural agrees with exact |
| {"current_player... | 2 | 0 | False | 0.042248 | -1.0 | True | False | neural prefers wrong child |
| {"current_player... | 3 | 3 | True | -0.030534 | 1.0 | False | False | neural agrees with exact |
| {"current_player... | 1 | 1 | True | -0.166705 | 1.0 | False | False | neural agrees with exact |
| {"current_player... | 4 | 4 | True | -0.001406 | 1.0 | False | True | neural agrees with exact |
| {"current_player... | 3 | 5 | False | -0.095022 | 0.0 | True | False | neural prefers wrong child |
| {"current_player... | 0 | 4 | False | -0.014352 | -1.0 | True | False | neural prefers wrong child |
| {"current_player... | 0 | 3 | False | -0.437654 | -1.0 | True | False | neural prefers wrong child |
| {"current_player... | 0 | 0 | True | 0.013746 | 1.0 | False | False | neural agrees with exact |
| {"current_player... | 3 | 5 | False | -0.083362 | -1.0 | True | False | neural prefers wrong child |
| {"current_player... | 1 | 2 | False | -0.179479 | -1.0 | True | False | neural prefers wrong child |
| {"current_player... | 1 | 1 | True | -0.05117 | 0.0 | False | False | neural agrees with exact |
| {"current_player... | 0 | 4 | False | -0.220131 | 0.0 | True | False | neural prefers wrong child |
| ... and 50 more rows | ... | ... | ... | ... | ... | ... | ... | ... | ... |

## 8. Target/control/holdout split

| candidate_hash | assigned_role | failure_class | exact_signal_class | puct_failure_budget | value_rank_error | reason | notes |
|---|---|---|---|---|---|---|---|
| {"current_player... | target_candidate | exact_clean_control | exact_unique_clear_margin | value_only | True | neural value rank error even though PUCT passes |  |
| {"current_player... | preservation_control | exact_clean_control | exact_unique_clear_margin | None | False | preserves under PUCT |  |
| {"current_player... | target_candidate | exact_clean_control | exact_unique_clear_margin | value_only | True | neural value rank error even though PUCT passes |  |
| {"current_player... | low_budget_diagnostic_only | low_budget_exact_failure | exact_unique_clear_margin | 64 | True | low-budget only failure |  |
| {"current_player... | preservation_control | exact_clean_control | exact_unique_clear_margin | None | False | preserves under PUCT |  |
| {"current_player... | preservation_control | exact_clean_control | exact_unique_clear_margin | None | False | preserves under PUCT |  |
| {"current_player... | low_budget_diagnostic_only | low_budget_exact_failure | exact_unique_clear_margin | 64 | False | low-budget only failure |  |
| {"current_player... | target_candidate | exact_clean_control | exact_unique_clear_margin | value_only | True | neural value rank error even though PUCT passes |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | target_candidate | persistent_exact_failure | exact_unique_clear_margin | 128 | False | PUCT fails persistently at mid-high budget |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | low_budget_diagnostic_only | low_budget_exact_failure | exact_unique_clear_margin | 64 | True | low-budget only failure |  |
| {"current_player... | target_candidate | persistent_exact_failure | exact_unique_clear_margin | 64 | False | PUCT fails persistently at mid-high budget |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | low_budget_diagnostic_only | low_budget_exact_failure | exact_unique_clear_margin | 64 | True | low-budget only failure |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | target_candidate | exact_clean_control | exact_unique_clear_margin | value_only | True | neural value rank error even though PUCT passes |  |
| {"current_player... | target_candidate | persistent_exact_failure | exact_unique_clear_margin | 64 | True | PUCT fails persistently at mid-high budget |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | target_candidate | exact_clean_control | exact_unique_clear_margin | value_only | True | neural value rank error even though PUCT passes |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | low_budget_diagnostic_only | low_budget_exact_failure | exact_unique_clear_margin | 64 | False | low-budget only failure |  |
| {"current_player... | target_candidate | persistent_exact_failure | exact_unique_clear_margin | 64 | True | PUCT fails persistently at mid-high budget |  |
| {"current_player... | low_budget_diagnostic_only | low_budget_exact_failure | exact_unique_clear_margin | 256 | True | low-budget only failure |  |
| {"current_player... | target_candidate | medium_budget_exact_failure | exact_unique_clear_margin | 64 | False | PUCT fails at medium budget, recovers |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | target_candidate | exact_clean_control | exact_unique_clear_margin | value_only | True | neural value rank error even though PUCT passes |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | target_candidate | medium_budget_exact_failure | exact_unique_clear_margin | 64 | True | PUCT fails at medium budget, recovers |  |
| {"current_player... | target_candidate | exact_clean_control | exact_unique_clear_margin | value_only | True | neural value rank error even though PUCT passes |  |
| {"current_player... | target_candidate | exact_clean_control | exact_unique_clear_margin | value_only | True | neural value rank error even though PUCT passes |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | target_candidate | exact_clean_control | exact_unique_clear_margin | value_only | True | neural value rank error even though PUCT passes |  |
| {"current_player... | low_budget_diagnostic_only | low_budget_exact_failure | exact_unique_clear_margin | 64 | True | low-budget only failure |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | target_candidate | exact_clean_control | exact_unique_clear_margin | value_only | True | neural value rank error even though PUCT passes |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | target_candidate | exact_clean_control | exact_unique_clear_margin | value_only | True | neural value rank error even though PUCT passes |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | target_candidate | exact_clean_control | exact_unique_clear_margin | value_only | True | neural value rank error even though PUCT passes |  |
| {"current_player... | low_budget_diagnostic_only | low_budget_exact_failure | exact_unique_clear_margin | 64 | False | low-budget only failure |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | target_candidate | exact_clean_control | exact_unique_clear_margin | value_only | True | neural value rank error even though PUCT passes |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | target_candidate | exact_clean_control | exact_unique_clear_margin | value_only | True | neural value rank error even though PUCT passes |  |
| {"current_player... | target_candidate | medium_budget_exact_failure | exact_unique_clear_margin | 64 | False | PUCT fails at medium budget, recovers |  |
| {"current_player... | target_candidate | exact_clean_control | exact_unique_clear_margin | value_only | True | neural value rank error even though PUCT passes |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | low_budget_diagnostic_only | low_budget_exact_failure | exact_unique_clear_margin | 64 | True | low-budget only failure |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | target_candidate | persistent_exact_failure | exact_unique_clear_margin | 64 | False | PUCT fails persistently at mid-high budget |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| {"current_player... | holdout_candidate | exact_clean_control | exact_unique_clear_margin | None | False | holdout |  |
| ... and 20 more rows | ... | ... | ... | ... | ... | ... | ... | ... |

## 9. Final targetability decision

| classification | target_candidate_count | preservation_control_count | holdout_count | low_budget_diagnostic_count | excluded_count | dominant_mechanism | next_action |
|---|---|---|---|---|---|---|---|
| harder_tablebase_exact_target_family_ready | 32 | 3 | 56 | 9 | 0 | search_failure_at_medium_high_budget | run tablebase-backed local search/value diagnostics on selected hard rows before any artifact/training |

## 10. Exactly one recommended next action

Recommendation: **run tablebase-backed local search/value diagnostics on selected hard rows before any artifact/training**

### Acceptance criteria

- No training was run.
- No arena was run.
- No model was promoted.
- Active references were not mutated.
- Exhausted families were excluded from selection.
- Teacher-conflict filtering was used.
- Selected rows are metadata candidates only, not replay artifacts.
- Final report recommends exactly one next branch.