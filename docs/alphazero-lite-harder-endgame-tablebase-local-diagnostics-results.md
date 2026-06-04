# Harder Endgame Tablebase Local Diagnostics — Results

**Date:** 2026-06-04
**Family:** `harder_fresh_endgame_tablebase`
**Script:** `ml/alphazero_lite/run_harder_endgame_tablebase_local_diagnostics.py`

## 1. Context

- Run classification: exact_tablebase_diagnostic_artifact_ready
- Selected family: harder_fresh_endgame_tablebase (from PR #73)
- Current artifact: storage/ai/alphazero_lite/current
- Active references not mutated
- Selected rows loaded: 91
- Valid rows: 91
- Invalid rows: 0
- No training was run.
- No arena was run.
- No model was promoted.
- Active references were not mutated.

## 2. Why PR #73 found a promising exact-teacher family

PR #73 mined harder endgame positions with four strategies: deeper self-play, low-budget PUCT failure prescreening, adversarial near-threshold sampling, and PUCT-vs-tablebase disagreement sampling. It found 32 target candidates with exact-tablebase-unique optimal moves, 7 persistent failures at 384+1200, 5 medium-budget failures, and 32 value rank errors. All rows are metadata candidates only — no replay artifacts were created.

This diagnostics run determines the causal mechanism for each failure: neural value rank error, PUCT selection pressure, child PUCT error, policy prior underweighting, backup/perspective issue, or low-budget noise.

## 3. Row validation

| candidate_id | assigned_role_from_pr73 | legal_moves | tablebase_available | unique_optimal | exhausted_overlap | teacher_conflict | valid | notes |
|---|---|---|---|---|---|---|---|---|
| harder_17_None | target_candidate | 2 | true | true | false | false | true | ok |
| harder_18_None | preservation_control | 2 | true | true | false | false | true | ok |
| harder_19_None | target_candidate | 2 | true | true | false | false | true | ok |
| harder_21_None | preservation_control | 2 | true | true | false | false | true | ok |
| harder_22_None | preservation_control | 2 | true | true | false | false | true | ok |
| harder_24_None | target_candidate | 2 | true | true | false | false | true | ok |
| harder_25_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_26_None | target_candidate | 2 | true | true | false | false | true | ok |
| harder_27_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_28_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_29_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_30_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_32_None | target_candidate | 2 | true | true | false | false | true | ok |
| harder_33_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_34_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_35_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_36_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_38_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_39_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_40_None | target_candidate | 2 | true | true | false | false | true | ok |
| harder_41_None | target_candidate | 2 | true | true | false | false | true | ok |
| harder_42_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_43_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_44_None | target_candidate | 2 | true | true | false | false | true | ok |
| harder_45_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_47_None | target_candidate | 2 | true | true | false | false | true | ok |
| harder_49_None | target_candidate | 2 | true | true | false | false | true | ok |
| harder_50_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_51_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_52_None | holdout_candidate | 3 | true | true | false | false | true | ok |
| harder_53_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_54_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_55_None | target_candidate | 2 | true | true | false | false | true | ok |
| harder_56_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_57_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_58_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_59_None | target_candidate | 2 | true | true | false | false | true | ok |
| harder_60_None | target_candidate | 2 | true | true | false | false | true | ok |
| harder_61_None | target_candidate | 2 | true | true | false | false | true | ok |
| harder_62_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_63_None | target_candidate | 2 | true | true | false | false | true | ok |
| harder_65_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_66_None | target_candidate | 2 | true | true | false | false | true | ok |
| harder_67_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_68_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_69_None | target_candidate | 2 | true | true | false | false | true | ok |
| harder_70_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_71_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_72_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_73_None | target_candidate | 2 | true | true | false | false | true | ok |
| harder_75_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_76_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_77_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_78_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_79_None | holdout_candidate | 3 | true | true | false | false | true | ok |
| harder_80_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_81_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_82_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_83_None | target_candidate | 2 | true | true | false | false | true | ok |
| harder_84_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_85_None | target_candidate | 2 | true | true | false | false | true | ok |
| harder_86_None | target_candidate | 3 | true | true | false | false | true | ok |
| harder_87_None | target_candidate | 2 | true | true | false | false | true | ok |
| harder_88_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_89_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_91_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_92_None | target_candidate | 2 | true | true | false | false | true | ok |
| harder_93_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_94_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_95_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_96_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_97_None | target_candidate | 2 | true | true | false | false | true | ok |
| harder_98_None | holdout_candidate | 3 | true | true | false | false | true | ok |
| harder_99_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_100_None | target_candidate | 3 | true | true | false | false | true | ok |
| harder_101_None | target_candidate | 2 | true | true | false | false | true | ok |
| harder_102_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_103_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_104_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_105_None | target_candidate | 2 | true | true | false | false | true | ok |
| harder_106_None | target_candidate | 2 | true | true | false | false | true | ok |
| harder_107_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_108_None | holdout_candidate | 3 | true | true | false | false | true | ok |
| harder_109_None | holdout_candidate | 3 | true | true | false | false | true | ok |
| harder_110_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_111_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_112_None | holdout_candidate | 2 | true | true | false | false | true | ok |
| harder_113_None | target_candidate | 2 | true | true | false | false | true | ok |
| harder_114_None | target_candidate | 2 | true | true | false | false | true | ok |
| harder_115_None | target_candidate | 2 | true | true | false | false | true | ok |
| harder_116_None | target_candidate | 5 | true | true | false | false | true | ok |

## 4. Exact tablebase re-enumeration

| candidate_id | root_value | optimal_move | second_best_move | best_minus_second_best | forced_win_or_loss | exact_signal_class | notes |
|---|---|---|---|---|---|---|---|
| harder_17_None | 1.000000 | 2 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_18_None | 0.000000 | 5 | -1.0 | 1.000000 | false | exact_unique_clear_margin | ok |
| harder_19_None | 0.000000 | 3 | -1.0 | 1.000000 | false | exact_unique_clear_margin | ok |
| harder_21_None | 1.000000 | 5 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_22_None | 1.000000 | 5 | 0.0 | 1.000000 | true | exact_unique_clear_margin | ok |
| harder_24_None | 1.000000 | 3 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_25_None | 1.000000 | 2 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_26_None | 1.000000 | 2 | 0.0 | 1.000000 | true | exact_unique_clear_margin | ok |
| harder_27_None | 1.000000 | 0 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_28_None | 1.000000 | 3 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_29_None | 1.000000 | 0 | 0.0 | 1.000000 | true | exact_unique_clear_margin | ok |
| harder_30_None | 1.000000 | 1 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_32_None | 1.000000 | 4 | 0.0 | 1.000000 | true | exact_unique_clear_margin | ok |
| harder_33_None | 1.000000 | 0 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_34_None | 1.000000 | 4 | 0.0 | 1.000000 | true | exact_unique_clear_margin | ok |
| harder_35_None | 0.000000 | 4 | -1.0 | 1.000000 | false | exact_unique_clear_margin | ok |
| harder_36_None | 1.000000 | 5 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_38_None | 1.000000 | 1 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_39_None | 1.000000 | 0 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_40_None | 1.000000 | 4 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_41_None | 1.000000 | 4 | 0.0 | 1.000000 | true | exact_unique_clear_margin | ok |
| harder_42_None | 1.000000 | 0 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_43_None | 1.000000 | 3 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_44_None | 1.000000 | 0 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_45_None | 1.000000 | 4 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_47_None | 1.000000 | 3 | 0.0 | 1.000000 | true | exact_unique_clear_margin | ok |
| harder_49_None | 1.000000 | 0 | 0.0 | 1.000000 | true | exact_unique_clear_margin | ok |
| harder_50_None | 1.000000 | 4 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_51_None | 1.000000 | 3 | 0.0 | 1.000000 | true | exact_unique_clear_margin | ok |
| harder_52_None | 1.000000 | 5 | 0.0 | 1.000000 | true | exact_unique_clear_margin | ok |
| harder_53_None | 1.000000 | 4 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_54_None | 1.000000 | 3 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_55_None | 1.000000 | 2 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_56_None | 1.000000 | 3 | 0.0 | 1.000000 | true | exact_unique_clear_margin | ok |
| harder_57_None | 1.000000 | 1 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_58_None | 1.000000 | 4 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_59_None | 1.000000 | 3 | 0.0 | 1.000000 | true | exact_unique_clear_margin | ok |
| harder_60_None | 1.000000 | 0 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_61_None | 1.000000 | 0 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_62_None | 1.000000 | 0 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_63_None | 1.000000 | 3 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_65_None | 0.000000 | 1 | -1.0 | 1.000000 | false | exact_unique_clear_margin | ok |
| harder_66_None | 1.000000 | 0 | 0.0 | 1.000000 | true | exact_unique_clear_margin | ok |
| harder_67_None | 1.000000 | 5 | 0.0 | 1.000000 | true | exact_unique_clear_margin | ok |
| harder_68_None | 1.000000 | 3 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_69_None | 1.000000 | 5 | 0.0 | 1.000000 | true | exact_unique_clear_margin | ok |
| harder_70_None | 1.000000 | 4 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_71_None | 1.000000 | 2 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_72_None | 1.000000 | 4 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_73_None | 1.000000 | 3 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_75_None | 1.000000 | 0 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_76_None | 1.000000 | 4 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_77_None | 1.000000 | 3 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_78_None | 1.000000 | 2 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_79_None | 1.000000 | 0 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_80_None | 1.000000 | 3 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_81_None | 1.000000 | 2 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_82_None | 1.000000 | 2 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_83_None | 1.000000 | 4 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_84_None | 1.000000 | 2 | 0.0 | 1.000000 | true | exact_unique_clear_margin | ok |
| harder_85_None | 1.000000 | 2 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_86_None | 1.000000 | 3 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_87_None | 1.000000 | 1 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_88_None | 1.000000 | 1 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_89_None | 1.000000 | 0 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_91_None | 1.000000 | 0 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_92_None | 1.000000 | 2 | 0.0 | 1.000000 | true | exact_unique_clear_margin | ok |
| harder_93_None | 1.000000 | 4 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_94_None | 1.000000 | 1 | 0.0 | 1.000000 | true | exact_unique_clear_margin | ok |
| harder_95_None | 1.000000 | 4 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_96_None | 1.000000 | 4 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_97_None | 0.000000 | 0 | -1.0 | 1.000000 | false | exact_unique_clear_margin | ok |
| harder_98_None | 1.000000 | 2 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_99_None | 1.000000 | 5 | 0.0 | 1.000000 | true | exact_unique_clear_margin | ok |
| harder_100_None | 0.000000 | 0 | -1.0 | 1.000000 | false | exact_unique_clear_margin | ok |
| harder_101_None | 0.000000 | 5 | -1.0 | 1.000000 | false | exact_unique_clear_margin | ok |
| harder_102_None | 1.000000 | 1 | 0.0 | 1.000000 | true | exact_unique_clear_margin | ok |
| harder_103_None | 1.000000 | 2 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_104_None | 1.000000 | 5 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_105_None | 1.000000 | 3 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_106_None | 1.000000 | 4 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_107_None | 1.000000 | 4 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_108_None | 1.000000 | 2 | 0.0 | 1.000000 | true | exact_unique_clear_margin | ok |
| harder_109_None | 0.000000 | 5 | -1.0 | 1.000000 | false | exact_unique_clear_margin | ok |
| harder_110_None | 0.000000 | 0 | -1.0 | 1.000000 | false | exact_unique_clear_margin | ok |
| harder_111_None | 1.000000 | 2 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_112_None | 1.000000 | 4 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_113_None | 1.000000 | 1 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_114_None | 1.000000 | 0 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_115_None | 1.000000 | 3 | -1.0 | 2.000000 | true | exact_unique_clear_margin | ok |
| harder_116_None | 1.000000 | 3 | 0.0 | 1.000000 | true | exact_unique_clear_margin | ok |

## 5. Detailed PUCT budget sweep

- Rows with persistent failures (384+1200): 7
- Rows with medium-budget failures (384 only): 5
- Rows with low-budget failures (32/64/128/256 only): 6
- Clean controls (all budgets pass): 73

| candidate_id | budget | optimal_move | selected_move | selected_is_optimal | optimal_visit_share | selected_visit_share | selected_minus_optimal_q_margin | optimal_policy_rank | first_budget_optimal_selected | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| harder_17_None | 32 | 2 | 2 | true | 0.968750 | 0.968750 | 0.000000 | 0 | true | deterministic PUCT baseline |
| harder_17_None | 64 | 2 | 2 | true | 0.984375 | 0.984375 | 0.000000 | 0 | true | deterministic PUCT baseline |
| harder_17_None | 128 | 2 | 2 | true | 0.984375 | 0.984375 | 0.000000 | 0 | true | deterministic PUCT baseline |
| harder_17_None | 256 | 2 | 2 | true | 0.988281 | 0.988281 | 0.000000 | 0 | true | deterministic PUCT baseline |
| harder_17_None | 384 | 2 | 2 | true | 0.992188 | 0.992188 | 0.000000 | 0 | true | deterministic PUCT baseline |
| harder_17_None | 768 | 2 | 2 | true | 0.993490 | 0.993490 | 0.000000 | 0 | true | deterministic PUCT baseline |
| harder_17_None | 1200 | 2 | 2 | true | 0.995000 | 0.995000 | 0.000000 | 0 | true | deterministic PUCT baseline |
| harder_17_None | 2400 | 2 | 2 | true | 0.996250 | 0.996250 | 0.000000 | 0 | true | deterministic PUCT baseline |
| harder_17_None | 5000 | 2 | 2 | true | 0.997200 | 0.997200 | 0.000000 | 0 | true | deterministic PUCT baseline |
| harder_18_None | 32 | 5 | 5 | true | 0.968750 | 0.968750 | 0.000000 | 0 | true | deterministic PUCT baseline |
| harder_18_None | 64 | 5 | 5 | true | 0.984375 | 0.984375 | 0.000000 | 0 | true | deterministic PUCT baseline |
| harder_18_None | 128 | 5 | 5 | true | 0.992188 | 0.992188 | 0.000000 | 0 | true | deterministic PUCT baseline |
| harder_18_None | 256 | 5 | 5 | true | 0.984375 | 0.984375 | 0.000000 | 0 | true | deterministic PUCT baseline |
| harder_18_None | 384 | 5 | 5 | true | 0.989583 | 0.989583 | 0.000000 | 0 | true | deterministic PUCT baseline |
| harder_18_None | 768 | 5 | 5 | true | 0.994792 | 0.994792 | 0.000000 | 0 | true | deterministic PUCT baseline |
| harder_18_None | 1200 | 5 | 5 | true | 0.995000 | 0.995000 | 0.000000 | 0 | true | deterministic PUCT baseline |
| harder_18_None | 2400 | 5 | 5 | true | 0.996667 | 0.996667 | 0.000000 | 0 | true | deterministic PUCT baseline |
| harder_18_None | 5000 | 5 | 5 | true | 0.997600 | 0.997600 | 0.000000 | 0 | true | deterministic PUCT baseline |
| harder_19_None | 32 | 3 | 3 | true | 0.500000 | 0.500000 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_19_None | 64 | 3 | 3 | true | 0.750000 | 0.750000 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_19_None | 128 | 3 | 3 | true | 0.875000 | 0.875000 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_19_None | 256 | 3 | 3 | true | 0.937500 | 0.937500 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_19_None | 384 | 3 | 3 | true | 0.958333 | 0.958333 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_19_None | 768 | 3 | 3 | true | 0.977865 | 0.977865 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_19_None | 1200 | 3 | 3 | true | 0.981667 | 0.981667 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_19_None | 2400 | 3 | 3 | true | 0.987083 | 0.987083 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_19_None | 5000 | 3 | 3 | true | 0.990800 | 0.990800 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_21_None | 32 | 5 | 5 | true | 0.812500 | 0.812500 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_21_None | 64 | 5 | 5 | true | 0.875000 | 0.875000 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_21_None | 128 | 5 | 5 | true | 0.906250 | 0.906250 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_21_None | 256 | 5 | 5 | true | 0.933594 | 0.933594 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_21_None | 384 | 5 | 5 | true | 0.945312 | 0.945312 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_21_None | 768 | 5 | 5 | true | 0.960938 | 0.960938 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_21_None | 1200 | 5 | 5 | true | 0.969167 | 0.969167 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_21_None | 2400 | 5 | 5 | true | 0.977917 | 0.977917 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_21_None | 5000 | 5 | 5 | true | 0.984600 | 0.984600 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_22_None | 32 | 5 | 5 | true | 0.906250 | 0.906250 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_22_None | 64 | 5 | 5 | true | 0.921875 | 0.921875 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_22_None | 128 | 5 | 5 | true | 0.937500 | 0.937500 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_22_None | 256 | 5 | 5 | true | 0.957031 | 0.957031 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_22_None | 384 | 5 | 5 | true | 0.963542 | 0.963542 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_22_None | 768 | 5 | 5 | true | 0.973958 | 0.973958 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_22_None | 1200 | 5 | 5 | true | 0.979167 | 0.979167 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_22_None | 2400 | 5 | 5 | true | 0.985000 | 0.985000 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_22_None | 5000 | 5 | 5 | true | 0.989600 | 0.989600 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_24_None | 32 | 3 | 1 | false | 0.281250 | 0.718750 | -0.360632 | 1 | false | deterministic PUCT baseline |
| harder_24_None | 64 | 3 | 3 | true | 0.640625 | 0.640625 | 0.000000 | 1 | false | deterministic PUCT baseline |
| harder_24_None | 128 | 3 | 3 | true | 0.820312 | 0.820312 | 0.000000 | 1 | false | deterministic PUCT baseline |
| harder_24_None | 256 | 3 | 3 | true | 0.910156 | 0.910156 | 0.000000 | 1 | false | deterministic PUCT baseline |
| harder_24_None | 384 | 3 | 3 | true | 0.940104 | 0.940104 | 0.000000 | 1 | false | deterministic PUCT baseline |
| harder_24_None | 768 | 3 | 3 | true | 0.967448 | 0.967448 | 0.000000 | 1 | false | deterministic PUCT baseline |
| harder_24_None | 1200 | 3 | 3 | true | 0.974167 | 0.974167 | 0.000000 | 1 | false | deterministic PUCT baseline |
| harder_24_None | 2400 | 3 | 3 | true | 0.981667 | 0.981667 | 0.000000 | 1 | false | deterministic PUCT baseline |
| harder_24_None | 5000 | 3 | 3 | true | 0.987200 | 0.987200 | 0.000000 | 1 | false | deterministic PUCT baseline |
| harder_25_None | 32 | 2 | 2 | true | 0.875000 | 0.875000 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_25_None | 64 | 2 | 2 | true | 0.906250 | 0.906250 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_25_None | 128 | 2 | 2 | true | 0.929688 | 0.929688 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_25_None | 256 | 2 | 2 | true | 0.949219 | 0.949219 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_25_None | 384 | 2 | 2 | true | 0.958333 | 0.958333 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_25_None | 768 | 2 | 2 | true | 0.971354 | 0.971354 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_25_None | 1200 | 2 | 2 | true | 0.976667 | 0.976667 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_25_None | 2400 | 2 | 2 | true | 0.983333 | 0.983333 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_25_None | 5000 | 2 | 2 | true | 0.988400 | 0.988400 | 0.000000 | 1 | true | deterministic PUCT baseline |
| harder_26_None | 32 | 2 | 2 | true | 0.968750 | 0.968750 | 0.000000 | 0 | true | deterministic PUCT baseline |
| harder_26_None | 64 | 2 | 2 | true | 0.593750 | 0.593750 | 0.000000 | 0 | true | deterministic PUCT baseline |
| harder_26_None | 128 | 2 | 5 | false | 0.296875 | 0.703125 | 0.355541 | 0 | true | deterministic PUCT baseline |
| harder_26_None | 256 | 2 | 5 | false | 0.148438 | 0.851562 | 0.298687 | 0 | true | deterministic PUCT baseline |
| harder_26_None | 384 | 2 | 5 | false | 0.098958 | 0.901042 | 0.282644 | 0 | true | deterministic PUCT baseline |
| harder_26_None | 768 | 2 | 5 | false | 0.049479 | 0.950521 | 0.269153 | 0 | true | deterministic PUCT baseline |
| harder_26_None | 1200 | 2 | 5 | false | 0.031667 | 0.968333 | 0.263773 | 0 | true | deterministic PUCT baseline |
| harder_26_None | 2400 | 2 | 5 | false | 0.017500 | 0.982500 | 0.249142 | 0 | true | deterministic PUCT baseline |
| harder_26_None | 5000 | 2 | 5 | false | 0.012200 | 0.987800 | 0.190863 | 0 | true | deterministic PUCT baseline |
| harder_27_None | 32 | 0 | 0 | true | 0.968750 | 0.968750 | 0.000000 | 0 | true | deterministic PUCT baseline |
| harder_27_None | 64 | 0 | 0 | true | 0.984375 | 0.984375 | 0.000000 | 0 | true | deterministic PUCT baseline |
| harder_27_None | 128 | 0 | 0 | true | 0.992188 | 0.992188 | 0.000000 | 0 | true | deterministic PUCT baseline |
| harder_27_None | 256 | 0 | 0 | true | 0.992188 | 0.992188 | 0.000000 | 0 | true | deterministic PUCT baseline |
| harder_27_None | 384 | 0 | 0 | true | 0.992188 | 0.992188 | 0.000000 | 0 | true | deterministic PUCT baseline |
| harder_27_None | 768 | 0 | 0 | true | 0.994792 | 0.994792 | 0.000000 | 0 | true | deterministic PUCT baseline |
| harder_27_None | 1200 | 0 | 0 | true | 0.995000 | 0.995000 | 0.000000 | 0 | true | deterministic PUCT baseline |
| harder_27_None | 2400 | 0 | 0 | true | 0.996667 | 0.996667 | 0.000000 | 0 | true | deterministic PUCT baseline |
| ... and 739 more rows | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

## 6. Root policy-prior audit

| candidate_id | optimal_move | selected_move | optimal_policy_probability | selected_policy_probability | optimal_policy_rank | optimal_prior_minus_selected_prior | prior_classification | notes |
|---|---|---|---|---|---|---|---|---|
| harder_17_None | 2 | 2 | 0.823038 | 0.823038 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_18_None | 5 | 5 | 0.840708 | 0.840708 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_19_None | 3 | 3 | 0.475022 | 0.475022 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_21_None | 5 | 5 | 0.125803 | 0.125803 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_22_None | 5 | 5 | 0.376820 | 0.376820 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_24_None | 3 | 3 | 0.266117 | 0.266117 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_25_None | 2 | 2 | 0.261417 | 0.261417 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_26_None | 2 | 5 | 0.675544 | 0.324456 | 0 | 0.351087 | policy_prior_supports_optimal |  |
| harder_27_None | 0 | 0 | 0.856164 | 0.856164 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_28_None | 3 | 3 | 0.116386 | 0.116386 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_29_None | 0 | 0 | 0.943065 | 0.943065 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_30_None | 1 | 1 | 0.438033 | 0.438033 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_32_None | 4 | 1 | 0.002874 | 0.997126 | 1 | -0.994251 | policy_prior_neutral |  |
| harder_33_None | 0 | 0 | 0.616659 | 0.616659 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_34_None | 4 | 4 | 0.366460 | 0.366460 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_35_None | 4 | 4 | 0.095553 | 0.095553 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_36_None | 5 | 5 | 0.561271 | 0.561271 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_38_None | 1 | 1 | 0.383171 | 0.383171 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_39_None | 0 | 0 | 0.494125 | 0.494125 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_40_None | 4 | 4 | 0.782336 | 0.782336 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_41_None | 4 | 2 | 0.098643 | 0.901357 | 1 | -0.802715 | policy_prior_neutral |  |
| harder_42_None | 0 | 0 | 0.719888 | 0.719888 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_43_None | 3 | 3 | 0.374916 | 0.374916 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_44_None | 0 | 0 | 0.046328 | 0.046328 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_45_None | 4 | 4 | 0.076901 | 0.076901 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_47_None | 3 | 0 | 0.367683 | 0.632317 | 1 | -0.264635 | policy_prior_neutral |  |
| harder_49_None | 0 | 0 | 0.149314 | 0.149314 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_50_None | 4 | 4 | 0.372200 | 0.372200 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_51_None | 3 | 3 | 0.807687 | 0.807687 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_52_None | 5 | 5 | 0.281084 | 0.281084 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_53_None | 4 | 4 | 0.190746 | 0.190746 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_54_None | 3 | 3 | 0.903029 | 0.903029 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_55_None | 2 | 2 | 0.255951 | 0.255951 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_56_None | 3 | 3 | 0.375138 | 0.375138 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_57_None | 1 | 1 | 0.967126 | 0.967126 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_58_None | 4 | 4 | 0.387087 | 0.387087 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_59_None | 3 | 3 | 0.111625 | 0.111625 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_60_None | 0 | 0 | 0.647720 | 0.647720 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_61_None | 0 | 0 | 0.805046 | 0.805046 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_62_None | 0 | 0 | 0.926054 | 0.926054 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_63_None | 3 | 3 | 0.972829 | 0.972829 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_65_None | 1 | 1 | 0.499496 | 0.499496 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_66_None | 0 | 0 | 0.966052 | 0.966052 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_67_None | 5 | 5 | 0.277387 | 0.277387 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_68_None | 3 | 3 | 0.086989 | 0.086989 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_69_None | 5 | 5 | 0.622479 | 0.622479 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_70_None | 4 | 4 | 0.086904 | 0.086904 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_71_None | 2 | 2 | 0.809891 | 0.809891 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_72_None | 4 | 4 | 0.015097 | 0.015097 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_73_None | 3 | 3 | 0.321194 | 0.321194 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_75_None | 0 | 0 | 0.560094 | 0.560094 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_76_None | 4 | 4 | 0.292934 | 0.292934 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_77_None | 3 | 3 | 0.038833 | 0.038833 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_78_None | 2 | 2 | 0.146954 | 0.146954 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_79_None | 0 | 0 | 0.652470 | 0.652470 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_80_None | 3 | 3 | 0.254368 | 0.254368 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_81_None | 2 | 2 | 0.209165 | 0.209165 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_82_None | 2 | 2 | 0.673767 | 0.673767 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_83_None | 4 | 4 | 0.033270 | 0.033270 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_84_None | 2 | 2 | 0.919092 | 0.919092 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_85_None | 2 | 2 | 0.882651 | 0.882651 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_86_None | 3 | 3 | 0.097644 | 0.097644 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_87_None | 1 | 1 | 0.487775 | 0.487775 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_88_None | 1 | 1 | 0.937290 | 0.937290 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_89_None | 0 | 0 | 0.867639 | 0.867639 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_91_None | 0 | 0 | 0.877064 | 0.877064 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_92_None | 2 | 1 | 0.270551 | 0.729449 | 1 | -0.458897 | policy_prior_neutral |  |
| harder_93_None | 4 | 4 | 0.056157 | 0.056157 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_94_None | 1 | 1 | 0.995114 | 0.995114 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_95_None | 4 | 4 | 0.064269 | 0.064269 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_96_None | 4 | 4 | 0.073236 | 0.073236 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_97_None | 0 | 0 | 0.501051 | 0.501051 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_98_None | 2 | 2 | 0.706563 | 0.706563 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_99_None | 5 | 5 | 0.684469 | 0.684469 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_100_None | 0 | 3 | 0.122321 | 0.029335 | 1 | 0.092986 | policy_prior_neutral |  |
| harder_101_None | 5 | 5 | 0.675534 | 0.675534 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_102_None | 1 | 1 | 0.994554 | 0.994554 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_103_None | 2 | 2 | 0.236419 | 0.236419 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_104_None | 5 | 5 | 0.401843 | 0.401843 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_105_None | 3 | 3 | 0.973884 | 0.973884 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_106_None | 4 | 4 | 0.014769 | 0.014769 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_107_None | 4 | 4 | 0.064408 | 0.064408 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_108_None | 2 | 2 | 0.765919 | 0.765919 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_109_None | 5 | 5 | 0.368897 | 0.368897 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_110_None | 0 | 0 | 0.712015 | 0.712015 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_111_None | 2 | 2 | 0.930019 | 0.930019 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_112_None | 4 | 4 | 0.093037 | 0.093037 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_113_None | 1 | 1 | 0.310629 | 0.310629 | 1 | 0.000000 | policy_prior_supports_optimal |  |
| harder_114_None | 0 | 0 | 0.936974 | 0.936974 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_115_None | 3 | 3 | 0.563281 | 0.563281 | 0 | 0.000000 | policy_prior_supports_optimal |  |
| harder_116_None | 3 | 1 | 0.177759 | 0.181979 | 2 | -0.004220 | policy_prior_underweights_optimal |  |

## 7. Neural value rank audit

| candidate_id | exact_optimal_move | neural_best_child_move | neural_best_is_exact_optimal | exact_optimal_child_neural_value | neural_best_child_exact_value | neural_value_rank_error | neural_value_sign_error | value_error_class | notes |
|---|---|---|---|---|---|---|---|---|---|
| harder_17_None | 2 | 4 | false | -0.383965 | -1.000000 | true | true | neural_prefers_nonoptimal_child |  |
| harder_18_None | 5 | 5 | true | 0.146777 | 0.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_19_None | 3 | 5 | false | -0.389206 | -1.000000 | true | false | neural_prefers_nonoptimal_child |  |
| harder_21_None | 5 | 5 | true | -0.006068 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_22_None | 5 | 5 | true | 0.390392 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_24_None | 3 | 1 | false | -0.341648 | -1.000000 | true | false | neural_prefers_nonoptimal_child |  |
| harder_25_None | 2 | 2 | true | -0.117187 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_26_None | 2 | 2 | true | 0.125029 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_27_None | 0 | 0 | true | 0.017276 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_28_None | 3 | 3 | true | -0.219676 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_29_None | 0 | 0 | true | -0.175728 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_30_None | 1 | 1 | true | -0.005144 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_32_None | 4 | 4 | true | -0.315468 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_33_None | 0 | 0 | true | 0.102191 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_34_None | 4 | 4 | true | 0.201506 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_35_None | 4 | 4 | true | 0.284909 | 0.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_36_None | 5 | 5 | true | 0.131153 | 1.000000 | false | true | neural_sign_error |  |
| harder_38_None | 1 | 1 | true | -0.073993 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_39_None | 0 | 0 | true | -0.212941 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_40_None | 4 | 3 | false | -0.021845 | -1.000000 | true | false | neural_prefers_nonoptimal_child |  |
| harder_41_None | 4 | 2 | false | -0.087578 | 0.000000 | true | false | neural_prefers_nonoptimal_child |  |
| harder_42_None | 0 | 0 | true | 0.027798 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_43_None | 3 | 3 | true | 0.248172 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_44_None | 0 | 1 | false | -0.117187 | -1.000000 | true | false | neural_prefers_nonoptimal_child |  |
| harder_45_None | 4 | 4 | true | -0.100892 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_47_None | 3 | 0 | false | -0.259929 | 0.000000 | true | false | neural_prefers_nonoptimal_child |  |
| harder_49_None | 0 | 0 | true | -0.247356 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_50_None | 4 | 4 | true | -0.015209 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_51_None | 3 | 3 | true | -0.145048 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_52_None | 5 | 5 | true | 0.022819 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_53_None | 4 | 4 | true | 0.088060 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_54_None | 3 | 3 | true | 0.032611 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_55_None | 2 | 0 | false | 0.042248 | -1.000000 | true | false | neural_prefers_nonoptimal_child |  |
| harder_56_None | 3 | 3 | true | -0.030534 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_57_None | 1 | 1 | true | -0.166705 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_58_None | 4 | 4 | true | -0.001406 | 1.000000 | false | true | neural_sign_error |  |
| harder_59_None | 3 | 5 | false | -0.095022 | 0.000000 | true | false | neural_prefers_nonoptimal_child |  |
| harder_60_None | 0 | 4 | false | -0.014352 | -1.000000 | true | false | neural_prefers_nonoptimal_child |  |
| harder_61_None | 0 | 3 | false | -0.437654 | -1.000000 | true | false | neural_prefers_nonoptimal_child |  |
| harder_62_None | 0 | 0 | true | 0.013746 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_63_None | 3 | 5 | false | -0.083362 | -1.000000 | true | false | neural_prefers_nonoptimal_child |  |
| harder_65_None | 1 | 1 | true | -0.051170 | 0.000000 | false | false | neural_ok_search_fails |  |
| harder_66_None | 0 | 4 | false | -0.220131 | 0.000000 | true | false | neural_prefers_nonoptimal_child |  |
| harder_67_None | 5 | 5 | true | -0.012440 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_68_None | 3 | 3 | true | 0.150314 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_69_None | 5 | 4 | false | 0.073555 | 0.000000 | true | false | neural_prefers_nonoptimal_child |  |
| harder_70_None | 4 | 4 | true | -0.205575 | 1.000000 | false | true | neural_sign_error |  |
| harder_71_None | 2 | 2 | true | 0.067553 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_72_None | 4 | 4 | true | -0.005144 | 1.000000 | false | true | neural_sign_error |  |
| harder_73_None | 3 | 1 | false | -0.187336 | -1.000000 | true | false | neural_prefers_nonoptimal_child |  |
| harder_75_None | 0 | 0 | true | -0.047068 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_76_None | 4 | 4 | true | 0.049495 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_77_None | 3 | 3 | true | -0.017327 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_78_None | 2 | 2 | true | -0.021064 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_79_None | 0 | 0 | true | 0.170094 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_80_None | 3 | 3 | true | -0.151228 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_81_None | 2 | 2 | true | -0.016589 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_82_None | 2 | 2 | true | -0.167601 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_83_None | 4 | 3 | false | -0.103807 | -1.000000 | true | true | neural_prefers_nonoptimal_child |  |
| harder_84_None | 2 | 2 | true | -0.003539 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_85_None | 2 | 4 | false | -0.040586 | -1.000000 | true | false | neural_prefers_nonoptimal_child |  |
| harder_86_None | 3 | 3 | true | 0.174629 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_87_None | 1 | 2 | false | -0.210410 | -1.000000 | true | false | neural_prefers_nonoptimal_child |  |
| harder_88_None | 1 | 1 | true | 0.132837 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_89_None | 0 | 0 | true | -0.010498 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_91_None | 0 | 0 | true | 0.050319 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_92_None | 2 | 2 | true | -0.097863 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_93_None | 4 | 4 | true | -0.187961 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_94_None | 1 | 1 | true | 0.214105 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_95_None | 4 | 4 | true | 0.316852 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_96_None | 4 | 4 | true | 0.303028 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_97_None | 0 | 3 | false | -0.256854 | -1.000000 | true | false | neural_prefers_nonoptimal_child |  |
| harder_98_None | 2 | 2 | true | -0.112092 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_99_None | 5 | 5 | true | -0.102943 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_100_None | 0 | 3 | false | -0.351219 | -1.000000 | true | false | neural_prefers_nonoptimal_child |  |
| harder_101_None | 5 | 4 | false | -0.355247 | -1.000000 | true | false | neural_prefers_nonoptimal_child |  |
| harder_102_None | 1 | 1 | true | 0.094984 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_103_None | 2 | 2 | true | -0.233788 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_104_None | 5 | 5 | true | 0.212132 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_105_None | 3 | 4 | false | -0.457007 | -1.000000 | true | true | neural_prefers_nonoptimal_child |  |
| harder_106_None | 4 | 1 | false | -0.356361 | -1.000000 | true | false | neural_prefers_nonoptimal_child |  |
| harder_107_None | 4 | 4 | true | 0.269788 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_108_None | 2 | 2 | true | 0.140386 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_109_None | 5 | 5 | true | 0.184517 | 0.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_110_None | 0 | 0 | true | -0.026756 | 0.000000 | false | false | neural_ok_search_fails |  |
| harder_111_None | 2 | 2 | true | -0.078026 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_112_None | 4 | 4 | true | 0.128369 | 1.000000 | false | false | neural_underestimates_optimal_child |  |
| harder_113_None | 1 | 0 | false | -0.206798 | -1.000000 | true | false | neural_prefers_nonoptimal_child |  |
| harder_114_None | 0 | 3 | false | -0.097767 | -1.000000 | true | false | neural_prefers_nonoptimal_child |  |
| harder_115_None | 3 | 0 | false | -0.236048 | -1.000000 | true | false | neural_prefers_nonoptimal_child |  |
| harder_116_None | 3 | 3 | true | -0.163961 | 1.000000 | false | false | neural_underestimates_optimal_child |  |

## 8. Child PUCT audit

| candidate_id | child_branch | budget | child_exact_value_root | child_puct_value_root | child_puct_error | child_selected_is_tablebase_optimal | classification | notes |
|---|---|---|---|---|---|---|---|---|
| harder_26_None | optimal_child | 384 | 1.000000 | 0.672416 | -0.327584 | true | child_puct_value_error |  |
| harder_26_None | selected_child | 384 | 0.000000 | 0.023997 | 0.023997 | false | child_puct_ok_root_selection_fails |  |
| harder_26_None | optimal_child | 1200 | 1.000000 | 0.840172 | -0.159828 | true | child_puct_value_error |  |
| harder_26_None | selected_child | 1200 | 0.000000 | 0.006884 | 0.006884 | false | child_puct_ok_root_selection_fails |  |
| harder_26_None | optimal_child | 2400 | 1.000000 | 0.886626 | -0.113374 | true | child_puct_value_error |  |
| harder_26_None | selected_child | 2400 | 0.000000 | 0.004691 | 0.004691 | false | child_puct_ok_root_selection_fails |  |
| harder_32_None | optimal_child | 384 | 1.000000 | 0.858062 | -0.141938 | true | child_puct_value_error |  |
| harder_32_None | selected_child | 384 | 0.000000 | -0.000810 | -0.000810 | true | child_puct_inconclusive |  |
| harder_32_None | optimal_child | 1200 | 1.000000 | 0.936960 | -0.063040 | true | child_puct_value_error |  |
| harder_32_None | selected_child | 1200 | 0.000000 | -0.000259 | -0.000259 | true | child_puct_inconclusive |  |
| harder_32_None | optimal_child | 2400 | 1.000000 | 0.965731 | -0.034269 | true | child_puct_inconclusive |  |
| harder_32_None | selected_child | 2400 | 0.000000 | -0.000130 | -0.000130 | true | child_puct_inconclusive |  |
| harder_41_None | optimal_child | 384 | 1.000000 | 0.696859 | -0.303141 | true | child_puct_value_error |  |
| harder_41_None | selected_child | 384 | 0.000000 | -0.009767 | -0.009767 | true | child_puct_inconclusive |  |
| harder_41_None | optimal_child | 1200 | 1.000000 | 0.864493 | -0.135507 | true | child_puct_value_error |  |
| harder_41_None | selected_child | 1200 | 0.000000 | -0.008959 | -0.008959 | true | child_puct_inconclusive |  |
| harder_41_None | optimal_child | 2400 | 1.000000 | 0.921167 | -0.078833 | true | child_puct_value_error |  |
| harder_41_None | selected_child | 2400 | 0.000000 | -0.006563 | -0.006563 | true | child_puct_inconclusive |  |
| harder_47_None | optimal_child | 384 | 1.000000 | 0.592435 | -0.407565 | true | child_puct_value_error |  |
| harder_47_None | selected_child | 384 | 0.000000 | 0.011607 | 0.011607 | false | child_puct_ok_root_selection_fails |  |
| harder_47_None | optimal_child | 1200 | 1.000000 | 0.843841 | -0.156159 | true | child_puct_value_error |  |
| harder_47_None | selected_child | 1200 | 0.000000 | 0.007651 | 0.007651 | false | child_puct_ok_root_selection_fails |  |
| harder_47_None | optimal_child | 2400 | 1.000000 | 0.909107 | -0.090893 | true | child_puct_value_error |  |
| harder_47_None | selected_child | 2400 | 0.000000 | 0.008409 | 0.008409 | false | child_puct_ok_root_selection_fails |  |
| harder_49_None | optimal_child | 384 | 1.000000 | 0.865915 | -0.134085 | true | child_puct_value_error |  |
| harder_49_None | selected_child | 384 | 0.000000 | -0.000539 | -0.000539 | true | child_puct_inconclusive |  |
| harder_49_None | optimal_child | 1200 | 1.000000 | 0.951951 | -0.048049 | true | child_puct_inconclusive |  |
| harder_49_None | selected_child | 1200 | 0.000000 | -0.000172 | -0.000172 | true | child_puct_inconclusive |  |
| harder_49_None | optimal_child | 2400 | 1.000000 | 0.973602 | -0.026398 | true | child_puct_inconclusive |  |
| harder_49_None | selected_child | 2400 | 0.000000 | -0.000086 | -0.000086 | true | child_puct_inconclusive |  |
| harder_59_None | optimal_child | 384 | 1.000000 | 0.957575 | -0.042425 | true | child_puct_inconclusive |  |
| harder_59_None | selected_child | 384 | 0.000000 | 0.041170 | 0.041170 | false | child_puct_ok_root_selection_fails |  |
| harder_59_None | optimal_child | 1200 | 1.000000 | 0.983568 | -0.016432 | true | child_puct_inconclusive |  |
| harder_59_None | selected_child | 1200 | 0.000000 | 0.027817 | 0.027817 | false | child_puct_ok_root_selection_fails |  |
| harder_59_None | optimal_child | 2400 | 1.000000 | 0.988762 | -0.011238 | true | child_puct_inconclusive |  |
| harder_59_None | selected_child | 2400 | 0.000000 | 0.020807 | 0.020807 | false | child_puct_ok_root_selection_fails |  |
| harder_86_None | optimal_child | 384 | 1.000000 | 0.868727 | -0.131273 | true | child_puct_value_error |  |
| harder_86_None | selected_child | 384 | -1.000000 | 0.032165 | 1.032165 | true | child_puct_value_error |  |
| harder_86_None | optimal_child | 1200 | 1.000000 | 0.930009 | -0.069991 | true | child_puct_value_error |  |
| harder_86_None | selected_child | 1200 | -1.000000 | 0.072510 | 1.072510 | true | child_puct_value_error |  |
| harder_86_None | optimal_child | 2400 | 1.000000 | 0.953802 | -0.046198 | true | child_puct_inconclusive |  |
| harder_86_None | selected_child | 2400 | -1.000000 | -0.235364 | 0.764636 | true | child_puct_value_error |  |
| harder_92_None | optimal_child | 384 | 1.000000 | 0.003503 | -0.996497 | true | child_puct_value_error |  |
| harder_92_None | selected_child | 384 | 0.000000 | 0.020314 | 0.020314 | false | child_puct_ok_root_selection_fails |  |
| harder_92_None | optimal_child | 1200 | 1.000000 | 0.004215 | -0.995785 | true | child_puct_value_error |  |
| harder_92_None | selected_child | 1200 | 0.000000 | 0.016700 | 0.016700 | false | child_puct_ok_root_selection_fails |  |
| harder_92_None | optimal_child | 2400 | 1.000000 | 0.007114 | -0.992886 | true | child_puct_value_error |  |
| harder_92_None | selected_child | 2400 | 0.000000 | 0.013767 | 0.013767 | false | child_puct_ok_root_selection_fails |  |
| harder_97_None | optimal_child | 384 | 0.000000 | 0.005167 | 0.005167 | false | child_puct_ok_root_selection_fails |  |
| harder_97_None | selected_child | 384 | -1.000000 | -0.344382 | 0.655618 | false | child_puct_value_error |  |
| harder_97_None | optimal_child | 1200 | 0.000000 | 0.003320 | 0.003320 | false | child_puct_ok_root_selection_fails |  |
| harder_97_None | selected_child | 1200 | -1.000000 | -0.774217 | 0.225783 | false | child_puct_value_error |  |
| harder_97_None | optimal_child | 2400 | 0.000000 | 0.002493 | 0.002493 | false | child_puct_ok_root_selection_fails |  |
| harder_97_None | selected_child | 2400 | -1.000000 | -0.880965 | 0.119035 | false | child_puct_value_error |  |
| harder_100_None | optimal_child | 384 | 0.000000 | 0.080498 | 0.080498 | true | child_puct_value_error |  |
| harder_100_None | selected_child | 384 | -1.000000 | -0.016161 | 0.983839 | true | child_puct_value_error |  |
| harder_100_None | optimal_child | 1200 | 0.000000 | 0.048987 | 0.048987 | true | child_puct_inconclusive |  |
| harder_100_None | selected_child | 1200 | -1.000000 | -0.014513 | 0.985487 | true | child_puct_value_error |  |
| harder_100_None | optimal_child | 2400 | 0.000000 | 0.026520 | 0.026520 | true | child_puct_inconclusive |  |
| harder_100_None | selected_child | 2400 | -1.000000 | -0.010472 | 0.989528 | true | child_puct_value_error |  |
| harder_106_None | optimal_child | 384 | 1.000000 | 0.915420 | -0.084580 | true | child_puct_value_error |  |
| harder_106_None | selected_child | 384 | -1.000000 | -0.706007 | 0.293993 | false | child_puct_value_error |  |
| harder_106_None | optimal_child | 1200 | 1.000000 | 0.971811 | -0.028189 | true | child_puct_inconclusive |  |
| harder_106_None | selected_child | 1200 | -1.000000 | -0.879398 | 0.120602 | false | child_puct_value_error |  |
| harder_106_None | optimal_child | 2400 | 1.000000 | 0.985630 | -0.014370 | true | child_puct_inconclusive |  |
| harder_106_None | selected_child | 2400 | -1.000000 | -0.924808 | 0.075192 | false | child_puct_value_error |  |
| harder_116_None | optimal_child | 384 | 1.000000 | -0.022462 | -1.022462 | true | child_puct_value_error |  |
| harder_116_None | selected_child | 384 | 0.000000 | -0.000856 | -0.000856 | true | child_puct_inconclusive |  |
| harder_116_None | optimal_child | 1200 | 1.000000 | -0.008125 | -1.008125 | true | child_puct_value_error |  |
| harder_116_None | selected_child | 1200 | 0.000000 | -0.000354 | -0.000354 | true | child_puct_inconclusive |  |
| harder_116_None | optimal_child | 2400 | 1.000000 | -0.004828 | -1.004828 | true | child_puct_value_error |  |
| harder_116_None | selected_child | 2400 | 0.000000 | -0.000594 | -0.000594 | true | child_puct_inconclusive |  |

## 9. Root counterfactual diagnostics

| candidate_id | intervention | budget | selected_move | selected_is_optimal | optimal_visit_share | selected_minus_optimal_q_margin | flipped_to_optimal | notes |
|---|---|---|---|---|---|---|---|---|
| harder_26_None | original | 384 | 5 | false | 0.098958 | 0.282644 | false | no intervention |
| harder_26_None | original | 1200 | 5 | false | 0.031667 | 0.263773 | false | no intervention |
| harder_26_None | uniform_legal_prior | 384 | 5 | false | 0.072917 | 0.112217 | false | uniform legal prior |
| harder_26_None | uniform_legal_prior | 1200 | 5 | false | 0.023333 | 0.094009 | false | uniform legal prior |
| harder_26_None | equalize_optimal_selected_priors | 384 | 5 | false | 0.072917 | 0.112217 | false | equalize optimal/selected priors |
| harder_26_None | equalize_optimal_selected_priors | 1200 | 5 | false | 0.023333 | 0.094009 | false | equalize optimal/selected priors |
| harder_26_None | policy_prior_boost_optimal | 384 | 5 | false | 0.109375 | 0.270883 | false | diagnostic small boost to optimal move prior |
| harder_26_None | policy_prior_boost_optimal | 1200 | 5 | false | 0.035000 | 0.251736 | false | diagnostic small boost to optimal move prior |
| harder_26_None | tablebase_child_value_override | 384 | 5 | false | 0.072917 | 0.081927 | false | override child backup values with exact tablebase child values |
| harder_26_None | tablebase_child_value_override | 1200 | 5 | false | 0.025000 | 0.125994 | false | override child backup values with exact tablebase child values |
| harder_32_None | original | 384 | 1 | false | 0.002604 | 0.313640 | false | no intervention |
| harder_32_None | original | 1200 | 1 | false | 0.000833 | 0.314884 | false | no intervention |
| harder_32_None | uniform_legal_prior | 384 | 4 | true | 0.971354 | 0.000000 | true | uniform legal prior |
| harder_32_None | uniform_legal_prior | 1200 | 4 | true | 0.982500 | 0.000000 | true | uniform legal prior |
| harder_32_None | equalize_optimal_selected_priors | 384 | 4 | true | 0.971354 | 0.000000 | true | equalize optimal/selected priors |
| harder_32_None | equalize_optimal_selected_priors | 1200 | 4 | true | 0.982500 | 0.000000 | true | equalize optimal/selected priors |
| harder_32_None | policy_prior_boost_optimal | 384 | 1 | false | 0.002604 | 0.313640 | false | diagnostic small boost to optimal move prior |
| harder_32_None | policy_prior_boost_optimal | 1200 | 1 | false | 0.001667 | 0.130458 | false | diagnostic small boost to optimal move prior |
| harder_32_None | tablebase_child_value_override | 384 | 1 | false | 0.046875 | 0.003429 | false | override child backup values with exact tablebase child values |
| harder_32_None | tablebase_child_value_override | 1200 | 1 | false | 0.015000 | 0.004015 | false | override child backup values with exact tablebase child values |
| harder_41_None | original | 384 | 2 | false | 0.005208 | 0.171138 | false | no intervention |
| harder_41_None | original | 1200 | 2 | false | 0.003333 | 0.087971 | false | no intervention |
| harder_41_None | uniform_legal_prior | 384 | 2 | false | 0.028646 | 0.045203 | false | uniform legal prior |
| harder_41_None | uniform_legal_prior | 1200 | 2 | false | 0.017500 | 0.015855 | false | uniform legal prior |
| harder_41_None | equalize_optimal_selected_priors | 384 | 2 | false | 0.028646 | 0.045203 | false | equalize optimal/selected priors |
| harder_41_None | equalize_optimal_selected_priors | 1200 | 2 | false | 0.017500 | 0.015855 | false | equalize optimal/selected priors |
| harder_41_None | policy_prior_boost_optimal | 384 | 2 | false | 0.007812 | 0.094763 | false | diagnostic small boost to optimal move prior |
| harder_41_None | policy_prior_boost_optimal | 1200 | 2 | false | 0.004167 | 0.062606 | false | diagnostic small boost to optimal move prior |
| harder_41_None | tablebase_child_value_override | 384 | 4 | true | 0.945312 | 0.000000 | true | override child backup values with exact tablebase child values |
| harder_41_None | tablebase_child_value_override | 1200 | 4 | true | 0.968333 | 0.000000 | true | override child backup values with exact tablebase child values |
| harder_47_None | original | 384 | 0 | false | 0.023438 | 0.008804 | false | no intervention |
| harder_47_None | original | 1200 | 0 | false | 0.020833 | 0.036259 | false | no intervention |
| harder_47_None | uniform_legal_prior | 384 | 3 | true | 0.966146 | 0.000000 | true | uniform legal prior |
| harder_47_None | uniform_legal_prior | 1200 | 3 | true | 0.982500 | 0.000000 | true | uniform legal prior |
| harder_47_None | equalize_optimal_selected_priors | 384 | 3 | true | 0.966146 | 0.000000 | true | equalize optimal/selected priors |
| harder_47_None | equalize_optimal_selected_priors | 1200 | 3 | true | 0.982500 | 0.000000 | true | equalize optimal/selected priors |
| harder_47_None | policy_prior_boost_optimal | 384 | 0 | false | 0.023438 | 0.008804 | false | diagnostic small boost to optimal move prior |
| harder_47_None | policy_prior_boost_optimal | 1200 | 0 | false | 0.020833 | 0.036259 | false | diagnostic small boost to optimal move prior |
| harder_47_None | tablebase_child_value_override | 384 | 3 | true | 0.960938 | 0.000000 | true | override child backup values with exact tablebase child values |
| harder_47_None | tablebase_child_value_override | 1200 | 3 | true | 0.977500 | 0.000000 | true | override child backup values with exact tablebase child values |
| harder_49_None | original | 384 | 2 | false | 0.007812 | 0.029699 | false | no intervention |
| harder_49_None | original | 1200 | 0 | true | 0.581667 | 0.000000 | true | no intervention |
| harder_49_None | uniform_legal_prior | 384 | 0 | true | 0.971354 | 0.000000 | true | uniform legal prior |
| harder_49_None | uniform_legal_prior | 1200 | 0 | true | 0.982500 | 0.000000 | true | uniform legal prior |
| harder_49_None | equalize_optimal_selected_priors | 384 | 0 | true | 0.971354 | 0.000000 | true | equalize optimal/selected priors |
| harder_49_None | equalize_optimal_selected_priors | 1200 | 0 | true | 0.982500 | 0.000000 | true | equalize optimal/selected priors |
| harder_49_None | policy_prior_boost_optimal | 384 | 2 | false | 0.179688 | -0.329237 | false | diagnostic small boost to optimal move prior |
| harder_49_None | policy_prior_boost_optimal | 1200 | 0 | true | 0.737500 | 0.000000 | true | diagnostic small boost to optimal move prior |
| harder_49_None | tablebase_child_value_override | 384 | 0 | true | 0.947917 | 0.000000 | true | override child backup values with exact tablebase child values |
| harder_49_None | tablebase_child_value_override | 1200 | 0 | true | 0.970000 | 0.000000 | true | override child backup values with exact tablebase child values |
| harder_59_None | original | 384 | 5 | false | 0.385417 | -0.834698 | false | no intervention |
| harder_59_None | original | 1200 | 3 | true | 0.803333 | 0.000000 | true | no intervention |
| harder_59_None | uniform_legal_prior | 384 | 3 | true | 0.966146 | 0.000000 | true | uniform legal prior |
| harder_59_None | uniform_legal_prior | 1200 | 3 | true | 0.982500 | 0.000000 | true | uniform legal prior |
| harder_59_None | equalize_optimal_selected_priors | 384 | 3 | true | 0.966146 | 0.000000 | true | equalize optimal/selected priors |
| harder_59_None | equalize_optimal_selected_priors | 1200 | 3 | true | 0.982500 | 0.000000 | true | equalize optimal/selected priors |
| harder_59_None | policy_prior_boost_optimal | 384 | 3 | true | 0.664062 | 0.000000 | true | diagnostic small boost to optimal move prior |
| harder_59_None | policy_prior_boost_optimal | 1200 | 3 | true | 0.892500 | 0.000000 | true | diagnostic small boost to optimal move prior |
| harder_59_None | tablebase_child_value_override | 384 | 3 | true | 0.945312 | 0.000000 | true | override child backup values with exact tablebase child values |
| harder_59_None | tablebase_child_value_override | 1200 | 3 | true | 0.968333 | 0.000000 | true | override child backup values with exact tablebase child values |
| harder_86_None | original | 384 | 1 | false | 0.018229 | 0.116790 | false | no intervention |
| harder_86_None | original | 1200 | 3 | true | 0.625000 | 0.000000 | true | no intervention |
| harder_86_None | uniform_legal_prior | 384 | 3 | true | 0.960938 | 0.000000 | true | uniform legal prior |
| harder_86_None | uniform_legal_prior | 1200 | 3 | true | 0.976667 | 0.000000 | true | uniform legal prior |
| harder_86_None | equalize_optimal_selected_priors | 384 | 3 | true | 0.963542 | 0.000000 | true | equalize optimal/selected priors |
| harder_86_None | equalize_optimal_selected_priors | 1200 | 3 | true | 0.980000 | 0.000000 | true | equalize optimal/selected priors |
| harder_86_None | policy_prior_boost_optimal | 384 | 1 | false | 0.200521 | -0.646044 | false | diagnostic small boost to optimal move prior |
| harder_86_None | policy_prior_boost_optimal | 1200 | 3 | true | 0.743333 | 0.000000 | true | diagnostic small boost to optimal move prior |
| harder_86_None | tablebase_child_value_override | 384 | 3 | true | 0.945312 | 0.000000 | true | override child backup values with exact tablebase child values |
| harder_86_None | tablebase_child_value_override | 1200 | 3 | true | 0.968333 | 0.000000 | true | override child backup values with exact tablebase child values |
| harder_92_None | original | 384 | 1 | false | 0.132812 | 0.012541 | false | no intervention |
| harder_92_None | original | 1200 | 1 | false | 0.042500 | 0.012796 | false | no intervention |
| harder_92_None | uniform_legal_prior | 384 | 2 | true | 0.968750 | 0.000000 | true | uniform legal prior |
| harder_92_None | uniform_legal_prior | 1200 | 2 | true | 0.787500 | 0.000000 | true | uniform legal prior |
| harder_92_None | equalize_optimal_selected_priors | 384 | 2 | true | 0.968750 | 0.000000 | true | equalize optimal/selected priors |
| harder_92_None | equalize_optimal_selected_priors | 1200 | 2 | true | 0.787500 | 0.000000 | true | equalize optimal/selected priors |
| harder_92_None | policy_prior_boost_optimal | 384 | 1 | false | 0.132812 | 0.012541 | false | diagnostic small boost to optimal move prior |
| harder_92_None | policy_prior_boost_optimal | 1200 | 1 | false | 0.042500 | 0.012796 | false | diagnostic small boost to optimal move prior |
| harder_92_None | tablebase_child_value_override | 384 | 2 | true | 0.955729 | 0.000000 | true | override child backup values with exact tablebase child values |
| harder_92_None | tablebase_child_value_override | 1200 | 1 | false | 0.359167 | 0.010639 | false | override child backup values with exact tablebase child values |
| harder_97_None | original | 384 | 3 | false | 0.497396 | -0.222652 | false | no intervention |
| harder_97_None | original | 1200 | 0 | true | 0.839167 | 0.000000 | true | no intervention |
| harder_97_None | uniform_legal_prior | 384 | 3 | false | 0.497396 | -0.222652 | false | uniform legal prior |
| harder_97_None | uniform_legal_prior | 1200 | 0 | true | 0.839167 | 0.000000 | true | uniform legal prior |
| harder_97_None | equalize_optimal_selected_priors | 384 | 3 | false | 0.497396 | -0.222652 | false | equalize optimal/selected priors |
| harder_97_None | equalize_optimal_selected_priors | 1200 | 0 | true | 0.839167 | 0.000000 | true | equalize optimal/selected priors |
| harder_97_None | policy_prior_boost_optimal | 384 | 3 | false | 0.497396 | -0.222652 | false | diagnostic small boost to optimal move prior |
| harder_97_None | policy_prior_boost_optimal | 1200 | 0 | true | 0.839167 | 0.000000 | true | diagnostic small boost to optimal move prior |
| harder_97_None | tablebase_child_value_override | 384 | 0 | true | 0.973958 | 0.000000 | true | override child backup values with exact tablebase child values |
| harder_97_None | tablebase_child_value_override | 1200 | 0 | true | 0.984167 | 0.000000 | true | override child backup values with exact tablebase child values |
| harder_100_None | original | 384 | 3 | false | 0.005208 | 0.211127 | false | no intervention |
| harder_100_None | original | 1200 | 3 | false | 0.430000 | -0.087661 | false | no intervention |
| harder_100_None | uniform_legal_prior | 384 | 0 | true | 0.770833 | 0.000000 | true | uniform legal prior |
| harder_100_None | uniform_legal_prior | 1200 | 0 | true | 0.921667 | 0.000000 | true | uniform legal prior |
| harder_100_None | equalize_optimal_selected_priors | 384 | 3 | false | 0.005208 | 0.212495 | false | equalize optimal/selected priors |
| harder_100_None | equalize_optimal_selected_priors | 1200 | 3 | false | 0.313333 | -0.097382 | false | equalize optimal/selected priors |
| harder_100_None | policy_prior_boost_optimal | 384 | 3 | false | 0.010417 | 0.061086 | false | diagnostic small boost to optimal move prior |
| harder_100_None | policy_prior_boost_optimal | 1200 | 0 | true | 0.683333 | 0.000000 | true | diagnostic small boost to optimal move prior |
| harder_100_None | tablebase_child_value_override | 384 | 0 | true | 0.723958 | 0.000000 | true | override child backup values with exact tablebase child values |
| harder_100_None | tablebase_child_value_override | 1200 | 0 | true | 0.873333 | 0.000000 | true | override child backup values with exact tablebase child values |
| harder_106_None | original | 384 | 1 | false | 0.460938 | -1.176633 | false | no intervention |
| harder_106_None | original | 1200 | 4 | true | 0.827500 | 0.000000 | true | no intervention |
| harder_106_None | uniform_legal_prior | 384 | 4 | true | 0.747396 | 0.000000 | true | uniform legal prior |
| harder_106_None | uniform_legal_prior | 1200 | 4 | true | 0.919167 | 0.000000 | true | uniform legal prior |
| harder_106_None | equalize_optimal_selected_priors | 384 | 4 | true | 0.747396 | 0.000000 | true | equalize optimal/selected priors |
| harder_106_None | equalize_optimal_selected_priors | 1200 | 4 | true | 0.919167 | 0.000000 | true | equalize optimal/selected priors |
| harder_106_None | policy_prior_boost_optimal | 384 | 4 | true | 0.635417 | 0.000000 | true | diagnostic small boost to optimal move prior |
| harder_106_None | policy_prior_boost_optimal | 1200 | 4 | true | 0.883333 | 0.000000 | true | diagnostic small boost to optimal move prior |
| harder_106_None | tablebase_child_value_override | 384 | 4 | true | 0.945312 | 0.000000 | true | override child backup values with exact tablebase child values |
| harder_106_None | tablebase_child_value_override | 1200 | 4 | true | 0.968333 | 0.000000 | true | override child backup values with exact tablebase child values |
| harder_116_None | original | 384 | 1 | false | 0.020833 | 0.236484 | false | no intervention |
| harder_116_None | original | 1200 | 1 | false | 0.017500 | 0.169360 | false | no intervention |
| harder_116_None | uniform_legal_prior | 384 | 1 | false | 0.023438 | 0.262306 | false | uniform legal prior |
| harder_116_None | uniform_legal_prior | 1200 | 1 | false | 0.015833 | 0.173255 | false | uniform legal prior |
| harder_116_None | equalize_optimal_selected_priors | 384 | 1 | false | 0.023438 | 0.261325 | false | equalize optimal/selected priors |
| harder_116_None | equalize_optimal_selected_priors | 1200 | 1 | false | 0.017500 | 0.169360 | false | equalize optimal/selected priors |
| harder_116_None | policy_prior_boost_optimal | 384 | 1 | false | 0.023438 | 0.261091 | false | diagnostic small boost to optimal move prior |
| harder_116_None | policy_prior_boost_optimal | 1200 | 1 | false | 0.019167 | 0.181075 | false | diagnostic small boost to optimal move prior |
| harder_116_None | tablebase_child_value_override | 384 | 1 | false | 0.052083 | 0.100679 | false | override child backup values with exact tablebase child values |
| harder_116_None | tablebase_child_value_override | 1200 | 1 | false | 0.020833 | 0.155853 | false | override child backup values with exact tablebase child values |

## 10. Row mechanism classifications

| candidate_id | row_mechanism | supporting_evidence | recommended_role | notes |
|---|---|---|---|---|
| harder_17_None | value_error_but_search_compensates | neural value rank error (neural_prefers_nonoptimal_child) but PUCT selects optimal at 384+ |  |  |
| harder_18_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_19_None | value_error_but_search_compensates | neural value rank error (neural_prefers_nonoptimal_child) but PUCT selects optimal at 384+ |  |  |
| harder_21_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_22_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_24_None | low_budget_search_noise | fails only below 384, passes at 384+ |  |  |
| harder_25_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_26_None | child_puct_value_error | no counterfactual flips selection -> child PUCT or backup issue |  |  |
| harder_27_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_28_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_29_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_30_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_32_None | root_selection_pressure | prior/selection pressure intervenable but not prior-underweight alone |  |  |
| harder_33_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_34_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_35_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_36_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_38_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_39_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_40_None | value_error_but_search_compensates | neural value rank error (neural_prefers_nonoptimal_child) but PUCT selects optimal at 384+ |  |  |
| harder_41_None | value_rank_error_drives_failure | value/backup error: tablebase child value override flips selection |  |  |
| harder_42_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_43_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_44_None | value_error_but_search_compensates | neural value rank error (neural_prefers_nonoptimal_child) but PUCT selects optimal at 384+ |  |  |
| harder_45_None | low_budget_search_noise | fails only below 384, passes at 384+ |  |  |
| harder_47_None | value_rank_error_drives_failure | value/backup error: tablebase child value override flips selection |  |  |
| harder_49_None | value_rank_error_drives_failure | value/backup error: tablebase override flips selection at 384 |  |  |
| harder_50_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_51_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_52_None | low_budget_search_noise | fails only below 384, passes at 384+ |  |  |
| harder_53_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_54_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_55_None | value_error_but_search_compensates | neural value rank error (neural_prefers_nonoptimal_child) but PUCT selects optimal at 384+ |  |  |
| harder_56_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_57_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_58_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_59_None | value_rank_error_drives_failure | value/backup error: tablebase override flips selection at 384 |  |  |
| harder_60_None | value_error_but_search_compensates | neural value rank error (neural_prefers_nonoptimal_child) but PUCT selects optimal at 384+ |  |  |
| harder_61_None | value_error_but_search_compensates | neural value rank error (neural_prefers_nonoptimal_child) but PUCT selects optimal at 384+ |  |  |
| harder_62_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_63_None | value_error_but_search_compensates | neural value rank error (neural_prefers_nonoptimal_child) but PUCT selects optimal at 384+ |  |  |
| harder_65_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_66_None | value_error_but_search_compensates | neural value rank error (neural_prefers_nonoptimal_child) but PUCT selects optimal at 384+ |  |  |
| harder_67_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_68_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_69_None | value_error_but_search_compensates | neural value rank error (neural_prefers_nonoptimal_child) but PUCT selects optimal at 384+ |  |  |
| harder_70_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_71_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_72_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_73_None | value_error_but_search_compensates | neural value rank error (neural_prefers_nonoptimal_child) but PUCT selects optimal at 384+ |  |  |
| harder_75_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_76_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_77_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_78_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_79_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_80_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_81_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_82_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_83_None | low_budget_search_noise | fails only below 384, passes at 384+ |  |  |
| harder_84_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_85_None | value_error_but_search_compensates | neural value rank error (neural_prefers_nonoptimal_child) but PUCT selects optimal at 384+ |  |  |
| harder_86_None | value_rank_error_drives_failure | value/backup error: tablebase override flips selection at 384 |  |  |
| harder_87_None | value_error_but_search_compensates | neural value rank error (neural_prefers_nonoptimal_child) but PUCT selects optimal at 384+ |  |  |
| harder_88_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_89_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_91_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_92_None | value_rank_error_drives_failure | value/backup error: tablebase child value override flips selection |  |  |
| harder_93_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_94_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_95_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_96_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_97_None | value_rank_error_drives_failure | value/backup error: tablebase override flips selection at 384 |  |  |
| harder_98_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_99_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_100_None | value_rank_error_drives_failure | value/backup error: tablebase child value override flips selection |  |  |
| harder_101_None | value_error_but_search_compensates | neural value rank error (neural_prefers_nonoptimal_child) but PUCT selects optimal at 384+ |  |  |
| harder_102_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_103_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_104_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_105_None | value_error_but_search_compensates | neural value rank error (neural_prefers_nonoptimal_child) but PUCT selects optimal at 384+ |  |  |
| harder_106_None | value_rank_error_drives_failure | value/backup error: tablebase override flips selection at 384 |  |  |
| harder_107_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_108_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_109_None | low_budget_search_noise | fails only below 384, passes at 384+ |  |  |
| harder_110_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_111_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_112_None | inconclusive | row does not fail at 384 or 1200 |  |  |
| harder_113_None | low_budget_search_noise | fails only below 384, passes at 384+ |  |  |
| harder_114_None | value_error_but_search_compensates | neural value rank error (neural_prefers_nonoptimal_child) but PUCT selects optimal at 384+ |  |  |
| harder_115_None | value_error_but_search_compensates | neural value rank error (neural_prefers_nonoptimal_child) but PUCT selects optimal at 384+ |  |  |
| harder_116_None | child_puct_value_error | no counterfactual flips selection -> child PUCT or backup issue |  |  |

## 11. Refined clean split

| bucket | row_count | rows | recommended_use | risks | next_action |
|---|---|---|---|---|---|
| holdout_candidate | 56 | harder_25_None, harder_27_None, harder_28_None, harder_29_None, harder_30_None... | unseen holdout for evaluation | do not use in training | set aside for final validation |
| preservation_control | 3 | harder_18_None, harder_21_None, harder_22_None | control group for testing | distribution must match targets | use as clean controls in artifact |
| production_candidate_later | 12 | harder_26_None, harder_32_None, harder_41_None, harder_47_None, harder_49_None... | train policy+value with exact targets | may need larger dataset for generalization | build tiny diagnostic artifact with controls |
| value_only_candidate | 20 | harder_17_None, harder_19_None, harder_24_None, harder_40_None, harder_44_None... | value calibration only | not useful for policy targets | include in value-only diagnostic artifact |

## 12. Final decision

| classification | supporting_evidence | rejected_alternatives | next_action |
|---|---|---|---|
| exact_tablebase_diagnostic_artifact_ready | production_candidate_later=12, value_only=20, search_only=0, persistent_failures=7, value_rank_errors=26, dominant_mechanism=inconclusive | see mechanism distribution | build a tiny train-only exact-tablebase diagnostic artifact with controls; no arena until local exact metrics improve |

## 13. Exactly one recommended next action

Recommendation: **build a tiny train-only exact-tablebase diagnostic artifact with controls; no arena until local exact metrics improve**

### Acceptance criteria

- No training was run.
- No arena was run.
- No model was promoted.
- Active references were not mutated.
- Exhausted families were excluded from selection.
- Teacher-conflict filtering was used.
- Exact tablebase labels were used only diagnostically.
- Final report recommends exactly one next branch.