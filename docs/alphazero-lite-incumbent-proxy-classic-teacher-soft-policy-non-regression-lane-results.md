# AlphaZero-lite Incumbent Proxy Classic Teacher Soft-Policy Non-Regression Lane Results

## 1. Context

- This branch is prepared because the dedicated low-LR lane still failed the strict PUCT non-regression gate.
- It softens Classic policy targets while keeping the same current-checkpoint initializer and low learning rate.
- No arena is part of this branch.

## 2. Prepared Variants

| artifact_name | reference_mass | epochs | lr | status |
| --- | --- | --- | --- | --- |
| classic_all_soft_policy_r70_low_lr | 0.7 | 1, 2, 4 | 0.00025 | prepared |
| classic_all_soft_policy_r60_low_lr | 0.6 | 1, 2, 4 | 0.00025 | prepared |

## 3. Strict Gate Template

| artifact_name | epoch | classic_gain_count | puct_lost_selection_384_count | puct_lost_selection_1200_count | heavy_regression_count | excluded_drift_count | gate_pass | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classic_all_soft_policy_r70_low_lr | 1 | 4 | 1 | 2 | 3 | 0 | false | puct_lost_selection_384,puct_lost_selection_1200,heavy_puct_regression |
| classic_all_soft_policy_r70_low_lr | 2 | 4 | 5 | 4 | 4 | 0 | false | puct_lost_selection_384,puct_lost_selection_1200,heavy_puct_regression |
| classic_all_soft_policy_r70_low_lr | 4 | 4 | 4 | 4 | 6 | 0 | false | puct_lost_selection_384,puct_lost_selection_1200,heavy_puct_regression |
| classic_all_soft_policy_r60_low_lr | 1 | 4 | 1 | 2 | 2 | 0 | false | puct_lost_selection_384,puct_lost_selection_1200,heavy_puct_regression |
| classic_all_soft_policy_r60_low_lr | 2 | 4 | 3 | 3 | 4 | 0 | false | puct_lost_selection_384,puct_lost_selection_1200,heavy_puct_regression |
| classic_all_soft_policy_r60_low_lr | 4 | 4 | 4 | 5 | 5 | 1 | false | puct_lost_selection_384,puct_lost_selection_1200,heavy_puct_regression,excluded_drift |

## 4. Decision

- Classification: `soft_policy_gate_pending`.
- best prepared candidate so far is classic_all_soft_policy_r60_low_lr at epoch 1 by the gate heuristic

## 5. Exactly One Recommended Next Action

Recommendation: **use this prepared softened-policy branch for the next diagnostic run; do not advance any replay lane until one softened-policy variant clears the strict PUCT gate.**
