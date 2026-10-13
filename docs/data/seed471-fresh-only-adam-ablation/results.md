# Seed471 fresh-only Adam learning-rate ablation

Exploratory analysis informed by previously inspected evaluation results. The authenticated evaluation occurrences are not a new holdout or independent confirmation.

**Decision:** `close_small_step_fresh_only_branch`

Gate outcomes (reported separately):

- `fresh_ce_vs_initializer_exposure_weighted`: **FAIL**
- `fresh_ce_vs_C_exposure_weighted`: **PASS**
- `pooled_ce_vs_initializer_exposure_weighted`: **PASS**
- `pooled_mse_vs_initializer_exposure_weighted`: **PASS**
- `fresh_ce_vs_initializer_equal_input`: **FAIL**
- `fresh_ce_vs_C_equal_input`: **PASS**
- `pooled_ce_vs_initializer_equal_input`: **FAIL**
- `pooled_mse_vs_initializer_equal_input`: **PASS**
- `training_J_decreases`: **PASS**

| Arm | Cohort | Weighting | Policy CE | Value MSE |
|---|---|---|---:|---:|
| initializer | fresh | exposure_weighted | 1.221102100 | 0.719704730 |
| initializer | fresh | equal_input | 1.223056465 | 0.724215707 |
| initializer | pooled | exposure_weighted | 1.170699161 | 0.476410666 |
| initializer | pooled | equal_input | 1.198160451 | 0.616998044 |
| C | fresh | exposure_weighted | 1.243069917 | 0.703736825 |
| C | fresh | equal_input | 1.243895663 | 0.709592116 |
| C | pooled | exposure_weighted | 1.234441587 | 0.503898016 |
| C | pooled | equal_input | 1.239370967 | 0.621390739 |
| T | fresh | exposure_weighted | 1.230509348 | 0.711643257 |
| T | fresh | equal_input | 1.233073079 | 0.716381686 |
| T | pooled | exposure_weighted | 1.169023414 | 0.479675461 |
| T | pooled | equal_input | 1.203746383 | 0.614485909 |

No arena execution, promotion, runtime export, or strength claim is authorized. A passing decision authorizes only a separately designed fresh confirmation.
