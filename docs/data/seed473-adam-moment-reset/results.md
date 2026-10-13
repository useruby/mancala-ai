# Seed473 Adam moment-reset ablation

Exploratory analysis using a reused evaluation cohort; this is not independent confirmation.

**Decision:** `close_moment_reset_branch`

Gates:
- `fresh_ce_vs_initializer_exposure_weighted`: **FAIL**
- `fresh_ce_vs_C_exposure_weighted`: **PASS**
- `pooled_ce_vs_initializer_exposure_weighted`: **PASS**
- `pooled_mse_vs_initializer_exposure_weighted`: **PASS**
- `fresh_ce_vs_initializer_equal_input`: **FAIL**
- `fresh_ce_vs_C_equal_input`: **PASS**
- `pooled_ce_vs_initializer_equal_input`: **PASS**
- `pooled_mse_vs_initializer_equal_input`: **PASS**
- `training_J_decreases`: **PASS**
- `exact_C_reproduction`: **PASS**

No searches, games, runtime export, promotion, or strength claim is authorized.
