# PUCT Teacher-Quality Audit

Classification: `mcts_teacher_quality_budget_limited`.

## Frozen Inputs

- Incumbent weights SHA-256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- Cohort SHA-256: `c5fc9432baece48e56d81c93a740d4b8057dfe49cb6d7397ec8bef5c519a8d0b`
- Exact provenance: `native_hybrid_exact pr281_production_v1; exact holdout lanes`

## Production Search Configuration

```json
{
  "c_puct": 1.25,
  "dirichlet_alpha": 0.3,
  "dirichlet_epsilon": 0.3,
  "fpu_mode": "zero",
  "input_encoding": "kalah_v3",
  "normal_simulations": 192,
  "normalize_values": false,
  "opening_min_simulations": 384,
  "policy_target_mode": "sharpened",
  "policy_target_noise_mode": "denoised",
  "root_policy_mode": "visit_count",
  "tactical_root_bias": 0.0,
  "tree_reuse": true,
  "value_transform": null,
  "value_trust_schedule": "disabled"
}
```

## Policy Results

| phase | budget | network exact-set | search exact-set | lift | wrong->correct | correct->wrong |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| opening | 48 | 0.473 | 0.483 | 0.010 | 37 | 27 |
| opening | 96 | 0.473 | 0.497 | 0.024 | 60 | 36 |
| opening | 192 | 0.473 | 0.529 | 0.056 | 87 | 31 |
| opening | 384 | 0.473 | 0.570 | 0.097 | 129 | 32 |
| opening | 768 | 0.473 | 0.605 | 0.132 | 166 | 34 |
| opening | 1200 | 0.473 | 0.631 | 0.158 | 196 | 38 |
| midgame | 48 | 0.354 | 0.447 | 0.093 | 127 | 34 |
| midgame | 96 | 0.354 | 0.472 | 0.118 | 156 | 38 |
| midgame | 192 | 0.354 | 0.514 | 0.160 | 201 | 41 |
| midgame | 384 | 0.354 | 0.536 | 0.182 | 232 | 50 |
| midgame | 768 | 0.354 | 0.565 | 0.211 | 272 | 61 |
| midgame | 1200 | 0.354 | 0.598 | 0.244 | 301 | 57 |

## Raw Visits Versus Stored Target

| phase | budget | raw exact-set | target exact-set | raw optimal mass | target optimal mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| opening | 48 | 0.483 | 0.483 | 0.401 | 0.450 |
| opening | 96 | 0.497 | 0.497 | 0.406 | 0.458 |
| opening | 192 | 0.529 | 0.529 | 0.424 | 0.480 |
| opening | 384 | 0.570 | 0.570 | 0.452 | 0.516 |
| opening | 768 | 0.605 | 0.605 | 0.489 | 0.556 |
| opening | 1200 | 0.631 | 0.631 | 0.518 | 0.587 |
| midgame | 48 | 0.447 | 0.447 | 0.402 | 0.446 |
| midgame | 96 | 0.472 | 0.472 | 0.421 | 0.471 |
| midgame | 192 | 0.514 | 0.514 | 0.448 | 0.503 |
| midgame | 384 | 0.536 | 0.536 | 0.481 | 0.537 |
| midgame | 768 | 0.565 | 0.565 | 0.511 | 0.564 |
| midgame | 1200 | 0.598 | 0.598 | 0.534 | 0.594 |

## Value Error

| phase | budget | network MAE | PUCT root-Q MAE | MAE improvement |
| --- | ---: | ---: | ---: | ---: |
| opening | 48 | 0.895 | 0.899 | -0.004 |
| opening | 96 | 0.895 | 0.895 | 0.000 |
| opening | 192 | 0.895 | 0.890 | 0.005 |
| opening | 384 | 0.895 | 0.883 | 0.012 |
| opening | 768 | 0.895 | 0.875 | 0.020 |
| opening | 1200 | 0.895 | 0.868 | 0.026 |
| midgame | 48 | 0.820 | 0.796 | 0.024 |
| midgame | 96 | 0.820 | 0.783 | 0.037 |
| midgame | 192 | 0.820 | 0.768 | 0.051 |
| midgame | 384 | 0.820 | 0.753 | 0.066 |
| midgame | 768 | 0.820 | 0.735 | 0.084 |
| midgame | 1200 | 0.820 | 0.722 | 0.098 |

## Adjacent-Budget Stability

| pair | top changes | correctness changes | wrong->correct | correct->wrong | TVD |
| --- | ---: | ---: | ---: | ---: | ---: |
| 192->384 | 0.121 | 0.082 | 0.057 | 0.025 | 0.091 |
| 384->768 | 0.125 | 0.088 | 0.060 | 0.028 | 0.095 |
| 48->96 | 0.114 | 0.070 | 0.045 | 0.025 | 0.083 |
| 768->1200 | 0.095 | 0.066 | 0.048 | 0.018 | 0.071 |
| 96->192 | 0.120 | 0.079 | 0.058 | 0.021 | 0.082 |

## Compute

| budget | wall seconds | states/sec | simulations/sec | x96 |
| ---: | ---: | ---: | ---: | ---: |
| 48 | 8.6 | 231.95 | 11134 | 0.5 |
| 96 | 19.5 | 102.77 | 9866 | 1.0 |
| 192 | 35.9 | 55.74 | 10702 | 2.0 |
| 384 | 73.4 | 27.26 | 10467 | 4.0 |
| 768 | 155.8 | 12.84 | 9860 | 8.0 |
| 1200 | 252.0 | 7.94 | 9525 | 12.5 |

## Decision

- Recommended self-play simulation budget: `1200`
- Next experiment: `controlled phase-specific self-play simulation-budget ablation`
