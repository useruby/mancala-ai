# Seed461 E2–E4 checkpoint averaging

| Order | A score | B score | B−A |
|---:|---:|---:|---:|
| 38611 | 0.4414 | 0.5352 | +0.0938 |
| 38612 | 0.4014 | 0.6074 | +0.2061 |
| 38613 | 0.4756 | 0.5127 | +0.0371 |
| 38614 | 0.5684 | 0.5352 | -0.0332 |
| 38615 | 0.6406 | 0.6201 | -0.0205 |

Mean paired effect: **+0.0566** (95% shared-opening bootstrap interval +0.0443 to +0.0688).
Worst pair: -0.0332; minimum A/B: 0.4014/0.5127; range A/B: 0.2393/0.1074.
Decision: **retain_constant_lr_0.001**. Failed criteria: at_least_four_nonnegative_pairs.

Parameter distances are descriptive only. Inference is conditional on these five orders and this dataset; PR #386's cosine rejection remains unchanged. No model is promoted.

Reproduce from repository root:
```sh
.venv/bin/python -m ml.alphazero_lite.run_seed461_e2_e4_average_arena
.venv/bin/python -m ml.alphazero_lite.analyze_seed461_e2_e4_average
```
