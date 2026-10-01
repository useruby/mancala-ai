# Seed461 cosine learning-rate decay at fixed E4

Primary evaluation uses E4 from all ten trajectories; best-validation selections are descriptive only.

| Order | A score | B score | B−A |
|---:|---:|---:|---:|
| 38611 | 0.4395 | 0.5273 | +0.0879 |
| 38612 | 0.3975 | 0.6084 | +0.2109 |
| 38613 | 0.4697 | 0.5215 | +0.0518 |
| 38614 | 0.5645 | 0.6230 | +0.0586 |
| 38615 | 0.6387 | 0.5703 | -0.0684 |

Mean effect: **+0.0682** (95% shared-opening bootstrap interval +0.0574 to +0.0791).
Worst pair: -0.0684. Minimum scores A/B: 0.3975/0.5215; ranges A/B: 0.2412/0.1016.
Decision: **retain_constant_lr_0.001**. Failed criteria: worst_pair_at_least_minus_0.05.

Best-validation selections (descriptive only; arena results did not select checkpoints):
- 38611: A E4, B E4
- 38612: A E2, B E4
- 38613: A E4, B E4
- 38614: A E1, B E4
- 38615: A E4, B E4

Reproduce from the repository root:
```sh
.venv/bin/python -m ml.alphazero_lite.run_seed461_cosine_lr_ablation register
.venv/bin/python -m ml.alphazero_lite.run_seed461_cosine_lr_ablation train
.venv/bin/python -m ml.alphazero_lite.run_seed461_cosine_lr_ablation publish-training
.venv/bin/python -m ml.alphazero_lite.run_seed461_cosine_lr_arena
.venv/bin/python -m ml.alphazero_lite.analyze_seed461_cosine_lr_ablation
```

Inference is conditional on these five training orders and this frozen evaluation dataset. PR #384's rejection remains in force; no model is promoted.
