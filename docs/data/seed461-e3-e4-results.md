# Seed461 constant-LR E3 vs E4

| Order | A score | B score | B−A |
|---:|---:|---:|---:|
| 38611 | 0.4512 | 0.5742 | +0.1230 |
| 38612 | 0.4072 | 0.5576 | +0.1504 |
| 38613 | 0.4717 | 0.5225 | +0.0508 |
| 38614 | 0.5742 | 0.4941 | -0.0801 |
| 38615 | 0.6484 | 0.5449 | -0.1035 |

Mean paired effect: **+0.0281** (95% shared-opening bootstrap interval +0.0111 to +0.0455).
Worst pair: -0.1035; minimum A/B: 0.4072/0.4941; range A/B: 0.2412/0.0801.
Decision: **retain_e4_baseline**. Failed criteria: mean_effect_at_least_0.03, at_least_four_nonnegative_pairs, worst_pair_at_least_minus_0.05.

Inference is conditional on these five orders and this dataset; PR #386's cosine and PR #388's averaging rejections remain unchanged. No model is promoted.

Reproduce from repository root:
```sh
.venv/bin/python -m ml.alphazero_lite.run_seed461_e3_e4_arena
.venv/bin/python -m ml.alphazero_lite.analyze_seed461_e3_e4
.venv/bin/python -m ml.alphazero_lite.reproduce_seed461_score_matrix docs/data/seed461-e3-e4-opening-score-matrix.json --seed 389
```
