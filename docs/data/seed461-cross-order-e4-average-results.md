# Cross-order constant-LR E4 checkpoint averaging

Prospective evaluation: one model P, the uniform arithmetic mean of five order-independent A-arm E4 checkpoints. A1–A5 are the individual E4 baselines. Each model played 512 games (two seat-paired games per shared opening) against frozen seed455; total 3,072 games.

| Model | Mean opening score | P−model |
|---|---:|---:|
| P | 0.5547 | — |
| A1 | 0.4316 | +0.1230 |
| A2 | 0.3896 | +0.1650 |
| A3 | 0.4658 | +0.0889 |
| A4 | 0.5596 | -0.0049 |
| A5 | 0.6318 | -0.0771 |

Primary P−mean(A1–A5): **+0.0590** (95% shared-opening bootstrap interval +0.0467 to +0.0713).
P absolute score: **0.5547** (95% interval 0.5420 to 0.5674). Worst comparison: **P-A5 -0.0771**.
Decision: **reject_cross_order_E4_averaging**. Failed conditions: P_score_at_least_four_of_five_baselines, worst_P_minus_Ai_at_least_minus_0.05.

Inference is scoped to these five source trajectories and this dataset. P is one constructed model; its score vector is reused once in all comparisons and shared-opening bootstrap resamples. There is no order-stability claim. Construction used zero training runs; future construction requires five runs. Inference remains single-network.
No model is promoted. PR #386, #388, and #389 rejections remain preserved.

The compact six-model opening-score matrix is bound to report/game hashes in `docs/data/seed461-cross-order-e4-average-opening-score-matrix.json` (SHA-256 `2c78f1eed1f59316dcdc59f64ff67168e5b86babedaabf5e87f967b42683590c`).

Independently reproduce the point estimates and intervals from the bound score matrix:
```sh
.venv/bin/python -m ml.alphazero_lite.reproduce_seed461_cross_order_e4_average docs/data/seed461-cross-order-e4-average-opening-score-matrix.json --seed 390
```
