# Seed48 Search-Target Exact Quality Audit

Classification: `search_targets_budget_saturated`.

This evaluation-only audit uses 200 unique KVTB1 tier-21 states from standard-start, seed-340 trajectories. The trajectory searches use deterministic evaluation settings (384 PUCT, `c_puct=1.25`, zero FPU, no subtree reuse, no value normalization, no tactical bias, no noise); a seed-340 sampler draws trajectory actions from their noise-free visit policies to produce a reproducible corpus. All corpus rows have `training_eligible: false` and were not added to replay.

KVTB1 action values are final player-0 pit margins. The audit adds the root stores and negates for a player-1 root before ranking actions. Tied best exact scores are retained as the optimal set.

| Method | Top-1 optimal | Optimal mass | Top-1 regret | Expected regret |
| --- | ---: | ---: | ---: | ---: |
| Raw | 0.715 | 0.630 | 1.550 | 2.284 |
| MCTS-96 | 0.750 | 0.702 | 1.110 | 1.586 |
| MCTS-384 | 0.770 | 0.711 | 1.030 | 1.508 |
| MCTS-1200 | 0.755 | 0.720 | 1.130 | 1.440 |

| Active stones | States | Raw optimal mass | MCTS-1200 optimal mass | Raw expected regret | MCTS-1200 expected regret |
| --- | ---: | ---: | ---: | ---: |
| 20-21 | 50 | 0.546 | 0.663 | 2.662 | 1.367 |
| 16-19 | 50 | 0.585 | 0.705 | 2.839 | 1.911 |
| 11-15 | 50 | 0.619 | 0.693 | 2.604 | 1.740 |
| 0-10 | 50 | 0.769 | 0.820 | 1.031 | 0.744 |

Paired bootstrap, 10,000 resamples, seed 340:

| Comparison | Optimal-mass delta, 95% CI | Expected-regret delta, 95% CI |
| --- | ---: | ---: |
| MCTS-96 minus raw | +0.072 [+0.051, +0.095] | -0.698 [-0.887, -0.518] |
| MCTS-384 minus raw | +0.081 [+0.052, +0.110] | -0.776 [-1.021, -0.541] |
| MCTS-1200 minus raw | +0.090 [+0.057, +0.124] | -0.844 [-1.122, -0.575] |
| MCTS-1200 minus MCTS-384 | +0.009 [-0.001, +0.020] | -0.067 [-0.138, +0.003] |

MCTS-1200 clearly improves both primary exact-distribution metrics over raw. Its 384-to-1200 paired intervals cross zero for both metrics, and no active-stone bucket regresses on either metric. Per the requested decision rule, do not change the budget in this work; run a controlled 384-vs-1200 self-play training-cost ablation before another full 1200-simulation iteration.

Artifacts and exact command are in `docs/data/alphazero-lite-seed48-target-quality/config.json`. The full machine-readable result includes by-bucket, legal-action-count, capture/extra-turn splits, all exact action values, and 20 ranked diagnostic disagreements.
