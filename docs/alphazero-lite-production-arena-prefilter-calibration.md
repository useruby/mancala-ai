# Production Arena Prefilter Calibration

Inherited: PR #327 `uniform1200_high_power_strength_confirmed`; PR #328 `uniform1200_promotion_arena_failed`; PR #329 `production_start_state_arena_disagreement_confirmed`.

## Known-Positive Calibration

| Pair | Label | Start score | Start pass | Canonical score | CI95 | Canonical 0.55 pass | Positive evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P47 | known_positive | 0.5000 | False | 0.6973 | [0.6641, 0.7305] | True | True |
| P48 | known_positive | 1.0000 | True | 0.7490 | [0.7188, 0.7793] | True | True |
| P49 | known_positive | 0.5000 | False | 0.7432 | [0.7129, 0.7744] | True | True |
| P50 | known_positive | 1.0000 | True | 0.7246 | [0.6943, 0.7549] | True | True |
| P51 | known_positive | 1.0000 | True | 0.7188 | [0.6865, 0.7510] | True | True |
| P52 | known_positive | 1.0000 | True | 0.7305 | [0.7002, 0.7607] | True | True |
| P_INC | known_positive | 0.5000 | False | 0.8896 | [0.8652, 0.9131] | True | True |

## Identity Controls

| Pair | Start 384/256 | Canonical 384/256 | CI95 | Pure search-budget effect | Material |
| --- | --- | --- | --- | --- | --- |
| I_INC | 0.5000 | 0.6035 | [0.5703, 0.6367] | +0.1035 | True |
| I_48 | 0.5000 | 0.5801 | [0.5518, 0.6084] | +0.0801 | True |

## Production Repeated Start

| Pair | W/D/L | Score | P0 score | P1 score | Unique trajectories | Repetition factor | Unique fraction | Seat explanation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P47 | 60/0/60 | 0.5000 | 1.0000 | 0.0000 | 2 | 60.0 | 0.0167 | deterministic P0 win / P1 loss |
| P48 | 120/0/0 | 1.0000 | 1.0000 | 1.0000 | 2 | 60.0 | 0.0167 | both same |
| P49 | 60/0/60 | 0.5000 | 1.0000 | 0.0000 | 2 | 60.0 | 0.0167 | deterministic P0 win / P1 loss |
| P50 | 120/0/0 | 1.0000 | 1.0000 | 1.0000 | 2 | 60.0 | 0.0167 | both same |
| P51 | 120/0/0 | 1.0000 | 1.0000 | 1.0000 | 2 | 60.0 | 0.0167 | both same |
| P52 | 120/0/0 | 1.0000 | 1.0000 | 1.0000 | 2 | 60.0 | 0.0167 | both same |
| P_INC | 60/0/60 | 0.5000 | 1.0000 | 0.0000 | 2 | 60.0 | 0.0167 | deterministic P0 win / P1 loss |
| I_INC | 60/0/60 | 0.5000 | 1.0000 | 0.0000 | 2 | 60.0 | 0.0167 | deterministic P0 win / P1 loss |
| I_48 | 60/0/60 | 0.5000 | 1.0000 | 0.0000 | 1 | 120.0 | 0.0083 | deterministic P0 win / P1 loss |

Every repeated-start arena has one start state. Its nominal 120 games therefore do not correspond to 120 distinct position/game observations; the table reports direct observed trajectory duplication only, not an effective-sample-size estimate.

## Canonical Results

| Pair | W/D/L | Pair score | Median | CI95 | P0/P1 | Margin | Unique trajectories |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P47 | 338/38/136 | 0.6973 | 0.7500 | [0.6641, 0.7305] | 0.7441/0.6504 | 5.37 | 509 |
| P48 | 367/33/112 | 0.7490 | 0.7500 | [0.7188, 0.7793] | 0.8027/0.6953 | 6.74 | 512 |
| P49 | 366/29/117 | 0.7432 | 0.7500 | [0.7129, 0.7744] | 0.7832/0.7031 | 5.88 | 511 |
| P50 | 354/34/124 | 0.7246 | 0.7500 | [0.6943, 0.7549] | 0.8027/0.6465 | 4.96 | 509 |
| P51 | 351/34/127 | 0.7188 | 0.7500 | [0.6865, 0.7510] | 0.7520/0.6855 | 5.49 | 509 |
| P52 | 355/38/119 | 0.7305 | 0.7500 | [0.7002, 0.7607] | 0.7891/0.6719 | 6.20 | 510 |
| P_INC | 443/25/44 | 0.8896 | 1.0000 | [0.8652, 0.9131] | 0.9141/0.8652 | 11.05 | 512 |
| I_INC | 282/54/176 | 0.6035 | 0.5000 | [0.5703, 0.6367] | 0.6660/0.5410 | 2.22 | 489 |
| I_48 | 279/36/197 | 0.5801 | 0.5000 | [0.5518, 0.6084] | 0.6133/0.5469 | 1.57 | 478 |

## Opening-Pair Distribution

| Pair | Challenger favored | Neutral | Current favored | Challenger both | Current both | Split seats | Draw pairs |
| --- | --- | --- | --- | --- | --- | --- |
| P47 | 129 | 114 | 13 | 95 | 9 | 114 | 38 |
| P48 | 146 | 104 | 6 | 117 | 2 | 104 | 33 |
| P49 | 141 | 110 | 5 | 115 | 2 | 110 | 29 |
| P50 | 133 | 118 | 5 | 105 | 3 | 116 | 32 |
| P51 | 135 | 111 | 10 | 105 | 6 | 111 | 34 |
| P52 | 139 | 109 | 8 | 107 | 2 | 109 | 38 |
| P_INC | 212 | 43 | 1 | 188 | 0 | 43 | 25 |

Cross-pair opening consistency for only the six uniform-vs-control pairs, kept separate from P_INC: 0/6=21, 1/6=22, 2/6=37, 3/6=53, 4/6=66, 5/6=38, 6/6=19.

## Design And Calibration

Frozen manifest has seven known-positive labels before results: P47-P52 inherited from PR #327 and P_INC from PR #329. The two identity controls have no positive/negative label and are excluded from every /7 sensitivity denominator.
Third suite SHA256: `811feb7d208467cdfa9a42244eb75adcc8877db5bc03f87dacbd3379e08bd4bf`; population 942, historical overlap 0, remaining 430, selected overlap PR #327/#329 0/0. It uses lexicographic representatives and stable-hash selection with version `production_prefilter_calibration_unique_v1` and seed 330, without model input.
Production contract: 120 games, threshold 0.55, 384/256, seed 42, seed contract azlite_eval_seed_v2, zero opening plies, deterministic PUCT, c_puct 1.25, zero FPU, no subtree reuse/normalization, root temperature and tactical bias 0, no root-prior/value transforms or opening cache.
False negatives: `3/7`; both pass `4/7`; start fail/canonical pass `3/7`; start pass/canonical fail `0/7`; both fail `0/7`.
Start/canonical distinct scores: 2/7; variances 0.061224/0.003476. Descriptive Spearman historical-vs-start/historical-vs-canonical/start-vs-canonical: -0.5774/0.7857/-0.1443; historical populations differ, so these are descriptive only.
Median repeated-start unique-trajectory fraction: `0.0167`. Both canonical seats exceed 0.50 for `7/7` known-positive pairs.

## Classification

`production_prefilter_budget_advantage_material`

Next experiment: Compare the same frozen calibration pairs under 384/256 versus equal-budget 384/384 on the SAME frozen calibration suite.
