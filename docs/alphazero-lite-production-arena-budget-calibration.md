# Production Arena Budget Calibration

Inherited PR #330 classification: `production_prefilter_budget_advantage_material`.

Frozen canonical suite SHA256: `811feb7d208467cdfa9a42244eb75adcc8877db5bc03f87dacbd3379e08bd4bf`. Frozen pairs: P47, P48, P49, P50, P51, P52, P_INC, I_INC, I_48.

Only semantic arena change: `current_simulations: 256 -> 384`; challenger remains 384 and all other search settings, suite, seed contract, opening-pair setup, and seats are unchanged.

## Scores

| Pair | 384/256 | 384/384 CI95 | Budget effect CI95 | 0.55 asymmetric/equal |
| --- | --- | --- | --- | --- |
| P47 | 0.6973 | 0.6133 [0.5811, 0.6455] | +0.0840 [+0.0518, +0.1162] | True/True |
| P48 | 0.7490 | 0.6807 [0.6465, 0.7148] | +0.0684 [+0.0352, +0.1025] | True/True |
| P49 | 0.7432 | 0.6953 [0.6621, 0.7275] | +0.0479 [+0.0186, +0.0781] | True/True |
| P50 | 0.7246 | 0.6553 [0.6250, 0.6855] | +0.0693 [+0.0410, +0.0986] | True/True |
| P51 | 0.7188 | 0.6445 [0.6123, 0.6777] | +0.0742 [+0.0400, +0.1084] | True/True |
| P52 | 0.7305 | 0.6582 [0.6230, 0.6924] | +0.0723 [+0.0410, +0.1035] | True/True |
| P_INC | 0.8896 | 0.8389 [0.8115, 0.8662] | +0.0508 [+0.0283, +0.0742] | True/True |
| I_INC | 0.6035 | 0.5000 [0.5000, 0.5000] | +0.1035 [+0.0703, +0.1367] | True/False |
| I_48 | 0.5801 | 0.5000 [0.5000, 0.5000] | +0.0801 [+0.0518, +0.1094] | True/False |

## Equal-Budget Arena Metrics

| Pair | W/D/L | Raw/pair/median | P0/P1 | Margin | Trajectories | Favored/neutral/current | Both challenger/current/split/draw |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P47 | 292/44/176 | 0.6133/0.6133/0.5000 | 0.6602/0.5664 | 3.12 | 509 | 91/144/21 | 59/13/142/42 |
| P48 | 334/29/149 | 0.6807/0.6807/0.5000 | 0.7363/0.6250 | 4.38 | 512 | 116/124/16 | 95/10/123/28 |
| P49 | 337/38/137 | 0.6953/0.6953/0.5000 | 0.7461/0.6445 | 4.48 | 511 | 125/116/15 | 96/6/116/38 |
| P50 | 310/51/151 | 0.6553/0.6553/0.5000 | 0.7383/0.5723 | 3.45 | 509 | 110/134/12 | 69/8/131/48 |
| P51 | 309/42/161 | 0.6445/0.6445/0.5000 | 0.6914/0.5977 | 3.56 | 510 | 105/130/21 | 75/11/129/41 |
| P52 | 314/46/152 | 0.6582/0.6582/0.5000 | 0.7148/0.6016 | 4.50 | 510 | 116/116/24 | 83/13/115/45 |
| P_INC | 417/25/70 | 0.8389/0.8389/1.0000 | 0.8535/0.8242 | 9.16 | 512 | 186/70/0 | 161/0/70/25 |
| I_INC | 221/70/221 | 0.5000/0.5000/0.5000 | 0.5605/0.4395 | 0.00 | 255 | 0/256/0 | 0/0/221/35 |
| I_48 | 237/38/237 | 0.5000/0.5000/0.5000 | 0.5254/0.4746 | 0.00 | 256 | 0/256/0 | 0/0/237/19 |

## Identity Controls

| Pair | Equal score | Seat P0/P1 | Bias removed |
| --- | --- | --- | --- |
| I_INC | 0.5000 | 0.5605/0.4395 | True |
| I_48 | 0.5000 | 0.5254/0.4746 | True |

Equal controls use deterministic trajectory pairing and their paired games reverse winners by seat; seed/search ledgers are retained in the equal-budget arena artifacts.

## Causal Effects

| Pair | Mean | Median | P10/P90 | Positive/zero/negative | Margin shift |
| --- | --- | --- | --- | --- |
| P47 | +0.0840 | +0.0000 | +0.0000/+0.5000 | 74/158/24 | +2.24 |
| P48 | +0.0684 | +0.0000 | -0.2500/+0.5000 | 67/159/30 | +2.36 |
| P49 | +0.0479 | +0.0000 | -0.2500/+0.5000 | 57/172/27 | +1.39 |
| P50 | +0.0693 | +0.0000 | +0.0000/+0.5000 | 67/165/24 | +1.52 |
| P51 | +0.0742 | +0.0000 | -0.1250/+0.5000 | 73/157/26 | +1.93 |
| P52 | +0.0723 | +0.0000 | +0.0000/+0.5000 | 62/169/25 | +1.70 |
| P_INC | +0.0508 | +0.0000 | +0.0000/+0.2500 | 46/196/14 | +1.89 |
| I_INC | +0.1035 | +0.0000 | -0.2500/+0.5000 | 88/137/31 | +2.22 |
| I_48 | +0.0801 | +0.0000 | +0.0000/+0.5000 | 62/177/17 | +1.57 |

Observed asymmetric advantage is `score_384_256 - 0.5`; equal-budget model advantage is `score_384_384 - 0.5`; budget shift is their difference. These are descriptive paired comparisons, not an exact additive correction.

## Retention And Threshold

| Pair | Direction | Asymmetric >=0.55 | Equal >=0.55 | Equal positive evidence |
| --- | --- | --- | --- |
| P47 | direction_statistically_supported | True | True | True |
| P48 | direction_statistically_supported | True | True | True |
| P49 | direction_statistically_supported | True | True | True |
| P50 | direction_statistically_supported | True | True | True |
| P51 | direction_statistically_supported | True | True | True |
| P52 | direction_statistically_supported | True | True | True |
| P_INC | direction_statistically_supported | True | True | True |

P_INC independent PR #329 holdout 384/256 score: `0.8838`; PR #330 calibration 384/256 score: `0.8896`; this suite's 384/384 score and paired effect are shown above. `seed48_incumbent_equal_budget_confirmed`: `True`.
Known positive evidence at equal budget: `7/7`; equal-budget 0.55 pass: `7/7`; asymmetry-dependent 0.55 pass: `0/7`.

## Opening Sensitivity

Positive budget-effect histogram across seven known-positive pairs: 0/7=47, 1/7=68, 2/7=73, 3/7=47, 4/7=15, 5/7=5, 6/7=1, 7/7=0.
Repeated budget-sensitive openings (>=5/7): `6`: [9, 74, 78, 87, 191, 208].
Outcome-transition counts and all per-opening effects are retained per pair in the aggregate/artifacts. Budget-shift heterogeneity mean/median/range: +0.0723/+0.0723/[+0.0479, +0.1035].

## Rules

Primary budget-causal rule: `True`. Model-strength retention rule: `True`.

## Classification

`production_prefilter_budget_asymmetry_confound_confirmed_strength_retained`

Exactly one next experiment: Implement an evaluation-only SHADOW canonical-opening prefilter in local_promotion_gate, disabled by default, using equal 384/384 search, the frozen calibration methodology, and opening-pair scoring. Run seed48 once through the complete downstream shadow gate.


## Outcome Transitions

`P47`: {"challenger_favored_to_neutral": 13, "current_favored_to_challenger_favored": 6, "current_favored_to_neutral": 10, "neutral_to_challenger_favored": 45, "neutral_to_current_favored": 8, "unchanged": 174}
`P48`: {"challenger_favored_to_current_favored": 2, "challenger_favored_to_neutral": 16, "current_favored_to_challenger_favored": 6, "current_favored_to_neutral": 8, "neutral_to_challenger_favored": 42, "neutral_to_current_favored": 2, "unchanged": 180}
`P49`: {"challenger_favored_to_neutral": 17, "current_favored_to_challenger_favored": 4, "current_favored_to_neutral": 7, "neutral_to_challenger_favored": 29, "neutral_to_current_favored": 1, "unchanged": 198}
`P50`: {"challenger_favored_to_neutral": 20, "current_favored_to_challenger_favored": 4, "current_favored_to_neutral": 4, "neutral_to_challenger_favored": 39, "neutral_to_current_favored": 1, "unchanged": 188}
`P51`: {"challenger_favored_to_current_favored": 4, "challenger_favored_to_neutral": 16, "current_favored_to_challenger_favored": 7, "current_favored_to_neutral": 11, "neutral_to_challenger_favored": 43, "neutral_to_current_favored": 3, "unchanged": 172}
`P52`: {"challenger_favored_to_current_favored": 1, "challenger_favored_to_neutral": 19, "current_favored_to_challenger_favored": 10, "current_favored_to_neutral": 8, "neutral_to_challenger_favored": 33, "neutral_to_current_favored": 1, "unchanged": 184}
`P_INC`: {"challenger_favored_to_current_favored": 1, "challenger_favored_to_neutral": 5, "neutral_to_challenger_favored": 32, "unchanged": 218}
`I_INC`: {"neutral_to_challenger_favored": 88, "neutral_to_current_favored": 31, "unchanged": 137}
`I_48`: {"neutral_to_challenger_favored": 62, "neutral_to_current_favored": 17, "unchanged": 177}
Descriptive Spearman equal-budget score versus budget shift: `-0.9286`; no mechanism is inferred.