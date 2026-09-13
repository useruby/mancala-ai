# True-Outcome Subtree Evaluator/Backup Attribution Audit

## Inherited Classification

`uniform1200_true_outcome_regression_confirmed` from PR #301. The primary population is exactly seed 44 / capture_available-025 and seed 45 / capture_available-018; capture-002 and capture-020 are excluded as margin-only regressions.

## Baseline And Coverage

All traced baselines reproduced their historical selected moves before post-hoc labels were requested. Checkpoint and oracle SHAs are in the machine artifact.

| Case | Selected | Regret | Leaf coverage | Internal coverage | Backup mismatches | Mechanism |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 44:control:capture_available-025 | 2 | 0 | 1.000 | 1.000 | 0 | `subtree_policy_error` |
| 44:uniform1200:capture_available-025 | 0 | 1 | 1.000 | 1.000 | 0 | `subtree_policy_error` |
| 45:control:capture_available-018 | 0 | 0 | 1.000 | 1.000 | 0 | `subtree_policy_error` |
| 45:uniform1200:capture_available-018 | 1 | 2 | 1.000 | 1.000 | 0 | `subtree_policy_error` |
| 44:parent:capture_available-025 | 0 | 1 | 1.000 | 1.000 | 0 | `subtree_policy_error` |
| 45:parent:capture_available-018 | 0 | 0 | 1.000 | 1.000 | 0 | `subtree_policy_error` |

## Same-Path Exact Backup

Exact leaf W/D/L values are applied only offline to the recorded paths; visits and selections remain historical.

- `44:control:capture_available-025`: repair=False, first stable repair=None, baseline Q={0: -0.027974786758808524, 2: 0.01008101922698504, 3: 0.0017230270979164226, 4: -0.00037114054020344686}, exact-leaf Q={0: 0.07608695652173914, 2: -0.05825242718446602, 3: -0.03296703296703297, 4: -0.40816326530612246}.
- `44:uniform1200:capture_available-025`: repair=False, first stable repair=4, baseline Q={0: 0.013315158479872618, 2: 0.026777479627666325, 3: 0.010537224122023153, 4: 0.0648708239120503}, exact-leaf Q={0: 0.08982035928143713, 2: 0.3076923076923077, 3: 0.2, 4: 0.19548872180451127}.
- `45:control:capture_available-018`: repair=False, first stable repair=32, baseline Q={0: 0.045862175556261806, 1: 0.018961839167651328, 2: 0.05404505852438771, 3: -0.01611203442301626, 4: -0.003594988998664502}, exact-leaf Q={0: 0.4880239520958084, 1: 0.09090909090909091, 2: 0.625, 3: 1.0, 4: 1.0}.
- `45:uniform1200:capture_available-018`: repair=False, first stable repair=1, baseline Q={0: 0.09173798876448638, 1: 0.18742939770544093, 2: 0.20392636024214553, 3: 0.08425892216396377, 4: 0.1804668588448549}, exact-leaf Q={0: 0.671875, 1: 0.34558823529411764, 2: 0.41304347826086957, 3: 1.0, 4: 1.0}.
- `44:parent:capture_available-025`: repair=False, first stable repair=None, baseline Q={0: -0.006579515586677538, 2: 0.0020415790962606006, 3: 0.012430001659914994, 4: -0.03219113417159994}, exact-leaf Q={0: 0.03017241379310345, 2: -0.06521739130434782, 3: -0.017241379310344827, 4: -0.4583333333333333}.
- `45:parent:capture_available-018`: repair=False, first stable repair=9, baseline Q={0: -0.019541096272839602, 1: 0.01505237432719122, 2: -0.010174710104149373, 3: 0.0010567375961233944, 4: -0.017889499501121512}, exact-leaf Q={0: 0.09285714285714286, 1: -0.46551724137931033, 2: 1.0, 3: 0.4, 4: 0.4375}.

## Controls

Seed-46 control/uniform traces for both roots and available SHA-verified parents are retained in the machine artifact as negative and contextual controls.

## Classification

`true_outcome_subtree_policy_failure_primary`

## One Next Experiment

Exactly one next experiment: diagnose the first repeated internal outcome-degrading decision family; do not alter the root search yet.

