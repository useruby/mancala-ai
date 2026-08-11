# AlphaZero-Lite Policy-Target Noise Causal Closeout

- Classification: `policy_target_noise_not_primary_confirmed`
- Additional classifications: `none`

## PR #177 Reproduction

```json
{
  "denoised_js": 0.019356463036061033,
  "denoised_top1": 0.8346354166666666,
  "disagreement_count": 103,
  "disagreement_fraction": 0.13411458333333334,
  "noisy_js": 0.03525294751849997,
  "noisy_top1": 0.81640625,
  "probe_count": 768
}
```

## Frozen Inputs

- current weights SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- probe manifest SHA256: `36a83804e6eed1844506a4f17a1abbdf8105bcb824ae763ab474b55e3d1432d5`
- teacher records SHA256: `f49e3ccbbf12dbd1593e399bd262593c9485dec75eec361d811fdff0ccf192f7`
- code commit: `2944f455876339c47c813733bc5c8be718b805d5`
- disagreement states: `103`

## Forced Outcomes

| Budget | Mean outcome delta | Median | 95% CI | Mean / median store margin | Denoised / noisy / tied |
| --- | ---: | ---: | --- | --- | --- |
| 768 | -0.1553 | 0.0000 | [-0.3883, 0.0874] | -1.8835 / 0.0000 | 0.233 / 0.340 / 0.427 |
| 1200 | -0.1165 | 0.0000 | [-0.3204, 0.0971] | -2.3883 / -2.0000 | 0.194 / 0.272 / 0.534 |

## Seed Pairing

The base continuation identity contains only state hash, continuation budget, root player, and experiment seed. It excludes intervention label, forced move, and teacher identity. The same context is used for both forced interventions; post-move searches use deterministic state-dependent seeds with no Dirichlet noise.

## Disagreement Composition

```json
{
  "legal_move_count": {
    "2": 3,
    "3": 9,
    "4": 33,
    "5": 27,
    "6": 31
  },
  "phase": {
    "late": 10,
    "midgame": 5,
    "opening": 88
  },
  "player": {
    "0": 50,
    "1": 53
  },
  "source_domain": {
    "evaluation_opening_diagnostic": 54,
    "pr176_standard_start_pilot": 49
  }
}
```

## Teacher Reference Agreement

| D384 only matches D1200 | N384 only matches D1200 | Neither matches D1200 | P(D384 = D1200) | P(N384 = D1200) |
| ---: | ---: | ---: | ---: | ---: |
| 52 | 38 | 13 | 0.5049 | 0.3689 |

## Player, Phase, And Domain Slices

| Slice | Group | Budget | States | Mean outcome delta | 95% CI | Mean store-margin delta |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| phase | late | 768 | 10 | 0.0000 | [0.0000, 0.0000] | -0.2000 |
| phase | late | 1200 | 10 | 0.0000 | [0.0000, 0.0000] | -0.2000 |
| phase | midgame | 768 | 5 | -0.4000 | [-0.8000, 0.0000] | -4.4000 |
| phase | midgame | 1200 | 5 | -0.2000 | [-0.6000, 0.0000] | -11.2000 |
| phase | opening | 768 | 88 | -0.1591 | [-0.4432, 0.1250] | -1.9318 |
| phase | opening | 1200 | 88 | -0.1250 | [-0.3636, 0.1250] | -2.1364 |
| player | 0 | 768 | 50 | -0.2400 | [-0.5800, 0.1000] | -1.4800 |
| player | 0 | 1200 | 50 | -0.2200 | [-0.5400, 0.1000] | -3.8400 |
| player | 1 | 768 | 53 | -0.0755 | [-0.4151, 0.2642] | -2.2642 |
| player | 1 | 1200 | 53 | -0.0189 | [-0.3019, 0.2642] | -1.0189 |
| domain_group | evaluation_diagnostic | 768 | 54 | -0.2593 | [-0.6296, 0.1111] | -3.1111 |
| domain_group | evaluation_diagnostic | 1200 | 54 | -0.3148 | [-0.6481, 0.0185] | -3.7037 |
| domain_group | self_play_pilot | 768 | 49 | -0.0408 | [-0.3265, 0.2449] | -0.5306 |
| domain_group | self_play_pilot | 1200 | 49 | 0.1020 | [-0.1429, 0.3469] | -0.9388 |

## D1200-Match Causal Analysis

| Group | Budget | States | D384 minus N384 outcome | 95% CI | Store-margin delta |
| --- | ---: | ---: | ---: | --- | ---: |
| denoised_matches_reference | 768 | 52 | 0.1346 | [-0.1923, 0.4615] | 3.1154 |
| denoised_matches_reference | 1200 | 52 | 0.0385 | [-0.2692, 0.3269] | 0.3077 |
| neither_matches_reference | 768 | 13 | -0.2308 | [-0.8462, 0.3846] | -2.0000 |
| neither_matches_reference | 1200 | 13 | 0.1538 | [-0.3077, 0.6923] | 2.0000 |
| noisy_matches_reference | 768 | 38 | -0.5263 | [-0.9211, -0.1316] | -8.6842 |
| noisy_matches_reference | 1200 | 38 | -0.4211 | [-0.7632, -0.0789] | -7.5789 |

D1200 agreement is directionally predictive here: when N384 alone matches D1200, D384-minus-N384 is negative at both budgets, including an upper CI below zero. The converse group is positive but its intervals include zero, so this is not a positive causal confirmation for denoising.

## Decision Rules

- `policy_target_noise_not_primary_confirmed`: no positive causal evidence, both global CIs include zero, and no sufficiently large robust phase/domain benefit.
- `denoised_policy_target_rejected`: negative mean and upper 95% CI below zero at either budget.
- `denoised_disagreement_moves_causally_better`: at least 64 states, positive means and lower CI >= 0 at both budgets, with no >=24-state player/domain mean below -0.10.
- `target_noise_effect_phase_localized`: inconclusive global result but one >=32-state phase has positive lower CI without corresponding benefit elsewhere.
- `strong_search_reference_not_causally_predictive`: D1200 matching fails to correspond to superior forced outcomes.

Per-state forced-continuation records, including forced move, outcome, store margin, trajectory hash, and paired seed-context hash, remain in the workdir rather than this report.

## Next Action

Audit denoised PUCT target convergence at D128, D384, D768, and D1200 before policy training.
