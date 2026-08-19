# AlphaZero-Lite Supervised Target-Residual Audit

**Classification:** `targets_individually_sound_objective_distillation_failure`

**Primary question:** are the states driving the harmful PR #197 supervised update asking the network to prefer causally better moves and well-grounded value outcomes?

## PR197 statistical repair

- Whole-game between-shard Adam cosine, median estimator: 0.6199 (95% CI [0.6180, 0.6219])
- Whole-game between-shard Adam cosine, mean estimator (labeled mean): 0.6166 (95% CI [0.6147, 0.6185])

Direct between-minus-within bootstrap (no marginal-CI inference):

| Effective rows | Difference (mean) | 95% CI |
| ---: | ---: | ---: |
| 512 | -0.0307 | [-0.0520, -0.0102] |
| 1024 | -0.0640 | [-0.0844, -0.0443] |
| 2048 | -0.0796 | [-0.0933, -0.0658] |
| 4096 | -0.1099 | [-0.1224, -0.0983] |
| 8192 | -0.1416 | [-0.1511, -0.1319] |

## Frozen probe provenance

- 1024 unique training states, stratified 16x64 (policy-residual quartile x value-residual quartile), sub-stratified by phase/player and spread across policy-target entropy.
- policy-residual quartile boundaries: [0.1549, 0.4627, 1.0495]
- value-residual quartile boundaries: [0.4174, 0.7694, 0.9945]
- stratum counts: `{'p0_v0': 64, 'p0_v1': 64, 'p0_v2': 64, 'p0_v3': 64, 'p1_v0': 64, 'p1_v1': 64, 'p1_v2': 64, 'p1_v3': 64, 'p2_v0': 64, 'p2_v1': 64, 'p2_v2': 64, 'p2_v3': 64, 'p3_v0': 64, 'p3_v1': 64, 'p3_v2': 64, 'p3_v3': 64}`

## Policy residual distribution

- policy residual KL: mean 0.6994, median 0.4647, p90 1.7149, max 3.9784
- value residual |target - value|: mean 0.7136, median 0.7699, p90 1.1749, max 1.5998

## Policy top-move causal quality

- raw-policy vs replay top move: 0.616; raw-policy vs D384: 0.659; raw-policy vs D1200: 0.613; replay vs D384: 0.846; replay vs D1200: 0.919
- replay-target top move vs current D384 top move (forced, current/current continuation), disagreement states: 158

| Budget | Margin delta (replay - current) | 95% CI | Outcome delta |
| ---: | ---: | ---: | ---: |
| 768 | +0.0248 | [-0.0090, +0.0583] | +0.0823 |
| 1200 | +0.0240 | [-0.0111, +0.0593] | +0.1709 |

Margin delta by policy-residual quartile (1200 budget):

| Quartile | Margin delta |
| ---: | ---: |
| 0 | -0.0688 |
| 1 | +0.0929 |
| 2 | +0.0135 |
| 3 | +0.0218 |

## Distribution-level gradient/quality alignment

- bounded subset states: 256
- gradient-quality alignment (cosine of replay mass-shift vs centered quality): mean +0.2879 (95% CI [+0.2121, +0.3648])
- pairwise concordance: 0.5052 (95% CI [0.4590, 0.5514])
- expected causal-quality change under infinitesimal policy step: mean +0.0322 (95% CI [+0.0230, +0.0418])

## Value residual confirmation

- residual sign agreement with D1200 outcome: 0.913 (high-residual: 0.911)
- residual sign agreement with D768 outcome: 0.897
- squared-error change if following replay target (vs D1200 reference): mean -0.4291

## Exploratory-vs-deterministic value-target analysis

- replay immediate move matches current D384 search: 0.791
- replay move matches D384: sign agreement 0.941 (n=785)
- replay move differs from D384: sign agreement 0.792 (n=207)
- early high-temp region: sign agreement 0.689 (n=193)
- later low-temp region: sign agreement 0.965 (n=831)

## Harmful-gradient attribution

- policy aligned rows (n=967): cosine with harmful trunk gradient +0.7350
- policy misaligned rows (n=57): cosine with harmful trunk gradient -0.1230
- value confirmed rows (n=935): cosine with harmful trunk gradient +0.0773
- value disagrees rows (n=89): cosine with harmful trunk gradient -0.0781

## Counterfactual objective-quality estimate

- G_all (exact normal objective): cosine with harmful +0.7210, norm 3.6312, Adam-step cosine +0.4790
- G_policy_aligned_only: cosine with harmful +0.7376, norm 3.8236, Adam-step cosine +0.4829
- G_value_confirmed_only: cosine with harmful +0.7204, norm 3.6468, Adam-step cosine +0.4782
- G_both_filtered: cosine with harmful +0.7369, norm 3.8396, Adam-step cosine +0.4833

## Exact classification evidence

| Signal | Value |
| --- | ---: |
| policy_disagreement_states | 158 |
| policy_negative_at_both_budgets | False |
| policy_ci_zero_at_least_one_budget | False |
| policy_misalignment_stronger_highest_quartile | False |
| policy_misaligned_gradient_aligns_harmful | False |
| value_high_residual_agreement_d1200 | 0.9107 |
| value_disagreement_worse_off_policy | True |
| value_questionable_gradient_aligns_harmful | False |

## One next action

`investigate function-space/search-aware constraints on distillation rather than new targets`

Full evidence: `docs/data/alphazero-lite-supervised-target-residual-audit-summary.json`.
