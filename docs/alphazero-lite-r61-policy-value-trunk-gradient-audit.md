# R61 Policy vs Value Shared-Trunk Gradient Audit

Inherited #315: `anchor_drift_shared_trunk_primary`. Inherited #316: `heads_only_no_anchor_rescue`. Inherited #317: `last_block_policy_no_anchor_rescue`.

## Classification

`shared_trunk_loss_gradient_not_explanatory`

Exactly one next experiment: move to activation/representation drift on the frozen cluster across the original full T61/T63 trajectories.

## Protocol And Reproduction

The only live runs were original full-scope R61/T61 and R61/T63. Both use G0, dynamic R61 plus unchanged fixed replay at weights 1,1,2, residual_v3 96x3, Adam LR 0.001, batch 512, four epochs, global clip 1.0, no scheduler, and historical best-state restoration. No self-play, replay mutation, target change, frozen-state injection, search change, promotion, scope change, or detach intervention occurred.

The mechanically observed loss was `L_total = L_policy + 0.3 * L_value`; pairwise and behavior-anchor weights were zero on every live batch. The maximum additive-identity errors were 3.91e-7 max / 6.03e-9 RMS for T61 and 6.31e-7 max / 7.60e-9 RMS for T63.

| seed | reference SHA | instrumented SHA | selected best epoch | parity |
| --- | --- | --- | ---: | --- |
| T61 | `50736a6063cd0558329c433ad7a6c60b741c83ceee75d36e21e703749af8c9de` | `50736a6063cd0558329c433ad7a6c60b741c83ceee75d36e21e703749af8c9de` | 4 | True |
| T63 | `bab60465a9bc93ac569ed28b356ef2abde2c1cbb0987a4a474e26117f9a57324` | `bab60465a9bc93ac569ed28b356ef2abde2c1cbb0987a4a474e26117f9a57324` | 4 | True |

The instrumented runs also matched epoch permutations, every recorded anchor margin, and validation-loss history. The audit is chronological; the displayed final checkpoints are restored-best states.

## Gradient And Conflict Results

All component vectors use the historical common clip scalar, not independently clipped components. Across 432 steps, shared-trunk policy/value cosine is weakly aligned, not systematically conflicting.

| seed | policy norm | weighted value norm | mean cosine | p10/p90 cosine | conflicting fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| T61 | 3.3103 | 0.1122 | 0.0950 | -0.0704 / 0.2737 | 0.46% |
| T63 | 3.3703 | 0.1134 | 0.0920 | -0.0688 / 0.2642 | 0.69% |

Blockwise mean cosine was positive in both seeds: input 0.0948/0.0903, block 0 0.1146/0.1099, block 1 0.0804/0.0861, and block 2 0.0461/0.0477 (T61/T63). No block shows a stable, source-specific conflict concentration.

## First-Order Anchor And Cluster Attribution

Positive anchor effect denotes an increase in action-0 margin under the component's negative clipped gradient step. Frozen-cluster effect uses the evaluation-only exact-optimal legal-policy probe; it was never added to training.

| seed | policy helpful anchor | policy harmful anchor | value helpful anchor | value harmful anchor |
| --- | ---: | ---: | ---: | ---: |
| T61 | 7645.16 | 5813.36 | 108.99 | 119.52 |
| T63 | 8741.15 | 5359.00 | 90.80 | 112.40 |

| seed | policy harmful cluster | value harmful cluster | policy net cluster | value net cluster |
| --- | ---: | ---: | ---: | ---: |
| T61 | 1064.24 | 19.43 | -731.37 | -0.92 |
| T63 | 1239.60 | 19.07 | -1020.50 | -1.47 |

Policy accounts for approximately 98% of T61 negative anchor first-order magnitude, but its cluster harm is larger in T63, not T61. Value harm is only modestly larger for T61. Thus neither source differentiates the failing trajectory on the frozen cluster as required by the dominance rule. Per-step matched-control probe telemetry is retained under `control_groups` in the machine-readable artifact.

## Timing And Adam Alignment

The first opposite correctness signs occur at optimizer step 3: T61 margin 2.0738 and T63 margin -3.6654. Before that point, T61's mean shared-trunk policy/value cosine is -0.1312 and T63's is 0.0723; after it they are 0.0961 and 0.0920. This brief early difference does not persist as systematic policy/value opposition.

The artifact records, for every fine trunk group and shared trunk, cosine of actual Adam delta against negative policy, negative weighted value, and negative combined gradient, plus `dot(delta_actual, g_margin)`. This connects first-order raw-gradient attribution to the real Adam update rather than treating gradients as updates.

## Local Adam Counterfactuals

Nineteen deterministic selected T61 steps were audited (the union of ten largest negative observed moves and ten strongest negative predicted component effects), with same-progress T63 counterparts. Clone A reproduced every selected historical next tensor exactly: maximum absolute error 0.0.

| seed | remove-policy rescue rate | remove-value rescue rate | mean remove-policy margin effect | mean remove-value margin effect |
| --- | ---: | ---: | ---: | ---: |
| T61 | 100.0% | 73.7% | 0.7326 | 0.0034 |
| T63 | 52.6% | 68.4% | 0.1446 | -0.0016 |

Removing policy from the trunk produces substantially larger local T61 margin improvements, but removing value also clears the 70% rescue threshold. Because the opposite removal shows an equally strong rescue rate, the primary-dominance rule rejects a single-source claim. No causal training ablation was run.

## Artifacts

Machine-readable per-step component telemetry and selected-step counterfactuals: `.tmp/r61-policy-value-trunk-gradient-audit/result.json`. The artifact includes common-clip norms/fractions, residual-block cosines, anchor and frozen-cluster effects, matched-control probe effects, optimizer-delta alignment, validation history, permutation hashes, and cloned-Adam variants.
