# R61 Policy vs Value Shared-Trunk Gradient Audit

Inherited #315: `anchor_drift_shared_trunk_primary`. Inherited #316: `heads_only_no_anchor_rescue`. Inherited #317: `last_block_policy_no_anchor_rescue`.

## Classification

`shared_trunk_loss_gradient_not_explanatory`

Exactly one next experiment: move to activation/representation drift on the frozen cluster across the original full T61/T63 trajectories.

## Baseline Reproduction

The only live runs were full-scope R61/T61 and R61/T63 with G0, unchanged dynamic/fixed replay and weights 1,1,2, residual_v3 96x3, Adam LR 0.001, B512, four epochs, clip 1.0, and no scheduler. No self-play, replay mutation, target injection, scope change, detach intervention, search change, or promotion occurred.

| seed | reference SHA | instrumented SHA | parity |
| --- | --- | --- | --- |
| T61 | `50736a6063cd0558329c433ad7a6c60b741c83ceee75d36e21e703749af8c9de` | `50736a6063cd0558329c433ad7a6c60b741c83ceee75d36e21e703749af8c9de` | True |
| T63 | `bab60465a9bc93ac569ed28b356ef2abde2c1cbb0987a4a474e26117f9a57324` | `bab60465a9bc93ac569ed28b356ef2abde2c1cbb0987a4a474e26117f9a57324` | True |

## Exact Loss

`L_total = L_policy + 0.3 * L_value`; pairwise and behavior-anchor weights were asserted zero on every live minibatch. The machine-readable artifact records pre/post-common-clip component norms, additive identity errors, residual-block cosines, anchor/cluster first-order effects, actual Adam-update alignment, and chronological validation history.

Maximum total-gradient additive-identity error was 3.91e-7 max / 6.03e-9 RMS for T61 and 6.31e-7 max / 7.60e-9 RMS for T63. The instrumented run matched permutations, anchor margins, validation history, restored-best epoch 4, and final checkpoint bytes in both seeds.

## Shared-Trunk Attribution

| seed | policy harmful anchor | value harmful anchor | policy helpful | value helpful |
| --- | ---: | ---: | ---: | ---: |
| T61 | 5813.36 | 119.522 | 7645.16 | 108.993 |
| T63 | 5359 | 112.4 | 8741.15 | 90.7978 |

## Frozen Cluster And Conflict

| seed | mean trunk cosine | p10/p90 | conflicting | policy harmful cluster | value harmful cluster |
| --- | ---: | ---: | ---: | ---: | ---: |
| T61 | 0.0950 | -0.0704 / 0.2737 | 0.46% | 1064.24 | 19.4266 |
| T63 | 0.0920 | -0.0688 / 0.2642 | 0.69% | 1239.6 | 19.0681 |

Mean shared-trunk policy/value norms were 3.3103/0.1122 (T61) and 3.3703/0.1134 (T63). The positive mean cosines and sub-1% conflicting fractions do not support systematic policy/value conflict. The artifact retains matched-control probe effects separately.

## Blockwise And Timing

Per-step JSON contains input/residual-block 0/1/2 and full shared-trunk norms, fractions, common-clipped effects, and Adam alignment. The first opposite anchor-margin correctness sign occurs at step 3.
Neither seed has a residual block with stable source-specific conflict concentration; the full blockwise aggregates and pre/post-divergence split are retained in the JSON artifact.

Before divergence, T61/T63 mean shared-trunk policy/value cosine was -0.1312/0.0723; post-divergence it was 0.0961/0.0920. Actual-update telemetry records cosine with negative policy, negative weighted value, and negative combined gradients plus `dot(delta_actual, g_margin)` for every trunk group.

## Counterfactual Validation

Every selected clone-A historical step reproduced the next live tensors before B/C interpretation. B/C include anchor, frozen-cluster, and matched-control probe values.

| seed | selected steps | remove-policy margin rescue | remove-value margin rescue |
| --- | ---: | ---: | ---: |
| T61 | 19 | 100.0% | 73.7% |
| T63 | 19 | 52.6% | 68.4% |

The T61 policy component is the larger first-order anchor term, but value removal also rescues at least 70% of selected T61 steps. The opposite source therefore shows equally strong local rescue under the pre-registered primary-dominance rule; no source-specific training intervention is justified.

Clone-A maximum next-tensor error was 0.0 for every selected step. Mean T61 removal effects were 0.7326 for policy and 0.0034 for value; matched-control and frozen-cluster effects are retained with each variant.


## Target Quality

The artifact records policy entropy, value-target distribution, phase bucket, outcome, and replay identity for the ten strongest harmful policy and value minibatches per seed. These are descriptive only and do not initiate replay investigation.

## Artifact

Machine-readable per-step telemetry and selected-step Adam counterfactuals are retained at `.tmp/r61-policy-value-trunk-gradient-audit/result.json`.
