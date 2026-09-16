# R61 Activation Representation Audit

Inherited #315: `anchor_drift_shared_trunk_primary`; #316: `heads_only_no_anchor_rescue`; #317: `last_block_policy_no_anchor_rescue`; #318: `shared_trunk_loss_gradient_not_explanatory`.

## Baseline And Activation Parity

{
  "T61": {
    "expected_sha256": "50736a6063cd0558329c433ad7a6c60b741c83ceee75d36e21e703749af8c9de",
    "reference_sha256": "50736a6063cd0558329c433ad7a6c60b741c83ceee75d36e21e703749af8c9de",
    "instrumented_sha256": "50736a6063cd0558329c433ad7a6c60b741c83ceee75d36e21e703749af8c9de",
    "reproduced": true
  },
  "T63": {
    "expected_sha256": "bab60465a9bc93ac569ed28b356ef2abde2c1cbb0987a4a474e26117f9a57324",
    "reference_sha256": "bab60465a9bc93ac569ed28b356ef2abde2c1cbb0987a4a474e26117f9a57324",
    "instrumented_sha256": "bab60465a9bc93ac569ed28b356ef2abde2c1cbb0987a4a474e26117f9a57324",
    "reproduced": true
  }
}

A3 was checked against `trunk_features`; A3-to-policy reconstruction and self-patching passed for every frozen state and endpoint. Both A3 directions also exactly matched the #315 shared-trunk hybrids.

## Representation And Specificity

Matched-progress per-stage T61/T63 distance, G0 drift, support disagreement, and cluster/control separation are retained at every landmark in the JSON artifact. Live per-step scalar activation telemetry covers all frozen states.

## Temporal Patching

First same-progress anchor bidirectional causal representation step: 108.

## Final Patching

Earliest causal activation stage: `A0`.

| stage | anchor rescue | cluster rescue | transfer | control degradation |
| --- | --- | ---: | ---: | ---: |
| A0 | True | 100.0% | 0.883 | 0.0% |
| A1 | True | 100.0% | 1.711 | 0.0% |
| A2 | True | 100.0% | 1.036 | 0.0% |
| A3 | True | 100.0% | 0.888 | 0.0% |
| A4 | True | 100.0% | 1.018 | 0.0% |

## Block And Policy Path

Native residual deltas, same-input block-function comparisons, policy-hidden path cross-products, secondary value-hidden distances, and statewise Spearman associations are in the machine-readable artifact.

## Classification

`cluster_representation_drift_early_trunk_primary`

Exactly one next experiment: audit which ordinary R61 replay minibatches produce the largest EARLY-TRUNK activation movement on the frozen cluster, using the already-recorded historical trajectory.
