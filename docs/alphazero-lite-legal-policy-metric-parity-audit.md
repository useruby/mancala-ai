# Legal Policy Metric Parity Audit

Inherited PR #314 result: `historical_parity_audit_inconclusive`. It established identical T61/T63 checkpoint bytes under the shared runtime and identified the checkpoint-vs-in-memory policy semantic mismatch.

## Semantics

`numpy_full_softmax` retains all six CheckpointEvaluator probabilities. `numpy_legacy_legal_zeroed` zeros illegal mass without renormalizing. `numpy_legal_normalized` renormalizes legal mass. `torch_legal_normalized` masks illegal logits before softmax. PUCT uses the legal-normalized interpretation.

## Primary Anchor

| checkpoint | illegal mass | legacy optimal mass | normalized optimal mass | legacy delta | normalized delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| G0 | 0.031877778 | 0.387194991 | 0.399944328 | 0.000000000 | 0.000000000 |
| T61 | 0.421765201 | 0.158261746 | 0.273698066 | -0.228933245 | -0.126246262 |
| T63 | 0.015000299 | 0.546428978 | 0.554750400 | 0.159233987 | 0.154806072 |

T61 assigns `0.421765201` probability to its two illegal actions. Dividing its legacy optimal mass by legal mass exactly produces the canonical `0.273698066`; the normalized T61 delta is `-0.126246262`. This fully reconciles the PR #313 in-memory result without a weight difference.

## Forward And PUCT Parity

NumPy/PyTorch maximum legal-policy difference: `3.54e-07`. PUCT maximum root-prior difference: `4.24e-08`. Every normalization identity holds numerically; all PUCT legal-child prior sums are one.

## Full Frozen Set Distortion

The machine artifact persists all four modes, legal counts, legal/illegal mass, both optimal masses, top-one correctness, and both entropy definitions for every G0/T61/T63 frozen state. Distortion increases as legal mass decreases; it does not alter legal top-action ranking.

## Entropy Semantics

`legacy_zeroed_entropy` is the historical entropy of an unnormalized zeroed vector. `legal_normalized_entropy` is the entropy of the actual legal policy. Neither historical hard decision rule used entropy; it was descriptive only.

## Historical Reclassification

PR #306: `internal_cluster_learning_dynamics_heterogeneous`. PR #307 corrected 3x3 classification: `anchor_forgetting_replay_optimizer_interaction`. PR #308 saved checkpoint/trace pairs are byte-identical and their legal-normalized final anchor masses agree.

PR #309, #310, and #311 reuse their saved checkpoints and existing success-rule functions. Arena, forensic, losses, stability, and clipping telemetry are unchanged; only policy-mass-dependent fields are recomputed.

## Historical Decision Dependency

| PR | original classification | depends on legacy policy mass? | corrected classification | decision flips? |
| --- | --- | --- | --- | --- |
| #306 | `internal_cluster_learning_dynamics_heterogeneous` | True | `internal_cluster_learning_dynamics_heterogeneous` | False |
| #307 | `anchor_forgetting_replay_optimizer_interaction` | True | `anchor_forgetting_replay_optimizer_interaction` | False |
| #308 | `anchor_interference_diffuse_optimization` | False | `anchor_interference_diffuse_optimization` | False |
| #309 | `reduced_lr_no_stability_gain` | True | `reduced_lr_no_stability_gain` | False |
| #310 | `larger_batch_no_stability_gain` | True | `larger_batch_no_stability_gain` | False |
| #311 | `grad_clip_relaxation_unstable` | True | `grad_clip_relaxation_unstable` | False |
| #312 | `baseline_reproduction_unexplained` | False | `baseline_reproduction_unexplained` | False |

## Canonical Metric

Use `legal_normalized_metric` for new frozen checkpoint evaluations. Historical output retains `legacy_metric`; legacy-zeroed entropy and legal-normalized entropy are reported separately and are not comparable.

## Hard Classification

`legal_policy_metric_mismatch_confirmed_no_decision_flip`

## Next Experiment

Resume PR #312 parameter-subspace drift audit using one canonical legal-normalized metric and checkpoint-based final evaluation.
