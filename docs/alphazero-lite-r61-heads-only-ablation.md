# R61 Heads-Only Shared-Trunk Ablation

Inherited PR #315 classifications: `legal_policy_metric_mismatch_confirmed_no_decision_flip`; `anchor_drift_shared_trunk_primary`.

## Artifact And Scope Manifest

- G0 SHA-256: `4cd77f12319935d776b68c5e597b8d399fd68f15ba7ea25b3d66bbe786292ed4`
- Frozen-set SHA-256: `5f86367fdb75af02294a1ace1c08dc2938919f0a5f0419376bdf89d127b6257a`
- Canonical checkpoint metric: `legal_normalized_metric`.
- Guardrails: no self-play, replay mutation, frozen-state injection, exact labels, search changes, optimizer changes, or promotion.

## Results

| seed | scope | top | optimal mass | P(0) | margin | trunk immutable |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| T61 | all | 3 | 0.2737 | 0.2737 | -0.2410 | None |
| T63 | all | 0 | 0.5548 | 0.5548 | 0.9733 | None |
| T61 | heads_only | 4 | 0.2456 | 0.2456 | -0.3374 | True |
| T63 | heads_only | 4 | 0.2251 | 0.2251 | -0.5334 | True |

## Classification

`heads_only_no_anchor_rescue`

Exactly one next experiment: test last_block_policy on frozen R61 as the smallest existing scope that permits limited representation adaptation.
