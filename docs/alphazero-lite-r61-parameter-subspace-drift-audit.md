# R61 Parameter Subspace Drift Audit

Inherited PR #311 classification: `grad_clip_relaxation_unstable`.

## Baseline Reproduction

The required historical-baseline reproduction gate failed, so this diagnostic stopped before collecting or interpreting trajectory and hybrid-swap evidence.

| seed | expected anchor optimal-mass delta | observed | final top action |
| --- | ---: | ---: | ---: |
| T61 | -0.2289 | -0.1262 | 3 |
| T63 | +0.1592 | +0.1548 | 0 |

The fixed artifacts were SHA-verified: G0 `4cd77f12319935d776b68c5e597b8d399fd68f15ba7ea25b3d66bbe786292ed4`, R61 `6218a63b4817a58ce41df9c701c3816739976d26a1629aa32b158f3d0650aa87`, and frozen set identity `5f86367fdb75af02294a1ace1c08dc2938919f0a5f0419376bdf89d127b6257a`.

## Audit Design

The runner mechanically partitions all residual_v3 trainable tensors into `trunk_input`, `trunk_residual`, `policy_hidden`, `policy_readout`, `value_hidden`, and `value_readout`, rejecting any missing or overlapping membership. It supports compact per-step telemetry, drift probes, offline whole-group hybrids, and the value-path policy-logit negative control, but none were executed after the reproduction failure.

## Classification

`parameter_drift_baseline_not_reproduced`

No next intervention was run or selected. No self-play, replay mutation, replay-weight change, optimizer change, anchor injection, search change, or promotion occurred.
