# R61 Last-Block Policy Ablation

## Classification

Inherited PR #315: `anchor_drift_shared_trunk_primary`. Inherited PR #316: `heads_only_no_anchor_rescue`.

`last_block_policy_no_anchor_rescue`

Exactly one next experiment: run a policy-vs-value gradient attribution audit on the shared trunk under the original full R61 T61/T63 runs, using the existing full-training trajectories.

## Configuration

- G0 SHA-256: `4cd77f12319935d776b68c5e597b8d399fd68f15ba7ea25b3d66bbe786292ed4`.
- Frozen evaluation set SHA-256: `5f86367fdb75af02294a1ace1c08dc2938919f0a5f0419376bdf89d127b6257a`.
- SHA-verified inherited full checkpoints: T61 `50736a6063cd0558329c433ad7a6c60b741c83ceee75d36e21e703749af8c9de`; T63 `bab60465a9bc93ac569ed28b356ef2abde2c1cbb0987a4a474e26117f9a57324`.
- SHA-verified inherited heads-only checkpoints: T61 `684d4ab7c0e244534157af09dffa8a12749815e5015dd2eb492474be958692ba`; T63 `707a0e06cae9ab5eee8db7fef09d337e1e5e20b47bb84bcecf2403ba144dcf1e`.
- R61 dynamic replay plus unchanged fixed sources, weights `1,1,2`, sharpened policy/value targets, residual_v3 96 x 3, Adam, LR 0.001, batch 512, four epochs, clip 1.0, unchanged validation split and restored-best selection.
- No self-play, replay mutation/weight change, frozen-state injection, exact labels as targets, search change, optimizer/hyperparameter/architecture change, or promotion.

## Scope And Invariants

The model had three residual blocks, so the mechanically derived final prefix was `residual_layers.2.`.

- Trainable: `residual_layers.2.*`, `policy_hidden_layer.*`, `policy_head.*` (28,518 parameters).
- Frozen: `input_layer.*`, `residual_layers.0.*`, `residual_layers.1.*`, `value_hidden_layer.*`, `value_head.*` (44,641 parameters).
- Every frozen tensor was bit-identical after each epoch and after restored-best selection for T61 and T63. No `last_block_policy_scope_leak` occurred.
- Both lanes consumed the same deterministic epoch permutations as their corresponding PR #316 heads-only lanes: T61 `42e3f787`, `4997dff1`, `92559a49`, `82d16ced`; T63 `2c2c0d9a`, `994e33a4`, `6d9012c8`, `3117f2ab`.

## Anchor Trajectory

All values use `legal_normalized_metric`.

| seed | checkpoint | top | optimal mass | P(0) | margin | entropy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| both | G0 | 0 | 0.3999 | 0.3999 | 0.3967 | 1.8307 |
| T61 | epoch 1 | 4 | 0.2541 | 0.2541 | -0.3697 | 1.9339 |
| T61 | epoch 2 | 4 | 0.1736 | 0.1736 | -0.7318 | 1.8921 |
| T61 | epoch 3 | 4 | 0.2689 | 0.2689 | -0.4609 | 1.8473 |
| T61 | epoch 4 / restored best | 4 | 0.1625 | 0.1625 | -0.9612 | 1.8539 |
| T63 | epoch 1 | 4 | 0.3084 | 0.3084 | -0.0156 | 1.9351 |
| T63 | epoch 2 | 4 | 0.1341 | 0.1341 | -1.1899 | 1.7899 |
| T63 | epoch 3 | 4 | 0.1361 | 0.1361 | -1.1350 | 1.8021 |
| T63 | epoch 4 / restored best | 4 | 0.1182 | 0.1182 | -1.2156 | 1.8189 |

T61 did not repair and its mass fell by 0.1112 relative to matched full training. T63 did not retain its successful full-training anchor behavior.

## Movement And Direction

Final trainable-path drift was non-zero: T61 norm 3.3188 (19.29% relative), T63 norm 3.2792 (19.06% relative). Thus this was not a frozen-model outcome.

| restricted vector | full T61 cosine / norm ratio | full T63 cosine / norm ratio |
| --- | ---: | ---: |
| T61 final block | 0.5928 / 1.7809 | 0.5616 / 1.7713 |
| T63 final block | 0.5838 / 1.7516 | 0.5824 / 1.7422 |

The directions do not show a selective convergence toward the successful T63 full-training direction.

## Frozen Family And Value

| seed | cluster mass | cluster top-1 | degrading mass | entropy | value MAE | forgotten / repaired |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| T61 last-block | 0.6193 | 0.8333 | 0.1171 | 1.4517 | 0.6571 | 1 / 0 |
| T63 last-block | 0.6504 | 0.8333 | 0.1237 | 1.3731 | 0.6355 | 1 / 0 |

Both cluster masses were materially below their matched full controls under the 0.02 threshold. Final validation value loss was 0.04622 (T61) and 0.04582 (T63); value outputs still changed through the trainable final block despite frozen value-head parameters. T61 was worse than heads-only on both anchor and cluster mass; T63 was worse on anchor mass but better on cluster mass.

## Exact Forensic Safety

The outcome-aligned exact forensic shadow reported no new critical regression against matched full controls.

| seed | overall accuracy / regret / blunder | capture available | sparse endgame |
| --- | --- | --- | --- |
| T61 | 0.8545 / 0.2582 / 0.1455 | 0.7500 / 0.3750 / 0.2500 | 0.8750 / 0.2083 / 0.1250 |
| T63 | 0.8545 / 0.2582 / 0.1455 | 0.7500 / 0.4167 / 0.2500 | 0.9167 / 0.1250 / 0.0833 |

## Arena

The fixed lightweight arena was run against the unchanged current baseline; no candidate was promoted.

| seed | W-D-L | score | Wilson 95% CI | effect from 0.5 | material regression |
| --- | --- | ---: | --- | ---: | --- |
| T61 | 15-0-15 | 0.5000 | [0.3315, 0.6685] | 0.0000 | false |
| T63 | 30-0-0 | 1.0000 | [0.8865, 1.0000] | 0.5000 | false |

The lack of arena regression does not overcome the failure of both anchor requirements and the material frozen-cluster regression.
