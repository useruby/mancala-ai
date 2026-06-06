# AlphaZero-Lite Move-Factorized Policy Head Screen Results

**Date:** 2026-06-06
**Status:** Architecture added; results pending

## Summary

**Classification:** Pending — run the separability screen to produce results.

This document tracks the move-factorized policy head architecture screen
(`residual_v4_move_factorized`) vs baseline `residual_v3` for exact-tablebase
production/control separability.

## Architecture Change

### `residual_v4_move_factorized`

- **Shared trunk:** identical to `residual_v3` (input layer, 3 residual blocks, 96-dim)
- **Value head:** identical to `residual_v3` (value_hidden → tanh → value_head)
- **Policy head (changed):** instead of a single global `nn.Linear(96, 6)`, the
  policy head uses 6 independent per-move linear projections:
  ```
  h → policy_hidden (96→96, ReLU) → move_projections[0..5] (96→1 each) → concatenate → logits[6]
  ```
- Each move gets its own weight vector, allowing move-specific pattern learning
  from the shared trunk representation.
- **Checkpoint keys:** `w_policy_move_0..5`, `b_policy_move_0..5` (replaces `w_policy`, `b_policy`)
- **Metadata:** `move_factorized: true` in architecture block
- **Parameter count:** same as residual_v3 (73,159 total for 96/3)

### Partial Compatibility

- Loading a `residual_v3` checkpoint into `residual_v4_move_factorized`:
  - Trunk (`w_input`, `w_residual_*`): loaded
  - Value head (`w_value_hidden`, `w_value`): loaded
  - Policy hidden (`w_policy_hidden`): loaded
  - `w_policy`, `b_policy`: skipped (reported)
  - Move projections: retain random init
- Loading a `residual_v4_move_factorized` checkpoint into `residual_v3`:
  - Compatible keys loaded; `w_policy_move_*` skipped (reported)

## Running the Screen

```bash
.venv/bin/python ml/alphazero_lite/run_architecture_separability_screen.py \
  --workdir /tmp/azlite_move_factorized_policy_head_screen \
  --init-current storage/ai/alphazero_lite/current \
  --architectures residual_v3,residual_v4_move_factorized \
  --trainable-scopes policy_head,last_block_policy,all \
  --epochs 1,2,4 \
  --learning-rates 2.5e-5 \
  --input-encoding kalah_v3
```

## Lanes

| Lane | Architecture | Trainable Scope | Description |
|------|-------------|----------------|-------------|
| 1 | residual_v3 | — | Current baseline (eval only) |
| 2 | residual_v3 | policy_head | Head-only fine-tune (control lane) |
| 3 | residual_v3 | last_block_policy | Last-block fine-tune (control lane) |
| 4 | residual_v3 | all | Full fine-tune (control lane) |
| 5 | residual_v4_move_factorized | policy_head | **Test lane:** frozen trunk, train only move-factorized policy |
| 6 | residual_v4_move_factorized | last_block_policy | **Test lane:** train last block + move-factorized policy |
| 7 | residual_v4_move_factorized | all | Comparison lane only |

## Metrics Reported

For each lane/checkpoint:
- architecture, trainable_scope, learning_rate, epoch
- checkpoint SHA256, trainable/frozen parameter count
- production_optimal_1200, production_optimal_2400
- production_gain_vs_current
- control_regression_count, control_regression_rate
- policy_loss, value_loss

Representation metrics:
- mean/median production-control nearest-neighbor distance
- nearest-neighbor conflicting-optimal-move count/rate
- production/control mean entropy
- production/control optimal-action logit margin
- current top-move matches exact optimal for production/control rows

## Acceptance Criteria

### Promising
- policy_head or last_block_policy gains at least +15 production_optimal_1200 over current
- control_regression_count stays <= 2
- conflict-rate or NN-distance metrics improve versus residual_v3
- optimal-action logit margin improves on production without worsening controls materially

### Reject
- v4 only improves production by unfreezing more capacity and causes the same control-regression slope as residual_v3
- v4 head-only saturates near +9 like residual_v3
- control regressions exceed residual_v3 at comparable production gain
- implementation requires large Ruby/runtime changes before any signal is available

## Guardrails

- Do not promote any model.
- Do not overwrite storage/ai/alphazero_lite/current.
- Do not run full 1600-game self-play.
- Do not add tablebase overlay generation.
- Do not add exact production patch training beyond the existing diagnostic production/control rows.
- Do not change kalah_v3 encoding in the same PR.
- Do not change promotion gate thresholds.

## Unit Tests

```
ml/alphazero_lite/tests/test_trainable_scope.py
  ResidualV4MoveFactorizedTest (12 tests)
    - Forward pass shapes
    - Move projections existence
    - No global policy head
    - Trainable scope: policy_head, last_block_policy
    - Parameter counts
    - Gradient flow
    - Freeze verification
  ResidualV4CheckpointTest (5 tests)
    - v3 checkpoint round-trip (unchanged)
    - v4 checkpoint round-trip
    - v3 → v4 partial load with key skip reporting
    - v4 → v3 partial load with key skip reporting
    - v3 export artifact key contract unchanged
```
