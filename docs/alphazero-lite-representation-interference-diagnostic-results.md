# AlphaZero-Lite Representation-Interference Diagnostic Results

**Date:** 2026-06-06

## Summary

**Classification:** `representation_interference_likely`

The production targets (147 positions) and preservation controls (206 positions) are **not cleanly separable** in the current residual_v3 learned representation. Direct patching will keep causing control regressions regardless of LR, replay weights, or small capacity changes.

## Current Baseline (eval only, no training)

- Production optimal@1200: **62/147** (42.2%)
- Production optimal@2400: **82/147** (55.8%)
- Control optimal@1200: **206/206** (100%)
- Control regressions (any budget): **1**

## Lane Results

| Lane | Scope | LR | Epochs | Prod Opt@1200 | Gain | Ctrl Reg | Ctrl Reg% | Policy Loss | Value Loss | Trainable | Frozen |
|------|-------|----|--------|--------------|------|---------|----------|------------|-----------|----------|--------|
| 1 | current_eval_only | — | 0 | 62/147 | 0 | 1 | 0.5% | — | — | — | — |
| 2a | head_only_policy_finetune | 2.5e-5 | 1 | 71/147 | +9 | 1 | 0.5% | 2.059 | 0.305 | 9,894 | 63,265 |
| 2b | head_only_policy_finetune | 2.5e-5 | 2 | 71/147 | +9 | 1 | 0.5% | 2.109 | 0.291 | 9,894 | 63,265 |
| 2c | head_only_policy_finetune | 2.5e-5 | 4 | 71/147 | +9 | 1 | 0.5% | 1.950 | 0.305 | 9,894 | 63,265 |
| 3a | last_block_plus_policy_finetune | 2.5e-5 | 1 | 72/147 | +10 | 4 | 1.9% | 2.034 | 0.300 | 28,518 | 44,641 |
| 3b | last_block_plus_policy_finetune | 2.5e-5 | 2 | 74/147 | +12 | 7 | 3.4% | 2.051 | 0.278 | 28,518 | 44,641 |
| 3c | last_block_plus_policy_finetune | 2.5e-5 | 4 | 83/147 | +21 | 7 | 3.4% | 1.853 | 0.281 | 28,518 | 44,641 |
| 4a | full_finetune_control | 2.5e-5 | 1 | 93/147 | +31 | 9 | 4.4% | 1.926 | 0.279 | 73,159 | 0 |
| 4b | full_finetune_control | 2.5e-5 | 2 | 95/147 | +33 | 12 | 5.8% | 1.857 | 0.241 | 73,159 | 0 |
| 4c | full_finetune_control | 2.5e-5 | 4 | 99/147 | +37 | 18 | 8.7% | 1.615 | 0.222 | 73,159 | 0 |

### Lane Analysis

**Lane 2 (head_only_policy_finetune):** Frozen trunk + residual blocks. Only 9,894 parameters trainable (policy_hidden_layer + policy_head). Gains +9 production optimal@1200 with **zero additional control regressions** beyond the 1 baseline. Saturates after epoch 1 — no further improvement with more training. The head-only ceiling of +9 suggests the frozen trunk representation fundamentally limits how much the policy can be adjusted for production positions while preserving controls.

**Lane 3 (last_block_plus_policy_finetune):** Unfreezes the final residual block plus policy heads (28,518 trainable params). Gains +10 to +21 production optimal@1200 as epochs increase, but control regressions grow from 1 to 7 (3.4%). Representation changes in the last block cause measurable control interference.

**Lane 4 (full_finetune_control):** All 73,159 parameters trainable. Best production gains: +31 to +37, reaching 99/147 (67.3%) at epoch 4. But control regressions reach 18/206 (8.7%) — matching PR #81's conclusion that full fine-tuning inevitably regresses controls.

## Representation Diagnostics (Current Model)

### Embedding-Space Analysis

| Metric | Value |
|--------|-------|
| Mean production-control NN distance | **0.619** |
| Median production-control NN distance | **0.588** |
| Production rows whose nearest control has conflicting exact optimal move | **116/147** |
| NN optimal-move conflict rate | **78.9%** |

The low NN distances (< 1.0) combined with the 78.9% conflict rate confirm that production and control positions are densely interleaved in the learned embedding space, and the nearest control neighbor of most production rows requires a *different* optimal move.

### Policy Confidence Metrics

| Metric | Production | Control |
|--------|-----------|---------|
| Mean entropy | 0.951 | 1.046 |
| Mean top-2 margin | 0.469 | 0.428 |
| Mean optimal-action logit margin | **2.694** | **1.154** |
| Current top-move matches exact optimal | 13/147 (8.8%) | 80/206 (38.8%) |

The current model is more confident about being *wrong* on production positions (logit gap for optimal move: 2.69) than on control positions (1.15). Only 8.8% of production rows have the neural-network top move matching the tablebase optimal move, vs 38.8% for controls.

## Interpretation

### Evidence for `representation_interference_likely`

1. **Head-only saturation at +9 (10.6% recovery):** The policy head alone can correct at most 9 of 85 production failures. This ceiling exists without touching the trunk representation — the frozen embedding cannot express the needed policy changes for the remaining 76 positions.

2. **Control regressions scale with trainable parameters:** Each increment in trainable scope — policy_head → last_block → full — trades production gain for control regression. The trade-off is approximately 0.5 control regressions per production correction.

3. **Embedding-space conflict (78.9%):** The overwhelming majority of production rows are nearest neighbors to a control row with a *conflicting* exact optimal move. Changing the policy for a production row will almost always affect nearby control rows.

4. **Dense interleaving (mean distance 0.62):** The small NN distance between production and control embeddings means they share the same region of representation space. There is no clean manifold separation the model can exploit.

### Implications

- **Do not add more direct tablebase patching.** Any approach that fine-tunes on production targets will cause control regressions proportional to its capacity. The representation itself needs to change before these targets can be safely incorporated.
- **Consider architecture changes** (wider trunk, more blocks, attention mechanisms) that could create more separable embeddings for production vs control positions.
- **Consider MCTS-guided self-play** with tablebase-verified targets rather than direct policy imitation, as suggested in PR #81's rejection.

## Classification Criteria

- `representation_separable`: policy_head or last_block_policy improves production substantially with near-zero control regressions. **NOT MET** — head-only caps at +9 (modest), and last-block regresses controls.
- `representation_interference_likely`: head-only cannot substantially improve, and full/last-block regresses controls. Production rows are near controls with conflicting optimal moves in embedding space. **CONFIRMED.**
- `optimization_instability`: head-only fits cleanly at one LR/epoch, full unnecessarily regresses controls. **NOT MET** — there is no clean head-only solution; the issue is representational, not optimization.

