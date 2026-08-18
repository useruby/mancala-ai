# AlphaZero-Lite Policy-Value Channel Attribution

Generated from immutable evaluator-output composition only. No model was trained or modified.

## Canonical Arena

| Step | Budget | Treatment | Paired effect vs C/C | 95% opening bootstrap CI |
| ---: | --- | --- | ---: | --- |
| 3 | 384:256 | C | +0.0000 | [+0.0000, +0.0000] |
| 3 | 384:256 | DP_DV | -0.0195 | [-0.0391, +0.0000] |
| 3 | 384:256 | DP_CV | -0.0781 | [-0.1035, -0.0534] |
| 3 | 384:256 | CP_DV | -0.0059 | [-0.0133, +0.0055] |
| 3 | 1200:1200 | C | +0.0000 | [+0.0000, +0.0000] |
| 3 | 1200:1200 | DP_DV | +0.0098 | [+0.0020, +0.0156] |
| 3 | 1200:1200 | DP_CV | +0.0000 | [+0.0000, +0.0000] |
| 3 | 1200:1200 | CP_DV | +0.1211 | [+0.0941, +0.1513] |
| 5 | 384:256 | C | +0.0000 | [+0.0000, +0.0000] |
| 5 | 384:256 | DP_DV | -0.0117 | [-0.0234, +0.0020] |
| 5 | 384:256 | DP_CV | -0.0879 | [-0.1224, -0.0586] |
| 5 | 384:256 | CP_DV | -0.1406 | [-0.1771, -0.1078] |
| 5 | 1200:1200 | C | +0.0000 | [+0.0000, +0.0000] |
| 5 | 1200:1200 | DP_DV | +0.0410 | [+0.0254, +0.0547] |
| 5 | 1200:1200 | DP_CV | +0.0000 | [+0.0000, +0.0000] |
| 5 | 1200:1200 | CP_DV | +0.0566 | [+0.0391, +0.0778] |
| 12 | 384:256 | C | +0.0000 | [+0.0000, +0.0000] |
| 12 | 384:256 | DP_DV | -0.3242 | [-0.3688, -0.2773] |
| 12 | 384:256 | DP_CV | -0.2012 | [-0.2477, -0.1557] |
| 12 | 384:256 | CP_DV | -0.2461 | [-0.2919, -0.1992] |
| 12 | 1200:1200 | C | +0.0000 | [+0.0000, +0.0000] |
| 12 | 1200:1200 | DP_DV | +0.0020 | [-0.0442, +0.0524] |
| 12 | 1200:1200 | DP_CV | +0.0059 | [-0.0055, +0.0133] |
| 12 | 1200:1200 | CP_DV | +0.0469 | [+0.0160, +0.0778] |

Each treatment uses both forced challenger seats over 128 unique openings; the matched control is C/C with the same artifact channels, seed contract, and budget.

## Fixed-State PUCT And Causal Audit

Phase C evaluated all six immutable evaluator compositions on the exact 256-state probe at `384:256`, `768:768`, and `1200:1200`, with deterministic zero-Dirichlet roots and the canonical c_puct schedule. Full per-step metrics are persisted in the JSON summary.

Phase E had no eligible forced-move treatment: neither primary single-channel intervention changed 32 or more roots in any probe context. No current/current forced continuations were therefore run.

## Classification

`both_channels_independently_harmful`

At step 12 / `384:256`, detached policy with current value regressed by `-0.2012` and current policy with detached value regressed by `-0.2461`, with bootstrap intervals excluding zero. Fully detached output was `-0.3242`. Do not attempt an anchoring treatment from this evidence alone.
