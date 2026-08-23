# Partial Lagged-Parent Root-Q Blending

**Classification:** `partial_shadow_loses_game_safety`

The preregistered smallest passing frozen-screen weight was `lambda_075`: 30/40 amplified roots were rescued and 0/40 washed controls newly diverged. Exact parent-Q (`lambda_100`) rescued 39/40.

| Lambda | Frozen rescue | New control divergences |
| --- | ---: | ---: |
| 0.00 | 0/40 | 0/40 |
| 0.25 | 20/40 | 0/40 |
| 0.50 | 27/40 | 0/40 |
| 0.75 | 30/40 | 0/40 |
| 1.00 | 39/40 | 0/40 |

Selected-lambda replay retained candidate-specific moves on 9/26 differing roots at 384 and 10/40 at 1200. However, the canonical 1200:1200 paired effect was -0.0195 (95% CI [-0.0391, -0.0039]), failing the preregistered game-safety gate.

**Recommended follow-up:** Investigate stabilization in training or search-target generation rather than inference-time partial shadow-Q.
