# seed446 — prediction-space decomposition

Retrospective decomposition of seed445's already-known value-MSE residual, using archived float32 predictions and targets with float64 arithmetic. No models were evaluated and no gradients were computed.

**B classification under both weightings: `squared_prediction_movement_dominant`.**

| Weighting | Q = sum(q) | N = sum(n) | R = Q + N | Q/R | N/R |
|---|---:|---:|---:|---:|---:|
| exposure_weighted | 0.00640667399828 | -0.0011628618539 | 0.00524381214438 | 1.2217589 | -0.22175887 |
| equal_input | 0.00701290603703 | -0.000948365995731 | 0.00606454004129 | 1.1563789 | -0.15637888 |

`q` is nonnegative squared prediction movement. `n = c - s` is the signed output-response remainder relative to the independently recomputed parameter-space first-order dot; it includes nonlinear and numerical effects and is not pure network curvature or a causal intervention. This retrospective result proposes no training constraint and authorizes no training, gate change, strength claim, or promotion. The historical `finite_step_value_residual_dominant` and `close_kl_capped_step_branch` findings are preserved.
