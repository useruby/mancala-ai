# seed445 — paired policy/value finite-step attribution

Retrospective attribution of the already-known frozen seed442 paths. The endpoints and failed gates predate this diagnostic; this is not prospective outcome evidence.

**B unseen value-MSE classification: `finite_step_value_residual_dominant`.**

## Primary B value MSE

| Weighting | D | S | R | endpoint difference |
|---|---:|---:|---:|---:|
| exposure_weighted | 0.00458619126125 | -0.000657620883135 | 0.00524381214438 | 0.00458619126125 |
| equal_input | 0.00389002257397 | -0.00217451746732 | 0.00606454004129 | 0.00389002257397 |

## Descriptive A endpoint totals

A value MSE: `{"equal_input": {"D": 0.004019646375743591, "R": 0.0544475838293586, "S": -0.05042793745361502, "endpoint_difference": 0.004019646375743591}, "exposure_weighted": {"D": 0.008461960633129428, "R": 0.045433234711903, "S": -0.036971274078773576, "endpoint_difference": 0.008461960633129428}}`

Policy CE totals for A and B are in the complete step ledger.

The classification follows the frozen signed rule under both weightings: B has D>0 and R≥0.75D in both; S is negative in both. Endpoint metrics reproduce #442. The trajectory classification remains `close_kl_capped_step_branch`.

Group contributions decompose only first-order dots; residuals are not assigned to groups and are not pure curvature estimates. The independent verifier recomputes predictions, gradients, dots, totals and classification. `verifier-amendment-1.json` transparently records a post-execution correction to match the runner's prediction batch boundaries; it did not change the objective, arithmetic protocol, runner bytes, or published arrays. The classification is retrospective and is not prospective outcome evidence. No classification authorizes training, changes historical gates, or establishes playing strength.
