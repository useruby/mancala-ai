# Seed470 fixed-initializer finite-step transport

Scale-one directions are first-order predictions; each scale is independently written from the same float32 initializer.

| Objective / direction / weighting | Classification | Predicted descent, observed ascent at 1 |
|---|---|---:|
| P/adam/exposure_weighted | `finite_step_overshoot_consistent` | 15 |
| P/adam/equal_input | `finite_step_overshoot_consistent` | 14 |
| P/negative_gradient/exposure_weighted | `finite_step_overshoot_consistent` | 13 |
| P/negative_gradient/equal_input | `finite_step_overshoot_consistent` | 13 |
| J/adam/exposure_weighted | `finite_step_overshoot_consistent` | 14 |
| J/adam/equal_input | `finite_step_overshoot_consistent` | 14 |
| J/negative_gradient/exposure_weighted | `finite_step_overshoot_consistent` | 14 |
| J/negative_gradient/equal_input | `finite_step_overshoot_consistent` | 13 |

Descriptive fixed-initializer finite-step diagnostic; not independent confirmation, a causal explanation of a historical trajectory, a learning-rate choice, or a model-strength claim. No training trajectory was run.
