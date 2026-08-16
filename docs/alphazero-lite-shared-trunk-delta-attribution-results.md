# AlphaZero-Lite Shared-Trunk Delta Attribution

**Classification:** `mixed_additive_harm_no_exclusive_primary`

Phase B's offline replay fit is descriptive only; it does not override paired arena effects. In Phase E, the current policy trunk gradient is 7.94x the weighted value gradient, so `policy_gradient_dominates_trunk_learning` applies. Its cosine CI [-0.0618, 0.0134] crosses zero, so `gradient_conflict` does not apply. At 1200:1200, both isolated deltas are materially harmful (T-C and JH-C), supporting `mixed_additive_harm` but no exclusive primary cause. The 384:256 T-C effect is also harmful, so this is not a practical-versus-high-search `tradeoff`. Next action: `do_not_run_another_unrestricted_joint_update; investigate_value_preserving_or_loss_decoupled_representation_updates`. No model was trained, promoted, or modified.

## Phase F

| Context | Contrast | Paired candidate effect | 95% opening bootstrap CI |
| --- | --- | ---: | --- |
| 384:256 | T-C | -0.1504 | [-0.1914, -0.1094] |
| 384:256 | JH-C | -0.1504 | [-0.1914, -0.1094] |
| 384:256 | J-JH | -0.0449 | [-0.0781, -0.0117] |
| 384:256 | J-T | -0.1074 | [-0.1504, -0.0645] |
| 768:768 | T-C | +0.0156 | [-0.0117, +0.0469] |
| 768:768 | JH-C | -0.0215 | [-0.0527, +0.0137] |
| 768:768 | J-JH | +0.1191 | [+0.0820, +0.1562] |
| 768:768 | J-T | +0.1387 | [+0.1016, +0.1777] |
| 1200:1200 | T-C | -0.0527 | [-0.0840, -0.0234] |
| 1200:1200 | JH-C | -0.0703 | [-0.0898, -0.0508] |
| 1200:1200 | J-JH | -0.0449 | [-0.0820, -0.0078] |
| 1200:1200 | J-T | -0.0352 | [-0.0586, -0.0098] |
| 256:768 | T-C | +0.0547 | [+0.0195, +0.0918] |
| 256:768 | JH-C | +0.0078 | [+0.0020, +0.0156] |
| 256:768 | J-JH | -0.0410 | [-0.0723, -0.0117] |
| 256:768 | J-T | -0.0332 | [-0.0625, -0.0020] |
