# AlphaZero-Lite Optimizer-Aware Trunk Dynamics Audit

**Classification:** `raw_gradient_dominance_not_optimizer_dominance, gradient_conflict_emerges_during_training, search_harm_tracks_gradient_conflict`

- PR191 replay hash: `df9a7b83763f530f7dfd05a8b5799d8a85e0b4053c126aa725bf1f12f084ea57`
- expected joint hash matched: `True`
- snapshots: `[0, 5, 12, 23, 35, 46]`
- confirmed conflict steps: `[5, 12, 23, 35, 46]`
- next action: `retire_the_7_94x_raw_gradient_argument_before_changing_loss_weights; compare_normal_joint_trunk_vs_one_prespecified_conflict_mitigated_trunk_gradient`

## PUCT Trajectory

| Step | Context | Move change | Visit JS | Root-value delta | Q-rank change | Visit-margin delta |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 384:256 | 0.0000 | 0.0000 | +0.0000 | +0.0000 | +0.00 |
| 0 | 768:768 | 0.0000 | 0.0000 | +0.0000 | +0.0000 | +0.00 |
| 0 | 1200:1200 | 0.0000 | 0.0000 | +0.0000 | +0.0000 | +0.00 |
| 5 | 384:256 | 0.0586 | 0.0031 | +0.0001 | +0.0039 | +3.54 |
| 5 | 768:768 | 0.0625 | 0.0059 | +0.0002 | -0.0156 | +9.18 |
| 5 | 1200:1200 | 0.0391 | 0.0035 | +0.0014 | -0.0117 | +12.69 |
| 12 | 384:256 | 0.0762 | 0.0061 | +0.0010 | +0.0215 | +5.19 |
| 12 | 768:768 | 0.0859 | 0.0110 | +0.0023 | +0.0039 | +11.99 |
| 12 | 1200:1200 | 0.0664 | 0.0097 | +0.0026 | +0.0078 | +23.37 |
| 23 | 384:256 | 0.1133 | 0.0104 | +0.0028 | +0.0234 | +7.27 |
| 23 | 768:768 | 0.1055 | 0.0165 | +0.0052 | -0.0098 | +14.25 |
| 23 | 1200:1200 | 0.0957 | 0.0144 | +0.0064 | +0.0039 | +25.59 |
| 35 | 384:256 | 0.1152 | 0.0117 | +0.0036 | -0.0020 | +10.82 |
| 35 | 768:768 | 0.1289 | 0.0194 | +0.0085 | +0.0176 | +17.28 |
| 35 | 1200:1200 | 0.0996 | 0.0160 | +0.0077 | +0.0234 | +28.87 |
| 46 | 384:256 | 0.1191 | 0.0132 | +0.0047 | +0.0156 | +10.73 |
| 46 | 768:768 | 0.1191 | 0.0203 | +0.0093 | +0.0098 | +19.38 |
| 46 | 1200:1200 | 0.0977 | 0.0179 | +0.0090 | +0.0352 | +34.20 |

## Sparse Arena

| Step | Context | Paired effect | 95% CI |
| ---: | --- | ---: | --- |
| 5 | 384:256 | -0.0176 | [-0.0312, -0.0039] |
| 5 | 1200:1200 | -0.0586 | [-0.0898, -0.0293] |
| 46 | 384:256 | -0.0742 | [-0.1230, -0.0254] |
| 46 | 1200:1200 | -0.1016 | [-0.1445, -0.0586] |
| 0 | 384:256 | +0.0000 | [+0.0000, +0.0000] |
| 0 | 1200:1200 | +0.0000 | [+0.0000, +0.0000] |

Full fixed-batch gradients, isolated Adam virtual updates, real trajectory alignment, frozen-probe data, and PUCT records are in `docs/data/alphazero-lite-optimizer-aware-trunk-dynamics-summary.json`.
