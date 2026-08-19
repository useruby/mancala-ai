# AlphaZero-Lite Function-Space Null-Direction Audit

**Classification:** `function_space_null_direction_refuted`

**Question:** is the harmful PR #197 trunk update dominated by a function-space null direction (parameter rotation invisible on the training probe but visible to held-out/search states)?

## Harmful-step decomposition

- step null fraction: 0.772 (visible 0.635)
- cosine(null, step): +0.772; cosine(visible, step): +0.635

## Raw-gradient decomposition (pre-Adam)

- gradient null fraction: 0.046 (visible 0.999)

The raw grand-mean gradient is almost entirely function-space (null fraction 0.046); Adam's sign-normalization reshapes it into a step whose null component carries 0.597 of the squared norm.

## Singular-value spectrum

- condition number: 7519.4
- eigenvalue min/max: 4.44e-02 / 2.51e+06
- bottom-half energy fraction: 0.166
- output dimension: 2391

## Search-effect contrast

| Model | Probe move change | Validation move change |
| --- | ---: | ---: |
| full step | 0.0205 | 0.0234 |
| visible only | 0.0205 | 0.0234 |
| null only | 0.0000 | 0.0020 |

The null component reproduces essentially none of the full step's move changes (probe and validation), while the visible component reproduces all of them. The harmful step's function-space null direction is inert.

## Classification evidence

| Signal | Value |
| --- | ---: |
| gradient_null_fraction | 0.0462 |
| step_null_fraction | 0.7724 |
| probe_null_move_change | 0.0000 |
| probe_visible_move_change | 0.0205 |
| probe_full_move_change | 0.0205 |
| validation_null_move_change | 0.0020 |
| validation_visible_move_change | 0.0234 |
| validation_full_move_change | 0.0234 |

## Next action

`the harmful step's large null component is inert; pursue search-aware or outcome-grounded distillation constraints, not a null-space projection`

Full evidence: `docs/data/alphazero-lite-function-space-null-direction-summary.json`.
