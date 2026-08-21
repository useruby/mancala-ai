# Fresh P1 Parent-Additive Policy Adapter

**Classification:** `adapter_not_safe_or_insufficient_fit`

**Recommended follow-up:** do not promote or tune this adapter; investigate the high-budget regression before another parameterization change.

## Contract

- Fresh P1 replay: 700 games, 27,642 rows, SHA `892827d8ee67a66e6324a2aaec7011df1a21625fc3f6bcd87cab39ce655d2a88`.
- Parent checkpoint SHA: `e4a9c95302c17b63405107d26b4cd4f90755d2b21980846ac6e9f0baabf7efe9`.
- Adapter: zero-initialized `Linear(96, 6)` residual logits from the frozen trunk.
- Trainable tensors: `policy_adapter.weight`, `policy_adapter.bias` only.
- Training: beta 0.95, Adam, lr `1e-5`, weight decay 0, clip 1.0, batch size 512, 46 optimizer steps.
- No promotion occurred.

## Invariants

- Zero adapter output equals P1 exactly: `true`.
- All inherited parent parameters are byte-identical after every checkpoint: `true`.

## Training Metrics

| Step | CE(search) | CE(P1) | CE(beta095) | Search improvement | Fit fraction | L1 mean/p99/max | JS | Top-1 disagreement |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | 1.100826 | 0.947235 | 0.954914 | +0.000017 | 0.0182 | 0.000120/0.000332/0.000392 | 3.32e-09 | 0.0000 |
| 4 | 1.100762 | 0.947235 | 0.954911 | +0.000082 | 0.0864 | 0.000500/0.001364/0.001492 | 5.55e-08 | 0.0004 |
| 16 | 1.100536 | 0.947237 | 0.954902 | +0.000308 | 0.3260 | 0.001754/0.004886/0.004995 | 6.73e-07 | 0.0018 |
| 46 | 1.100130 | 0.947246 | 0.954891 | +0.000713 | 0.7557 | 0.003667/0.009228/0.010425 | 2.89e-06 | 0.0032 |

## Arena Matrix

| Step | Budget | Effect vs P1 | 95% CI | Seat A | Seat B | W/D/L | Safe |
| ---: | --- | ---: | --- | ---: | ---: | --- | ---: |
| 16 | 384:256 | -0.0098 | [-0.0195, -0.0020] | -0.0195 | +0.0000 | 390/48/74 | true |
| 16 | 1200:1200 | -0.0195 | [-0.0391, -0.0039] | -0.0391 | +0.0000 | 180/132/200 | false |
| 46 | 384:256 | -0.0195 | [-0.0391, -0.0039] | -0.0391 | +0.0000 | 390/38/84 | false |
| 46 | 1200:1200 | -0.0234 | [-0.0430, -0.0078] | -0.0469 | +0.0000 | 180/128/204 | false |

Steps 16 and 46 met the meaningful-fit threshold. Neither was safe at both budgets, so the cumulative P0 gate was not eligible and no transitive P0 claim is made.
