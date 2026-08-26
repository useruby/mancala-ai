# On-Policy Lagged-Parent Shadow Replay

**Classification:** `onpolicy_data_not_distillable`

This run used the user-authorized regenerated canonical suite
`57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04`, rather
than the unavailable historical `ff21...` suite. It is therefore not directly
comparable to the original canonical-arena evidence.

## Fixed Contract

- P1 checkpoint SHA-256: `e4a9c95302c17b63405107d26b4cd4f90755d2b21980846ac6e9f0baabf7efe9`
- A16 artifact SHA-256: `74e160734554dba9ecf4ecf3ac13b3e44526c4f6e7e5e52d9fe8bde0054eb789`
- Fresh-P1 self-play protocol: 700 games, seed 44, 24 workers, 384 simulations,
  c_puct 1.25, visit-count roots, noisy targets, Dirichlet alpha/epsilon 0.3,
  tree reuse, and `kalah_v3` input.
- All lanes started from A16 step 16 and its Adam state. Only the policy adapter
  trained, with beta 0.95, Adam LR `1e-5`, clipping 1.0, and new replay only.
- No shadow evaluator was used for arena evaluation. The frozen 40+40
  ordinary-search diagnostic was not rerun because the canonical-suite
  replacement was authorized after training began.

## Replay And Exclusion Diagnostics

| Lane | Rows | Unique states | Complete trajectory agreement vs P1 | Mean first divergence ply | State Jaccard vs P1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ordinary | 27,642 | 23,314 | 1.0000 | 39.49 | 1.0000 |
| shadow .75 | 27,879 | 23,120 | 0.0143 | 7.95 | 0.0787 |
| shadow 1.0 | 26,711 | 21,999 | 0.0071 | 6.49 | 0.0577 |

Neither shadow replay is parent-like under the preregistered criterion.

| Lane | Amplified | Washed | Held-out | Canonical replacement | Eligible rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| ordinary | 41 | 47 | 2 | 202 | 27,350 |
| shadow .75 | 4 | 9 | 2 | 267 | 27,597 |
| shadow 1.0 | 4 | 7 | 2 | 273 | 26,425 |

## Step-16 Training

| Lane | CE(search) | CE(P1) | CE(beta095) | Fit | L1 vs P1 | JS vs P1 | Top-1 | Adapter norm | Delta A16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ordinary | 1.095691 | 0.945453 | 0.952965 | 1.0502 | 0.002901 | 1.84e-06 | 0.0026 | 0.004470 | 0.002072 |
| shadow .75 | 1.087476 | 0.933134 | 0.940851 | 0.9859 | 0.003035 | 1.99e-06 | 0.0021 | 0.004537 | 0.002148 |
| shadow 1.0 | 1.105071 | 0.932688 | 0.941307 | 0.9872 | 0.003072 | 2.08e-06 | 0.0030 | 0.004569 | 0.002173 |

All checkpoints at steps 1, 4, and 16 had fit above 0.25 and retained bit-identical inherited parameters.

## Ordinary-PUCT Arena

All arena rows used 128 openings, seat swap, two games per opening, seed 42, and
10,000-opening bootstrap intervals.

| Lane | Step | 384:256 effect / CI | 1200:1200 effect / CI |
| --- | ---: | --- | --- |
| ordinary | 1 | -0.0098 [-0.0195, -0.0020] | -0.0195 [-0.0391, -0.0039] |
| ordinary | 4 | -0.0098 [-0.0195, -0.0020] | -0.0195 [-0.0391, -0.0039] |
| ordinary | 16 | -0.0195 [-0.0391, -0.0039] | +0.0410 [+0.0137, +0.0664] |
| shadow .75 | 1 | -0.0098 [-0.0195, -0.0020] | -0.0195 [-0.0391, -0.0039] |
| shadow .75 | 4 | -0.0098 [-0.0195, -0.0020] | -0.0195 [-0.0391, -0.0039] |
| shadow .75 | 16 | -0.0195 [-0.0391, -0.0039] | -0.0195 [-0.0391, -0.0039] |
| shadow 1.0 | 1 | -0.0098 [-0.0195, -0.0020] | -0.0195 [-0.0391, -0.0039] |
| shadow 1.0 | 4 | -0.0098 [-0.0195, -0.0020] | -0.0195 [-0.0391, -0.0039] |
| shadow 1.0 | 16 | -0.0195 [-0.0391, -0.0039] | -0.0195 [-0.0391, -0.0039] |

No checkpoint was safe at both budgets, so the P0 gate was not eligible.

## Interpretation

Shadow generation changed the encountered trajectories and produced meaningful
teacher fitting, but it did not transfer ordinary-PUCT safety. The lone positive
high-budget ordinary checkpoint was still unsafe at low budget and remained
within a near-parent policy displacement. No candidate is promoted.
