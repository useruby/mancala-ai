# Seed475: fresh-policy gradient attribution

Retrospective and exploratory frozen-checkpoint diagnostic. This is not causal evidence, a reweighting recommendation, or a strength claim.

| Checkpoint | Unseen weighting | Classification | cosine with -g_full | <=16 contribution | 17–32 contribution | >32 contribution |
|---|---|---|---:|---:|---:|---:|
| initializer | exposure | full_population_descent | -0.259968822 | 0.16474384 | -0.183017776 | -0.668449266 |
| initializer | equal_input | full_population_descent | -0.242378833 | 0.212085185 | -0.189033653 | -0.656342944 |
| seed474_P | exposure | full_population_descent | -0.545130785 | 0.735416846 | -0.888363509 | -0.562249874 |
| seed474_P | equal_input | full_population_descent | -0.543800365 | 0.746337913 | -0.897308874 | -0.56499055 |

## Finding

At both checkpoints and under both unseen weightings, the full negative fresh-training-policy gradient predicts first-order descent on the unseen >32 policy loss (`full_population_descent`); it does not oppose that local improvement direction. All 859 strict-unseen occurrences and their own targets were retained (851 exact inputs). The shared fresh-training policy denominator is 60,015 across all components.

The descriptive bucket contributions show the >32 component predicting descent in every setting. The <=16 component predicts ascent, while the 17–32 component predicts descent. Therefore the specified pattern “both <=32 buckets predict ascent while >32 predicts descent” does not occur. These local gradients do not establish causality and do not automatically justify reweighting.

Fresh training population: 60,015 occurrences, partitioned into 17,882 at <=16 stones, 20,121 at 17–32, and 22,012 at >32. Their coefficient masses use the same 60,015 total denominator. Complete occurrence counts, exact-input counts, loss numerators, gradient vectors, norms, pairwise geometry, signed contributions, and additive residuals are in `evidence.json` and `gradients.npz`.

No training, optimizer sweep, searches, games, runtime export, promotion, or strength claim was performed.
