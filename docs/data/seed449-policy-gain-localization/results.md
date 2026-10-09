# Seed449 policy-gain localization

**Retrospective descriptive analysis.** Seed447's aggregate endpoint results were already known. This analysis does not establish a causal benefit from changing training weights.

Classification: `frequency_concentrated_policy_gain`. Seed447's historical classification remains `close_joint_output_cap_branch`.

Population: 2607 validation exposures; 1242 exact float32 identities; 1256 compact source rows; 1242 canonical identities.

Registered training replay-copy weight sum: 18 (source weights {'fresh': 1, 'generic_bootstrap': 4, 'random_teacher': 1, 'opening_disagreement': 8, 'stability': 4}); these are not validation exposure counts.

## Primary T − initializer

- Exposure mean change: -0.023404300703
- Equal-identity mean change: -0.00131050223811
- Difference: -0.0220937984649
- Population covariance: -0.0463756301111
- Covariance / mean exposure: -0.0220937984649
- Gross improvement mass: -188.666674253
- Gross worsening mass: 127.65166232

Signed contribution values are additive quantities, not probabilities.

## Fixed exposure-count strata

| Stratum | Identities | Exposures | Exposure mean | Equal-identity mean | Exposure contribution | Equal contribution | Median | Improve / worsen / unchanged |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 848 | 848 | 0.017151926187271217 | 0.017151926187271217 | 0.005579145917455309 | 0.01171081594750885 | 0.01542384922504425 | (0.44693396226415094, 0.5507075471698113, 0.0023584905660377358) |
| 2-4 | 344 | 1360 | -0.03754009947125964 | -0.03803977778257239 | -0.019583634553476455 | -0.010535977099198793 | -0.03225298970937729 | (0.6102941176470589, 0.3897058823529412, 0.0) |
| >=5 | 50 | 399 | -0.06141681718945802 | -0.0617358725865682 | -0.009399812066971136 | -0.002485341086415789 | -0.06194370985031128 | (0.7343358395989975, 0.2656641604010025, 0.0) |

## Other comparisons

| Comparison | Exposure mean change | Equal-identity mean change | Exposure − equal | Population covariance |
|---|---:|---:|---:|---:|
| C minus initializer | -0.0148163920809 | -0.00163298008581 | -0.0131834119951 | -0.0276724275936 |
| T minus C | -0.00858790862206 | 0.000322477847708 | -0.00891038646977 | -0.0187032025175 |

C − initializer is a descriptive control; T − C is descriptive and is not a causal treatment effect.

## Seed447 aggregate reconciliation

All reconstructed exposure and equal-identity policy CE endpoints are within seed447's atol=2e-6, rtol=2e-6 tolerance. Residual numerical differences are recorded in `results.json` (maximum absolute residual below 3.1e-9); archived float32 outputs were preserved. These tiny residuals can arise from historical forward batching and float32-vs-float64 reduction.

## Method and ledgers

The ordered row ledger retains every exposure, target, legal mask, source-row reference, arm loss, and row change. The identity ledger averages row losses within exact float32 input identity, retaining duplicate rows even when targets differ. Identity strata are n=1, n=2–4, and n≥5. Population covariance is across identities; the covariance reconciliation uses unrounded values.

## Limitations

- Retrospective descriptive analysis; aggregate seed447 outcomes were known.
- Does not establish a causal benefit from changed training weights.
- No new model forward passes, gradients, training, games, or promotion.
- Labels authorize no training, weighting changes, strength evaluation, additional experiment, or promotion.
