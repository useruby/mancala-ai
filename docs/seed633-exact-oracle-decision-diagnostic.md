# Seed 633 exact-oracle decision-quality diagnostic

**Decision: `imitation_decoupled_from_exact_quality`.** Better teacher-target
cross-entropy is not a robust proxy for better exact policy decisions on this
cohort. It sometimes accompanies better decisions, but the overall exact-optimal
action mass moved in the wrong direction, while top-policy accuracy and regret
were essentially unchanged. There is no evidence here about playing strength
after search.

## Design and evidence freeze

This retrospective, paired analysis uses the merged Seed 633 authenticated
cohort and the already-published B512/S615 NumPy policy metrics. Before
calculating diagnostic outcomes, the analysis froze the source run, cohort and
metric hashes; both checkpoint identities; exact-oracle inclusion rule;
tie-handling; tolerances; subgroup definitions; and bootstrap procedure.
Seed 633's original independent verifier authenticated the 4,521-state source
bundle and independently reconstructed model metrics. This diagnostic verifier
then independently reconstructs all derived exact-only metrics and summaries.

Primary analysis includes only the 1,481 rows labeled
`exact_root_tablebase` with the authenticated full legal-action margin map.
Action margins and optimal sets are rechecked for exact action correspondence;
all actions tied at the highest margin are treated as optimal. Policy top action
is the unsearched legal-masked model argmax, with NumPy's deterministic
lowest-action-index tie break. Search-selected actions are unavailable and are
not inferred. Regret is the exact authenticated best margin minus the chosen
top-policy action's margin, in recorded outcome-point units.

Numerical rules frozen in `freeze.json`:

- Teacher-imitation improvement: S615-minus-B512 CE `< -1e-6` nats; worse is
  `> +1e-6` nats.
- Exact-quality direction is determined lexicographically by top-policy regret
  (strict integer-point difference), then optimal-action correctness, then
  optimal-action probability mass with a ±0.005 tolerance.
- Pairwise uncertainty uses 20,000 source-opening-cluster resamples, seed 633,
  and percentile 2.5%/97.5% bounds. These are descriptive stability intervals;
  the position rows are not treated as independent observations.

The authentic historical source-opening labels are used as clusters. No
crosswalk to Seed 624 `game_id` or `opening_identity` is assumed. Membership
classes are the original four Seed 633 labels. The holdout is also reported as
`neither_training_membership`; inherited P-E4 exposure remains unresolved.

## Results

All differences below are paired S615 minus B512, on identical exact-oracle
states:

| Exact-oracle cohort | States / groups | Teacher CE (nats) | Exact-optimal mass | Top-policy optimal | Top-policy regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 1,481 / 182 | −0.001589 [−0.008835, 0.004427] | −0.001880 [−0.003554, −0.000272] | +0.001350 [−0.006378, 0.009273] | −0.005402 [−0.052706, 0.039244] |
| Protected holdout | 567 / 36 | −0.004517 [−0.020970, 0.006981] | −0.002397 [−0.005948, 0.000792] | +0.001764 [−0.013233, 0.017545] | −0.056437 [−0.165821, 0.035589] |

Brackets are source-opening-cluster bootstrap intervals. For the full cohort,
S615 has slightly lower mean CE, but the interval includes zero; the optimal
mass interval is below zero. Top-policy accuracy and regret intervals include
zero. In the protected holdout, all four intervals include zero; optimal mass
again trends lower while accuracy/regret trend slightly better. Thus the
holdout does not support a robust transfer claim.

Descriptive paired-state Pearson correlations between CE change and exact
quality change were −0.593 for optimal-action mass, −0.202 for top-action
correctness, and +0.157 for regret (n=1,481). Because lower CE and lower regret
are both better, the latter is directionally consistent with imitation
improvement often accompanying worse regret. These unclustered correlations
are descriptive only and not inferential evidence.

The predeclared joint categories count 239 states with improved imitation and
better exact quality, 19 with improved imitation but worse exact quality, and
5 with worse imitation but improved exact quality. 926 had no meaningful exact
quality movement under the frozen thresholds. For completeness, 292 had both
worse imitation and worse exact quality; they do not fit the four discordance /
no-change cases named in the request. Every row has its own category in the
per-state output.

By original membership class, exact-optimal mass changed −0.007229 on B512-only
(402 rows), +0.004354 on S615-only (402), −0.002449 on shared (110), and
−0.002397 on protected holdout (567). CE changed +0.025230, −0.022990,
−0.006295, and −0.004517 respectively. This reciprocal in-sample pattern does
not extend to a consistent direction on held-out states and membership was not
randomized.

All exact-oracle states are in `at_most_16` phase (1,481 rows); there is no
exact-oracle coverage in the earlier phases, so phase variation cannot be
assessed. By legal-action count, sample sizes are 328, 471, 378, 207, 83, and 14
for 1–6 legal actions. The teacher-entropy quantile cuts collapse because all
exact-root targets have zero entropy; the only populated quartile is bin 4
(1,481 states), so there is no informative entropy-stratum comparison. There
are 846 unique-optimum and 635 tied-optimum states. Mean exact-optimal mass
change is −0.001054 on unique-optimum rows and −0.002980 on tied rows; top
accuracy changes +0.004728 and −0.003150 respectively. Full subgroup summaries
and cluster counts are in `summary.json`.

## Interpretation

**Does improved teacher imitation translate into better exact decisions, or are
we optimizing a surrogate objective that is weakly connected to playing
strength?** In this retrospective comparison, the evidence says teacher CE is
a weak and inconsistent surrogate for exact decision quality: improvements
occurred with exact-quality gains on some rows, losses on others, and mostly no
meaningful movement. The aggregate CE shift is tiny and uncertain, optimal
action mass declines, while top-action accuracy and regret do not show reliable
change. Protected holdout estimates are uncertain. The conservative conclusion
is decoupling on this cohort, not a causal statement or a general playing-
strength result.

**Final classification: `imitation_decoupled_from_exact_quality`.**

## Reproduction and verification

The frozen bundle is `docs/data/seed633-exact-oracle-decision-diagnostic-final/`.
Its source bundle is Seed 633 attempt 5, whose historical independent
verification record is preserved alongside its frozen cohort and per-state
source metrics. Reproduce the derived bundle with:

```sh
python -m ml.alphazero_lite.seed633_exact_oracle_decision_diagnostic \
  --source docs/data/seed633-teacher-target-learning/attempt-0005 \
  --output docs/data/seed633-exact-oracle-decision-diagnostic-new
python -m ml.alphazero_lite.verify_seed633_exact_oracle_decision_diagnostic \
  --source docs/data/seed633-teacher-target-learning/attempt-0005 \
  --output docs/data/seed633-exact-oracle-decision-diagnostic-new
```

The verifier rechecks legal actions and action-margin/optimal-set correspondence,
the model checkpoint hashes bound by the authenticated source freeze, every
per-state derived metric, and all subgroup/bootstrap results. Regression tests
cover altered oracle labels, action masks, model-row identity, and membership
preservation. Per-state policies and outputs are top-policy diagnostics only;
no games, model training, search, or new solving were performed.
