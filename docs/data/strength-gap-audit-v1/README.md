# Training-to-strength gap audit v1

## Executive finding

The evidence supports a **decision-relevance / search-mediation** explanation more than a simple “loss improved, therefore policy improved” account, but does not identify a single causal mechanism. On the #557 fixed-root cohort, T512 has worse raw-prior exact-optimal frequency than serving P (62.50% vs 65.63%) and higher raw-prior mean regret (+0.1563 seeds paired), while post-PUCT regret is effectively tied (T512−P −0.0078; 95% opening-bootstrap interval [−0.1797, +0.1406]). This means development loss improvements did not translate into better decisions on this exact-root sample; it does **not** show the changes were erased by search or explain full-game results.

The #558 value-only comparison finds no detected strength change: replacing the initializer value path with D432 changed 9/256 selections, with zero mean regret change (95% interval [−0.2578, +0.1797]). This is evidence that this value substitution did not materially affect decisions under this search/cohort, not evidence that value quality is generally irrelevant. #577 similarly did not qualify: treatment-minus-control regret −0.0709 seeds, 95% CI [−0.1844, +0.0284], with WDL downgrades 10/423 vs 9/423. Its per-root published payload supports paired regret and changed-action counts but does not publish exact root state descriptors or complete action-margins; it cannot support the same strategic subgroup or raw-prior oracle analysis as #557.

## A. Are better priors choosing better moves before search?

Not on #557 relative to P. The published exact-prior-optimal rate is 62.50% for T512 versus 65.63% for P. The paired mean prior regret contrast is +0.1563 seeds (20,000 opening-cluster bootstrap; 95% percentile interval [−0.1250, +0.4453]). T512 assigns 2.95 percentage points less prior mass to exact-optimal actions on average (paired interval [−3.85, −2.06] percentage points). These are descriptive results from 256 roots selected from #556 trajectories, not a census of encountered positions.

## B. Does PUCT preserve, erase, or reverse differences?

PUCT improves both policies’ root selections overall: the published exact-optimal rate rises from 62.50% to 76.95% for T512 and 65.63% to 78.13% for P. Its “improved prior” rates are 17.58% and 15.23%; “worsened prior” rates are 4.30% and 2.73%, respectively. Post-search mean regret is 1.1406 vs 1.1484 seeds; paired contrast −0.0078 (95% CI [−0.1719, +0.1406]). Thus search closes the small raw-regret point difference in this sample, but uncertainty includes either direction. These tables do not show whether the small paired advantage is a specific systematic correction or sampling variation.

For #558, policy priors are identical by construction. D432 value changed the selected PUCT action on 9/256 roots (3.52%), while mean regret remained exactly equal; the published exact-optimal rate was slightly lower for D432 (75.78% vs 78.13%). The #577 qualification cohort changed selected actions on 32/423 roots; paired regret is not a statistically resolved improvement and WDL downgrade count increases by one.

## C. Are mistakes concentrated in strategically important positions?

The #557 roots are exclusively 17–21 active pit-stone positions with at least two legal moves. Existing subgroup summaries show counts of 12, 16, 21, 36, and 171 by stone count, so the 21-stone stratum dominates and the smaller strata are exploratory. The audit retains per-root stone count and legal-move count, plus oracle-derived regret and optimal-prior mass, in `seed557-roots.jsonl`; no confirmatory subgroup claim is made from sparse slices. No quality label for “strategically important” was preregistered. Exact best-versus-second-best action separation can be computed from full oracle margins, but a small margin is not itself a validated measure of strategic importance. The #558 inherited exact oracle records do not contain state descriptors in its paired root metrics; #577 likewise lacks root descriptors and full action-margin vectors. Consequently, neither cohort can be validly pooled into a positional-concentration analysis.

## D. Is value quality the main bottleneck?

Not established. #558 is a controlled value-path substitution with identical priors and only 9 changed decisions; its zero mean regret contrast and broad interval provide no positive evidence that D432 value improves decisions on this cohort. That weighs against the narrow hypothesis that simply swapping in this value checkpoint is sufficient. It does not distinguish weak value signal from search insensitivity, calibration/magnitude issues, data mismatch, or the possibility that value matters mainly on other states. #577 does not isolate value. #557 uses a shared initializer value model across policy arms.

## E. One next experiment

Run one **matched fixed-root policy-prior sensitivity test** on a newly frozen, independently sourced public cohort, with P and one candidate policy, keeping value model, search budget and settings fixed. Hypothesis: policy loss gains fail to improve strength because the candidate’s changed prior mass is not preferentially moved onto exact-optimal actions in high-regret roots; if true, paired optimal-action prior mass will not improve and post-PUCT regret will remain nonnegative. Use complete exact legal-action oracle vectors, per-root prior vectors and PUCT outcomes; stratify prospectively by legal-action count and exact best/runner-up outcome margin, without crossing cohorts.

Estimated compute: no training; 2×128 PUCT simulations per root. At 256 roots this is 65,536 simulations plus exact tablebase lookups already required to score roots. Success metric: candidate-minus-P paired mean exact post-PUCT margin regret with opening-cluster 95% CI wholly below zero, alongside improvement in exact-optimal prior mass. Stop if frozen evidence validation fails or if, at the preregistered sample size, the interval includes zero; do not promote from this diagnostic alone. This small experiment tests the prior-to-decision link directly and does not run here.

## Data provenance and scope

`aggregate.json` gives source SHA-256 digests and cohort-specific paired results; `seed557-roots.jsonl`, `seed558-roots.jsonl`, and `seed577-roots.jsonl` are the reproducible root-level tables. Re-run from repository root with `python -m ml.alphazero_lite.strength_gap_audit`; verify with `python -m unittest ml.alphazero_lite.test_strength_gap_audit -v`. The script checks #557 complete legal-action oracle coverage, cohort/policy completeness, published aggregate reproduction, #558 root uniqueness and #577 pairing. Its inputs are the published diagnostic evidence, not newly generated inference.

Development loss and strength roots are different populations. On #542 development evidence, update 432 pooled policy CE is 0.6622 vs initializer 0.7781 and actual-value MSE is 0.3466 vs 0.3749 (equal production-float32-input weighting). Yet the #547 arena score was 0.48169 against P; #552 was 0.48511 and #556 T512 was 0.49780. These associations establish a metric/strength disconnect, not a per-position correspondence or causal explanation. #546’s published development evaluation, #547/#552/#556 arena results, #557 retrospective roots, #558 value swap, and #577 qualification cohort must remain distinct. No access to sealed holdout was made; historical overlap limitations, including #483, are not resolved by this audit.

The v1 analysis intentionally reuses #557/#558 validated aggregates instead of republishing their analyses. No training, self-play, arena games, model inference, or promotion was run.
