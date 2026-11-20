# Seed 620 — Controlled search-decision attribution feasibility

**Status:** review-only feasibility assessment. No games, MCTS, exact solving,
training, or checkpoint edits were performed. No candidate is selected or
promoted.

## Recommendation

**Revise, then proceed conditionally.** The required ingredients exist to
compare B512 and S615 at frozen identical states: authenticated checkpoint
exports, deterministic fixed-root PUCT, complete historical oracle cohorts,
and prior examples that bind cohorts, seeds, budgets, and full search traces.
The exact-quality cohort should be drawn entirely from authenticated #557 and
#559 roots; it must not depend on overlap with Seed 615 states. Separately
sample Seed 615 opening and first-divergence states for prior/search-action
diagnostics if useful, but make exact-regret claims on those states only where
independently authenticated targets already exist. Distinguish full-checkpoint
effects from a controlled policy-prior intervention with a fixed value
evaluator; a difference between full checkpoints alone cannot be attributed
to policy priors.

## Evidence reviewed

- PR #613 (merged 2026-11-18) published authenticated Seed 604's 4,096-game
  evaluation and verification; both strength gates failed (`neither`).
- PR #618 (merged 2026-11-20) executed Seed 615's preregistered 1,024-opening,
  4,096-game B512/S615 comparison. Both strength gates failed. Its execution
  freeze is `ba30f5e3efadf3358db9f03c7b4eba4f69b23e2f95dcb4089723a1fb9664f32b`.
- PR #619's read-only retrospective authenticated the B512 checkpoint
  `bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c`, S615
  `cdf1e36c4c9374f28b182757b12471b3f00567eb59825bb938435cb56cf8d780`, and
  P-E4 `2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad`.
  Raw priors differed on 58/1,024 registered opening states and 209/676
  deduplicated first-divergence states. Mean TV was 0.033264 and 0.058269,
  respectively. In 2,048 matched opening/seat pairs, 715 diverged first and
  the first diverging action was always the candidate's. But paired game seeds
  differ by candidate identity. The retrospective explicitly says priors are
  not searched actions or quality measures and no mechanism was established.
- #557 froze 256 roots before policy evaluation, used fresh one-worker,
  128-simulation PUCT trees and matched per-root seeds across policies, and
  authenticated exact margins for every legal action. It reports raw-prior
  and post-PUCT exact-optimality/regret separately. This is a strong reusable
  design precedent, although its old model set is not B512/S615.
- #558 reused #557's exact 256 roots and oracle rows byte-for-byte, holding the
  policy fixed while swapping the value/trunk route. It verified identical
  policy priors and isolated value routing. It demonstrates swap architecture
  and verifier discipline, but not the requested policy swap.
- #559 selected 512 different first-eligible roots from authenticated #556
  trajectories (256 per source side), used fresh 128-simulation PUCT with
  exact root/leaf solving disabled, and reports complete exact legal-action
  margins at all 512 roots. This is another potential source of fixed states;
  it is not an independent population from #556 games.
- Seed 615's opening and first-divergence state files include policy-prior
  evidence only. Its `exact_root_eligible` count (178) is a tactical/stone
  threshold classification, not an authenticated oracle-coverage count.

## State coverage and selection

Authenticated exact-action coverage currently established by these sources:

| Source | Authenticated exact coverage | Provenance / caveat |
|---|---:|---|
| #557 | 256 roots, all legal actions | #556 archived continuations; frozen cohort and oracle records |
| #558 | same 256 roots | Reuses #557 cohort and oracle byte-for-byte; no new states |
| #559 run-0003 | 512 roots, all legal actions | #556 continuations; distinct from #557 by its selection/exclusion contract |
| Seed 615 | No per-state exact action-margin targets in the retrospective | 178 first-divergence roots are only classified as at/below 16 active pit stones |

These counts must not be added as if disjoint, and low stone count alone does
not prove a state was solved or that the target uses the right state
orientation. For an exact-quality experiment, select and freeze a cohort
directly from the union of #557 and #559 roots, deduplicating canonically and
retaining source provenance; authenticate each oracle row and validate legal
actions, player-to-move/seat orientation, and exact-margin perspective. This
cohort is viable regardless of whether it overlaps any Seed 615 state. A
separate Seed 615 trajectory cohort can answer whether actions differ on those
states, but it is not a prerequisite and cannot inherit oracle coverage by
stone count or nearby-state similarity. Do not run an oracle query to fill a
coverage gap.

First divergence is outcome-conditioned: a state enters only after two
candidate-specific game trajectories have already separated. It overweights
positions where actions differed, is influenced by the different game seeds,
and omits all agreeing states after the first divergence. Testing only that
subset estimates a selected disagreement effect, not the typical causal
effect across the evaluation domain.

Proposed exact-quality cohort: up to 128 unique exact-covered states selected
from the deduplicated #557/#559 union, with at least 32 raw-prior agreement and
32 raw-prior disagreement states if available. The agreement/disagreement
stratum is determined by B512/S615 prior inference on the frozen eligible pool
under a preregistered deterministic rule, before any search outputs are
observed. Freeze the full eligible pool, source membership, selection seed,
selected identities, and oracle joins. If the pool cannot supply at least 64
exact-covered states, including at least 16 in each stratum, report that
feasibility failure; do not solve states or condition the cohort on Seed 615
overlap. Use sampling weights or stratum-specific estimates because
disagreement enrichment is not representative by default. Preserve the
historical opening/game cluster identities for uncertainty estimation.

Optionally freeze a distinct Seed 615 trajectory cohort (up to 128 unique
states, balanced between registered opening and first-divergence sources and
including prior agreements as well as disagreements). It supports prior and
post-search action-disagreement analyses. Include exact regret for these rows
only if they independently join to an authenticated existing oracle; no
coverage is implied by the #557/#559 cohort or stone count. Keep its results
separate from the exact-quality cohort and do not require overlap between the
two cohorts.

## Separate full-checkpoint and policy-prior estimands

The main **full-checkpoint effect** compares normal B512 inference/search with
normal S615 inference/search on identical states. Both policy and value
outputs (and any shared-trunk representation) may differ. This answers
whether replacing the whole checkpoint changes searched actions/quality; it
does not identify policy priors as the cause.

A distinct **policy-prior effect** is feasible as an explicit search
intervention without transplanting tensors: at every network expansion,
obtain the prior vector from B512 in one arm and S615 in the other, while both
arms use the same fixed B512 value evaluator for all leaf/root values. Keep
rules, legal-action masking, and search code identical. The treatment variable
is the checkpoint-sourced prior function; the value path is held fixed. Because
the networks have a shared trunk, S615's prior can reflect its changed trunk
representation as well as policy-head weights. Thus this estimates the causal
effect of switching the *prior function* within the specified search, not the
isolated effect of the policy-head tensor alone, nor the full trained S615
policy/value system. Search routing must explicitly discard S615's value
output; verify this in traces and tests.

A literal head-only transplant is a narrower, optional ablation. Seed 615's
training report records four value tensors matching B512 and 18 policy/trunk
tensors differing, so the shared trunk makes head-only routing potentially
out-of-distribution. Do not call it a trained-policy effect. It is feasible
only if a read-only tensor dependency audit confirms compatible names, shapes,
dtypes and architecture, and the report labels the fixed trunk and
interpretation. No checkpoint is to be modified. The full-checkpoint and
fixed-value prior-swap contrasts must be reported separately.

For both comparisons, identical-state paired search is feasible. Fix one
search-runtime commit/export and identical rules, encoding,
legal actions, `c_puct`, FPU, root settings, worker count, simulation budget,
and stopping semantics. Use a fresh tree per arm and state. Pair randomization
by supplying the same state-derived seed and same deterministic random stream
to both arms. Record RNG draw semantics: policy-dependent legal-child
expansion can consume random draws differently, so a shared initial seed is
not sufficient evidence of common random numbers. Prefer counter-based draws
keyed by `(state_id, simulation_index, decision_depth, purpose)` so control and
treatment receive the same exogenous tie/noise variates despite different
tree paths. If refactoring to this contract is disproportionate, use fixed
deterministic PUCT with no stochastic root noise and explicit deterministic
tie-breaks; then pair by identical state/runtime rather than claiming matched
random paths.

Record (1) raw network prior top action and optimal-action prior mass before
search; (2) post-search selected action (maximum visits with frozen tie-break)
and visit vector for each estimand; and (3) exact regret only on authenticated
oracle states. Equal simulation counts must be enforced, not merely equal
wall time. Report prior-swap and full-checkpoint outcomes separately, including
when the full-checkpoint effect is not reproduced by the fixed-value
prior-swap intervention.

## Hypothesis, outcomes, and confounders

**Primary hypothesis:** on the exact-covered historical cohort, switching the
prior function from B512 to S615 while holding the B512 value evaluator fixed
changes some post-search actions and reduces mean exact regret. This is the
policy-prior intervention estimand. Separately, the full S615 checkpoint may
change decisions relative to B512; that contrast includes policy and value
pathways and is not attributed to priors. Secondary hypothesis: prior-swap
effects are larger in states where B512/S615 raw-prior top actions disagree
than where they agree.

Primary metrics on the exact-covered cohort: for the fixed-value prior swap,
paired exact-regret difference (S615 prior minus B512 prior; negative favors
S615), exact-optimal action rate and WDL-inferior action rate. Report the same
metrics separately for the full-checkpoint contrast. Exact regret is exact
best final-score margin minus selected-action final-score margin,
perspective-normalized; ties in exact-optimal actions all count optimal.
Secondary metrics are post-search action disagreement and prior optimal mass.
On optional Seed 615 states without exact targets, report action differences
only; do not infer action quality from the historical cohort's oracle labels.

Confounders to bind or disclose: checkpoint/export mismatch; architecture or
tensor routing; full-checkpoint value differences changing PUCT exploitation;
incorrectly allowing S615 values into the fixed-value prior-swap arm;
shared-trunk effects on prior outputs and any head-only representation shift;
input orientation and legal mask; search-runtime drift; stochastic seed
semantics; nondeterministic kernels; tie-breaking; simulation count and
termination; search cache/tree reuse; oracle perspective/tie semantics;
duplicate states and clustered provenance; post-hoc cohort selection; and
policy disagreement enrichment. Full-model search effects do not establish
policy-only causality. First-divergence observational game outcome differences
do not establish action value.

## Prospective registration (no execution authorized)

Before any run, register and freeze:

1. **Cohort:** state serialization, canonical and production-float32 identity,
   player-to-move, legal moves, source archive/file/line and hash, game/opening
   cluster, ply/prefix, stratum, selection procedure/seed, exclusions, and
   exact-oracle join identity. Freeze up to 128 from the #557/#559 union,
   independent of Seed 615 state overlap; minimum 64 exact-covered with at
   least 16 prior agreements and 16 disagreements. Optionally freeze a
   separate Seed 615 trajectory cohort for action-only analyses.
2. **Models and runtime:** checkpoint SHA-256s and authenticated source
   registration for B512/S615; export hashes; architecture/config and tensor
   equality audit; code commit/source hashes; Python/framework/device/runtime
   versions; input encoding and rule implementation. Specify full-checkpoint
   full-checkpoint arms and fixed-value prior-source arms separately. If a
   head-only ablation is registered, bind the tensor dependency audit and
   disclose its fixed-trunk estimand.
3. **Search:** fixed simulation budget (recommend 128, matching #557/#559),
   one worker, fresh tree per arm/state, exact-root/leaf solve disabled,
   fixed PUCT/FPU/root parameters, no noise unless explicitly frozen,
   maximum-visit action rule and deterministic tie-break. Freeze paired
   counter-based randomness or deterministic no-noise semantics as above.
4. **Metrics and analysis:** archive priors, visits, selected actions, RNG
   keys/draw audit, per-action exact margins, and per-state regret. Estimate
   paired state-level differences and uncertainty by resampling provenance
   clusters (opening/game), not actions or individual search simulations.
   Report state-level paired points plus 95% cluster-bootstrap intervals and
   exact-count tables for each estimand. Keep prior, fixed-value prior-swap,
   full-checkpoint, and optional action-only endpoints distinct.
5. **Predeclared decisions:** (a) evidence that prior-source intervention
   affects search decisions requires post-search action-disagreement rate
   above 5% with 95% interval lower bound above 0% in the fixed-value prior
   swap. Report full-checkpoint action disagreement separately as a
   whole-checkpoint effect; it does not establish prior attribution. (b)
   Action-quality improvement for the
   prior-priority question requires S615-prior-minus-B512-prior mean exact
   regret at most -0.25 seeds and 95% upper bound below 0, with no increase
   in WDL-inferior rate. Apply the same gate separately to full-checkpoint
   effects; do not attribute a full-checkpoint pass to priors. Otherwise
   classify as no demonstrated quality change or inconclusive according to
   interval width.
   Treat the disagreement-stratum result as secondary, not a rescue gate.
   These are diagnostic thresholds, not promotion criteria.

The fixed 0.25-seed threshold follows the scale used in #558's preregistered
regret gate. At 64–128 states the interval may be wide; state this in advance.
Do not extend the cohort after inspecting outcomes. Any underpowered result is
inconclusive, not evidence of equivalence.

## Reusable components and estimated cost

Reuse #557/#558/#559 cohort/oracle schemas, authenticated source identities,
fixed-root PUCT and maximum-visit tie verification, export/runtime binding,
per-state records, and independent replay-verifier patterns. Reuse Seed 615's
checkpoint exports, frozen policy-prior comparison rows, and source hashes.
The existing diagnostic needs no new self-play or arena games.

Expected implementation: approximately 1–2 engineering days to freeze the
deduplicated #557/#559 cohort and authenticate its oracle joins; 2–4 days for
paired full-checkpoint and fixed-value prior-routing search, RNG
contract/instrumentation and full trace schema; 2–3 days for an independent
verifier, tamper tests and registered analysis; plus review. A head-only
transplant is optional, could add 2–5 days, and may be rejected after the
compatibility audit. These estimates assume the existing export/runtime supports both checkpoints;
they exclude any solver work. If deterministic fixed-root search is already
reusable without changes, the runner effort may be at the low end.

Computational cost at 128 states, four search arms (two full checkpoints and
two fixed-value prior-source arms), 128 simulations per arm is 65,536 total
simulations plus inference and verification. At one worker and similar network
inference costs to #557/#559 this is a small diagnostic (likely minutes to
hours depending on accelerator/CPU batching and deterministic kernel setup),
not an arena-scale run. A smaller registered cohort reduces this linearly. Do
not launch until resource/runtime bounds and reproducibility environment are
frozen.

## Independent verification and reproducibility

Require an independent verifier to work from a physical copy of frozen
inputs/outputs and bundled runtime assets, without consulting mutable source
paths. It must reconstruct every state identity and oracle join; validate
legal actions, orientations, margins and exact-optimal sets; authenticate all
model/runtime/source hashes; replay every search from frozen RNG contract;
check simulation counts, visit totals, selected-action/tie rule, priors,
post-search action and per-state regret; and independently regenerate all
aggregates, intervals and decision labels. Reject missing, duplicate,
out-of-cohort, altered, or unbound rows. Include tests for player perspectives,
multiple optimal moves, action ties, agreement/disagreement strata, and
counterfactual RNG alignment. Publish the registration, frozen artifacts,
commands, environment lock, hashes, raw per-state evidence, verifier report,
and analysis code. Independent review must confirm no holdout-state training
reuse and no candidate promotion.

## Decision

**Revise.** Freeze exact-quality states directly from the authenticated #557/#559
union, without requiring any Seed 615 overlap. Keep any Seed 615 trajectory
cohort separate and action-only unless its own states already have authenticated
targets. Compare full checkpoints as a whole-model estimand and separately
intervene on prior source while holding the value evaluator fixed; do not infer
policy causality from full-checkpoint differences. Proceed to experiment
registration only after cohort, oracle, and runtime-routing gates pass. This
report authorizes no execution.
