# Seed 620 — Controlled search-decision attribution feasibility

**Status:** review-only feasibility assessment. No games, MCTS, exact solving,
training, or checkpoint edits were performed. No candidate is selected or
promoted.

## Recommendation

**Revise, then proceed conditionally.** The required ingredients exist to
compare B512 and S615 at frozen identical states: authenticated checkpoint
exports, matching model layouts, deterministic fixed-root PUCT, and prior
examples that bind cohorts, seeds, budgets, and full search traces. Exact
action-margin targets exist for the frozen #557 256-root cohort and #559
512-root cohort. However, their exact overlap with the 676 first-divergence
states and the 1,024 Seed 615 opening states is not established by stone count
or by the published aggregate summaries. A read-only identity join and cohort
freeze must precede registration. If fewer than 64 exact-covered states can be
selected with sufficient disagreement and agreement cases, run a prior-only
diagnostic on the larger fixed cohort and defer exact-regret claims; do not
solve new states under this task.

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
orientation. Before a prospective freeze, join canonical state identity and
player-to-move/seat orientation against each authenticated oracle record,
validate legal-action sets and exact-margin perspective, and report exact
coverage by candidate source. Do not run an oracle query to fill any gap.

First divergence is outcome-conditioned: a state enters only after two
candidate-specific game trajectories have already separated. It overweights
positions where actions differed, is influenced by the different game seeds,
and omits all agreeing states after the first divergence. Testing only that
subset estimates a selected disagreement effect, not the typical causal
effect across the evaluation domain.

Proposed fixed cohort: up to 128 unique canonical states sampled before the
new inference, stratified as 64 from the registered Seed 615 opening roots
(including at least 32 policy top-action agreements and 32 disagreements,
when available) and 64 from first-divergence roots with the same agreement /
disagreement balance. Deduplicate across strata; where a source stratum is
short, backfill from the other stratum without changing balance targets if
possible. Freeze the entire eligible pool, selection seed, selected identities,
and provenance first. Report an unweighted overall estimate only if the
selection probabilities support it; otherwise report stratum-level estimates
and the fixed-cohort average as descriptive. Preserve opening/game cluster
identities for uncertainty estimation. If exact coverage in a stratum is
insufficient, do not silently substitute stone-count eligibility.

The 64/64 split deliberately includes both pre-existing disagreement and
agreement states; it is a diagnostic contrast, not a representative estimate
unless sampling weights justify that interpretation. Ideally 128 exact-covered
states; minimum feasibility gate is 64 exact-covered states with at least 16
in each agreement/disagreement class. Otherwise revise to a larger prior-only
cohort and label exact-quality conclusions unsupported.

## Controlled counterfactual and policy-only swap

Both checkpoints derive from the same AlphaZero-lite residual architecture;
Seed 615's training report records four value tensors matching B512 and 18
policy/trunk tensors differing. This supports a plausible aligned
policy-head comparison, but the shared trunk means the true intervention is
not simply the final policy head: changing policy/trunk activations may also
change values. Loading full B512 and full S615 into the same runtime and
comparing those outputs isolates checkpoint identity, not policy alone.

A controlled policy-only experiment is technically feasible only after a
read-only tensor-layout and serving-export audit confirms identical names,
shapes, dtypes, architecture/configuration, input encoding, legal-action
masking, and value convention. Construct two compositions with a single
common frozen value/trunk route and swap only the policy component if
architecture permits. Because policy and value share the trunk in these
models, a literal policy-head-only transplant may not represent the trained
policy function. The clean estimand for existing full checkpoints is therefore
**full-checkpoint effect on search decisions**. A policy-only ablation can be
a separate, explicitly labeled diagnostic: hold B512's trunk/value path fixed
and route B512 versus S615 policy-head weights only, but it may be invalid if
the policy head expects its own trained trunk representation. Do not declare
that swap feasible until runtime exports and tensor dependency graph are
verified; no new checkpoint should be created or modified.

For the main full-checkpoint comparison, identical-state paired search is
feasible. Fix one search-runtime commit/export and identical rules, encoding,
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

Record both (1) raw network prior top action and optimal-action prior mass,
before search, and (2) post-search selected action (maximum visits with
frozen tie-break), visit vector, and exact regret. Equal simulation counts
must be enforced, not merely equal wall time. Report where the post-search
action changed relative to the raw-prior top action.

## Hypothesis, outcomes, and confounders

**Primary hypothesis:** on identical fixed states and under identical fixed
search semantics, S615's altered priors change post-search actions more often
than expected from paired search variation, and reduce mean exact regret
relative to B512 on the fixed exact-covered cohort. Secondary hypothesis:
the effect is larger in states where B512/S615 raw-prior top actions disagree
than where they agree.

Primary metrics on exact-covered states: paired difference in exact regret
(S615 minus B512; negative favors S615), exact-optimal action rate, and
WDL-inferior action rate. Exact regret is exact best final-score margin minus
selected-action final-score margin, perspective-normalized; ties in exact
optimal actions all count optimal. Secondary: post-search action disagreement
rate and prior optimal mass. Prior-only metrics on the full cohort remain
separate from searched outcomes.

Confounders to bind or disclose: checkpoint/export mismatch; architecture or
tensor routing; model-specific value outputs changing PUCT exploitation;
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
   exact-oracle join identity. Freeze up to 128; minimum 64 exact-covered with
   at least 16 agreements and 16 disagreements.
2. **Models and runtime:** checkpoint SHA-256s and authenticated source
   registration for B512/S615; export hashes; architecture/config and tensor
   equality audit; code commit/source hashes; Python/framework/device/runtime
   versions; input encoding and rule implementation. Specify full-checkpoint
   primary arms. Register any policy-only decomposition separately and only
   after compatibility passes.
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
   exact-count tables. Keep prior and post-search endpoints distinct.
5. **Predeclared decisions:** (a) evidence that priors affect search decisions
   requires post-search action-disagreement rate above 5% with 95% interval
   lower bound above 0%; (b) action-quality improvement requires S615-minus-
   B512 mean exact regret at most -0.25 seeds and 95% upper bound below 0,
   with no increase in WDL-inferior rate; (c) otherwise classify as no
   demonstrated quality change or inconclusive according to interval width.
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

Expected implementation: approximately 2–4 engineering days for an immutable
cohort/oracle join and coverage report; 2–4 days for a paired fixed-root runner,
RNG contract/instrumentation and full trace schema; 2–3 days for an independent
verifier, tamper tests and registered analysis; plus review. A policy-only
transplant could add 2–5 days or be rejected after compatibility inspection.
These estimates assume the existing export/runtime supports both checkpoints;
they exclude any solver work. If deterministic fixed-root search is already
reusable without changes, the runner effort may be at the low end.

Computational cost at 128 states, two arms, 128 simulations is 32,768 total
simulations plus inference and verification. At one worker and similar network
inference costs to #557/#559 this is a small diagnostic (likely minutes to
hours depending on accelerator/CPU batching and deterministic kernel setup),
not an arena-scale run. Do not launch until resource/runtime bounds and
reproducibility environment are frozen.

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

**Revise.** The causal fixed-state search question is feasible for full
checkpoints, but exact coverage must first be explicitly joined and policy-only
causality is not yet established by architecture compatibility. Freeze a
balanced cohort including agreements before inference; use authenticated
existing targets only. Proceed to an experiment registration only after the
coverage and compatibility gates pass. This report authorizes no execution.
