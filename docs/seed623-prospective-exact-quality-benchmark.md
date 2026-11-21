# Seed 623 — prospective exact-quality benchmark for B512 and S615

**Status:** protocol and feasibility assessment only. No state generation,
model inference, search, oracle query, solver run, training, or evaluation was
performed. This document authorizes no execution and is not a candidate-
promotion evaluation.

## Recommendation

**Revise.** A prospective protocol is technically plausible and avoids
Seed 620–622's outcome-conditioned, highly selected historical roots. However,
the missing identity-complete P-E4 ancestry prevents a claim that newly generated
states are absent from undocumented historical training. Obtain separate
authorization for a feasibility-only solver benchmark and a provenance review;
register the full benchmark only if both gates pass and the intended
known-exposure estimand is accepted. If the requirement is proof of no training
exposure, reject this design until the P-E4 ancestry gap is repaired.

## Research questions and estimands

The target domain is reachable nonterminal states from the stated generation
distribution with 17–21 stones remaining in pits (the exact tier-21 domain),
at least two legal actions, and complete exact legal-action coverage. Claims
are conditional on this domain and generation distribution; they do not
generalize to all Kalah positions, opening play, higher-stone positions, full
games, or playing strength.

1. **Primary representative decision quality:** paired difference in final-score
   exact regret between the S615 full checkpoint (arm B) and B512 full
   checkpoint (arm A), averaged over the primary population. Positive values
   favor B512; report S615-minus-B512 as the signed contrast (negative favors
   S615). This estimates the whole-checkpoint effect, including policy, value,
   and shared-trunk changes, after fixed-budget search.
2. **Prior-source intervention:** paired S615-prior/B512-value (arm C) minus
   B512-prior/B512-value (arm A) exact regret. This estimates switching the
   checkpoint-sourced prior function under the B512 evaluator, not an isolated
   policy-head tensor effect.
3. **Secondary disagreement diagnostic:** among a separately declared,
   disagreement-enriched subset of the frozen eligible population, estimate
   prior and post-search action disagreement and paired regret contrasts. This
   is conditional diagnostic evidence, not an unbiased population estimate.

The benchmark measures decisions against exact final-score margins at a fixed
128-simulation budget. Any diagnostic improvement is not evidence of improved
playing strength or candidate promotion.

## Review of prior evidence and reusable infrastructure

- **PR #620** documents feasibility, separates full-checkpoint from
  fixed-value prior-source effects, and correctly notes that first-divergence
  states are outcome-conditioned. Its proposed historical-root cohort is now
  superseded by the Seed 621/622 clearance findings.
- **PR #621** authenticated 768 unique roots across #557/#559 with full exact
  legal-action coverage and model-prior reconstructions. It found two S615
  membership overlaps, two protected-holdout overlaps, 745 prior agreements,
  and 19 disagreements among the remaining direct membership nonmatches.
  Those nonmatches were not cleared for contamination.
- **PR #622** reconstructed 512-update B512/S615 consumed-state schedules and
  confirmed both inherit P-E4. It classified 4 roots as confirmed exposure,
  764 as unresolved, and none as cleared. P-E4 has no identity-complete
  inherited training ledger; training-game/opening identifiers also cannot be
  fully crosswalked to #621 source groups.
- **PR #557** provides frozen roots, canonical selection, fixed-root deterministic
  PUCT, exact per-action margins, action/tie evidence, and a verifier. **PR
  #559** provides 17–21-stone roots, fresh 128-simulation trees, exact-solving
  disabled during search, and a native tier-21 oracle path. Their roots derive
  from #556 trajectories and are retrospective, not a representative
  prospective population.
- The authenticated B512/S615 checkpoint exports and Seed 615 evaluation and
  training manifests provide model/source identities and known membership
  inventories. The #621 `ComposedArtifactEvaluator` supports C's S615 prior
  and B512 value routing. It computes both source outputs before selecting;
  independently assert the selected value is B512's at every expansion and
  deliberately test against conflicting S615 values.
- Reuse `kalah_v3` production encoding, canonical Kalah state serialization,
  legal-action masking, deterministic PUCT behavior, #557/#559 maximum-visit
  tie convention, exact oracle record schema, and independent physical-copy
  verifier patterns. Historical exclusion contracts and Seed 615 opening
  generation are useful membership and reproducibility precedents, but do not
  prove absence from undocumented P-E4 ancestry.

The prior evidence gives no prospective exact-quality result for B512/S615.
Neither prior disagreement nor arena outcomes may determine primary selection.

## Deterministic, model-independent state generation

This is a proposed protocol, not an instruction to generate states now.

1. **Rules and identities:** bind the canonical rules implementation and source
   commit/hash, rule version, initial state, legal move semantics (including
   extra turns and terminal sweep), state canonicalization, and `kalah_v3`
   encoder source/hash. Pin Python/runtime and PRNG implementation/version.
   State serialization is canonical sorted-key JSON with integer pit/store
   counts, side to move, and explicit rule ID; separately store SHA-256 of
   canonical state bytes and of the exact production float32 input bytes
   (little-endian, fixed feature order and shape). Reject a state if canonical
   identity or encoded float32 identity duplicates any earlier selected row.
2. **Generation seeds:** preregister master seed `62320261121`; derive seeds
   using SHA-256 over a versioned domain separator plus `(opening_index,
   trajectory_index, phase)`, taking a fixed 64-bit prefix. Freeze algorithm,
   byte order, and seed derivation before generation. These are prospective
   suggested constants and must be validated and frozen in a separately
   authorized execution freeze; never replace individual seeds after observing
   states or model outputs.
3. **Opening/game coverage:** define 128 independently seeded opening
   clusters. For each opening, create two independent legal random-play
   continuations, for 256 source games total. Use a specified deterministic
   uniform-over-legal-actions policy for every move; no model, checkpoint,
   evaluation result, or model prior is available to the generator. Apply
   opening seed to the source-game identity; use the standard initial state
   with a fixed zero-ply prefix to avoid an additional opening policy. Continue
   until terminal or a fixed 200-ply cap. The game state engine, random draw
   schedule, and termination behavior are hash-bound.
4. **Stone/position coverage:** collect all eligible visits in each of five
   active-pit-stone bins: 17, 18, 19, 20, 21. Within each game and bin,
   select at most one encountered state using a deterministic hash rank of
   `(master seed, game ID, canonical state hash)`; then retain at most two
   states per game, selected by ascending hash rank over that game's selected
   bin candidates. Quotas
   across the primary sample are fixed at 102 or 103 states per stone count
   (512 total), allocated by a preregistered deterministic rank within bin,
   with 128 openings represented and no opening contributing more than four
   states. Require at least 80 states per stone bin and 100 opening clusters;
   otherwise report generation shortfall without relaxing criteria. Every
   candidate must replay legally from the initial state and be nonterminal
   with at least two legal actions.
5. **Freeze before inference/search:** publish the full generated candidate
   inventory, source trajectories, opening/game cluster IDs, legal-action
   sets, stone counts, both identity hashes, seeds, and exclusions. Freeze
   it and compute hashes before any B512/S615 model is loaded. Check the entire
   candidate inventory against authenticated known training and protected
   holdout inventories before drawing the primary sample. Selection, quotas,
   exclusions, and state IDs are fixed without prior disagreement, inference,
   exact margins, or outcomes. If fewer than 512 remain after documented
   overlap exclusions, stop and report the shortfall; do not fill based on
   models or outcomes.

The random legal policy defines a reproducible reachable-state distribution,
not a uniform distribution over all legal Kalah states or necessarily the
production-game distribution. Report state inclusion probabilities and
stone-bin/opening weights. Keep all states generated from one opening/game
cluster together in analysis. Full candidate pool and exclusions are published
so later auditors can reconstruct selection.

## Known-training-exposure exclusion contract

Before any B512/S615 inference, compare each candidate to the complete
authenticated **available** identities for B512 training membership, S615
training membership, and protected holdouts, using both canonical state and
production float32 input hashes. Also compare provenance against every
authenticated training-source game/opening ledger for which a complete
crosswalk exists. Remove exact identity matches and any proven source-game or
opening-cluster matches; record match class, source hash, and reason. A missing
inventory, ambiguous identity, or unavailable crosswalk is `unresolved`, never
`cleared`. Exclusion code and inventories are independently verified; do not
inspect models to classify or resolve exposure.

**P-E4 gap:** B512 and S615 both descend from P-E4 (`2ecc98ac…edbad`). Its
identity-complete training-state ancestry is unavailable. Therefore canonical
and float32 nonmatches establish only “no exact match in the authenticated
available B512/S615/protected-holdout inventories.” They cannot establish
absence from P-E4, its ancestors, transformed/augmented records, or unknown
source trajectories. New states do not repair this gap simply by being
generated prospectively. The strongest defensible label is **known-exposure
screened; inherited exposure unresolved**. Report separate counts of exact
matches removed, provenance-linked clusters removed, and unresolved rows. No
“training-disjoint,” “unseen,” contamination-free, or generalization claim is
permitted. The estimand applies to the generated population under this explicit
exposure limitation. If that limitation invalidates the intended question,
reject before execution.

## Primary and secondary cohorts

### Primary: representative fixed-state cohort

Use the 512-state hash-selected sample above after only the fixed known-exposure
exclusions. Do not stratify, oversample, or replace states on prior or model
behavior. Preserve the prospective quotas and inclusion weights; report the
population estimate with those weights and a sensitivity analysis restricted
to equally weighted states. All three search arms run on the same frozen roots.
The primary inference is conditional on the reachable 17–21-stone population
and known-exposure-screened sample, not historical training-disjointness.

### Secondary: prior-disagreement diagnostic

Only after the entire eligible population and primary sample are frozen, run
deterministic legal-masked prior inference on the eligible population. Define
top-action disagreement using the existing deterministic prior tie rule and
also report total-variation distance (TV) continuously. The diagnostic subset
is the top 64 eligible rows by descending TV, capped at one per source game and
four per opening; ties use SHA-256 of the frozen state ID. If fewer than 32
unique source games qualify, report it as unavailable. Preserve a separately
selected random 64-state reference subset using the frozen hash rank, for
descriptive contrast only. Do not let either subset replace or reweight the
primary estimate; do not call the enriched subset representative.

No automatic 16-disagreement minimum is carried forward from Seed 620. The
secondary estimand concerns the effect in a defined high-TV tail, so it needs
at least 64 diagnostic states from 32 distinct games and 64 primary-population
reference observations (potentially overlapping primary roots) to describe
the selected-tail contrast. Report continuous TV and rank/selection rules;
report instability if caps alter the top-64 set. This gate is about diagnostic
precision and cluster support, not population disagreement prevalence or
quality.

## MCTS comparison and routing verification

| Arm | Prior source | Value source | Interpretation |
|---|---|---|---|
| A | B512 | B512 | Full B512 checkpoint |
| B | S615 | S615 | Full S615 checkpoint |
| C | S615 | B512 | S615-sourced prior function, fixed B512 evaluator |

The B512-prior/B512-value composed control is computationally and
semantically identical to A if the evaluator routes both outputs from the same
B512 inference, legal mask, and value perspective. Verify equality of model
identity, priors, value at every root and expansion, search traces, and selected
actions on a synthetic conflicting-source fixture. If all assertions pass,
run only A/B/C (not a redundant fourth search) and retain explicit control
assertions and their verifier evidence.

Use the existing deterministic PUCT implementation, pinned to a code commit
and source hashes, with 128 simulations per state/arm, fresh trees, one search
worker, no root noise, no subtree/cache reuse, identical `c_puct`/FPU and
selection/backup semantics, deterministic tie-breaking, exact-root and exact-
leaf solving disabled in search, and fixed maximum-visit action selection.
Freeze input orientation, terminal handling, legal masking, precision/device,
runtime, inference batch behavior, and stopping semantics. Verify exact
simulation count and visit totals per arm/state.

Independently verify prior and value routing at every network expansion, not
only roots: record source IDs and hashes, state hash, raw/legal prior vector,
value, side-to-move, and expansion index. C's expansion value must equal B512's
and must not reflect S615's value; test deliberately contradictory S615 values.
Verify arm A against the explicit B512/B512 composed control assertions. Arm B
is the full-checkpoint contrast; C−A is the fixed-value prior-function
contrast. B−A alone cannot identify a prior-source effect. Because S615 changed
shared-trunk tensors, C estimates switching the S615-sourced prior function,
including its changed representation, not an isolated policy-head intervention.

At 512 states × 128 simulations × 3 unique arms, planned work is 196,608
simulations plus inference/routing verification. Excluding an exactly
identical redundant control saves 65,536 simulations. Search is a diagnostic
and must not consume oracle values.

## Exact-oracle protocol and feasibility gates

Oracle work requires separate authorization after the model-independent state
population is frozen. No solver was run for this assessment.

- Reuse the authenticated native Kalah exact tablebase/probe used by #559 only
  if an independent audit proves exact compatibility with the canonical rules,
  tier-21 bound, state orientation, and objective. Bind binary/source hashes,
  tablebase SHA-256/LFS OID, build recipe/compiler/flags, platform, and license
  or provenance metadata. Reproducibility requires a clean rebuild or a
  separately archived authenticated executable and deterministic outputs.
- Query **every legal action** for every state and return exact final-score
  margins from the root side-to-move perspective. Apply each legal root action,
  solve the successor to terminal under the frozen rules, and transform
  perspective explicitly (including Kalah extra turns). Require no missing
  legal action, no fallback to network values, and no partial coverage.
- Define `best_margin=max(margin[action])`; exact regret for selected action
  `a` is `best_margin-margin[a]`. Every exact-maximum action is optimal. A
  WDL-inferior action has a strictly worse win/draw/loss class from the root
  perspective than the best legal action; equal-class score-margin losses are
  regret but not WDL-inferior. Assert tie consistency and no floating rounding
  before comparing integer margins.
- Independently verify by replaying state/action legality and checking all
  action values against a second implementation or independently rebuilt
  solver. Reconstruct exact-optimal sets, regrets, WDL classes, and aggregates
  from a physical copy of frozen inputs and outputs. A solver self-check alone
  is insufficient.
- **Resource gate:** before authorizing full solving, run a separately
  authorized calibration on a prespecified small set (e.g. 20 states spanning
  17–21 stones) to measure cold startup, tablebase prewarm, per-action query,
  peak RSS, and reproducibility. #559 reported 0.110 ms warm exact-decision
  mean and 0.127 ms p95 over tier-21 roots, but ~3,877.88 seconds tablebase
  prewarm; these are historical measurements, not promises for this population
  or runtime. Require projected full-run wall time within a separately
  registered 24-hour budget, memory within registered hardware limits, 100%
  action coverage, deterministic repeated calibration results, and independent
  solver agreement. If any fails, stop before full oracle execution. Calibration
  is itself a solver run and requires explicit authorization; do not perform it
  as part of this design task.

The domain must remain within 21 active pit stones because the authenticated
tier-21 tablebase is the known reusable exact resource. Do not silently expand
the domain or infer coverage from stone count alone.

## Statistical registration and decision rules

### Endpoints

- **Primary endpoint:** cluster- and inclusion-weighted mean paired exact
  regret difference B−A (S615 minus B512); negative favors S615. The sign,
  weighting, and comparison are fixed before oracle/action results.
- **Key secondary:** C−A mean paired exact regret; negative favors the
  S615-sourced prior function with B512 value fixed.
- **Secondary:** post-search selected-action disagreement for B−A and C−A;
  exact-optimal action rate; WDL-inferior action rate; prior top-action and
  optimal-action mass; prior TV; pre-search and post-search exact regret;
  per-stone-bin effects. Diagnostic-cohort effects are separately labeled and
  never substituted for the primary endpoint.
- Report exact-optimality with ties included. Report counts and rates of
  WDL-inferior choices plus cluster-aware intervals. Separate the oracle's
  exact optimal action set from post-search actions and raw-prior top actions.

### Size and precision rationale

Candidate primary size is **512 states in at least 100 opening clusters**.
This is larger and less selected than the historical 64–128-state suggestion,
and supports stone-bin coverage plus clustered uncertainty. Planning assumes
paired regret-difference SD 1.5 seeds and intracluster correlation 0.20 with at
most four states per opening: design effect `1 + (4−1)*0.20 = 1.6`, effective
sample about 320. For a two-sided 5% test, a true 0.25-seed mean difference
then has approximate standardized effect `0.25/(1.5/sqrt(320)) = 2.98`, about
85% power by normal approximation. Approximate 95% half-width is 0.16 seeds
under these assumptions. These are design assumptions, not empirical Seed
623 variance estimates; conduct a blinded, model-free design audit of cluster
counts and weights before authorization. If effective cluster sample size falls
below 250 or any opening dominates >4 observations, revise the design before
inference, not after viewing outcomes. No interim sample-size extension.

### Uncertainty and decisions

Use paired resampling of opening clusters (both games and every selected state
within a cluster travel together), with 20,000 bootstrap replicates and a
frozen analysis seed. Apply the fixed inclusion weights inside each replicate;
report percentile 95% intervals and a cluster-robust sensitivity interval.
Never bootstrap actions, simulations, or states as independent observations.
Report effective cluster count, largest cluster, state counts per stone bin,
and raw unweighted results. Predeclare multiplicity: B−A is the sole primary
contrast at two-sided 5%; C−A and all strata are secondary, with intervals
descriptive and no multiplicity-based promotion gate.

Decision labels:

- **Diagnostic improvement demonstrated:** primary B−A mean regret ≤ −0.25
  seeds and its 95% interval upper endpoint < 0, with no increase in
  WDL-inferior rate (difference's 95% upper bound ≤ 0). Separately apply this
  same labeled diagnostic rule to C−A; it does not replace B−A.
- **No demonstrated improvement:** primary interval includes zero and its
  upper/lower limits rule out the preregistered −0.25-seed benefit (interval
  lower endpoint > −0.25). This is not equivalence or proof of no effect.
- **Inconclusive:** interval spans both zero and −0.25, the effective cluster
  size/coverage gate fails, or exact oracle or routing verification is
  incomplete. No post-hoc expansion or endpoint substitution.

Regardless of diagnostic label, report exact-optimal and WDL-inferior action
rates with intervals. The benchmark does not test head-to-head game outcomes,
full-match strength, deployment readiness, or candidate promotion. Arena
strength improvement requires a separate prospectively registered evaluation.

Historical checkpoint exposure remains a major interpretive limitation: even a
positive prospective-state result cannot establish performance on genuinely
unseen states while P-E4 ancestry is unresolved. It can establish comparative
decision quality on the generated, known-exposure-screened population only.

## Stages and authorization gates

1. **Review this protocol:** independent review of estimands, game/rule/source
   identity, generation distribution, exclusion contract, and the P-E4
   limitation. No execution authorized here.
2. **Prepare a generation/exclusion freeze:** separately authorize deterministic
   model-independent generation and membership screening only. Bind code,
   PRNG, seeds, rules, serializers, hash-rank procedures, and protected/training
   membership snapshots before generation. Stop on missing identities,
   replay mismatch, or source hashes changing.
3. **Freeze and audit population:** independently replay all trajectories,
   verify canonical and float32 deduplication, reproduce quotas/selection,
   check exposure classifications, and publish inventory. No inference until
   review accepts the unresolved exposure contract and population gates pass.
4. **Prior classification and diagnostic subset:** separately authorize
   B512/S615 inference after state freeze. Freeze priors, classes, and selected
   diagnostic subset before any search. The primary sample remains unchanged.
5. **Solver calibration:** separate authorization for 20-state feasibility
   calibration after source/tablebase audit. Pass runtime, resources, exact
   coverage, determinism, and independent agreement gates before authorizing
   complete oracle work.
6. **Oracle freeze:** authorize complete all-legal-action solve and independent
   verification. Any gaps or disagreement stop the benchmark; no state
   replacement.
7. **Search registration and execution:** only after population, oracle,
   runtime, routing, analysis code, and resource caps are frozen; separate
   authorization required. Bind workers and runtime. Search workers remain
   one per state/arm; pair-level concurrency and CPU/memory limits must be
   frozen before execution under repository arena policy.
8. **Independent report/review:** verify from a physical copy, reproduce all
   records and estimates, disclose historical-exposure limits, and obtain
   independent review. No automatic merge, deployment, or candidate promotion.

## Feasibility summary

The state generator, legal Kalah transition engine, `kalah_v3` identity path,
B512/S615 exports, deterministic fixed-root PUCT, and exact tier-21 machinery
have strong precedents. The primary novelty is implementing and independently
auditing the prospective model-independent generator, exposure inventory join,
and a clustered sampling manifest. Search itself is modest at 196,608 total
simulations. Exactness is the resource risk: the historical ~3,878-second
prewarm dominates nominal warm query time and must be remeasured through an
authorized calibration. The most important scientific limitation is not
computational: unresolved P-E4 ancestry forbids a clean historical
non-exposure claim. Accordingly the recommendation remains **revise**, with
execution conditional on the stated gates and acceptance of the narrower
estimand.
