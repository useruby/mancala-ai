# Seed 623 — prospective exact-quality benchmark for B512 and S615

**Status:** proposed protocol amendment; not authoritative pending explicit
project-owner approval. Protocol and feasibility assessment only. No state
generation, model inference, search, oracle query, solver run, training, or
evaluation was performed. This document authorizes no execution and is not a
candidate-promotion evaluation.

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

### Amendment status

This document incorporates the proposed Seed 624 estimand amendment for review.
It becomes authoritative only after explicit project-owner approval is recorded
with the approver identity, approval timestamp, exact amended protocol
commit/hash, and decision to withdraw the inclusion-probability and
design-unbiased inference requirements. This documentation change itself grants
no execution authorization.

The target domain is reachable nonterminal states from the stated generation
distribution with 17–21 stones remaining in pits (the exact tier-21 domain),
at least two legal actions, and complete exact legal-action coverage. Claims
are conditional on this domain and generation distribution; they do not
generalize to all Kalah positions, opening play, higher-stone positions, full
games, or playing strength.

1. **Primary finite-cohort decision quality:** exact B−A contrast in final-score
   regret, S615 minus B512, over all 512 states in the frozen hash-selected
   population. Compute the mean within each of the five stone-count bins and
   standardize with fixed weight 1/5 per bin. Negative values favor S615. This
   is a finite-cohort descriptive contrast, not an estimate with design-unbiased
   inference over the broader generation process. It captures the whole-
   checkpoint effect, including policy, value, and shared-trunk changes, after
   fixed-budget search.
2. **Prior-source diagnostic:** exact finite-cohort paired C−A contrast in
   regret, standardized with the same fixed 1/5 weight for each stone-count
   bin. This describes switching the checkpoint-sourced prior function under
   the B512 evaluator, not an isolated policy-head tensor effect.
3. **Secondary disagreement diagnostic:** among a separately declared,
   disagreement-enriched subset of the frozen primary population, estimate
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
   identity or encoded float32 identity duplicates any earlier eligible
   candidate row.
2. **Generation seeds:** preregister master seed `62320261121`; derive seeds
   using SHA-256 over a versioned domain separator plus `(opening_index,
   trajectory_index, phase)`, taking a fixed 64-bit prefix. Freeze algorithm,
   byte order, and seed derivation before generation. These are prospective
   suggested constants and must be validated and frozen in a separately
   authorized execution freeze; never replace individual seeds after observing
   states or model outputs.
3. **Opening/game coverage and oversupply:** generate 256 distinct opening
   positions, each from an independent SHA-derived seed and exactly eight
   uniform legal actions from the standard initial position. These 256 opening
   positions—not trajectory batches—are the top-level provenance clusters.
   Deduplicate opening positions by canonical and production-float32 identity;
   continue generation only by consuming the next preregistered opening index
   from a frozen reserve of 512 opening seeds. For every retained opening
   position, generate eight independent continuation games (2,048 games at
   full capacity), each with its own derived PRNG stream and uniform-over-legal-
   actions policy. Continue to terminal or a 200-action cap. No model,
   checkpoint, evaluation result, or prior is available to this generator.
   Bind the game engine, prefix/game random draw schedules, and termination
   behavior. If fewer than 256 distinct opening positions survive the frozen
   512-seed reserve, stop with a generation shortfall.
4. **Candidate inventory and primary quotas:** in every continuation game,
   collect all eligible visits in each active-pit-stone bin (17, 18, 19, 20,
   21), with at least two legal actions. Rank occurrences by SHA-256 of the
   versioned domain separator, master seed, opening ID, game ID, bin, and
   canonical state hash; keep at most one candidate per game/bin, then deduplicate
   globally by canonical state and production-float32 input identity. Preserve
   all occurrence and source-game links. After exact-known-exposure exclusions,
   the **hard primary target** is exactly 512 states: 103 each in bins 17 and
   18, and 102 each in bins 19, 20, and 21. Select under exact constraints of
   at most two states per continuation game, at most four per opening cluster,
   and representation of at least 128 distinct opening clusters. A pinned,
   single-worker deterministic integer optimizer selects the feasible set
   lexicographically by the within-bin candidate hash ranks (canonical state
   hash breaks any hash-rank collision); bind optimizer name/version, settings,
   and source hash before generation. This makes the selection reproducible
   while satisfying all five exact quotas. The former 80-per-bin value is not
   a sufficient acceptance gate.
   A preferred oversupply gate is at least four times each bin's target
   (412/412/408/408/408 unique, eligible, known-exposure-screened candidates)
   before cluster caps. The hard minimum capacity gate is at least twice each
   target (206/206/204/204/204), plus a successful constrained selection to
   the exact target and the cluster/game limits above. The 4x gate is preferred,
   not mandatory; failure of a hard gate means no primary sample.
5. **Deterministic selection and shortfall:** freeze the full generated
   inventory, apply only the predeclared exact-membership/provenance exclusions,
   and select by the frozen hash rank and quotas. Do not use inferred priors,
   exact margins, model outcomes, or search outcomes to rank, exclude, or
   replace states. If the hard per-bin inventory, distinct opening-cluster,
   constrained-selection, deduplication, or exact quota gate fails, publish
   the shortfall and stop. Do not top up from other stone bins, draw ad hoc
   seeds, or replace rows; any resampling requires a new preregistration,
   freeze, and authorization before generation.
6. **Freeze before inference/search:** publish candidate and selected-state
   inventories, complete trajectories, opening/game cluster IDs, legal-action
   sets, stone counts, canonical and float32 identities, seed derivations,
   exclusions, and selection ranks. Freeze and hash
   them before any B512/S615 model is loaded. Check all candidates against the
   authenticated available training and protected-holdout identities before
   selecting the primary cohort.

The random legal policy defines a reproducible reachable-state distribution,
not a uniform distribution over all legal Kalah states or necessarily the
production-game distribution. The amended primary target is the deterministic,
hash-standardized selected-state population: the exact 512 frozen states,
standardized to the equal-weighted mixture of the five eligible stone-count
bins with fixed weight 1/5 per bin. The constrained hash-ranked selection is not
a probability sample of the broader generation process. Candidate inclusion
probabilities and design weights are neither claimed, calculated, reported, nor
applied; results do not support design-unbiased inference over all states
reachable under that process. Report the opening-position generation
distribution, fixed bin weights, and each selected state's source game. Keep
opening clusters intact for descriptive stability resampling. Publish the
complete generated candidate inventory, all exclusions and ranks, exact
selected set, per-bin counts, and cluster membership for reconstruction.

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
permitted. The estimand applies to the frozen selected population under this
explicit exposure limitation. If that limitation invalidates the intended
question, reject before execution.

## Primary and secondary cohorts

### Primary: deterministic fixed-state cohort

Use the 512-state hash-selected cohort above after only the fixed
known-exposure exclusions. Do not select or replace states on prior or model
behavior. Preserve the exact per-bin quotas and fixed 1/5 bin weights;
within-bin selection follows the frozen rank and cluster-cap rule. Compute the
standardized primary contrast exactly over the complete selected cohort and
report unweighted descriptive results as sensitivity summaries only. All three
search arms run on the same frozen roots. The estimand is conditional on this
fixed reachable, equal-stone-bin, known-exposure-screened population, not
historical training-disjointness or the broader generation process.

### Secondary: prior-disagreement diagnostic

Only after the primary sample is frozen, run deterministic legal-masked prior
inference on its 512 states (the prior inference is already required for arms
A/B/C). Define
top-action disagreement using the existing deterministic prior tie rule and
also report total-variation distance (TV) continuously. The diagnostic subset
is restricted to the primary cohort: top 64 by descending TV, capped at one per
source game and four per opening; ties use SHA-256 of the frozen state ID. If
fewer than 32 unique source games qualify, report it as unavailable. The full
primary population is its descriptive reference. Do not introduce new states,
replace or reweight the primary estimate, or call the enriched subset
representative.

No automatic 16-disagreement minimum is carried forward from Seed 620. The
secondary estimand concerns the effect in a defined high-TV tail. Its
availability gate is 64 primary states from at least 32 distinct continuation
games, subject to the one-per-game/four-per-opening caps. If unavailable,
report the shortfall and retain the primary analysis. Report continuous TV,
top-action disagreement, and the rank/selection rule; the enriched comparison
is conditional diagnostic evidence and does not estimate population
disagreement prevalence or typical quality.

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

At 512 primary states × 128 simulations × 3 unique arms, planned work is
196,608 simulations. The secondary high-TV cohort is a subset of those same
states and uses their already planned exact margins and A/B/C searches, so it
adds zero solver queries and zero simulations. Prior inference, per-expansion
routing checks, and the arm-A composed-control assertions are included in the
same 512-state execution plan. Excluding an exactly identical redundant control
saves 65,536 simulations. Search is diagnostic and must not consume oracle
values.

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

- **Primary endpoint:** exact finite-cohort mean paired regret difference B−A
  (S615 minus B512), standardized as the mean of the five within-bin means with
  fixed weight 1/5 per bin; negative favors S615. Compute it over all 512
  selected states. The sign, weights, and comparison are fixed before
  oracle/action results.
- **Key secondary:** exact finite-cohort C−A mean paired regret, standardized
  with the same fixed 1/5 weight per bin; negative favors the S615-sourced prior
  function with B512 value fixed.
- **Secondary:** post-search selected-action disagreement for B−A and C−A;
  exact-optimal action rate; WDL-inferior action rate; prior top-action and
  optimal-action mass; prior TV; pre-search and post-search exact regret;
  per-stone-bin effects. Diagnostic-cohort effects are separately labeled and
  never substituted for the primary endpoint.
- Report exact-optimality with ties included. Report counts and rates of
  WDL-inferior choices plus descriptive stability intervals. Separate the
  oracle's exact optimal action set from post-search actions and raw-prior top
  actions.

### Finite-cohort size and structure

The primary cohort is exactly **512 selected states**, with the frozen per-bin
quotas and at least 128 distinct opening positions (the generation target is
256). Each opening position is generated by an independent eight-action prefix
and has up to eight independent continuation games; no more than four selected
states can come from one opening and no more than two from one game. These
constraints define the cohort and its cluster composition; they are not a
probability-sampling or inferential precision guarantee. Before model inference,
conduct a model-free audit of cluster counts, per-bin quotas, and cluster-size
limits. Do not apply an effective-sample-size or power gate to this finite-
cohort descriptive estimand. No interim sample-size extension.

### Uncertainty and decisions

Opening-cluster bootstrap results are **descriptive stability intervals**,
not confidence intervals. With 20,000 replicates and the frozen analysis seed,
resample each opening-position cluster (prefix, all continuation games, and
selected states together); calculate the five bin means in each replicate and
average them with fixed weight 1/5 per bin. Do not apply candidate inclusion
weights. Report the interval to show how the standardized contrast varies under
perturbation of the observed cluster composition. It has no confidence-coverage
or design-based interpretation for a larger stochastic population and is never
used as an inferential gate. Never bootstrap actions, simulations, or states as
independent observations. Report cluster count, largest cluster, state counts
per stone bin, and raw unweighted results. No multiplicity-based promotion
gate applies; C−A and strata remain secondary descriptive contrasts.

For the finite-cohort WDL-inferior action-rate difference, calculate the exact
S615-minus-B512 difference in the fraction of selected actions strictly worse
in win/draw/loss class than the exact-best legal action, standardized with fixed
weight 1/5 across bins. The +1 percentage-point threshold is a descriptive
decision threshold, not a non-inferiority margin or inferential test.

After exact action coverage, oracle, and routing requirements pass, assign the
finite-cohort descriptive label:

1. **Observed selected-population regret threshold met** if and only if the
   exact B−A finite-cohort mean regret difference is at most −0.25 seeds and
   the exact finite-cohort S615-minus-B512 WDL-inferior rate difference is at
   most +1 percentage point.
2. Otherwise, report **observed selected-population regret threshold not met**,
   with a separate flag if the finite-cohort WDL-inferior difference exceeds
   +1 percentage point.
3. If any required exact action coverage or outcome is missing, report
   **unavailable**; do not substitute a stability-interval endpoint or another
   population.

Report the cluster-resampling stability interval beside the exact finite-cohort
contrasts, but never use it to determine these labels. The labels describe only
the frozen selected population; they do not establish improvement, no
improvement, non-inferiority, or generalization for the broader generation
distribution. Any such stochastic-population claim requires a separately
registered sampling design and estimand. Report C−A and diagnostic-tail
contrasts as secondary estimates, not alternate primary decision routes.

Regardless of diagnostic label, report exact-optimal and WDL-inferior action
rates with descriptive stability intervals. The benchmark does not test
head-to-head game outcomes, full-match strength, deployment readiness, or
candidate promotion. Arena strength improvement requires a separate
prospectively registered evaluation.

Historical checkpoint exposure remains a major interpretive limitation: even a
positive prospective-state result cannot establish performance on genuinely
unseen states while P-E4 ancestry is unresolved. It can establish comparative
decision quality on the frozen selected, known-exposure-screened population
only.

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
