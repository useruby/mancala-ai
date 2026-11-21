# Seed 621 — exact-oracle cohort and prior feasibility audit

Read-only audit of existing #557/#559 oracle-covered roots against authenticated
B512/S615 checkpoint exports and the published Seed 599/615 training and
protected-holdout membership inventories. It runs deterministic network
inference only. It does not invoke search, a solver, or a game runner.

## Reproduce

```sh
python -m ml.alphazero_lite.seed621_oracle_cohort_audit
python -m ml.alphazero_lite.verify_seed621_oracle_cohort_audit
python -m unittest ml.alphazero_lite.test_seed621_oracle_cohort_audit
```

The published source verifiers are also prerequisites:

```sh
python -m ml.alphazero_lite.verify_seed557_retrospective
python -m ml.alphazero_lite.verify_seed559_fixed_position_feasibility \
  --data docs/data/seed559-fixed-position-feasibility/run-0003
```

`eligible-state-inventory.jsonl` contains all canonical unique states, complete
exact legal-action margins, both network prior comparisons, provenance cluster
identities, and state-level exclusion/contamination labels. `report.json` binds
source and membership hashes, totals, prior divergence summaries, and the
inventory hash.

The independent verifier does not import the audit producer. It authenticates
the frozen source, membership, and runtime identities, reconstructs cohort to
oracle joins, independently evaluates legal-masked B512/S615 priors, rebuilds
every inventory row, and compares row evidence plus all aggregate counts and
metrics. Its negative tests cover changed oracle margins, altered overlap
classification, changed prior class despite recomputed publication counts/hash,
and missing or substituted states.

## Results and decision

| Measure | Result |
| --- | ---: |
| Authenticated source records | 768 |
| Unique #557/#559 states | 768 |
| Cross-cohort duplicates | 0 |
| States with complete legal-action oracle coverage | 768 |
| Excluded for exact training/holdout membership overlap | 4 |
| Exact-membership nonmatches | 764 (contamination remains unresolved) |
| Raw-prior top-action agreement | 745 |
| Raw-prior top-action disagreement | 19 |

The exact-state inventory identifies 2 S615 training-state matches and 2
protected-holdout matches; there are no exact B512 training-state matches.
Direct membership nonmatches do **not** establish absence of contamination:
the historical evidence does not independently prove that all training input
representations, augmented/derived samples, and source trajectories are covered
by an identity-complete exclusion contract for this new cohort. Therefore all
764 remaining states are marked unresolved for contamination and are not
certified eligible. Priors were measured descriptively on these 764 nonmatching
states; the 745/19 class counts are feasibility counts, not a cleared selection
pool. In particular, the 19 observed disagreements do not satisfy the
acceptance gate until contamination is resolved.

Prior metrics are deterministic, legal-masked network priors. By historical
cohort, after excluding exact membership matches, #557 has 251 agreements and
4 disagreements (mean TV 0.021476; mean JS 0.000952 bits, n=255), while #559
has 494 agreements and 15 disagreements (mean TV 0.021571; mean JS 0.000996
bits, n=509). The #557/#559 cohorts are historically clustered within #556
games. Preserve source game/opening identities and use provenance-cluster
bootstrap intervals; do not treat 764 states as independent observations.
This diagnostic cohort is not representative of the full playing-strength
distribution.

## Authentication and orientation

The #557 source verifier reconstructed its 256 roots from receipt-bound #556
game batches and authenticated full legal-action oracle coverage. The #559
source verifier reconstructed its 512 roots and checked complete legal-action
oracle coverage. This audit independently rechecks canonical state hashes,
legal action sets, side-to-move state orientation, unique oracle joins, and
complete action-margin maps. Exact margins are retained in the native published
root perspective (side to move); no margin is converted or inferred from
stone-count eligibility. The report includes source file hashes and both
checkpoint identities. Training membership hashes are bound to the registered
Seed 599 B512 membership, Seed 615 candidate membership, and protected holdout
membership.

The standard verifier commands for #557 and #559 passed in this workspace.
The Seed 615 opening-diversity verifier could not be rerun here because its
training-source authentication imports PyTorch, which is absent from the local
Python environment. Its published membership and evaluation-freeze hashes
were checked by this audit. Re-run its independent verifier in its pinned
environment as part of external review.

## Search routing compatibility

`arena.ArtifactEvaluator.evaluate(game)` returns `(legal_masked_priors, value)`;
`ComposedArtifactEvaluator` independently selects policy and value source and
returns the selected pair. For arm C, `ComposedArtifactEvaluator(S615, B512,
policy_source="current", value_source="candidate")` provides S615's policy and
B512's value on each nonterminal expansion. The S615 value must be ignored.
The current evaluator computes both outputs from each source before selecting,
so the independent verifier must test the routed scalar value against B512
alone, deliberately give S615 a conflicting synthetic value, and reject any
trace or output reflecting S615 value. Also assert B512-prior/B512-value,
S615-prior/S615-value, legal masks and state orientation at root and non-root
expansions, terminal leaves bypass evaluator calls, and every expansion value
matches the specified source. No search was run for this audit.

The existing PUCT interface supports fresh fixed-root searches, 128 simulations,
one worker, no root noise, and deterministic expansion/action tie behavior;
prefer that implementation and freeze its runtime/source hashes. Before
registration, independently verify policy/value routing at every network
expansion, equality of arm C's value to B512 under all exercised states,
absence of S615-value leakage, fresh-tree isolation, exact search budget,
visit sums, selected maximum-visit action, and fixed tie-breaking. Report the
policy intervention as switching the S615-sourced prior function, including
its shared-trunk representation, not as an isolated policy-head effect.

## Selection and registration requirements

First resolve contamination with authenticated full training membership and
source-provenance evidence, including float32/production identity and all
derived/augmented training-state routes, or register a defensible exclusion
rule. Do not recover sample size by relaxing state, oracle, or contamination
rules. Then rerun prior classification over only the cleared exact-covered
inventory. If feasible, select up to 128 states, at least 32 from each prior
agreement class where available (hard minimum 64 total and 16 per class), by a
predeclared hash rank within class, capped for source-cluster dominance. Freeze
the entire eligible inventory, model/source hashes, class labels, selection
seed, selected IDs, oracle joins, cluster identities, and any sampling weights
before search output is observed. Estimate uncertainty by resampling source
opening/game clusters, not states, actions, or simulations.

**Recommendation: `revise`.** The oracle-coverage and raw-prior count gates
appear numerically feasible, and the PUCT interface provides a credible fixed-
value routing design. However, training contamination has not been excluded
for the proposed states. This is a registration blocker. No execution freeze
was created and no search arm was authorized or run.
