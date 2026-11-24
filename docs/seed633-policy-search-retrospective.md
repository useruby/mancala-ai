# B512/S615 policy-to-search historical evidence feasibility

**Decision: stop at the historical-evidence gate. Final classification:
`historical_evidence_insufficient`.** This bounded retrospective found only two
identical canonical states that connect Seed 615 raw policies and recorded
candidate actions to Seed 633 authenticated exact legal-action margins. That is
far below the preregistered adequacy requirements. No policy-to-search regret
contrast, mechanism classification, or game-outcome association was computed.

## Scope and authentication

This assessment uses the merged PR #619 and PR #633 evidence, plus the
Seed 633 attempt-0005 source cohort from PR #632, authenticated by the fixed
hashes in the merged attempt-0003 freeze. The attempt-0003 exact-oracle verifier was run
against that source and independently reconstructed all 1,481 exact states,
action sets, legal masks, and published metrics. Seed 615 artifacts are bound
to the PR #619 diagnostic and evaluation-freeze identities. Fixed source
identities and checkpoint hashes are recorded in
`docs/data/seed633-policy-search-retrospective/evidence-manifest.json`.

The analysis freeze (`docs/data/seed633-policy-search-retrospective/freeze.json`)
sets exact-identity matching, the minimum coverage gate, and practical effect
criteria before the cross-source join. Existing published PR reports were
consulted to inventory their content; this was a retrospective feasibility
exercise, not a prospective benchmark or a pre-outcome registration.

The join uses exact canonical identity strings, including the player
perspective encoded by the identity, and no approximate float matching. The two
matched Seed 615 rows were checked for agreement between their canonical root
player and the recorded state `current_player`. For these rows PR #619 records
both actual candidate game actions, distinct candidate-specific seeds, the
opening index, and the paired outcomes. These are recorded MCTS/game actions,
not reconstructed from network priors. PR #633 exact-oracle rows themselves
have no observed search-selected actions.

No crosswalk between historical source-opening labels and Seed 624 opening
identifiers is assumed. The Seed 560 root-search bundle is excluded: its
execution freeze binds a different P-E4/initializer composition and search
budgets, not the B512 and S615 checkpoints.

## Frozen adequacy criteria

Proceed only with at least 100 unique identical roots that have both raw
policies and both recorded search actions, at least 50 such roots with
authenticated exact action margins, and at least 20 authenticated Seed 615
opening groups. At least two membership classes and one phase stratum must
also have nonzero exact-margin coverage. These minima require a meaningful
paired set across multiple opening clusters and enough exact-valued roots to
avoid assigning a mechanism from a handful of observations. They are coverage
requirements, not claims about statistical power.

If the gate had passed, the frozen practical-equivalence margin would be one
outcome point; the equalization, preservation, amplification, and mixed-effect
rules are recorded in the freeze. The one-point margin is a conservative
minimum interpretable change on the exact score-margin scale. These criteria
were not applied to decision-quality results because the coverage gate failed.

## Coverage

| Evidence component | Unique states / occurrences | Exact margins | Both raw policies | Both recorded search actions | Opening groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| PR #619 policy-prior evidence | 1,700 unique policy states; 2,048 first-divergence occurrences | — | 1,700 | 676 states have paired first-divergence actions in the published diagnostic | 1,024 opening groups in the paired evaluation |
| PR #633 attempt-0003 exact-oracle cohort | 1,481 unique states | 1,481 | not used as a search-action source | 0 observed | 182 authenticated source-opening clusters |
| Exact-identity intersection | **2 unique states** | **2** | **2** | **2** | **2 Seed 615 opening indices** |

The PR #619 source reports 4,096 games and 2,048 paired trajectory comparisons.
The 676 paired first-divergence policy states are the unique action-disagreement
states in its policy evidence; the full first-divergence artifact also records
paired game outcomes and trajectory identities. The PR #633 exact cohort's
1,481 action-margin rows are all in phase `at_most_16`.

The authenticated Seed 615 evaluation freeze specifies 128 simulations per
move, `c_puct=1.25`, deterministic temperature-zero root selection, no root
noise, one search worker, exact-root solving enabled through 16 stones, and no
exact leaf solving. The paired candidate game seeds are model-specific by the
registered seed derivation; the two exact-identity matches each contain both
recorded candidate seeds. The Seed 560 root-search freeze instead binds a
P-E4/initializer policy composition, fresh-tree 128/512 searches, and shared
same-root search seeds; it cannot provide search actions for B512/S615.

### Exact intersection coverage by membership and phase

| Membership class | Phase | Matched states with policies, exact margins, and recorded actions |
| --- | --- | ---: |
| `S615_only` | `at_most_16` | 1 |
| `neither_training_membership` | `at_most_16` | 1 |
| `B512_only` | — | 0 |
| `shared_training` | — | 0 |

There is no matched exact-margin coverage outside `at_most_16`. The full
Seed 633 exact cohort has 402 `B512_only`, 402 `S615_only`, 110 `shared_training`,
and 567 `neither_training_membership` states. These source cohort counts do not
constitute corresponding Seed 615 trajectory overlap.

| Gate component | Required | Observed | Result |
| --- | ---: | ---: | --- |
| Unique matched roots with both policies and actual search actions | 100 | 2 | Fail |
| Matched roots with exact margins | 50 | 2 | Fail |
| Authenticated Seed 615 opening groups | 20 | 2 | Fail |
| Membership classes with nonzero exact coverage | 2 | 2 | Pass |
| Phase strata with nonzero exact coverage | 1 | 1 | Pass |

Coverage values are independently rebuilt in
`docs/data/seed633-policy-search-retrospective/coverage.json` by
`ml/alphazero_lite/verify_policy_search_historical_coverage.py`. The verifier
checks fixed source hashes, exact identity intersections, recorded action
availability, and player perspective. It does not import either producer's
aggregation implementation. Regression tests reject substituted source bytes,
incorrect identity matching, player-perspective errors, altered exact margins,
and substituted recorded actions.

## Missing or incompatible evidence

- The exact intersection is two states, not enough for a stable paired
  policy-to-search quality comparison or source-opening-cluster intervals.
- Search actions are absent from the Seed 633 exact-oracle bundle; only
  candidate actions in authenticated Seed 615 game/first-divergence evidence
  can qualify as actual recorded decisions.
- The two action-bearing matched roots are selected first-divergence states,
  not a representative or independently sampled decision cohort.
- Seed 615 assigns different numeric seeds to B512 and S615 candidate games.
  Their action and outcome differences therefore cannot be attributed solely
  to the model intervention.
- The historical source-opening to Seed 624 opening crosswalk remains
  unresolved.

The two matched rows have paired outcomes in the historical diagnostic, but
this selected two-state subset is not analyzed for searched-regret/outcome
association. The broader existing 2,048 paired game comparisons remain useful
for the original PR #619 descriptive outcome question; they do not supply exact
action margins for the trajectory decisions needed here. No causal,
independent-population, or playing-strength conclusion follows.

## Stop decision

Because the frozen matched-coverage thresholds fail, stages 3–5 stop here. Raw
top-policy regret, search-selected-action regret, search correction, optimal
agreement transitions, bootstrap intervals, and mechanism classification were
not calculated. No inference, MCTS, evaluation game, solving, or tablebase
probe was run, and no checkpoints or search settings were changed.

**Recommendation:** return to the already-existing Seed 623 prospective
benchmark protocol for a decision-quality comparison, subject to its existing
registration and execution safeguards. Do not start another retrospective
diagnostic. No additional experiment is initiated by this report.

**Final classification: `historical_evidence_insufficient`.**
