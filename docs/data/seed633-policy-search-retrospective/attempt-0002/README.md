# Attempt 0002 — full-trajectory evidence coverage

**Decision: stop at the coverage gate. Final classification:
`historical_evidence_insufficient`.** The authenticated full Seed 615 trajectory
replay found many same-root raw-policy/exact-margin occurrences, but all 155
one-to-one paired exact-margin occurrences with both models' raw policies were
selected through the exact-root solver. There were zero identical roots with
both an authenticated MCTS-selected B512 action and an authenticated
MCTS-selected S615 action plus exact margins. No policy/search regret,
optimal-action agreement, outcome association, or mechanism effect was
computed.

## Authenticated inventory and replay

Attempt 0002 is frozen at `freeze.json`; it supersedes only the conclusions in
attempt 0001 and leaves all attempt-0001 files unchanged (`supersedes.json`).
Inputs include all 4,096 registered Seed 615 games from merged PR #618, the
Seed 633 attempt-0005 raw per-state policy metrics from PR #632, and the
attempt-0003 exact-oracle evidence from PR #633. The manifest binds the PR merge
commits, checkpoint IDs, execution freeze, 4,096 raw game-file hashes, result
file hashes, policy metrics, exact margins, and the Kalah/arena/exact-root source
identities.

Each trajectory was replayed after its authenticated 12-move opening prefix
with an independent ordinary Kalah transition implementation. The replay
checks every action's legality and mover, state perspective, opening identity,
terminal/capped result, winner, margin, and result-file digest chain. The
reconstructed first-divergence states/actions/seeds/outcomes agree with all 2,048
published PR #619 paired trajectory rows. Every candidate decision occurrence
is recorded in `trajectory-decision-coverage.jsonl.gz`, including the canonical
state identity, candidate checkpoint, seat, action, opening/game identity,
seed, ply, raw-policy/exact-margin presence, and decision-source class.

Exact canonical identities are SHA-256 over sorted compact ASCII JSON of the
replayed player-relative state, including the player to move. Each occurrence
joins directly to attempt-0005 raw policies and attempt-0003 exact margins by
exact identity. Membership in the derived PR #619 policy-prior cohort is not a
join prerequisite. Repeated root occurrences are retained. Paired occurrences
are one-to-one within opening, seat, identity, and occurrence order; unmatched
duplicates are left unpaired rather than expanded as Cartesian products.

## Trajectory and decision-source coverage

| Coverage item | Observed |
| --- | ---: |
| Registered Seed 615 game records | 4,096 |
| Paired candidate trajectories | 2,048 |
| Identical trajectories | 1,333 |
| Divergent trajectories | 715 |
| Candidate decision occurrences | 60,618 (B512 30,384; S615 30,234) |
| Opponent decisions, not attributed to candidate | 60,118 |
| Candidate MCTS-selected actions | 32,454 (B512 16,239; S615 16,215) |
| Candidate exact-root-solver-selected actions | 28,164 (B512 14,145; S615 14,019) |
| Candidate actions with unresolved source | 0 |
| Other registered candidate decision mechanisms | 0 |
| Exact-root-eligible decisions, both players | 55,889 |
| Recorded exact-root handoff calls, both players | 55,889 |

The frozen arena source sends roots above 16 active pit stones down the PUCT
path. For roots at or below 16, eligibility alone was not taken as proof of
solver execution. Instead, every game’s `exact_root_handoff_count` and
`first_exact_root_handoff_ply` were reconciled with independently replayed
eligible plies. Counts and first handoff positions agreed throughout the
archive, establishing the 28,164 candidate exact-root selected actions and
27,725 opponent exact-root selected actions. No exact root was re-executed or
probed.

## Direct Seed 633 intersections and adequacy

The 60,618 candidate decision occurrences include 437 occurrences (133 unique
roots) with both attempt-0005 raw policies and exact margins. Exact margins are
all in phase `at_most_16`.

| Paired comparison cohort | Matched occurrences | Unique roots | Opening groups |
| --- | ---: | ---: | ---: |
| Same opening, seat, exact state, one-to-one occurrence order | 23,641 | 23,251 | — |
| Both raw policies | 155 | 92 | — |
| Both raw policies and exact margins | 155 | 92 | 94 |
| Both authenticated MCTS actions | 0 | 0 | 0 |
| Both authenticated MCTS actions and exact margins | **0** | **0** | **0** |
| Both exact-root-solver actions and exact margins | 155 | 92 | 94 |

Exact-margin paired-occurrence coverage by Seed 633 membership class is:

| Membership class | Occurrences | Phase |
| --- | ---: | --- |
| `B512_only` | 43 | `at_most_16` |
| `S615_only` | 31 | `at_most_16` |
| `neither_training_membership` | 60 | `at_most_16` |
| `shared_training` | 21 | `at_most_16` |

The unchanged feasibility requirements are at least 100 unique paired roots
with both raw policies and both MCTS-selected actions, at least 50 such roots
with exact margins, and at least 20 opening groups; membership and phase minima
are also retained. The MCTS-action cohort has 0 roots, so the gate fails. The
exact-margin intersections do not substitute for MCTS coverage: they are
exact-root solver decisions.

## Verification and stop

`verify_policy_search_historical_coverage_attempt_0002.py` independently
reconstructs the full replay, exact state identities, decision-source classes,
occurrence pairing, and coverage report from the fixed source hashes; it imports
no producer aggregation. Its `--verify-existing` mode recomputes both the
coverage JSON and compressed per-decision artifact. Negative tests cover
duplicate source identities, incomplete intersections, one-to-one duplicate
pairing, opponent attribution, player-perspective action errors, exact-root
eligibility without observed execution, and exact-solver actions being
misclassified as MCTS.

Re-run the published artifacts and targeted regressions with:

```sh
python -m ml.alphazero_lite.verify_policy_search_historical_coverage_attempt_0002 --verify-existing
python -m unittest ml.alphazero_lite.test_policy_search_historical_coverage_attempt_0002 ml.alphazero_lite.test_verify_policy_search_historical_coverage ml.alphazero_lite.test_seed633_exact_oracle_decision_diagnostic ml.alphazero_lite.test_seed615_policy_divergence_diagnostic
```

The retrospective stops at coverage. No new model inference, MCTS execution,
tablebase probe, state solving, game generation, or training was performed.
Seed-specific paired outcome differences are not used as causal evidence. The
source-opening to Seed 624 opening crosswalk remains unresolved.

**Recommendation:** use the existing Seed 623 prospective benchmark protocol
for a policy-to-MCTS decision-quality comparison, under its existing
registration and execution safeguards. No additional experiment is initiated.
