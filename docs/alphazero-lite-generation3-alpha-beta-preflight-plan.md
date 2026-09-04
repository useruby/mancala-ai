# Generation-3 Policy-Free Alpha-Beta Preflight Plan

## Status

Preregistered before execution. This is a diagnostic-only search-interface
experiment. It creates no training data, replay, arena result, artifact, or
evaluation suite.

## Hypothesis

At the same 384 nonterminal leaf-evaluator-call budget, deterministic
iterative-deepening alpha-beta using a fixed-root-perspective scalar evaluator
makes lower-regret root decisions than ordinary deterministic PUCT using the
frozen production artifact. The primary lane uses the artifact value head; the
secondary mechanism-control lane uses the existing heuristic scalar value.

## Frozen Identity

- Artifact path: `model-artifact/current` (read-only)
- Version: `azlite-balanced-w8s4-policy-head-e1`
- `weights.json` SHA-256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- Metadata field: `artifacts.weights_json_sha256`

The runner verifies the file hash and metadata before execution and the file
hash again after execution. It runs `validate_a16_lineage_closeout.py` before
and after execution.

## Corpus

- 64 fresh nonterminal states, reachable only through legal prefixes from the
  standard 4x6 start
- Corpus seed: `270001`
- At least two legal moves; canonical-state deduplicated
- Prefixes are enumerated without model selection, bounded deterministically,
  then sampled round-robin by phase, player, legal-action count, capture
  availability, and extra-turn availability before canonical-hash sorting.
- Every row is `diagnostic_only: true` and `not_training_eligible: true`.

## Search And Reference Budgets

- Alpha-beta leaf-evaluator budget: 384 nonterminal calls per root
- Alpha-beta defensive node cap: 100,000 nodes per root
- Ordinary deterministic PUCT neural-evaluation budget: 384 per root
- PUCT seeds: `270101`, `270102`, `270103`
- Forced-action ClassicMCTS continuation: 2,400 simulations per legal root
  action; no tablebase and no exact solve options
- ClassicMCTS reference seed: SHA-256-derived integer of
  `270001:{state_hash}:{action}`

## Search Semantics

Alpha-beta uses iterative-deepening minimax. Values always represent the fixed
root player: nodes for that player maximize, all other nodes minimize, and an
extra turn does not invert value. Each legal action is one ply; there are no
extensions. Terminal root-player utility is +1, 0, or -1. The selected result
is exclusively the last fully completed iteration; incomplete deeper work is
discarded for selection. Failure to complete depth one is explicit.

The primary evaluator calls `ArtifactEvaluator.evaluate(game)`, discards its
policy entirely, and converts the returned current-player value by player
identity. The control evaluator likewise uses only `HeuristicEvaluator`'s
scalar value. The sole move ordering is terminal, extra turn, decreasing
capture count, decreasing immediate store delta, and increasing relative move
index. A sufficient-depth transposition best move may precede this ordering.

## Metrics And Analysis

For each lane and PUCT pairing: selected reference value, regret, exact-best
and top-two agreement, catastrophic miss (regret >= 0.25), legal selection,
runtime, and search telemetry. Results are aggregated and sliced by phase,
player, legal-action count, capture availability, and extra-turn availability.

The paired hierarchical bootstrap samples states then one PUCT seed within
each sampled state. It uses 10,000 samples and seed `270001` for
`alpha_beta_regret - ordinary_puct_regret`; negative is better. At least the
first 12 canonically sorted corpus states are rerun, and selection plus all
non-runtime telemetry must be byte-equivalent.

## Classification

Each lane is `qualified` only if all invariants pass, PUCT mean regret is
positive, alpha-beta mean regret is at most 75% of PUCT mean regret, the paired
95% upper bound is below zero, exact-best agreement does not decrease,
catastrophic-miss rate does not increase, and no preregistered slice with at
least eight paired rows increases catastrophic misses by more than 0.02.

Otherwise: `regresses_subsets` for the failed slice condition,
`no_search_gain` for an invariant-preserving nonqualification,
`budget_contract_failed` for budget failure, or `invariant_failure`.
Top-level classification prioritizes the artifact-value qualified result, then
the heuristic mechanism result, subset regression, no gain, and failures.

## Guardrails And Stop Conditions

No training, fine-tuning, self-play, replay, arena, promotion, export, artifact
mutation, suite import or consumption, suite-registry change, closeout-ledger
change, or qualified-candidate/status change is permitted. PUCT and
ClassicMCTS behavior remains unmodified. No policy transform, learned ordering,
Q-head, threshold adjustment, corpus adjustment, evaluator adjustment, or
post-result budget adjustment is allowed. Existing Gumbel, tablebase, A16,
replay, target-mixing, and residual-v4 branches remain closed.

Stop after this fixed run. If an implementation correction is necessary, record
it in the result and rerun the complete corpus and analysis, never selected
rows. A qualified result authorizes only a separately reviewed next-stage
proposal; it never authorizes training, data generation, arena evaluation,
production integration, or lineage-ledger changes.
