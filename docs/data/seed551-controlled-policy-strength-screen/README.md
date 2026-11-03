# Seed551 controlled policy-strength screen

This is an exploratory comparison of the update-108 C and T policies. Each
policy keeps its own trunk; both sides use the same frozen initializer value
network and its own trunk through `ComposedArtifactEvaluator`. The comparison
measures the policy-strength difference associated with changing the source
self-play evaluator. It does not isolate policy-target quality, establish
superiority to serving P, or authorize promotion.

The preregistered suite contains exactly 1,024 indexed opening pairs and 2,048
games. T and C swap seats in every pair. It reuses seed511's indexed suite; the
seed461 continuation-overlap limitation remains unresolved. Canonical-state
and production-float32 overlap counts against both seed549 training populations
and the shared development population are in `overlap-summary.json` and
`overlap-per-opening.jsonl`; no suite rows were removed or replaced.

Scoring, opening-cluster bootstrap, quantiles, and decision thresholds are
frozen in `registration.json` before games. Games capped at 200 plies are
explicitly marked nonterminal and reported separately from terminal outcomes;
they use the arena's recorded current store-margin result for points.

The initial read-only input verification attempts and their corrective reruns
are preserved in `input-verification.json`. See `execution-freeze.json`,
`command-preflight.jsonl`, `commands.jsonl`, per-opening batch files, and
recovery/attempt ledgers for the complete execution trace. Independent
read-only verification reconstructs trajectory legality, winner perspective,
seat pairing, runtime contract, ledger, score, bootstrap, and decision.
The pre-game preparation attempt that surfaced the nested corpus receipt check
is preserved in `docs/data/seed551-controlled-policy-strength-screen-preparation-attempt-0001/`; it contains no arena games.

This screen is not a fresh holdout and cannot authorize deployment or
promotion. Non-advancement does not prove the source evaluator is universally
inferior.
