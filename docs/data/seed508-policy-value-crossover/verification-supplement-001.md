# Seed508 read-only verification supplement 001

This supplement adds a command/runtime-contract audit and independent recomputation
to the existing publication. It does not replace or amend the registration,
pre-execution freeze, source snapshots, commands, reports, outcomes, results, or
publication receipt.

The initially regenerated `publication-receipt.json` differed from the exact
receipt committed in PR #508. Its bytes are preserved as
`publication-receipt-regenerated-001.json`; the original `publication-receipt.json`
has been restored byte-for-byte. New correction files are bound by a separate
`verification-supplement-receipt-001.json`.

## Execution timing and authorization

The recorded execution occurred during the latest **preparation-only milestone**.
The execution command and outcome-bearing records establish that execution did
occur; this retrospective fact is separate from authorization. In particular,
the pre-execution freeze records a freeze before execution, not an approved
execution authorization. No approval is inferred or relabeled here.

## Verification scope and findings

The supplement independently checks all 1,024 complete command records against
their argv and corresponding arena report: bound current initializer, arm-specific
challenger base and policy/value artifacts, budgets, parent opening seed, suite
digest, paths, search options, random-opening plies, move cap, report-side runtime
settings, exact-root identities/settings, and outcome totals. Composition fields
added by the runner are treated as declarations, not proof of arena loading.
Every recorded result was also replay-verified by the existing verifier.
Receipt-rebound semantic tests accept an unmodified copied publication and reject
changed opponent, component assignment, search and exact-root settings, and result
content. Run them with `python -m pytest
ml/alphazero_lite/test_seed508_verification_supplement.py -q`. The audit supports
an explicit `--root` and `--out` so it can be run from an unrelated working
directory against a physical publication copy.

P-minus-V and both descriptive standalone statistics are recomputed from the
replay-verified outcome ledger, independently of published result JSON. The
published primary estimate and intervals and standalone counts, scores, and
intervals agree. The analysis remains limited by reuse of seed488's previously
examined suite and unresolved seed461 continuation-population overlap.

This is read-only verification. No games, training, promotion, fresh confirmation,
or 1,200-simulation test were run.
