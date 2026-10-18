# Seed488 pre-game execution correction

This freeze supersedes the **execution attempt only** from PR #491. The
registration, receipt, openings, failed invocation log, first batch, endpoint
exports, and manifest from that attempt remain unchanged and are authenticated
by `failure-provenance.json`. There were no games, reports, outcomes, pending
commits, or results in the failed attempt.

## Corrected option boundary

Arena opening behavior is configured independently from per-side search
overrides. Each override JSON is the registered search-options object with only
`random_opening_plies` removed. Arena receives `--random-opening-plies 0`
explicitly. The accepted override field set in `arena.py` is unchanged.

`execution-freeze.json` binds the untouched original registration, preparation
receipt, 512-opening suite, failed attempt, runtime artifact manifest and both
artifact directories. It also binds the corrected runner, independent verifier,
and synthetic test source hashes. The corrected runner writes any future
per-opening batches below `resume-corrected/`, so the failed opening-0 batch
under `resume/` is never reused or rewritten. A completed-run receipt will be
published separately as `execution-receipt.json`.

The no-game integration test feeds the runner's exact constructed argument
vector to arena's real command-line parser, runtime-contract resolver, opening
suite preflight, and per-side override parsers. It proves that the historical
malformed options are rejected and that the corrected command reaches the
executor boundary, where only the search/game loop is intercepted. No games or
searches are run by this test.

## Review gate and unchanged protocol

Read-only verification reports `ready_for_review_not_executed`; the runner still
requires explicit `--execute`. Do not execute before this correction is
reviewed. Candidate/initializer endpoints, 512 openings, 1,024 paired games,
seed derivation, 1,200 simulations per move, analysis, bootstrap, and
no-promotion rule are unchanged. This remains exploratory, not fresh
confirmation. Non-overlap with the unresolved seed461 continuation population
cannot be certified because authenticated state identities for its 1,129,643
receipt-matched continuation actions are unavailable.
