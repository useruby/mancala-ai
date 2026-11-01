# Append-only verifier supplement

This supplement addresses the three verification findings from PR #548 without
changing the frozen generation registration, raw games, attempt ledger, logs, or
published `results.json`. The machine-readable binding and source hash chain are
in `verification-supplement.json`.

## Read-only report reconciliation

Run the combined verifier with the registered interpreter and project root:

```bash
/home/alex/Mancala/ai/.tmp/seed542-continuation-prep/.tmp/seed542-recorded-runtime/bin/python -m ml.alphazero_lite.selfplay_evaluator_pilot verify-pilot
```

It authenticates attempt commands and output/log receipts, independently replays
both arm files, reconstructs the report including game counts, rows, lengths,
outcomes, bucket entropy/diversity/duplicate diagnostics, throughput, memory,
exact-root usage, and success status, then compares the full reconstructed
object with the existing `results.json`. Any field disagreement is rejected;
the command does not rewrite the report or evidence and checks hashes plus
nanosecond modification times around verification.

The combined verifier, independent shard replay, and attempt verifier also
passed in a physical copy while the original worktree path was inaccessible.
`physical-copy-verification-supplement.json` records those commands and the
evidence bytes/mtimes checked.

## Population and row accounting

The verifier rejects any game ID outside the registered range `0..63`. It
reconciles every source row as replayed, capped, or incomplete and independently
requires the sum to equal the JSONL row count; stone-bucket row totals must equal
the replayed-row count. The 64 expected game slots remain separately accounted
as completed, capped, or incomplete/unstarted.

## Full search-profile contract

For every completed row, both `search_profile` and `teacher_search_profile` are
compared in full to a reconstructed expected profile using the frozen
generation registration and authenticated serving `search_policy.json` sidecar.
The comparison includes simulations, `c_puct`, every search option, exact-root
enabled state and threshold, objective, tie rule, solver identity, exact-leaf
mode, native-probe identity, tablebase identity, and canonical profile hash.
Mutation tests alter objective, tie rule, and tablebase identity, recompute the
profile hashes, and confirm rejection for both action and teacher profiles.

These checks authenticate recorded telemetry and target-generation settings;
they do not independently prove policy-target optimality. No games were rerun.
