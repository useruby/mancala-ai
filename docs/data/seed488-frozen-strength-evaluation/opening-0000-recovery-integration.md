# Seed488 opening-zero recovery integration

The exact saved opening-zero pair was verified and imported once under the
matching `.venv-azlite` environment (Python 3.14.7, NumPy 2.5.2). No game was
rerun. The published recovery receipt fixes the saved game-log, report,
single-entry batch, and parent-suite hashes to PR #493's preserved evidence.
The continuation binding is prospective; the original registration's
trajectory-contract claim is left intact as historical evidence.

The recovery receipt records the original registration, execution-freeze, and
registered-source identities, corrected replay/rules source identities, and
both fully replayed outcomes: 33 plies, margin 0, draw for challenger seats 0
and 1. It explicitly states that opening prefixes are player-relative while
arena trajectory pit indices are absolute. Historical execution provenance is
retained on both imported ledger rows.

The one-time command sequence was:

```bash
. .venv-azlite/bin/activate
python -m ml.alphazero_lite.seed488_replay_recovery --publish-receipt
python -m ml.alphazero_lite.seed488_replay_recovery --publish-continuation-binding
python -m ml.alphazero_lite.seed488_replay_recovery --import-pair
```

The importer rejects altered saved artifacts, duplicate imports, conflicting
outcomes, foreign pending commits, and unrelated uncommitted batches. Read-only
verification reports `opening_zero_recovery_verified_continuation_incomplete`.
The remaining-pair runner requires this validated receipt and imported pair;
committed-pair resume validation and the independent verifier replay absolute
trajectory indices. The opening-zero writer tests invoke the actual arena
writer with deterministic stub evaluators and no search.

The 512-opening suite, 1,024-game plan, models, seed derivation, 1,200-simulation
budget, paired bootstrap (20,000 replicates, seed 488), exploratory overlap
limitation, and no-promotion rule are unchanged. The remaining 511 pairs were
not launched; continuation awaits review.
