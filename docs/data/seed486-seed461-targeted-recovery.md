# Seed486 targeted seed461 execution-evidence recovery

**Bounded result: incomplete; generation remains blocked.** The machine-readable
ledger is `seed486-seed461-recovery-ledger.json`, produced and checked by
`ml.alphazero_lite.seed486_recovery`. It references the 58-stream worklist from
seed485 and preserves every prior file and the seed483 confirmation rules.

## Recovered evidence

The original seed461 run directories are present under the listed `.tmp/seed461-*`
paths in this checkout. All 58 receipt-bound game logs have a matching local
copy and matching archived member. The ledger includes exact local paths and
SHA-256 hashes, receipt/binding links, available report hashes, registrations,
and matched suite links. Logs and reports are linked only through their matching
game-log receipt; nearby artifacts are not used to fill missing fields.

The first contradictory cosine-LR stream, `order_38611_A`, has a local report
and game log matching the receipt. The runner source describes its invocation,
and the arena report records configuration and suite digest, but neither binds
the source bytes or working Git revision to the execution. The trajectory rows
contain action strings, not explicit state traces. No execution command log,
source snapshot manifest linked to a game receipt, or resume/chunk mapping was
found in the enumerated locations. Existing game/opening indices and seat fields
are retained as recorded; no offsets or conventions are inferred.

Consequently no writer/rules replay contract is established, no trajectory is
replayed, and no continuation state identities are reconstructed or appended to
the exclusion union. This is a bounded “not found in these locations” result,
not a claim that the missing artifacts do not exist elsewhere. Fixed-E4 streams
retain the registration/suite gaps recorded by seed485.

## Exact artifact request

Provide for each game-log receipt: (1) execution-time `arena.py` and Kalah rules
source bytes or SHA-256 hashes, linked to the run by an execution manifest,
command/runtime receipt, or equivalent source-bound record; (2) original command
logs and resume/chunk mapping, including suite association, seat initialization,
and game ranges; and (3) explicit state traces tied to the same game-log receipt
if trajectory-independent identity reconstruction is expected. For fixed-E4
streams additionally provide the missing registration and opening-suite receipt
and bytes. Nearby Git revisions or source matches without a run-bound link are
not sufficient.

## Verification

```bash
python -m ml.alphazero_lite.seed486_recovery --root . --verify docs/data/seed486-seed461-recovery-ledger.json
```

The report leaves generation unauthorized and asserts no export, promotion, or
strength result.
