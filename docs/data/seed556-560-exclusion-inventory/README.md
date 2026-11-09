# #556–#560 exclusion inventory

This deterministic archive replay is **verified for #556–#560 only**. It is
not evidence of disjointness from the earlier training/development populations,
does not resolve the seed461 continuation overlap, does not reopen the #483
confirmation gate, and does not authorize real suite generation.

The committed source manifest binds #561's merge commit, the merged #556 and
#560 publications, receipt-bound #556 inputs, and exact #557–#560 cohort and
#560 freeze/continuation hashes. Every cohort row must contain a replay-checked
state before it is counted. `identity-inventory.json` preserves both identity indexes
and a replay-state provenance record for each identity in JSONL form (one
source/scheme/identity record per line). `count-report.json`
records counts and overlap independently for canonical-state SHA-256 and
little-endian `kalah_v3` float32 SHA-256.

Run from repository root:

```sh
PYTHONPATH=. python -m ml.alphazero_lite.seed556_560_exclusion_inventory
PYTHONPATH=. python -m unittest ml.alphazero_lite.test_seed556_560_exclusion_inventory -v
```

The verifier is read-only. `--write` is only for deterministic artifact
reconstruction during publication. It needs rules and the committed archives;
no model asset, search, game execution, or training is used.

The 16,107 #560 action count counts arm records (identical-action aliases are
repeated); replaying each of the 313 physical continuations once yields 9,878
actions. Both counts, and the 9,849 distinct identities per scheme, are
preserved in the report.
