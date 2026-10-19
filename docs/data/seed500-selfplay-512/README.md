# Seed500 512-game self-play corpus

The archived data consists of eight independent production-generator outputs:
`shard-503000.jsonl` through `shard-503007.jsonl`. The pre-generation recipe,
model/runtime identities, dependency versions, exact commands, and partition are
in [manifest.md](manifest.md). Each JSONL is archived byte-for-byte and its
SHA-256 and byte count are recorded in [accounting.json](accounting.json).
The original pilot file remains separate and unchanged.

## Accounting

- **512 complete games**, 0 capped, 0 incomplete; **18,281 rows**. All 18,281
  rows were accepted by `train.load_jsonl` and replayed successfully.
- Across all shards: 11,896 unique canonical states / 11,896 unique encoded
  inputs; 6,385 duplicate rows by each identity. Both counts agree for this
  corpus.
- PUCT, >32 stones: 6,315 rows, 3,975 within-subgroup repeat rows, mean legal
  policy entropy 0.560731 nats, one-hot 11.31%.
- PUCT, 17–32 stones: 5,795 rows, 1,181 within-subgroup repeat rows, mean
  legal policy entropy 0.083767 nats, one-hot 35.25%.
- Exact-root, ≤16 stones: 6,171 rows, 1,229 within-subgroup repeat rows, zero
  entropy, one-hot 100%. No rows occurred in the other teacher/bucket cells.
- Unique encoded-input growth by shard: 1,897; 1,630; 1,450; 1,308; 1,599;
  1,397; 1,418; 1,197 new identities, reaching 11,896 cumulative identities.

### Training/development source identity overlap

The designated source partition is six training-source shards (13,724 rows)
and two development-source shards. The 4,557 original development rows
contained 3,654 canonical identities and 3,654 exact-input identities. After
excluding every development row whose canonical state **or** exact encoded
input occurs in any training-source shard, 2,671 rows remain, with 2,615 unique
canonical and 2,615 unique encoded-input identities. Thus 1,886 development
rows overlap training by at least one identity. Development-source means only
the prospective shard assignment; no historical-unseen or untouched-holdout
claim is made.

## Execution and limits

Recorded generator time was **270.70 seconds** total (individual shard durations
are in `accounting.json`). An initial launch attempt failed before the generator
was invoked because `/usr/bin/time` was unavailable; it created no outputs. The
eight authorized invocations were then executed once each using Bash timing.
No capped or incomplete games were replaced. `accounting.json` records exact
per-shard hashes and the combined content digest.

Focused replay/loader and Ruff checks passed; accounting reproduction was run in
a separate checkout. This is an integrity and usability record only. It does
not authorize training, candidate evaluation, promotion, or a fresh-confirmation
claim. The closed F advancement decision and all historical limitations remain
in force.

Reproduce the accounting from the repository root:

```bash
.venv-azlite/bin/python -m ml.alphazero_lite.validate_seed500_pilot \
  docs/data/seed500-selfplay-512/shard-503000.jsonl \
  docs/data/seed500-selfplay-512/shard-503001.jsonl \
  docs/data/seed500-selfplay-512/shard-503002.jsonl \
  docs/data/seed500-selfplay-512/shard-503003.jsonl \
  docs/data/seed500-selfplay-512/shard-503004.jsonl \
  docs/data/seed500-selfplay-512/shard-503005.jsonl \
  docs/data/seed500-selfplay-512/shard-503006.jsonl \
  docs/data/seed500-selfplay-512/shard-503007.jsonl
```
