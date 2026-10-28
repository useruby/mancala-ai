# Seed542 expanded self-play continuation and preparation

This document records the completed continuation amendment to merged PR #542.
The original files `manifest.md`, `continuation-freeze.json`,
`execution-interruption.json`, `execution-inventory.jsonl`, `accounting.json`,
`partial-status.json`, `shard-542000.jsonl`, and the seed542001 staging file
remain unchanged. The amendment and its authenticated local runtime source are
recorded in `continuation-amendment.json`.
Its `generation_status` records the pre-launch state at amendment freeze; the
actual completed service result is recorded separately in `execution-service.json`.

## Generation and original-attempt accounting

The durable user service `seed542-amended-generation.service` ran the 62
previously unattempted original shards, seeds 542002–542063, and the separately
registered 13-game seed542064 supplement. All 63 invocations completed with 64
complete games per original shard and 13 complete games in the supplement. No
games were capped or incomplete in those new outputs. The original interrupted
seed542001 was skipped, not retried or overwritten.
The service invocation, constrained resources, exit status, and aggregate
completion counts are recorded in `execution-service.json`.

The original attempt remains accounted as 4,096 requested slots, 115 completed
games, zero capped, and 3,981 incomplete or not started: one complete shard,
one interrupted shard, and 62 unattempted shards. Its 4,096 slots plus the 13
supplement slots give 4,109 planned slots. The original plan now has 4,083
complete games (47 full original training shards, seed542001's 51 recovered
complete games, and 16 development shards), one incomplete game, and 12
unstarted games. Seed542001 is still recorded as an interrupted shard, not a
completed shard.

## Usable corpus and provenance

`recovered-seed542001-complete.jsonl` contains exactly the 1,768 original rows
from its 51 complete games. `recovered-seed542001-provenance.jsonl` binds each
derivative row to its original staging path, line number, source SHA-256, and
line SHA-256. Values and metadata are preserved byte-for-byte. The incomplete
game and unstarted games are absent from the derivative.

The explicit `source-manifest.json` records each nonuniform source file and its
hash: 47 full original training files × 64 games, the 51-game recovered
derivative, the 13-game seed542064 supplement, and 16 development files × 64
games. The resulting usable corpus contains 3,072 training games and 1,024
development games. The original accounting remains in `accounting.json`;
amended reconciled accounting is in `amended-accounting.json`.

All 4,096 usable games were replayed. The production loader accepted every row
with multiplicity preserved and rejected none. Training has 109,504 rows and
development has 36,844 rows. Both reports reconcile canonical and production
float32 duplicate counts, policy/value target modes, active-stone/teacher
cohorts, entropy, and one-hot rates.

## Common development cohort

`development-membership.json` records row-level source identity and hashes,
both state identities, inclusion, and exclusion reasons. Membership was
compared only against the original seed504 training data and the expanded
3,072-game training population. Of the 36,844 development-source rows, 21,485
were excluded and 15,359 retained. The retained cohort has zero overlap under
both canonical-state and production-float32-input identity.

## Frozen matched training protocol (not executed)

`training-protocol.json` freezes, but does not execute, arms C and D. C uses the
original seed504 13,724-row training population and its frozen permutations; D
uses the expanded 109,504-row training population and deterministic
without-replacement permutation seed542. Each arm has 108 updates and 54,896
row exposures using seed504's exact sequence, `[512] * 26 + [412]` repeated four
times. D covers 54,896 unique rows with maximum exposure one; C preserves its
four-pass seed504 exposure schedule. Schedule hashes and row identities are
published in the protocol file.

Initializer, architecture, joint policy/value objective, optimizer,
coefficients, clipping, and final-update-only candidate endpoint match the
registered comparison. Candidate training, evaluation, promotion, and
deployment were not performed.

## Runtime and verification

The recorded interpreter is the repository-local
`.tmp/seed542-recorded-runtime/bin/python` with Python 3.14.7, NumPy 2.5.3, and
PyTorch 2.14.0+cu130. The tablebase was reused from
`/home/alex/Mancala/ai/model-artifact/runtime/kalah_v1_21.kvtb`; its SHA-256 and
709,635,072-byte size matched the registered identity. It was not downloaded
through Git LFS.

`ml/alphazero_lite/verify_seed542_amended_corpus.py` independently validates
the amendment, inventory, source manifest, recovered provenance, games,
production loader, development membership, and schedules without writing to
evidence. `amended-verification.json` records the actual verifier hashes,
read-only byte/mtime checks, focused and historical test results, and quality
check outcomes.
