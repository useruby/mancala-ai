# Seed549 matched self-play corpus

## Scope and interpretation

This is an exploratory generation and data-preparation comparison. It does not
establish improved playing strength or a preferred evaluator. No training,
arena evaluation, promotion, or deployment was performed. The known seed461
historical overlap provenance remains unresolved; recovering it was outside this
task. Seed549 development rows also overlap the earlier #548 pilot, as reported
by exact canonical and production-float32 identities in `results.json`.

## Frozen design

`registration.json` and `execution-freeze.json` bind the sources, dependencies,
artifacts, commands, settings, seeds, resource limits, and partition rules.
Control is the complete frozen initializer. Treatment is frozen serving-P E4
policy and its own trunk plus initializer value and its own trunk. The exact
component, sidecar, initializer, native-probe, and tablebase hashes are in the
registration. Execution used Python 3.14.7, NumPy 2.5.3, and PyTorch 2.14.0+cu130.

Each arm has eight separate 64-game shards with base seeds 549000–549007. The
production worker's per-game RNG derivation is
`(shard_base_seed * 1_000_003) + game_index`. Shards 0–5 were designated training
before generation (384 games / arm); shards 6–7 were reserved for development
(128 games / arm). No pilot games were imported.

Both arms used the pilot settings unchanged: 128 PUCT simulations, standard
4×6 start, `c_puct=1.25`, temperature 1.0 through ply 9 then 0.1, Dirichlet
alpha/epsilon 0.3/0.25 while action noise is enabled, no progress schedule,
denoised default visit-count policy targets, selected-one-hot exact-root targets,
default terminal value targets, exact-root threshold 16, exact leaf solving
disabled, one worker, and 200-ply cap. The full command and remaining disabled
search options are frozen in `registration.json` and recorded per attempt.

## Generation outcomes

All 1,024 games completed without caps or incomplete trajectories. The
independent replay verifier accepted every game and the production loader
accepted all 36,984 rows.

| Measure | Control | Treatment |
| --- | ---: | ---: |
| Games / accepted rows | 512 / 18,474 | 512 / 18,510 |
| Mean game length (min–max) | 36.08 (13–60) | 36.15 (12–63) |
| PUCT / exact-root rows | 12,043 / 6,431 | 11,916 / 6,594 |
| Elapsed generation time | 3,535.92 s | 5,841.29 s |
| Throughput | 0.1448 games/s | 0.0877 games/s |
| Peak sampled process-tree RSS (4 GiB limit) | 2.326 GiB | 2.327 GiB |
| Canonical unique / float32 unique | 12,739 / 12,739 | 9,122 / 9,122 |
| Unique >32-stone identities | 2,408 | 1,578 |
| >32 unique identities / game | 4.70 | 3.08 |
| >32 unique identities / generation second | 0.681 | 0.270 |

These are generation characteristics only. In particular, higher concentration,
lower diversity, and lower target entropy are not evidence of better training
quality. Full per-game lengths and >32-stone identities, stone-bucket target
diagnostics, hashes, and throughput/memory receipts are in
`results.json` and `attempts.jsonl`.

## Durable recovery record

The first control shard-0 invocation exited 2 because the experiment adapter
omitted the production worker's required `--out` argument. Its exact command,
exit status, zero-byte output, and log hash are preserved in `attempts.jsonl`.
The narrow correction is bound by `execution-correction.json`; the same planned
shard and seed were retried as a distinct attempt because the failed invocation
produced no rows. The corrected output and its own command, timing, memory,
log/output hashes, and publication receipt are separately recorded. No partial
or outcome-bearing game attempt was overwritten or replaced.

## Prepared populations and schedules

Each arm has a source manifest and separate training/development occurrence
identity files. Occurrences retain source shard/game/move/row identity, canonical
state and exact production-float32 identity, policy target, value target, and
stone count. The shared development population contains both reserved partitions.
It excludes an occurrence if its canonical state **or** exact float32 input
overlaps either arm's training rows. Every occurrence and exclusion reason is
preserved in `shared-development-membership.jsonl`; repeated inputs retain their
occurrence-level targets, including disagreements.

`results.json` gives future development cohort sizes by source evaluator and
stone bucket. Known overlap with the pilot and with the registered seed511 v4 /
seed512 evaluation opening roots is disclosed there. Those are descriptive
development data and reused-suite identities, not a fresh historical holdout.

Both deterministic future schedules have 108 updates and 54,896 exposures, with
the established batch sizes `[512] * 26 + [412]` repeated four times. They draw
seeded shuffled passes without replacement within each pass, retain duplicate
row occurrences, and bind pass boundaries and exposure indices. No optimizer
update was run.

`post-generation-receipt.json` binds every derivative product separately from
the prospective generation registration. Append-only verifier corrections
preserve the original receipt and raw data bytes.

## Independent verification

Run the single read-only verifier with the recorded interpreter:

```bash
/home/alex/Mancala/ai/.tmp/seed542-continuation-prep/.tmp/seed542-recorded-runtime/bin/python \
  -m ml.alphazero_lite.seed549_derivatives verify
```

It authenticates the generation binding chain and frozen artifacts, validates
attempt commands and output/log receipts, independently replays every shard,
checks standard starts, trajectories, action and teacher search profiles,
terminal targets, production-loader acceptance, game/row accounting, then
rebuilds and reconciles source manifests, development membership, results, and
schedules. It verifies evidence bytes and nanosecond mtimes remain unchanged.

## Sources and checks

- `ml/alphazero_lite/seed549_corpus.py` — frozen orchestration, adapter dispatch,
  recovery controls, and source registration.
- `ml/alphazero_lite/seed549_derivatives.py` — occurrence identity builder and
  read-only independent verifier.
- `ml/alphazero_lite/test_seed549_corpus.py` and
  `ml/alphazero_lite/test_seed549_derivatives.py` — self-contained orchestration,
  recovery, exclusion, schedule, and accounting tests.

No model advancement is authorized by this corpus publication. A later controlled
training comparison is a separate task.
