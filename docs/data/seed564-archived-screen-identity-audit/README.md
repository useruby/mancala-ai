# Archived screen identity audit

## Seven-source rules-only replay

The seven completed sources are PRs #511, #512, #545, #547, #551, #552, and
#555. Each is pinned to its merged publication, final corrected receipt chain,
registered suite, outcomes ledger, all raw game batches, and recovered-pair
bindings in `seed564-seven-source-manifest.json`. PR #545 is pinned to merge commit
`56c5ae4e24ee73500f4f4ad027974cd186bba1ab`. Its final corrected completeness
verifier SHA-256 is
`7af938402a096fc9c81d747efc99f42a2d58abc8339b40e385710e0d3df43b6c`; the
published correction-results document is also hash-bound by
`seed545-source-manifest.json`. The #545 manifest is pinned in its public
verifier with SHA-256
`5664eb3ee58665de9370f105b2fd31d60448e5d0dd3174868ad7aead1f1a5c09` and binds
all 1,024 #545 raw game batches plus its registered opening suite, final
outcomes ledger, corrected publication evidence, and recovery receipts. The
seven-source manifest digest is pinned in
`ml/alphazero_lite/archived_screen_identity_seven.py`.

Run the read-only public entry point from the repository root:

```sh
python -m ml.alphazero_lite.archived_screen_identity_seven \
  --root . \
  --out docs/data/seed564-archived-screen-identity-audit
```

The default invocation verifies the already-published index/report bytes against
freshly recomputed output and rejects missing or corrupted artifacts without
writing them. `--write-indexes` is a separate explicit generation mode; it is
not the verification path.

Every source authenticated and replayed 1,024 opening pairs / 2,048 games. The
deterministic indexes contain 481,217 state occurrences across the seven
sources. Per-source canonical and production `kalah_v3` float32 distinct counts
match:

| PR | Distinct states | Overlap with #562+#563 | Additions beyond #562+#563 | Replayed state occurrences |
| --- | ---: | ---: | ---: | ---: |
| #511 | 56,329 | 26,837 | 29,492 | 67,052 |
| #512 | 60,399 | 9,464 | 50,935 | 72,024 |
| #545 | 50,076 | 30,031 | 20,045 | 68,166 |
| #547 | 53,161 | 28,445 | 24,716 | 68,287 |
| #551 | 55,015 | 21,805 | 33,210 | 68,631 |
| #552 | 51,158 | 32,126 | 19,032 | 68,100 |
| #555 | 49,646 | 30,294 | 19,352 | 68,957 |
| Seven-source union (deduplicated) | 207,422 | 52,166 | 155,256 | — |

The verified #562+#563 baseline was rebuilt from its published deterministic
identity indexes before comparison: 58,101 #562 identities plus the #563
training/development inventory produce a 147,146-identity union under each
scheme. All seven sources reuse the same 1,024 opening roots; root overlap is
reported separately from continuation overlap. The seven-source opening-root
union is 1,024 identities, with zero additions beyond the baseline. The
continuation union is 206,399 identities; 155,256 are additions beyond the
baseline. Pairwise root, continuation, and cross root-to-continuation overlaps
are published in `seven-source-count-report.json`.

Recovered openings 0/1 from #545, 0/1/2 from #547, and opening 1 from #551 are
accounted for once through their final recovery bindings and unique outcomes
ledger pairs. Failed or superseded prelaunch/preparation/verification attempts
are listed separately in the machine-readable report; they are not added to
committed game populations.

The run succeeded while the materialized tablebase was renamed inaccessible.
The verifier reads the already-recorded absolute moves and the pure Python
rules engine; it does not import Torch or NumPy and does not access model weights
or tablebase files. Evidence byte sizes, modification times, and SHA-256 values
are snapshotted before and after replay and must remain unchanged. A physical
copy test runs the public CLI from a disposable evidence-only checkout and an
unrelated working directory, with no materialized tablebase or original runtime
assets in the copy, and confirms the same evidence immutability.

The #545 and #547 published-verifier runs under `/home/alex/azlite-venv` remain
supporting cross-checks under a different runtime. Reproduction under the
recorded runtime is explicitly unverified and is not represented as a passing
gate here.

Focused tests:

```sh
/home/alex/azlite-venv/bin/python -m pytest \
  ml/alphazero_lite/test_archived_screen_identity_inventory.py -q
```

The focused suite has 8 passing tests, including rejection of corrupted
published indexes without repair, duplicate-seat pairs, reversed decisive
ledger outcomes, and altered archive bytes with a regenerated local receipt.

The verifier imports neither Torch nor NumPy and reads no model weights or
tablebase. It confirms evidence bytes and modification times are unchanged.
The physical-copy public-entrypoint test runs from an evidence-only fixture,
does not rename or move the source checkout, and compares deterministic index
bytes.

Remaining historical evaluation populations outside these seven sources remain
uncovered, and missing counts are unknown. Seed461 continuations remain
unresolved. The #483 fresh-confirmation gate remains closed.
