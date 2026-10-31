# Post-run verification and registered decision

## Completeness gates

The no-analysis verifier
`ml/alphazero_lite/verify_seed542_expanded_policy_strength_completeness.py`
(SHA-256
`3fd3447d2a06d1cb4b6a86c72f3fd4b5753a0bf3714256843867fbb4b72ae36a`)
passed both runs with:

```json
{
  "analysis_performed": false,
  "evidence_bytes_and_mtimes_unchanged": true,
  "evidence_files_snapshotted": 6433,
  "games": 2048,
  "openings": 1024,
  "pairs": 1024,
  "status": "all_1024_pairs_and_2048_games_independently_verified"
}
```

Gate A ran against the original screen worktree and output. Gate B ran from the
physical copy at `.tmp/seed542-screen-physical-copy`, with the original screen
worktree pathname renamed and inaccessible for the duration of verification.
The tablebase was materialized independently in the copy. After Gate B, all
6,269 pre-analysis files under the screen output directory were compared:
relative paths, byte sizes, modification times, and SHA-256 digests matched.
The sample `outcomes.jsonl` inodes differed, confirming it was a separate copy.

Frozen inputs at verification:

- Continuation registration: `c749bc760019fddc3a53755c41b4ea1bf28e87251cfc44c9f0cb71ac1088cb6b`.
- Continuation freeze: `9f44fdf3a9d4a5c87a29aa85aa4bb3677c1aa1ac7b8dd204affb61292440b4d4`.
- Complete 2,048-row ledger: `e3c54882fec44163fe1d8166241883728a491372f7c98021d2faf75c52f15e13`.
- Verifier plan: `84e6425b9bad76a05c99278211bba9ca0169c9f65b1339c4a02f31b0dea9d420`.

The initial combined verifier invocation stopped at its legacy preflight
comparison because the v2/v3 command-preflight records omit a `suite_sha256`
field. Each pair's completion receipt binds the frozen suite hash, and the
dedicated completeness verifier validated those receipts and replayed all pairs.
No frozen execution or verification source was edited to bypass that schema
assumption.

## Registered analysis and decision

Only after both completeness gates passed, the registered independent analysis
was calculated from the reconciled ledger using the frozen method: 20,000
opening-cluster bootstrap resamples, seed 545, replacement with multiplicity,
NumPy linear quantiles, and a one-sided 95% lower bound. The independent
published-analysis comparison passed for both the result and every opening row.

| Measure | Result |
| --- | ---: |
| Games / opening pairs | 2,048 / 1,024 |
| Wins / draws / losses | 896 / 201 / 951 |
| Score | 0.486572265625 |
| One-sided 95% lower bound | 0.4765625 |
| Descriptive two-sided 95% interval | [0.474609375, 0.498046875] |
| Registered advancement rule | lower bound strictly greater than 0.50 |
| Decision | **do not advance** |

The screen is exploratory on the reused seed511 suite, is not fresh
confirmation, and retains the unresolved seed461 historical-overlap limitation.
No 1,200-simulation proposal, training, promotion, or deployment was started.

Published registered outputs:

- `results.json` SHA-256:
  `27f12552842029d04c6220a815b59a39061157c0f2fd9049cb4c3ffed8ccaa1f`.
- `per-opening-results.jsonl` SHA-256:
  `f9de5f6ae2f2b55995eeefeaa64f652d9d92a588ccabeb916dd3ebb3fc99e626`.

The runner-only outputs remain separately named
`runner-provisional-analysis.json` and
`runner-provisional-per-opening.jsonl`; they were not used to calculate the
registered result.

## Append-only verifier correction

Review identified that the original, v2, and v3 command-preflight files needed
direct SHA-256 checks against their respective frozen
`command_preflight_sha256` values before their rows were used. The completeness
verifier now checks all three files explicitly; this also covers the v3 file,
which is not listed in the v3 authenticated-file map. The registration and
freeze files themselves remain unchanged.

The corrected verifier SHA-256 is
`7af938402a096fc9c81d747efc99f42a2d58abc8339b40e385710e0d3df43b6c`. Its
negative test alters each preflight file while keeping the freeze object intact;
all three tampering cases are rejected. Focused test result: **3 passed**.

The corrected verifier then passed the full 1,024-pair / 2,048-game read-only
replay against the original evidence and from a physical copy with the original
worktree path inaccessible. Both runs reported
`all_1024_pairs_and_2048_games_independently_verified` and confirmed unchanged
evidence bytes and mtimes. No games were rerun, and frozen execution sources,
registrations, freezes, preflight files, ledger rows, and raw game evidence were
left unchanged. The authenticated execution-status documents were restored to
their freeze-bound bytes before these runs.
