# Exact tactical-label diagnostic

This retrospective diagnostic asks whether the archived 128-simulation P move
can be proved inferior on early-game positions retained by merged PR #585. It
does not launch games, train a model, or alter serving. PRs #580 and #585 remain
unchanged, and #483 remains closed.

## Authenticated source and frozen cohort

The source is the merged #585 archive at
`docs/data/broader-opening-coverage-pilot/`. Its independent replay verifier
is run by the tactical verifier before it reads source rows; all 124 pairs
passed using the externally supplied evidence digest
`e1cf047b1e8587d673ae711c11e29eb9125240391d945a581a27ce47ce37cd75`; the
published source freeze digest is
`3f2ccca4fae605f826b4fb00e0adb6f88c8c87ad8ca1373152aea051133f75b5`.
Run that verification first:

```bash
python -m ml.alphazero_lite.verify_broader_opening_coverage_results \
  --expected-evidence-digest-sha256 \
  e1cf047b1e8587d673ae711c11e29eb9125240391d945a581a27ce47ce37cd75
```

`cohort.json` freezes 256 roots from 793 eligible retained P decisions, drawn
round-robin across 116 opening groups. The archived action is taken from the
replayed trajectory, not inferred from the visit target. Candidates were
deduplicated under both canonical-state and production `kalah_v3` float32
identities. Selection orders each group by
`SHA256("585-exact-tactical-v1:<opening_id>:<row_id>")`, then round-robins
sorted opening groups. Cohort SHA-256:
`6fcabe67cc4b36e41c1b084f330e9ac56524ac5e8a684160353b925750f583ab`.

Before search, the cohort bound 30 wall seconds per root, 5 seconds per legal
action, a 500,000-entry transposition table, ascending move order, the solver
and native-adapter source hashes, the native probe binary hash, and the
authenticated Kalah v1 21-stone tablebase hash. Full-depth alpha-beta uses
exact tablebase values for every eligible nonterminal descendant at 21 or
fewer active stones. Only completed exact action margins are certified;
timeouts stay unresolved. `evidence-progress.json` contains root-granular
resumable output and `evidence.json` contains the final per-action evidence.

## Result and gate

All 256 roots are unresolved under the frozen budgets. Of 1,199 legal-action
searches, 5 completed exactly and 1,194 timed out; no root had a certified
best action, and no archived move was proved inferior. The 5 exact action
claims were re-solved using the authenticated tablebase and the deadline-safe
probe; all five margins matched:

| Root | Row | Move | Original margin | Rechecked margin |
|---:|---|---:|---:|---:|
| 93 | `treatment-matched-0048-P0:0003` | 3 | -2 | -2 |
| 95 | `treatment-matched-0049-P1:0005` | 4 | +2 | +2 |
| 162 | `treatment-matched-0026-P1:0003` | 2 | -6 | -6 |
| 177 | `control-matched-0034-P0:0003` | 1 | -2 | -2 |
| 215 | `treatment-matched-0053-P0:0008` | 1 | -28 | -28 |

Outcome downgrades:
win-to-draw 0, win-to-loss 0, draw-to-loss 0. The planning gate (at least 32
certified roots and at least 8 proven inferior archived moves) fails on both
counts. Proof coverage is the immediate limiting factor; the missed-move count
is not estimable from unresolved roots and is not evidence of zero true missed
moves. Stop this branch; do not propose a training ablation from this result.

Evidence SHA-256:
`8c0636d41dae6fa14796ed95da38a7510d772cbe05c5b3329bd506e2d9c61dca`.
The unchanged five-claim tablebase recheck is pinned by SHA-256
`39efdb46a31b1f8abd50c696f70964886271957cff9a152c98c86f12bb6695db`.

## Reproduce and verify

The source tree and authenticated tablebase must match the hashes in the frozen
cohort. Materialize the registered Git LFS tablebase before verifying. To
continue from the saved root checkpoint or reproduce the run:

```bash
python -m ml.alphazero_lite.exact_tactical_label_diagnostic solve
python -m ml.alphazero_lite.exact_tactical_label_diagnostic recheck-exact-claims
python -m ml.alphazero_lite.verify_exact_tactical_label_diagnostic
python -m unittest ml.alphazero_lite.test_exact_tactical_label_diagnostic_regressions -v
```

`solve` verifies the frozen cohort, solver source, native probe, and tablebase
before searching. It resumes after the last fully recorded root. The verifier
first replays and authenticates #585, rebuilds the deterministic ordered cohort
from authenticated source rows, and compares every state, legal-action set and
archived move. It recomputes externally pinned cohort and evidence digests,
checks tablebase hashes, reconstructs both state identities, verifies the
legal-action set, root budgets, exact margin intervals and inferior-move proofs,
then validates the separate five-claim tablebase recheck and reconstructs gate
counts and certified outcome downgrades. The original `evidence.json` bytes and
digest are preserved; the recheck is an additional artifact.
