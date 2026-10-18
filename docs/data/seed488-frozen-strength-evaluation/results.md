# Seed488 exploratory strength evaluation

## Result

All 512 registered paired openings completed, for 1,024 independently replayed
games. F scored **0.4892578125** (448 wins, 106 draws, 470 losses; 0 games
reached the 200-ply cap). The primary score is `(F wins + 0.5 × draws) / 1,024`.

The registered 20,000-replicate paired-opening percentile bootstrap (seed 488)
has a one-sided 95% lower bound of **0.4755859375** and a descriptive two-sided
95% interval of **[0.47314453125, 0.50537109375]**. The lower bound does not
exceed 0.50, so the registered advance rule is not met. There is no promotion.

## Verification

- Independent complete-publication verifier:
  `.venv-azlite/bin/python -m ml.alphazero_lite.verify_seed488_strength_evaluation --root .`
  returned `verified_complete`, 1,024 games, score `0.4892578125`, and
  `advance: false`.
- Focused tests:
  `.venv-azlite/bin/python -m pytest -q ml/alphazero_lite/test_seed488_strength_evaluation.py ml/alphazero_lite/test_seed488_opening_replay.py`
  passed, 31 tests, from a physical worktree with the registered LFS-backed
  native probe/tablebase runtime files copied into it.
- Relevant regressions passed: `test_run_manifest`,
  `test_seed425_publication_portable`, `validate_forensic_reference_artifact`,
  and the native MTD(f) probe tests.
- Ruff lint passed for `ml/alphazero_lite script/ai`; Ruff format check passed
  (1,151 files); `git diff --check` passed.
- `pre-commit run --all-files` passed.
- The committed publication was rechecked from a separate physical checkout
  with the required native probe and tablebase files populated; this confirms
  the published evidence is checkout-portable and does not depend on the
  original execution working directory.

The complete execution receipt is `execution-receipt.json`; the independent
analysis summary is `analysis.json`. Per-opening paired raw arena game logs,
reports, and exact single-opening suites are retained under
`resume-corrected/` and are bound through the outcome ledger. The originally
failed pre-game attempt and opening-0 recovery evidence remain unchanged.

## Limitations

This is exploratory strength evidence on the registered suite, not fresh
confirmation or an independently untouched holdout. Non-overlap with the
unresolved seed461 continuation population cannot be certified because
authenticated state identities for its 1,129,643 receipt-matched continuation
actions are unavailable. Fresh confirmation remains unauthorized.
