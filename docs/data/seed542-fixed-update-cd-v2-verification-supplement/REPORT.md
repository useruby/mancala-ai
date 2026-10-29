# Seed542 v2 verification supplement

This append-only supplement addresses two publication cross-checks without
changing the frozen v2 runner, verifier, registration, training trace, or
checkpoints.

## Reconciled production boundary checkpoints

The supplement independently replayed C from the authenticated initializer and
published schedule, then compared all arrays (names, shapes, dtypes, and values)
against both C and current-runtime production checkpoints:

| Update | C SHA-256 | Production SHA-256 | Exact match |
|---:|---|---|---|
| 27 | `0af46e55368de075e69fa66b16ed91bd715443a98788262a8bd9df7f4daa29d9` | same | yes |
| 54 | `d0a74dc5719c382d426a3f351a18de1b953513cb691c62bad0138f75b89900cd` | same | yes |
| 81 | `5e906da79a3f5578a882ff92b22a2963d6e06156c1f1e0992781a4f93ba23e63` | same | yes |
| 108 | `8097083c8f8554535b2939150544ef90c7c7f1ed3ff15ffbe734ec9f9dd1a775` | same | yes |

The check reads the production checkpoints directly; it does not treat the
previously published equality flags or historical seed504 drift report as a
substitute for array reconciliation.

## Reconciled duplicated effects

The supplement independently recomputed development predictions, metrics, and
D−C / D−initializer effects, then required the duplicated `run-status.json`
effect object to match those results exactly. Its read-only verification passed.

## Mutation tests

- Changed a copied production E1 boundary parameter by one float32 step, rebuilt
  the descriptive historical-drift report to agree with that alteration, and
  confirmed the supplement rejects the production/C boundary mismatch.
- Changed a copied >32-stone exposure-weighted D−C policy-CE effect in
  `run-status.json` and confirmed the supplement rejects the status/evaluation
  mismatch.

The mutation tests use copied artifacts under repo-local `.tmp`; all original
v2 evidence remains untouched. The real supplement verifier passed from a
physically separate checkout and unrelated working directory while both
original experiment worktree paths were inaccessible. It reported unchanged
registered input and output bytes/mtimes. Supplement verifier SHA-256:
`f7d2c8fc99075abab3800e7c4645791dbce09bb155f4b92f488010771560caee`.
