# Seed560 PR #560 follow-up

This follow-up records the review corrections without changing the published
`run-0004` directory.

- The `run-0004/analysis.json` root records reproduce mean incremental latency
  **0.05476068352800212 seconds** (0.05476 s). This corrects the 0.05444 s
  figure previously quoted in the PR description.
- The verifier now reruns both root searches and every P continuation search
  against the authenticated P/initializer artifacts, native probe, and
  materialized tablebase. Legal replay alone is not treated as proof of the
  chosen search action.
- Future/resumed runs persist the root record and both arms in one fsynced,
  atomic per-opening commit. Aggregate JSONL ledgers are rebuilt from those
  commits after interruption; legacy partial roots without both arms are
  rejected before any opening is skipped.
- Serving artifacts are checked against the authenticated #556 component
  registration, #552 P model identity, #555 initializer and exact-root asset
  identities before evaluator construction and again after loading.

To reproduce search-choice verification, provide a materialized tablebase
whose SHA-256 and size match the authenticated #556 asset. The physical-copy
verifier copies this asset into the clone, hides both original paths during
verification, and then checks that the evidence bytes and mtimes stayed
unchanged. Its review receipt is stored alongside this note at
`physical-copy-verification-review.json`; the `run-0004` directory is not
rewritten.

```sh
PYTHONPATH=. python -m ml.alphazero_lite.verify_seed560_paired_first_action \
  --data docs/data/seed560-paired-first-action/run-0004 \
  --materialized-tablebase /path/to/kalah_v1_21.kvtb
PYTHONPATH=. python -m ml.alphazero_lite.verify_seed560_physical_copy \
  --materialized-tablebase /path/to/kalah_v1_21.kvtb
```
