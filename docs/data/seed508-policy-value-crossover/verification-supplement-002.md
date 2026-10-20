# Seed508 read-only verification supplement 002

This append-only supplement corrects the arithmetic and report-contract coverage
of supplement 001. It preserves the original registration, execution freeze,
source snapshots, commands, reports, outcomes, results, publication receipt, and
all supplement-001 artifacts.

The v2 checker derives standalone W/D/L, scores and paired-opening descriptive
percentile intervals, plus the matched P-minus-V estimate, 20,000-resample
seed-508 interval, and interpretation directly from the outcome ledger after
replay verification. Its RNG follows the frozen analysis conventions: NumPy
`default_rng(508)` for the primary matched bootstrap, and seeds 509 and 510 for
the P and V standalone intervals. It does not call the runner's `analyze()`.
Every primary and standalone published field agrees with independent arithmetic.

The checker also validates both command-resolved search-option JSON values,
report search options, `exact_leaf_solve_mode=disabled`, and policy/value sidecar
identity and component-file hashes against authenticated registration and
artifacts, retaining the existing opponent/component checks. The test suite
accepts a physical copy and rejects receipt-rebound primary interval,
interpretation, standalone interval, report search-option, exact-leaf-mode, and
sidecar-identity mutations with semantic error codes.

The independently recomputed findings agree with the published results; there
is no arithmetic discrepancy. Findings remain exploratory and do not imply
promotion or fresh confirmation. No new games, training, promotion, fresh
confirmation, or 1,200-simulation test was performed.

Run focused verification with `python -m pytest
ml/alphazero_lite/test_seed508_verification_supplement_v2.py -q`, and run the
checker with `python -m docs.data.seed508-policy-value-crossover.verify_supplement_v2
--root REPOSITORY --out PUBLICATION_COPY` from any working directory.
