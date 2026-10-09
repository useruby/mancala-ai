# seed450 — post-execution seed449 publication correction

This supplemental correction addresses a portability defect discovered after
seed449 analysis: seed449's frozen verifier passed the absolute derivative
paths in seed416 `registration-v3.json` to its replay loader. The correction
does not rewrite seed449's registration, snapshots, ledgers, results, receipt,
or analysis chronology.

## Portable verification

From any working directory, run:

```bash
.venv-azlite/bin/python /path/to/checkout/ml/alphazero_lite/verify_seed450_seed449_correction.py --root /path/to/checkout
```

The verifier reads the five committed compressed source snapshots beneath
`--root`, validates their compressed and decompressed identities, reconstructs
the frozen lane-A derivative JSONL serialization in a temporary directory
inside that checkout, verifies each registered derivative SHA256, and supplies
those temporary files to the production replay loader. The historical absolute
derivative strings remain provenance only; the verifier does not open them.
It prints its report and never generates or writes a receipt.

Fresh source raw line 24 declares `policy_target_actual_mode=exact_root_one_hot`
while `policy_target_mode=sharpened`. The training loader's actual-mode-first
resolver selects the exact-root one-hot validator branch; after common shape,
finiteness, normalization, and legality checks that branch requires a one-hot
target and returns before ordinary requested-mode matching. Its exact-root
metadata records selected action 2, optimal actions `[2, 5]`, and equal margins
of -24. Lane A copies the stored policy unchanged. This behavior is exercised
alongside ordinary mode matching, exact-root uniform metadata validation, and
invalid/conflicting metadata in the focused tests.

The production loader then checks its source-specific value modes, row
eligibility, policy-loss coefficients, and replay copy order. Publication
reconstruction independently checks ordered compact-row coverage, exact
float32 membership identities, legal masks, archived targets and prediction
ordering, row losses, identity aggregation, strata, contributions, covariance,
endpoint reconciliation (atol=2e-6, rtol=2e-6), and classification against the
seed449 ledgers and results.

## Verified interpretation

The corrected portable read-only verification confirms 2,607 exposures and
1,242 exact encoded-input identities, retains
`frequency_concentrated_policy_gain`, and preserves the historical
`close_joint_output_cap_branch` classification. Among singleton exposures,
840 are from `fresh` and eight from `random_teacher`, reconstructed by joining
the verified membership rows to source membership. Frequency and source are
confounded. This descriptive classification does not show that replay
duplication causes the observed difference and does not, by itself, justify a
change to training weights.

This is explicitly a post-execution correction. It changes no targets,
coefficients, training, gradients, model execution, search, games, or promotion.
It validates the historical publication without asserting that this correction
preceded analysis.

See `correction-report.json` for recorded verification results and
`correction-receipt.json` for the supplemental source/dependency/input binding.
