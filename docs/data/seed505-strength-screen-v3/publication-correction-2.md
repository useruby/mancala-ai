# Seed505 publication correction 2

The six corruption cases in the original correction-1 test module did **not**
establish semantic rejection: they accepted any `ValueError` and passed the
copied publication directory as the verifier root. As a result, missing source
identities or runtime dependencies, or an earlier unrelated validation error,
could satisfy those assertions.

The separate `test_seed505_publication_semantic_rejection.py` module first
verifies an unmodified publication copy using the actual repository root and
the copied publication as the verifier's explicit `out` argument. Each of the
six cases repeats that positive control, applies its corruption, rebinds the
ordinary publication receipt, and requires the corresponding semantic
rejection code (`published_result_mismatch`, `report_runtime:0`,
`suite_association:0`, or `ledger_replay:0:0`). The prior correction receipt
and all of its bound files remain preserved.

This test correction is verification-only. The failed seed505 strength-screen
decision is unchanged; no games, training, promotion, fresh confirmation, or
1,200-simulation test was performed.
