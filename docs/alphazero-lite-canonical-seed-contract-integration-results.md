# Canonical Seed Contract Integration

Classification: `evaluation_contract_promoted_candidate_rejected`.

`azlite_eval_seed_v1` is now used by arena evaluation, opening-suite benchmarking, and the seat-aware gate. Each search derives from the model-independent v1 serialization schema in `evaluation_seed_contract.py`; paths, model hashes, labels, worker identity, cache paths, and output paths are excluded.

## Compliance

| Path | Status |
| --- | --- |
| `arena.py` | Per-search v1 seed context and ordered ledger |
| `run_opening_suite_seat_benchmark.py` | Passes v1 contract and ledger options to arena |
| `seat_aware_promotion_gate` | Passes v1 contract and ledger options to arena |
| `run_paired_seed_evaluation_audit.py` | v1 reference path; opening-cluster bootstrap |

## Validation

- `ruff check ml/alphazero_lite script/ai`: passed.
- Required targeted unittest set: 106 passed.
- Duplicate-current delta: exactly `0.0`.
- Runtime profile: c_puct `1.25`, `768:768` override `0.90`, tactical root bias `0.0`, deterministic root selection, and no value normalization.
- Seed-42 ledger SHA256: `3efb17739dbef1a8c33aff0c865c4af8246b276241a10fd516345dcd0602adfb`.

## Medium Rebaseline

The primary interval uses 10,000 paired opening-cluster bootstrap samples over 128 openings, not repeated deterministic seed means.

| Budget | Mean DS delta | 95% CI | Decision relevance |
| --- | ---: | --- | --- |
| 384:256 | +0.03516 | [-0.05078, +0.12119] | Lower bound is not positive |
| 768:768 | -0.02734 | opening-level interval recorded | Within mean robustness limit |
| 1200:1200 | -0.00781 | opening-level interval recorded | Within mean robustness limit |
| 1200:256 | +0.03906 | opening-level interval recorded | Within mean robustness limit |

The 384:256 opening outcomes were 23 positive, 86 zero, and 19 negative. The candidate therefore fails medium robustness. Fixed-large, held-out suites, and the promotion gate were not run. No model was trained or promoted.

## Rechecks

Future rechecks are limited to PR #164 deterministic joint-head results, PR #168 composition results, search-calibration runners, and value-transform/value-blend runners.
