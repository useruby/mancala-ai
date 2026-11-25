# Seed 636 prospective population generation — attempt 0001

## Outcome and classification

The registered deterministic generation, authenticated known-exposure screening,
deduplication, and population selection completed. The final classification is
**known-exposure-screened; inherited P-E4 exposure unresolved**. This is a
model-independent finite-cohort population artifact only. No inference, MCTS,
exact solving, tablebase probing, training, evaluation games, or candidate
promotion was run.

## Authorization and frozen execution

- Registration PR: [#637](https://github.com/useruby/mancala-ai/pull/637), merged
  by `useruby` into `main` at `51e1699829f841e8c0cf0a483f66a4cbc9ba1071`.
- Attempt: `attempt-0001`; source commit exactly the owner merge commit above.
- Attempt manifest SHA-256:
  `712feba653d6b1c867bcd43045b5c97223d208386ef743dcf32d8c19a0121c7b`.
- Registration SHA-256:
  `5e177839ba3697fd1e85960f6b1a33717fbf1925d86f348ed26e752d93e8f658`.
- Random seed: `62320261121`; one execution worker and one selection optimizer
  worker. Generation used CPython 3.14.7 and NumPy 2.5.3.
- Registered resource limits were frozen: 14,400 wall seconds, 10,800 CPU
  seconds, 17,179,869,184 process-RSS bytes, and 21,474,836,480 output/temp
  bytes. The lifecycle runner recorded 68.54 elapsed seconds. Its RSS figure is
  a process-wide child high-water mark, not an attempt-specific peak.

## Registered criteria and observed evidence

| Criterion | Observed | Result |
| --- | ---: | --- |
| Distinct openings | 256 | Pass |
| Independent continuation games | 2,048 (8 per opening) | Pass |
| Candidate rows after canonical and production-float32 deduplication | 5,623 | Pass |
| Exact-known-exposure matches removed | 0 | Pass; all 5,623 rows classified as known-exposure nonmatches |
| Eligible supply, bins 17/18/19/20/21 | 1,079 / 1,140 / 1,138 / 1,130 / 1,136 | Pass; each exceeds its registered hard minimum 206/206/204/204/204 |
| Selected rows, bins 17/18/19/20/21 | 103 / 103 / 102 / 102 / 102 | Pass; 512 total |
| Selected opening clusters | 218 | Pass; minimum 128 |
| Maximum selected rows per opening | 4 | Pass |
| Maximum selected rows per continuation game | 2 | Pass |
| Authenticated B512/S615/protected-holdout membership | Seed 627 verifier status `verified`; 2,048 / 2,048 / 1,353 rows | Pass |
| Provenance interpretation | Authenticated, non-crosswalkable; inherited P-E4 unresolved | As registered; no disjointness claim |

The preferred four-times candidate supply was also achieved in all five bins.
The exact independent verifier reproduced opening/game generation, membership
screening, candidate supply, both deduplication identities, optimizer selection,
quotas, and cluster caps. Its output is recorded in `domain-verification.json`.

## Lifecycle and archive verification

The lifecycle verifier passed with registered inputs verified and one declared
output verified. The external archive contains 22 attempt files; archive,
sidecar, safe extraction, member inventory, and hashes passed verification.

- Archive: `/home/alex/mancala-ai-experiment-archives/seed636-prospective-population--attempt-0001.tar.zst`
- Archive size: 8,066,427 bytes
- Archive SHA-256:
  `d6e454f889e6155d7dccf3bf81e6769989896389d401fc5bf4cbd058059c8bf1`
- Archive sidecar:
  `/home/alex/mancala-ai-experiment-archives/seed636-prospective-population--attempt-0001.archive-manifest.json`
- Sidecar SHA-256:
  `057358aad2d72b733a29019c21f7a3f0a9b7d68e37cd5ea2954660e59c366bc5`
- Required-artifact inventory SHA-256:
  `a4ae4728e87e37a97a37d3abb51687c48f251736f77da1b1af82cd64c5dbd148`

## Limitations and next gate

The population is the registered deterministic, hash-selected finite cohort and
is not a probability sample of the broader reachable-state distribution. Exact
membership nonmatches cover only authenticated available B512, S615, and
protected-holdout inventories. They cannot clear undocumented P-E4 ancestry;
the result does not establish training disjointness, unseen-state generalization,
playing strength, or candidate qualification. Any inference, solver, or search
work requires its own registration PR and owner merge.

## Reproduction

1. Verify lifecycle records and outputs with:
   `python -m ml.alphazero_lite.experiment_lifecycle verify <attempt-dir>`.
2. Verify the external archive and its complete inventory using the published
   archive and sidecar paths above.
3. Independently verify the archived bundle with:
   `python -m ml.alphazero_lite.verify_seed624_prospective_benchmark <extracted-attempt>/population/population.json`.
4. Reauthenticate Seed 627 input evidence using the unchanged
   `ml.alphazero_lite.seed628_authenticated_screening.authenticate_snapshot_bundle`
   path under the pinned historical runtime and the frozen NumPy 2.5.3
   generation runtime.
