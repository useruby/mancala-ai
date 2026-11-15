# Seed 594 — run-0001 results

## Execution

- Outcome: completed exactly one frozen treatment, 512 updates, on the pinned
  CPU runtime (Python 3.12.8, NumPy 2.2.6, PyTorch 2.7.1+cpu; one thread).
- Training roots: 256; batch size: 32; training seed: 571.
- Execution authorization and all frozen bindings: exact match before start.
- Execution errors: none.
- Qualification inference, PUCT, games, and sealed-suite evaluations: not run.

## Published artifacts and identities

- `training-trace.jsonl`: all 512 update records; SHA-256
  `ef50d48aa1615b04582b7d78c307d92e6c970788875698c1ba15c0ab1feada25`.
- `seed594-exact-oracle-update-512.npz`: update-512 checkpoint; SHA-256
  `7fddf5d6cba7eea4e7b8e06ba89c5b936d28727470934a51c603a739768e9bdd`.
- Parent P-E4 initializer checkpoint SHA-256:
  `2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad`.
- Exact-target canonical identity SHA-256:
  `00cf696db627a441598a5c1d3e4ca8842e26fa68550588c1f6fc31aa36b36a28`.
- Exact-target file SHA-256:
  `85b1f6ca7fe44ae0d0c6db08cc0ce338a9e3398b4aead09c1388935a8d69b4c2`.
- `checkpoint-identity.json` records the checkpoint, target, parent, tensor
  identities, and frozen-parameter status.
- `verification-receipt.json` is the output of
  `verify_seed594_training_result`.

## Training summary

| Metric | Value |
| --- | ---: |
| Loss, update 1 | 1.1315083504 |
| Loss, update 512 | 0.8434216380 |
| Loss, mean across 512 updates | 1.0392785884 |
| Loss, range | 0.8265901804–1.4925214052 |
| Reported pre-clipping gradient norm, mean | 5.6917579817 |
| Reported pre-clipping gradient norm, range | 3.6232714653–10.2227745056 |
| Updates with pre-clipping norm above the frozen clip threshold of 1.0 | 512 |

## Verification

Structural/integrity status: **verified**. The receipt authenticates the frozen
inputs, source, targets, and initializer; validates the complete 512-record
batch/runtime trace and checkpoint keys, shapes, dtypes, finite values, and
identities; and confirms frozen parameters are unchanged. It is explicitly
structural/integrity-only and does not numerically replay optimizer updates.
