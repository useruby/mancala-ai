# Seed434 census source verification

This supplemental, read-only proof reconstructs the #432 compact rows from the
registered #416 sources and split, then checks the original #432 accounting and
#433 corrected results. It does not change historical publication files.

The append-only `receipt.json` binds the #416 registration and frozen split,
#432 manifest/results/archives, #433 correction receipt/results, the calculation
and normalization sources, and the supplemental verifier. Its recorded outcome
is that all 87,625 #432 rows and the #433 corrected result reproduce, with
149,448 expanded positions (134,502 train; 14,946 validation).

From a checkout containing committed evidence and source code, run:

```bash
python -m ml.alphazero_lite.verify_seed434_census_source --root /path/to/checkout
```

The verifier is read-only and does not require run-local replay files, training
checkpoints, or native runtime artifacts.
