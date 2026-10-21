# Seed516 semantic correction

Append-only correction to the seed515 independent verification. Run
`python -m ml.alphazero_lite.verify_seed516` from a checkout to execute the
seed515 independent replay plus finite-number and labeled-table checks. Ordinary
verification records no physical-copy or preservation test. A separately
captured execution record can be supplied with `--execution-record`; it must
contain the command, exit status, checkout identity, and before/after inventories.

The replay depends on the registered seed512 exact-root tablebase object. This
is an external runtime asset, authenticated by its registered pointer and object
hash. Seed516 does not perform searches, games, training, deployment, promotion,
or artifact replacement. Seed514/515 evidence and receipts remain unchanged.
