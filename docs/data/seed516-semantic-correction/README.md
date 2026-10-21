# Seed516 semantic correction

Append-only correction to the seed515 independent verification. Run
`python -m ml.alphazero_lite.verify_seed516` from a checkout to execute the
seed515 independent replay plus finite-number and labeled-table checks. Ordinary
verification records no physical-copy or preservation test. A separately
captured execution record can be supplied with `--execution-record`; it must
contain the command, exit status, checkout identity, and before/after inventories.

Seed516 rejects non-finite reported statistics and duplicate per-state entries,
then checks the labeled aggregate/stratum/configuration/column/unit cells in both
seed514 tables against the replay's independently recomputed values. Its tests
begin with the unchanged positive publication and exercise the real correction
entry point for malformed numeric summaries and table mutations.

The replay depends on the registered seed512 exact-root tablebase object. This
is an external runtime asset; the physical-copy run authenticated the external
registered object and was not self-contained. Ordinary semantic verification
does not claim a physical-copy or preservation check. The separately recorded
copy execution and before/after inventories are in
`seed516-correction-receipt.json`.

Seed516 performed no searches, games, training, deployment, promotion, or
artifact replacement. Historical seed514/515 evidence and receipts were not
regenerated; the receipt records matching SHA-256 and nanosecond-mtime
inventories before and after verification.
