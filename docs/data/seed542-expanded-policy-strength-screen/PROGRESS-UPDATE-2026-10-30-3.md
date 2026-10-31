# Durable continuation progress snapshot — update 3

As of this snapshot, the persistent continuation-v3 service is active under
the frozen 25% CPU / 4 GiB memory scope. Ten complete seat-swapped opening
pairs (20 games) have passed independent pair validation and were committed to
the ledger, through opening index 9. Opening 10 is in progress; openings
10–1023 remain.

The completed pairs are preserved with their original command receipts, raw
reports and trajectories, completion receipts, and ledger bindings. No
aggregate analysis or advancement decision has been calculated. Interim
pair-level scores are not used to change execution or decision rules.
