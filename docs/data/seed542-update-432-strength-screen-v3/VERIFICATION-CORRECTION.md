# Append-only verification correction

This correction addresses the two verifier findings on PR #547 without editing the frozen registrations, execution sources, suite, commands, raw games, receipts, ledger, or analysis.

## Per-opening suite and command binding

The supplemental verifier now compares every `batches/opening-NNNN.jsonl` byte-for-byte with line N of the registered `openings.jsonl`. It also compares each played `command-NNNN.json` with that index's frozen command-preflight row, including argv, registration and suite hashes, component assignments and hashes, runtime, and runtime-source hashes. The completion receipt is checked as self-consistent with the played opening, command, games, and report; this does not substitute for matching the frozen suite row.

The negative test copies opening 3's bytes to index 4, rebinds command and output-receipt hashes to those copied bytes, leaves the suite and freeze unchanged, and confirms rejection because the played bytes differ from frozen suite row 4. A second test alters the command while leaving its frozen preflight unchanged and confirms rejection.

## Continuation v4–v6 chain

The correction authenticates each continuation registration against its freeze, links v4 to the original registration/freeze, v5 to v4, and v6 to v5, and checks the frozen suite and update-432 checkpoint identity throughout. It binds v4/v5/v6 recovery evidence to opening pairs 0/1/2, reconciles the v5 two-game prefix and v6 four-game prefix with the final ledger, and checks the continuation source-digest maps against their freezes. The active v6 source bytes are also checked against the frozen v6 source hash; unchanged shared source files are checked against all continuation maps. Continuation pair bindings for all 1,024 openings identify their applicable continuation registration/freeze and bind the opening, preflight row, command, raw games, report, and completion receipt.

v4 and v5 are retained as historical continuation records whose orchestration sources were superseded by later append-only recovery revisions. Their original source digests remain cross-bound by their registration/freeze files and child continuation hashes; the original registered arena runner and game-command sources remain unchanged and are checked against the original execution freeze.

## Verification outcome

- Per-opening suite/preflight/command bindings: **1,024 verified**.
- Continuation bindings: **1,024 verified**.
- Continuation parents, recovery pairs, and ledger prefixes: **verified**.
- Independently replayed: **1,024 seat pairs / 2,048 games**.
- Registered bootstrap and published result: reproduced exactly.
- Separate physical copy: independently verified; evidence directory is byte-identical and game-file inodes differ.
- Focused supplement tests: **4 passed** (suite-row substitution with rebound receipts, command tampering, continuation source-map tampering, and continuation-binding tampering).

The decision is unchanged: **do not advance**. Score is 0.481689453125 and the registered one-sided 95% lower bound is 0.47021484375, below the strict 0.50 threshold.
