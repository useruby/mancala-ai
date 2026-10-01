# Opening replay contract erratum

The historical suite generators stored absolute pit indices in `prefix_moves`. The arena loader/replayer interprets those entries as player-relative actions. Because replay intentionally stops at the first illegal action, a registered opening could silently become a one- or two-ply opening.

The corrected reconstruction is in [`opening-replay-contract-erratum.json`](opening-replay-contract-erratum.json). It covers all available suites for the #383–391 order-evaluation sequence. Each registered 256-opening suite reached only 10 unique actual starts; the two #391 confirmation suites each declared 512 states, reached the same 10 actual starts, and replayed no entry to its declared state. Prefix lengths and evidence identities are included in the machine-readable audit. The old game rows did not record actual opening-state identity, so those identities are reconstructed from each runner's arena replay path.

Reproduce the suite counts, state identities, pairwise actual-state overlaps, and #391 raw evidence hash checks with `.venv/bin/python -m ml.alphazero_lite.audit_opening_replay_contract`.

The registrations, original scores, score matrices, and raw evidence are preserved. The #391 advancement claim is withdrawn. The affected holdouts cannot support treatment-benefit or training-order-generalization conclusions. Existing #386/#388/#389/#390 results remain historical records, not valid corrected-distribution evaluations.

The canonical production prefilter is unaffected: all 256 prefixes replay completely to their declared unique states. Its #392 rejection remains valid and seed455 remains the incumbent. No corrected diagnostic result overrides that canonical promotion failure.

New exports use `arena_player_relative_v2`: generators retain their absolute internal traces, then export conversion replays each absolute action and translates it relative to the player moving at that ply. Extra turns therefore retain the mover's frame. Versioned suites are completely replayed, compared against declared state/hash and screened for duplicate actual starts before arena launch. New game evidence records actual start identity and applied prefix length.

The frozen E4 diagnostic used seeds 393 and 394 and completed 2,048 games. A post-run audit found that its registration excluded prior actual states but missed 14 and 12 intersections with prior declared-state sets. Its games, matrix, and observed fail result are preserved, but the preregistered exclusion contract failed; see [`order38615-corrected-diagnostic/results.md`](order38615-corrected-diagnostic/results.md). No outcome-dependent replacement games were run.
