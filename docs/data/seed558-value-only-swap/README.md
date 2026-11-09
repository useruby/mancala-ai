# Seed558 paired value-only retrospective

This retrospective reuses all 256 roots and all exact legal-action margins from the merged #557 diagnostic, byte-for-byte. It performs 512 independent 128-simulation searches (two fresh trees per root), and runs no training or arena games. No serving asset is changed.

## Frozen compositions and provenance

The baseline composition is serving P's seed504 E4 policy head and E4 trunk with the frozen seed542 initializer value head and initializer trunk. The treatment keeps that exact P policy composition and routes value evaluation through the complete #546 D-update-432 checkpoint, including the D432 shared trunk and value head. The trained input checkpoint SHA-256 is `4555cba73d2c908d1eca829836f361b226ccc1072be28affd816b2a240123163`; the runtime export file hashes are in `execution-freeze.json` and the export files are bundled under `artifacts/`.

The freeze records #557 and #546 merge commits, the #557 cohort and oracle hashes, #557's receipt-bound runtime identities, the #546 registration/freeze/runtime-authentication digests, and the serving publication identity. The D432 artifact hash is checked against the published #546 checkpoint identity. Every paired root's computed policy prior vectors are identical, and the verifier recomputes those priors from the bundled policy and both compositions. The only routed output difference is the value head/trunk.

The #546 publication reports 12 canonical/production-float32 opening overlaps among its 1,024 seed511 roots for the historical seed542 expanded training corpus; it also flags the seed461 historical overlap limitation as unresolved. Those are disclosures for that strength screen, not claimed counts for this #557 near-cutoff cohort. Overlap between this diagnostic cohort and training data is not authenticated; no disjointness claim is made. The full frozen cohort is retained.

## Search and scoring protocol

- Both arms use the same per-root seed, deterministic ordinary PUCT, `c_puct=1.25`, 128 simulations, one search worker, and a fresh tree per arm/root.
- Exact-root and exact-leaf solving are disabled. The frozen oracle margins are read only after search for scoring.
- Each search record retains root legal moves, matched seed, priors, six-entry visits, selected action, and child visit/Q statistics. The verifier checks selected-action visit maxima and deterministic Q/prior/index tie-breaking.
- Primary contrast: D432-value minus initializer-value mean exact final-margin regret. Negative favors D432. The interval is a paired opening bootstrap, 20,000 resamples, seed 558.

## Result and preregistered decision

| Arm | Mean regret (seeds) | Exact-optimal rate | WDL downgrade rate |
|---|---:|---:|---:|
| Initializer value | 1.1484 | 78.13% | 4.30% |
| D432 value | 1.1484 | 75.78% | 4.30% |

- Paired contrast (D432 − initializer): **0.0000 seeds**; opening-bootstrap 95% interval **[−0.2578, 0.1797]**.
- The selected action changed on 9/256 roots (**3.52%**).
- The preregistered rule requires a mean improvement of at least 0.25 seeds, interval upper bound below zero, and no WDL downgrade increase. The result meets none of the regret improvement criteria; **this diagnostic does not support advancing the swap**.
- Preserve #557's inherited `do_not_advance_T512` decision. Even a qualifying result would support proposing a separate 128-simulation value-only arena only. This near-cutoff cohort does not establish full-game strength.

## Verification

From the repository root run:

```bash
python -m ml.alphazero_lite.verify_seed558_value_only_swap
python -m unittest ml.alphazero_lite.test_seed558_value_only_swap -v
```

The read-only verifier works on a physical copy of this directory and bundled component exports; it does not open original artifact paths. Tests verify fresh output and reject altered priors, a selected move contradicted by visits, and a substituted value artifact.

To execute a new run, provide a nonexistent output path, for example `python -m ml.alphazero_lite.seed558_value_only_swap --output .tmp/seed558-reproduction-01`. The runner rejects every existing output directory and has no refreeze mode; it never writes into this published evidence directory. Verification independently replays all 512 searches against the bundled arm compositions and checks visits, selected actions, child Q values, and priors against the recorded results.
