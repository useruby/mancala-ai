# Seed552: C108 versus frozen serving P

## Decision

**`do_not_advance_C108`**. The exploratory reused-suite screen completed with all 2,048 games valid and terminal. C108 scored **0.485107421875**; its opening-cluster bootstrap one-sided 95% lower bound was **0.474853515625**, below the strict 0.50 advancement threshold. The descriptive 95% interval was **[0.472894287109375, 0.497314453125]**. There were 894 C wins, 199 terminal draws, 955 losses, and no capped/nonterminal games.

The preregistered advancement rule was not met. This result does not authorize promotion or deployment. No 1,200-simulation follow-up was run.

## Protocol and interpretation

C is the newly registered exploratory candidate: the #550 initializer-self-play control policy at update 108, with its own trunk. P is historical seed504 E4 with its own trunk. Both use the same frozen initializer value network and initializer trunk through `ComposedArtifactEvaluator`. P was authenticated against the checked-in serving release and its component and policy-sidecar identities.

The screen ran exactly 1,024 seat-swapped opening pairs / 2,048 new games at 128 simulations per side, using the corrected #551 search/runtime protocol and the indexed seed511 v4 openings. It is exploratory evidence on a reused suite, not fresh confirmation. The unresolved seed461 continuation-overlap limitation remains. No conclusion about C versus P is inferred from C's earlier result against T, and #550's original candidate designation and #551's `do_not_advance_source_evaluator_branch` decision are preserved.

## Population overlap

All 1,024 suite rows are retained. Canonical-state and production-float32 overlap counts were:

| Population | Rows | Canonical overlap openings | Float32 overlap openings |
|---|---:|---:|---:|
| C policy training | 13,818 | 0 | 0 |
| C development | 4,738 | 1 | 1 |
| P policy training | 13,724 | 0 | 0 |

The P training population is authenticated by the seed504 registration and its six registered policy-training shards. No historical games or outcomes were imported.

## Verification

The read-only seed552 verifier reconstructed all trajectories and outcomes, seat pairs, raw-report runtime/search identities, per-opening scores, ledger accounting, bootstrap, and decision. It verified 1,024 pairs / 2,048 games. Physical-copy verification passed while both the frozen execution worktree and the verification worktree were inaccessible; the repository pointer and separately supplied materialized tablebase were checked, and evidence bytes/mtimes remained unchanged. See `verification.json`, `verification-receipt.json`, `physical-copy-verification-supersession-03.json`, and `execution-source-correction.json`. The full local checkout revision, remote ancestry, and tablebase representation provenance are recorded in `checkout-provenance.json`.
