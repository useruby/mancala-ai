# seed444 — frozen-checkpoint minibatch denominator audit

> **Post-observation corrected rerun.** The original registration and initial evidence are preserved byte-for-byte. The initial E=0 observations preceded discovery of the unused-parameter handling defect and are superseded for measurement purposes. Corrected execution hashes were bound after that observation and before this rerun.

## Coefficient coverage

- Training exposures N: **134,502** across 78,863 unique compact rows (55,639 repeated exposures retained).
- Original float32 coefficients: positive 134,502, zero 0, min/max 1/1; Q=134502.
- Expanded little-endian float32 coefficient-vector SHA256: `e3a1b2988e2a7be03d1760b9fcc374bce40346f4c3570623afeb68bdaf5a264c`.
- Independently enumerating every exposure found that all coefficients equal exactly 1.0f. Therefore S_b=n_b in every batch, Q=N, S_b/Q=n_b/N, and the two policy aggregate objectives are structurally identical at every parameter state. The numerical rerun still computed the differences; no result was forced to zero.

## Fixed-checkpoint measurements

Both checkpoints use all 134,502 training exposures in the first frozen epoch order, batch size 512, and the final partial batch (358 exposures). Values below are unclipped fixed-parameter gradients, not Adam updates or expectations over random permutations.

| Checkpoint | N | Q | ||G_global|| | ||G_batch|| | E | policy cosine | total cosine | direct reconciliation L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| initializer | 134502 | 134502.0 | 1.16982217235 | 1.16982217235 | 0 | 1 | 1 | 1.172e-07 |
| adam_a16 | 134502 | 134502.0 | 1.28631179294 | 1.28631179294 | 0 | 1 | 1 | 1.235e-07 |

The denominator ratio n_b·Q/(N·S_b) is 1 for every positive-mass batch at both checkpoints. Grouped squared differences are recorded for shared trunk, policy head, and value head and reconcile to the full-vector squared norm. The batch ledger includes exposure and coefficient counts, source/bucket counts, and ordered position/compact-row hashes.

## Equal-update weighting and unseen alignment

The equal-update mean across minibatches is reported separately. Its difference from the exposure-weighted batch mean isolates final-partial-batch weighting from denominator choice; per checkpoint its vector norm and relative effect appear in corrected evidence. Strict unseen >32 policy-CE alignment is descriptive only and reuses the corrected #438 vectors after validating their checkpoint, source-registration/split, population, weighting, parameter-layout, protocol, source and archive identities.

| Checkpoint | Weighting | dot(U,G_global) | cosine | unit-descent alignment |
|---|---|---:|---:|---:|
| initializer | exposure_weighted | 2.25268894226 | 0.840364314694 | -1.9256678455 |
| initializer | equal_input | 1.00477191521 | 0.594310455847 | -0.858909959951 |
| adam_a16 | exposure_weighted | 2.59081048382 | 0.856242322342 | -2.01413879437 |
| adam_a16 | equal_input | 1.61786913534 | 0.653189834419 | -1.25775814559 |

## Fixed decision and interpretation

**Original decision classification retained:** `denominator_mismatch_below_screen_threshold`. The structural finding is `true`: for this frozen dataset, changing between these denominators is an exact objective-aggregation no-op. This does not establish a training bug, causality, reopen a closed branch, or authorize a promotion or experiment. A material mismatch in other coefficient data would require a separately preregistered ablation.

No training, optimizer update, search, game, export, or promotion was performed.
