# Seed426 canonical-state overlap audit

**Decision:** `canonical_state_overlap_detected`. Of the 14,946 validation
weighted positions, 4,240 (28.37%) have canonical identities present in
training. In the frozen `>32` active-stone bucket, 3,166 / 5,773 (54.84%) of
validation weighted positions overlap. Under the fixed rule, recommend a
**separately authorized validation-metric correction using an identity-disjoint
holdout**. This recommendation authorizes no training, replay, checkpoint, or
arena work.

This is a retrospective, data-only census of the exact #416/#422 registered
mixture and the #416 `training-freeze-v3` split. Earlier in this audit, before
reconstructing the loader's exact `np.tile` position order, provisional figures
of 6,974 intersecting canonical identities and 5,109 / 5,973 high-stone
weighted validation positions were observed. Those figures used adjacent row
copies rather than the frozen loader's repeated complete compact-row sequence;
they are superseded by the hash-bound results here. The chronology is also
recorded in `manifest.json`.

## Frozen input and split reconciliation

The five registered sources, weights and value-target modes are:

| Source | Weight | Value target | Eligible compact rows | Train rows | Validation rows | Weighted train positions | Weighted validation positions |
|---|---:|---|---:|---:|---:|---:|---:|
| fresh | 1 | default | 66,733 | 60,015 | 6,718 | 60,015 | 6,718 |
| generic_bootstrap | 4 | sharpened | 13,941 | 12,588 | 1,353 | 50,352 | 5,412 |
| random_teacher | 1 | sharpened | 2,951 | 2,675 | 276 | 2,675 | 276 |
| opening_disagreement | 8 | sharpened | 2,000 | 1,780 | 220 | 14,240 | 1,760 |
| stability | 4 | sharpened | 2,000 | 1,805 | 195 | 7,220 | 780 |
| **Total** | — | — | **87,625** | **78,863** | **8,762** | **134,502** | **14,946** |

There are 149,448 weighted positions overall. The frozen loader was executed
from the exact registered `train.py` snapshot identity with
`policy_target_mode=sharpened`, `value_target_mode=default`, and the registered
per-source value modes. The frozen call has `exclude_buckets=None`: all raw
lines parsed and loaded, so the explicit skipped-raw-line list is empty. Raw
line numbers are per-source and 1-based; compact row IDs are global and
zero-based. Loaded state, policy and value arrays were checked and SHA256-bound
per source and for the combined load. The split's ordered train/validation
arrays, source-row membership sets, multiplicity indexes, disjointness and
complete position coverage reconcile to the freeze.

## Overlap census

Identity definitions are independent and reported separately:

* **Canonical state:** ordered integer `player_pits`, `opponent_pits`,
  `player_store`, `opponent_store`, and `current_player`. Stores and side to
  move are part of identity.
* **Network input:** exact 27-element `kalah_v3` vector encoded as consecutive
  little-endian IEEE-754 float32 bytes. Identity is byte equality.

In this census the definitions yield equal counts, but input identity does not
replace the canonical-state decision rule.

| Scope | Definition | Validation overlap / denominator | Fraction | Validation identities unseen in training |
|---|---|---:|---:|---:|
| Overall | Unique identities | 1,611 / 7,291 | 22.10% | 5,680 |
| Overall | Eligible source rows | 3,046 / 8,762 | 34.76% | — |
| Overall | Weighted positions | 4,240 / 14,946 | 28.37% | — |
| `>32` | Unique identities | 774 / 2,016 | 38.39% | 1,242 |
| `>32` | Eligible source rows | 2,020 / 3,276 | 61.66% | — |
| `>32` | Weighted positions | 3,166 / 5,773 | 54.84% | — |
| `17–32` | Unique identities | 421 / 2,743 | 15.35% | 2,322 |
| `17–32` | Eligible source rows | 531 / 2,863 | 18.55% | — |
| `17–32` | Weighted positions | 570 / 4,583 | 12.44% | — |
| `≤16` | Unique identities | 416 / 2,532 | 16.43% | 2,116 |
| `≤16` | Eligible source rows | 495 / 2,623 | 18.87% | — |
| `≤16` | Weighted positions | 504 / 4,590 | 10.98% | — |

The denominators are distinct: unique validation identities, distinct
validation compact/source rows, and expanded weighted validation positions.
Weighted copies contribute to the last measure as exposure, not as independent
observations. Active-stone buckets partition validation rows and positions.

For the 1,611 shared identities, the validation-position exposure histogram
by per-identity group size is in `results.json`; the largest group accounts for
302 / 4,240 (7.12%) of shared validation positions. The top 10 groups account
22.19%; identity HHI is 0.00908. There are 1,108 shared identities with one
validation weighted position.

### Source attribution

The additive attribution partitions shared identities by exact sets of
training-source and validation-source membership. In weighted validation
positions, the mutually exclusive attribution is:

| Membership pattern | Shared identities | Shared validation weighted positions |
|---|---:|---:|
| Within-source only | 1,360 | 2,208 |
| Cross-source only | 62 | 206 |
| Both within- and cross-source | 189 | 1,826 |
| **Total, reconciles to global canonical overlap** | **1,611** | **4,240** |

The source-pairwise exposure summary is 5,535 within-source and 7,203
cross-source pair exposures. These pairwise totals overlap when an identity
belongs to multiple sources; do not sum them as though mutually exclusive.
The 57 exact source-membership patterns in `results.json` are additive and
reconcile to the global census. Source lineage does not establish game-family
independence, so no such independence is inferred.

## Provenance, limits and commands

The gzip replay snapshots decompress byte-for-byte to the five original
registered SHA256s. `manifest.json` binds the #416 v3 registration, frozen
split, training loader, `kalah_v3` encoder, decoder, runner, pure analysis,
verifier, identity definitions, formulas and decision rules. The publication
is explicitly retrospective and separate from the historical training freeze.

The audit tests validation-position generalization only. It does not diagnose
target corruption, establish the cause of strength regressions, explain failed
strength interventions, or overturn any prior arena result. No training,
search, game, checkpoint evaluation, or target generation was part of this
audit.

From the repository root, regenerate after installing the project's data
dependencies (the runner checks the original sources when available and can
fall back to the published lossless snapshots):

```bash
python ml/alphazero_lite/run_seed426_overlap_audit.py
python ml/alphazero_lite/verify_seed426_overlap_audit.py
python -m pytest ml/alphazero_lite/test_seed426_overlap_analysis.py -q
ruff check ml/alphazero_lite/seed426_overlap_analysis.py ml/alphazero_lite/run_seed426_overlap_audit.py ml/alphazero_lite/verify_seed426_overlap_audit.py ml/alphazero_lite/test_seed426_overlap_analysis.py
ruff format --check ml/alphazero_lite/seed426_overlap_analysis.py ml/alphazero_lite/run_seed426_overlap_audit.py ml/alphazero_lite/verify_seed426_overlap_audit.py ml/alphazero_lite/test_seed426_overlap_analysis.py
python -m unittest ml.alphazero_lite.test_seed425_publication_portable -v
```

The read-only verifier uses only the Python standard library and public
hash-bound files. The test suite also runs that actual verifier entry point
from an unrelated working directory against a relocated fixture containing no
checkpoints, native executable, tablebase, or ignored runtime data.
