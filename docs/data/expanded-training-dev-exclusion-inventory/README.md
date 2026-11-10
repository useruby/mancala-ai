# Expanded training/development exclusion inventory

This read-only inventory is anchored at merged #562 (`3e493173`). It authenticates
and indexes three source families: archived #503 shards with #504's registered
split; the amended #543 seed542 corpus; and the matched #549 control/treatment
corpora with their shared development membership. Identity indexes are derived
from authenticated/replayed rule states using canonical-state SHA-256 and the
production little-endian `kalah_v3` float32 input SHA-256. Each occurrence keeps
its source family, population, file, and line provenance.

## Reconciled populations

| Family | Occurrences (training + raw development) | Distinct identities (both schemes) | Overlap with #562 | New beyond #562 |
| --- | ---: | ---: | ---: | ---: |
| #503/#504 | 18,281 | 11,896 | 374 | 11,522 |
| #543 seed542 | 146,348 | 70,614 | 1,639 | 68,975 |
| #549 control + treatment | 36,984 | 20,470 | 724 | 19,746 |
| Combined source-family union | — | 91,082 | 2,037 | 89,045 |

The two identity schemes have equal counts for these authenticated inputs. The
family totals are not additive: the combined union deduplicates cross-family
overlap. #562 contains 58,101 identities under each scheme; the union of #562
and these families is 147,146.

The #503/#504 counts distinguish 13,724 training occurrences, 4,557 raw
development occurrences, and 2,671 retained development occurrences that were
scored (1,886 raw development occurrences excluded by the registered overlap
rule). The #543 corpus contains 109,504 training and 36,844 raw development
occurrences; the retained development membership contains 15,359 occurrences.
The #542 recovered 51-game derivative (1,768 rows) and the 13-game supplement
(453 rows) are each counted once in the training population. The incomplete game
and unstarted slots are excluded. Total usable rows reconcile to 146,348.
Seed549 contains 13,818 control and 13,849 treatment training rows, 4,656 and
4,661 arm-specific raw development rows, and 4,738 retained shared-development
occurrences. The retained development populations are separately indexed and
are not added again to family totals.

The #550 registration's publication-pinned input hashes bind its control and
treatment training identity lists; both occurrence lists match the authenticated
#549 corpus memberships. #554's publication-pinned input-authentication record
and training-input binding authenticate all 12 derivative shards; their decoded
state occurrences match the 13,818-row #549 control training multiset. Changed
targets, exposure schedules, or checkpoints do not add state identities.
Neither derivative adds an independent state population.

The verifier pins the #504 source manifest and #500 accounting/results records,
both #549 arm manifests and shared membership, the #550 registration, and the
#554 input-authentication and training-input-binding records to the merged
publication bytes before accepting their local shard-hash maps.

## Artifacts and verification

- `source-manifest.json` binds source manifests, split/membership publications,
  and the #562 reference inventory.
- `identity-inventory.jsonl` preserves occurrence-level provenance and both
  identities for source training, raw development, and retained development.
- `count-report.json` contains per-population occurrence/distinct counts,
  family overlap/additions against #562, cross-family overlap, and explicit
  remaining gaps.

From repository root, run the verifier (read-only):

```sh
PYTHONPATH=. python -m ml.alphazero_lite.expanded_training_dev_exclusion_inventory
PYTHONPATH=. python -m unittest \
  ml.alphazero_lite.test_seed500_504_exclusion_inventory \
  ml.alphazero_lite.test_seed542_549_exclusion_inventory -v
```

`--write` reconstructs deterministic publication artifacts. The verifier
authenticates all shard bytes and source membership bindings, independently
reconstructs both indexes, reruns #562's authenticated census, and compares
exact artifact bytes. It uses no model assets, inference, training, or game
execution.

This inventory does not cover every #512–#555 historical evaluation source.
The unresolved seed461 continuation population remains unresolved, and the #483
confirmation gate remains closed. No suite generation or fresh-confirmation
claim is authorized.
