# Seed 622 — historical training provenance and contamination

This is a read-only audit of the 768 unique exact-oracle-covered roots in the
authenticated Seed 621 inventory. It performs no search, solving, training, or
game execution. Reproduce with:

```sh
python -m ml.alphazero_lite.seed622_training_provenance_audit
python -m ml.alphazero_lite.verify_seed622_training_provenance_audit
python -m unittest ml.alphazero_lite.test_seed622_training_provenance_audit
```

`state-classification-ledger.jsonl` retains every Seed 621 root, canonical and
production `kalah_v3` little-endian float32 identities, provenance labels,
exposure categories, and the limitations attached to each conclusion.
`report.json` summarizes the trace-authenticated training lineage and counts;
`verification-report.json` records the independent verifier result.

## Training lineage

B512 and S615 share the P-E4 parent checkpoint (SHA-256
`2ecc98ac…edbad`). B512 uses 2,048 Seed 599 membership rows over 512 updates
with 32-row minibatches. Its frozen trace shows each of the 2,048 row indexes
was consumed. S615 uses 2,048 states and the same frozen 512-update/32-row
index schedule, mapped against sorted S615 membership; its trace likewise
shows all 2,048 states consumed. Both use `kalah_v3` inputs. The frozen
executors do not define symmetry augmentation or replay expansion. The state
inventories selected from #580/#585 are actual training inputs only where they
occur in registered membership and recorded batches; the full source archives
are not treated as consumed training data.

Seed 599 membership derives from retained completed-game examples in #580/#585.
The membership rows carry source artifact, game, opening, and row provenance;
the source archives are larger than the selected membership. Those game/opening
relationships remain distinct risk dimensions and are not treated as exact
state leakage. The available provenance identifiers do not support a complete
crosswalk from the Seed 621 `source_path`/`opening_index` keys to every training
game/opening key. P-E4's earlier training ancestry is also not backed by an
identity-complete consumed-state ledger.

The Seed 599 preparation binds its source archive hash map, and this audit
rechecks all 252 listed #580/#585 source artifacts against that map. The
training freezes bind the historical executor/source hashes; receipts bind
the parent, membership, batch schedule, and execution freeze. The independent
Seed 621 source batch and canonical source-record hashes are also rechecked for
every inventory root.

## Results

| Classification | Count | Agreement | Disagreement |
| --- | ---: | ---: | ---: |
| `excluded_confirmed_exposure` | 4 | 4 | 0 |
| `cleared_under_verified_contract` | 0 | 0 | 0 |
| `unresolved_insufficient_provenance` | 764 | 745 | 19 |

The raw unresolved inventory contains 675 source-game/opening clusters, an
inverse-Simpson effective cluster count of 619.6 (largest observed cluster: 2
states). This descriptive figure does not resolve contamination and is not an
eligible effective sample size. By historical oracle cohort, #557 has 251
unresolved agreements and 4 disagreements, plus 1 excluded agreement; #559
has 494 unresolved agreements and 15 disagreements, plus 3 excluded agreements.

Four roots overlap the known S615 or protected holdout populations. The other
764 have no exact canonical or production-float32 membership match, but this
does not establish clearance: inherited P-E4 exposure and complete historic
source provenance are unavailable. The 19 disagreements therefore do not
survive as *cleared* disagreements. No eligible population can meet 64 total
states and 16 per prior class; source-cluster independence/effective sample
size is not estimable for a cleared cohort. The existing registration must not
be relaxed to manufacture clearance.

By direct contamination category, 2 roots match S615 training identities, 2
overlap the protected holdout, and 764 have neither membership match. There are
no direct B512 identity matches. The source-game/opening relationship remains
unresolved for all 768 roots because the historical identifiers cannot be
completely crosswalked; this category is kept separate from direct state
identity overlap.

**Recommendation: `stop`.** The B512/S615 consumed-state schedules are
reconstructable, but the shared parent’s inherited exposure is not. Available
historical records do not resolve that gap into a verifiable exclusion
contract for this cohort. No experimental cohort is selected or frozen.

## Verification boundary

The independent verifier authenticates Seed 621 inventory identity, both
training memberships, protected holdout, run traces/receipts/checkpoint hashes,
all 512 batch assignments, canonical identities, encoded float32 identities,
and every state-level classification and aggregate count. It fails closed on
omitted membership rows, trace/index changes, input identity mismatches, and
false clearance. Historical published artifacts are read-only and their
published digests are checked; this audit adds no evidence to or edits those
artifacts.

The historical Seed 615 independent verifier is not runnable in the current
workspace: its pinned PyTorch 2.13.0+cu130 environment is unavailable (`torch`
is not installed). The retained Seed 615 receipt, trace, integrity-report,
freeze, membership, and checkpoint hashes are bound into this audit, but this
does not substitute for rerunning that pinned-environment verification. The
absence of that dependency is recorded as an unresolved verification
limitation, alongside the missing P-E4 training-state ancestry.
