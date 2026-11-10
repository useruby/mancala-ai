# #561 historical exclusion inventory, version 1

This is a read-only, versioned inventory for the future holdback protocol. It
does not generate a real opening suite, execute games or searches, train a
model, or change serving behavior.

## Contents

- `source-manifest.json` pins merged #564 as the starting point, #511's exact
  registration bytes, and its complete `exclusion_source_sha256` mapping.
- `canonical-index.jsonl.gz` and `kalah-v3-float32-index.jsonl.gz` are
  deterministic gzip indexes. Each occurrence records its identity, source,
  and source provenance. Snapshot-derived identities are labelled as imported;
  arena/self-play states are labelled as independently replayed in the report.
- `reconciliation-report.json` gives source counts, overlap with the
  independently rebuilt #562–#564 subtotal, additions, and overlap with
  #563's #503/#504 family.

## Holdback bundle reference

The holdback verifier accepts the inventory through a versioned reference
document, rather than requiring expanded rule-state rows:

```json
{
  "schema": "seed561_holdback_inventory_reference_v1",
  "manifest_path": "historical/source-manifest.json",
  "manifest_sha256": "946fc52c75fa2e6c0499811fcb97ff773ea00ec6ec9b6d1bfb3ec1c2466a95ad"
}
```

Place the manifest and both indexes at the manifest's relative paths inside the
bundle. The trusted digest is pinned by `holdback_arena`; callers can also pass
an independently supplied trusted digest to `verify`. The loader checks both
scheme bindings, the decompressed JSONL digest, occurrence and distinct counts,
and derives the exclusion identity sets used while rebuilding and checking the
suite. Equivalent gzip headers/compression are accepted when the JSONL matches;
changed index content or a locally regenerated manifest is rejected.

## Verification

From a checkout containing the authenticated evidence files, run:

```bash
python -m ml.alphazero_lite.seed561_historical_exclusion_inventory \
  --root . \
  --out docs/data/seed561-historical-exclusion-inventory-v1
```

The verifier checks all 2,073 source byte hashes against #511's statically
pinned registration, runs the read-only #562–#564 verifier, reconstructs both
identity schemes, and compares the generated artifacts byte-for-byte. It does
not repair or rewrite files unless the explicit `--write` option is provided.

## Results and limitations

The #562–#564 subtotal is 302,402 identities under both schemes. The
#511-registered older-source union adds 231,631 identities, yielding 534,033
known identities under each scheme. The older sources comprise verified
#482–#484 identity snapshots plus independent replay of #488/#499/#505/#508
games and #500 self-play state rows. #500's 11,896 intersections with #563's
#503/#504 source family are already represented and are not counted again.

Seed461 continuation overlap remains unresolved. Other historical source
populations outside this exact #511-pinned scope remain uncovered with unknown
counts. The #483 confirmation gate remains closed.
