# Seed 627 authenticated membership snapshots

The `snapshots/` subdirectory contains immutable, create-only snapshots of B512 consumed
training membership, S615 consumed training membership, and protected-holdout
membership. `source-identity-lineage-manifest.json` records raw historical input
digests, pinned verifier outputs, dependency identities, row counts and snapshot
digests. Rows use Seed 624's exposure-screening fields:
`canonical_state`, `canonical_identity`, and
`production_float32_identity`; they also retain source membership indexes and
lineage evidence.

Verify the copied evidence bundle with:

```sh
.venv/bin/python -m ml.alphazero_lite.verify_seed627_membership_snapshots \
  /path/to/copied/seed627-authenticated-membership/snapshots
```

The verifier reconstructs from historical source records and traces; the
manifest's status is not treated as proof. It checks exact row contents, both
identity encodings, consumed-index reconstruction, aggregates and SHA-256
digests. Source-game/opening provenance is authenticated but not crosswalkable
to Seed 624 identifiers. P-E4 inherited ancestry remains unresolved.
