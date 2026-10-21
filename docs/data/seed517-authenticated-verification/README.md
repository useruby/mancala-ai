# Seed517 authenticated verification correction

This append-only correction authenticates the actual checked-out tablebase
bytes as either the registered LFS pointer or the registered object. When the
pointer is present, the separately materialized object must also match the
registered object SHA-256. An unrelated or missing checkout artifact cannot be
rescued by an available object cache. Seed515's replay and semantic checks are
retained.

The seed516 verifier CLI prints its report without writing by default. `--output
PATH` writes a new file only and refuses to replace an existing path. No
seed514–516 publication receipt or evidence was regenerated.

See `seed517-correction-receipt.json` for tests, quality checks, the physical
copy execution, dependency paths, and before/after preservation inventories.
No searches, games, training, deployment, promotion, or artifact replacement
were performed.
