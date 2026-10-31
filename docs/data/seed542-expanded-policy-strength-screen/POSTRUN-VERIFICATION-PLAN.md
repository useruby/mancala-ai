# Post-run verification and analysis sequence

## Gate A: completeness only, no aggregation

Run the separate read-only verifier module
`ml.alphazero_lite.verify_seed542_expanded_policy_strength_completeness` from
the frozen registered Python environment. Its SHA-256 is
`3fd3447d2a06d1cb4b6a86c72f3fd4b5753a0bf3714256843867fbb4b72ae36a`.

It authenticates the original and continuation freezes, suite import, overlap
disclosure, training receipts, current P and candidate artifacts, the full
search/runtime contract, all command/completion receipts, all 1,024 replayed
seat pairs, and the 2,048-row ledger. It explicitly performs no score
aggregation, bootstrap, or decision calculation.

Frozen identities at the gate:

- Continuation registration SHA-256:
  `c749bc760019fddc3a53755c41b4ea1bf28e87251cfc44c9f0cb71ac1088cb6b`.
- Continuation freeze SHA-256:
  `9f44fdf3a9d4a5c87a29aa85aa4bb3677c1aa1ac7b8dd204affb61292440b4d4`.
- Complete outcome ledger SHA-256:
  `e3c54882fec44163fe1d8166241883728a491372f7c98021d2faf75c52f15e13`.

## Gate B: physical-copy completeness verification

After Gate A passes, run the same verifier from a physically separate checkout
and unrelated working directory, with the original screen worktree inaccessible.
Provide the materialized registered tablebase in the copy's repo-local `.tmp`
runtime path. Snapshot evidence bytes and modification times before and after.
No official analysis is produced until both gates pass.

## Registered analysis, only after both gates

Use opening-cluster scores and the frozen 20,000-resample bootstrap (seed 545,
replacement with multiplicity, NumPy quantile method `linear`, one-sided 0.05
lower bound, descriptive 0.025/0.975 interval). Advance only if all 2,048 games
pass independent validation and the lower bound is strictly greater than 0.50.
Advancement permits proposing a separate 1,200-simulation evaluation only.

`runner-provisional-analysis.json` and
`runner-provisional-per-opening.jsonl` are retained separately as pre-verifier
runner outputs; they are not the registered result and will not be used for the
decision.
