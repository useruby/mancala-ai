# AlphaZero Lite Capture 002 Next-Branch Gap

## Current Frontier

- The trace harmonization branch advanced `capture_002` from `trace_insufficient` to `selection_score_pressure_confirmed`.
- The follow-on selection-pressure ablation branch then stopped at:
  - classification: `selection_pressure_persists`
  - decision: `stop_002_selection_pressure_ablation_inconclusive`

## What This Means

- We now have enough evidence to say the harmonized early selection-score lead persists across the tested search-pressure variants.
- The likely next conceptual branch is a non-residual mechanism review, consistent with the existing residual-ablation decision vocabulary:
  - `write_002_non_residual_mechanism_review_spec`

## Repo Gap

- The repo does not currently implement a runner or artifact family for that next branch.
- Existing downstream modules that might look nearby are not valid entrypoints here:
  - `capture_002_selection_score_residual_audit.py` requires a prior-pressure audit artifact with classification `selection_score_residual_lead` and trace artifacts classified `unresolved`.
  - `capture_002_residual_ablation.py` requires a residual-audit artifact plus the same unresolved-trace lineage.
- Our current branch no longer satisfies those contracts because the harmonized traces classify as `selection_score_pressure_confirmed`, not `unresolved`.

## Practical Conclusion

- No further supported automated branch exists in the current repo from this state.
- Any additional continuation now requires new product code for a dedicated `capture_002` non-residual review branch rather than more orchestration glue.
