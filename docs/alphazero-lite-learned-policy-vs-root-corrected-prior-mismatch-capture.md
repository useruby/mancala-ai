# Learned Policy vs Root-Corrected Prior Mismatch Capture

## Outcome

- classification: `reference_underweighting_and_wrong_extra_turn_overweighting_confirmed`
- decision: `write_learned_policy_mismatch_note`

## Row 002 Move Probabilities

| move | learned policy | zero-wrong corrected prior | swap corrected prior |
| --- | --- | --- | --- |
| `0` | `0.2972` | `0.4344` | `0.2972` |
| `1` | `0.3202` | `0.468` | `0.3202` |
| `2` | `0.3158` | `0.0` | `0.0668` |
| `4` | `0.0668` | `0.0976` | `0.3158` |

## Row 002 Rankings

- `original_prior`: policy `[1, 2, 0, 4]`, selection_score `[2, 0, 1, 4]`, final_visit `[2, 4, 1, 0]`, selected `2`
- `zero_wrong_extra_turn_prior`: policy `[1, 0, 4, 2]`, selection_score `[4, 1, 0, 2]`, final_visit `[4, 1, 0, 2]`, selected `4`
- `swap_reference_and_wrong`: policy `[1, 4, 0, 2]`, selection_score `[4, 1, 0, 2]`, final_visit `[4, 1, 0, 2]`, selected `4`

## Row 003 Preservation

- `original_prior`: selected `1`, reference visit share `0.9401`
- `zero_wrong_extra_turn_prior`: selected `1`, reference visit share `0.9688`
- `swap_reference_and_wrong`: selected `1`, reference visit share `0.9453`

## Conclusion

- row `002` needs root correction because the learned guarded-`w2` policy both underweights the reference move and leaves too much mass on the wrong extra-turn branch
- row `003` remains stable under the same corrections, so the mismatch is row-specific rather than a generic extra-turn suppression issue
