# AlphaZero-lite Learned Policy Mismatch Note

## Outcome

- classification: `reference_underweighting_and_wrong_extra_turn_overweighting_confirmed`
- decision: `note_learned_policy_mismatch_confirmed`

## Conclusion

- The guarded-w2 learned policy on row 002 both underweights the reference move and overweights the wrong extra-turn move, while row 003 remains stable under the same root corrections.
- focus row: `capture_available-002`
- preservation row: `capture_available-003`

Supporting artifact:

- `/tmp/azlite_learned_policy_vs_root_corrected_prior_mismatch_capture/learned-policy-vs-root-corrected-prior-mismatch-capture/learned_policy_vs_root_corrected_prior_mismatch_capture.json`
