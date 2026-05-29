# Guarded w2 Root-vs-Learned Prior Persistence Review

## Outcome

- classification: `root_override_effective_but_training_nonpersistent`
- decision: `write_guarded_w2_root_vs_learned_prior_persistence_spec`
- conclusion: root-only prior overrides were strong enough to flip guarded `w2` row `002`, but the learned prior-calibration retry did not persist that flip through final search selection and collapsed arena

## Root Override Review

- persistent root interventions at required budgets `[384, 1200]`: `zero_wrong_extra_turn_prior, swap_reference_and_wrong`

## Learned Retry Review

- row `002` prior improved: `True`
- row `002` visit share improved: `True`
- row `002` selected move still not fixed: `True`
- row `002` Q margin worsened: `True`
- arena collapsed: `True`
- train best-val-loss delta: `-0.001236`

## Recommendation

- do not retry replay-side prior reweighting on guarded `w2`
- next branch should stay diagnostic and explain why root-only prior correction does not persist through the learned policy/search stack
