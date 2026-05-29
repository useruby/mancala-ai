# PUCT1200 Root-Prior Transform Validation

## Context

This run validates the narrow root-prior transform on the broader forensic hard-state suite using direct artifact PUCT search at 1200 simulations.

- artifact: `/tmp/azlite_rule_conditioned_opening_full_guarded/rule-conditioned-opening-full-guarded/w2/versions/aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1`
- validation path: `ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json`
- root prior transform: `seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5`
- validation report: `/tmp/azlite_puct1200_root_prior_transform_validation/hard_state_validation_puct1200.json`

## Overall

- position count: `224`
- policy top1 agreement: `0.0`
- average regret: `0.0907`
- value calibration mae: `0.4405`
- bucket count: `8`

## Notes

- overall summary: `{'positions': 224, 'top1_agreement': 0.0, 'average_regret': 0.0907, 'blunder_rate': 0.4955, 'value_calibration_mae': 0.4405}`
