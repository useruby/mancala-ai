# AlphaZero-lite Corrected-Reference Targeted Hard-State Replay Results

## Outcome

The corrected-reference targeted hard-state replay lane ran to completion after fixing the `arena.py` subprocess import path.

All three replay weights completed:

- `w1`
- `w2`
- `w4`

No variant improved arena strength over the incumbent.

## Dataset Summary

- selected mined states: `64`
- labeled rows: `128`
- mining source: corrected-reference forensic failures only

Family composition of the mined set:

- `capture_available`: `14`
- `high_imbalance`: `9`
- `high_value_swing`: `9`
- `incumbent_proxy_disagreement`: `9`
- `opening_plies_1_8`: `9`
- `starvation_pressure`: `11`
- `sparse_endgame`: `3`

Key artifacts:

- run root: `/tmp/azlite_corrected_reference_targeted_hard_state_replay/corrected-reference-targeted-hard-state-replay`
- family quota summary: `/tmp/azlite_corrected_reference_targeted_hard_state_replay/corrected-reference-targeted-hard-state-replay/reports/family_quota_summary.json`
- report template: `/tmp/azlite_corrected_reference_targeted_hard_state_replay/corrected-reference-targeted-hard-state-replay/reports/hard_state_replay_experiment_report.json`

## Results Table

| variant | hard_state_weight | hard_state_average_regret | hard_state_value_calibration_mae | capture_available_average_regret | arena_score | arena_ci_low | arena_ci_high | unstable_decision | mcts1200_score | benchmark_pass | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `w1` | `1` | `0.0904` | `0.3528` | `0.0225` | `0.5000` | `0.4119` | `0.5881` | `true` | `0.6167` | `false` | slightly better calibrated than `w4` and better on `capture_available` than `w2`, but no arena lift |
| `w2` | `2` | `0.0904` | `0.3447` | `0.0303` | `0.5000` | `0.4119` | `0.5881` | `true` | `0.6167` | `false` | best value calibration, but weakest corrected-reference regret profile of the three |
| `w4` | `4` | `0.0841` | `0.3690` | `0.0211` | `0.5000` | `0.4119` | `0.5881` | `true` | `0.6167` | `false` | best hard-state regret overall and best capture-available regret, but still no arena lift |

## Interpretation

What improved:

- the corrected-reference mining lane now runs end to end
- the replay dataset is no longer contaminated by reference-integrity errors
- the mined set is meaningfully tilted toward the intended corrected failure families
- `w4` achieved the best overall hard-state regret and the best `capture_available` regret of the three variants
- `w2` achieved the best value calibration MAE of the three variants
- `w1` landed between `w2` and `w4`: tied `w2` on overall regret, but with weaker calibration than `w2` and weaker overall regret than `w4`

What did not improve:

- all three variants tied the same arena score: `0.5000`
- all three variants tied the same `MCTS1200` score: `0.6167`
- all three variants failed benchmark promotion checks because the arena gate failed

Practical readout:

- corrected-reference-only replay is cleaner than the earlier replay branch
- but it still reproduces the same basic pattern: hard-state metrics can move without producing any arena gain
- the replay weight tradeoff remains the same shape as before:
  - more weight helps regret a bit (`w4`)
  - moderate weight helps calibration more (`w2`)
  - lighter replay weight (`w1`) does not recover arena strength either

## Recommendation

Next action: **do not continue replay-weight sweeps from this lane. Return to corrected-reference failure-family diagnostics or policy-target construction work.**

Why:

- the lane answered the intended question under corrected references
- the result is still negative on arena strength
- further replay-side reweighting is not justified by these outcomes alone
