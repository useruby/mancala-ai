# AlphaZero-Lite Terminal-Outcome Replay-Balance Results

- classification: `pr155_value_training_not_reproducible`

## Replay Weighting

| stratum | rows | gradient fraction |
|---|---:|---:|
| player player_0 | 21479 | 0.5255 |
| player player_1 | 19392 | 0.4745 |

## Symmetry

- checked rows: `40871`
- invariant failures: `none`

## Training Lanes

| lane | sampler | epochs | probe pass | stop reasons |
|---|---|---:|---:|---|
| row_uniform_repro | row_uniform | 2 | False | player_0_sign_regressed, artifact_768_768_changed_rate, artifact_1200_1200_changed_rate, artifact_1200_256_changed_rate, symmetry_residual_materially_worse |
| game_balanced_e1 | game_balanced | 1 | False | player_0_sign_regressed, correlation_not_improved, player_delta_imbalance_not_lower, artifact_768_768_changed_rate, symmetry_residual_materially_worse |
| game_balanced_e2 | game_balanced | 2 | False | player_0_sign_regressed, player_delta_imbalance_not_lower, artifact_768_768_changed_rate, artifact_1200_1200_changed_rate, artifact_1200_256_changed_rate, symmetry_residual_materially_worse |
| game_seat_balanced_e1 | seat_outcome_balanced | 1 | True | none |
| game_seat_balanced_e2 | seat_outcome_balanced | 2 | False | artifact_768_768_changed_rate, artifact_1200_1200_changed_rate, artifact_1200_256_changed_rate |

## Row-Uniform Reproduction

- exact weights match: `False`
- max absolute value difference: `0.02441933`
- value sign agreement: `0.996582`
- value MAE difference: `0.00445703`
- failure reasons: `weights_not_identical, value_predictions_not_reproduced, search_changed_rate_not_reproduced`

## Value And Search Diagnostics

| lane | group | MAE | sign accuracy | Pearson | Spearman | mean delta |
|---|---|---:|---:|---:|---:|---:|
| row_uniform_repro | overall | 0.7028 | 0.7224 | 0.5842 | 0.5270 | +0.0428 |
| row_uniform_repro | player_0 | 0.7106 | 0.7009 | 0.5691 | 0.4814 | +0.0913 |
| row_uniform_repro | player_1 | 0.6939 | 0.7469 | 0.6033 | 0.5776 | -0.0125 |
| row_uniform_repro | phase:opening | 0.8708 | 0.5828 | 0.3012 | 0.2188 | +0.0998 |
| row_uniform_repro | phase:mid | 0.7591 | 0.6910 | 0.4975 | 0.4364 | +0.0305 |
| row_uniform_repro | phase:late | 0.5250 | 0.8511 | 0.7895 | 0.7131 | +0.0211 |
| game_balanced_e1 | overall | 0.7252 | 0.7080 | 0.5612 | 0.5079 | +0.0552 |
| game_balanced_e1 | player_0 | 0.7375 | 0.6885 | 0.5552 | 0.4697 | +0.1120 |
| game_balanced_e1 | player_1 | 0.7112 | 0.7302 | 0.5768 | 0.5494 | -0.0095 |
| game_balanced_e1 | phase:opening | 0.8876 | 0.5948 | 0.2687 | 0.1929 | +0.0961 |
| game_balanced_e1 | phase:mid | 0.7802 | 0.6682 | 0.4886 | 0.4337 | +0.0549 |
| game_balanced_e1 | phase:late | 0.5528 | 0.8301 | 0.7559 | 0.6901 | +0.0291 |
| game_balanced_e2 | overall | 0.7020 | 0.7144 | 0.5729 | 0.5106 | +0.0534 |
| game_balanced_e2 | player_0 | 0.7097 | 0.6959 | 0.5645 | 0.4790 | +0.1368 |
| game_balanced_e2 | player_1 | 0.6933 | 0.7354 | 0.6003 | 0.5710 | -0.0414 |
| game_balanced_e2 | phase:opening | 0.8651 | 0.5882 | 0.2922 | 0.2012 | +0.1218 |
| game_balanced_e2 | phase:mid | 0.7541 | 0.6853 | 0.4959 | 0.4280 | +0.0477 |
| game_balanced_e2 | phase:late | 0.5327 | 0.8315 | 0.7695 | 0.6967 | +0.0165 |
| game_seat_balanced_e1 | overall | 0.7322 | 0.7295 | 0.5727 | 0.5259 | +0.0011 |
| game_seat_balanced_e1 | player_0 | 0.7424 | 0.7202 | 0.5589 | 0.4688 | -0.0144 |
| game_seat_balanced_e1 | player_1 | 0.7206 | 0.7401 | 0.5826 | 0.5633 | +0.0187 |
| game_seat_balanced_e1 | phase:opening | 0.9008 | 0.6078 | 0.2349 | 0.1806 | -0.0035 |
| game_seat_balanced_e1 | phase:mid | 0.7977 | 0.6830 | 0.4749 | 0.4270 | -0.0060 |
| game_seat_balanced_e1 | phase:late | 0.5428 | 0.8652 | 0.7960 | 0.7211 | +0.0127 |
| game_seat_balanced_e2 | overall | 0.7152 | 0.7385 | 0.5894 | 0.5437 | -0.0009 |
| game_seat_balanced_e2 | player_0 | 0.7227 | 0.7307 | 0.5742 | 0.4851 | -0.0195 |
| game_seat_balanced_e2 | player_1 | 0.7065 | 0.7474 | 0.6017 | 0.5838 | +0.0203 |
| game_seat_balanced_e2 | phase:opening | 0.8840 | 0.6220 | 0.3076 | 0.2411 | +0.0141 |
| game_seat_balanced_e2 | phase:mid | 0.7810 | 0.6938 | 0.4856 | 0.4364 | -0.0165 |
| game_seat_balanced_e2 | phase:late | 0.5252 | 0.8687 | 0.8087 | 0.7277 | +0.0087 |

| lane | budget | artifact changed rate (95% CI) | budget-only changed rate (95% CI) | artifact KL | budget KL |
|---|---|---:|---:|---:|---:|
| row_uniform_repro | 384:256 | 0.0938 (0.0638, 0.1357) | 0.1055 (0.0735, 0.1491) | 0.0419 | 0.0175 |
| row_uniform_repro | 768:256 | 0.1133 (0.0800, 0.1580) | 0.1797 (0.1375, 0.2313) | 0.0461 | 0.0736 |
| row_uniform_repro | 768:768 | 0.1094 (0.0768, 0.1535) | 0.0000 (0.0000, 0.0148) | 0.1107 | 0.0000 |
| row_uniform_repro | 1200:1200 | 0.1016 (0.0703, 0.1447) | 0.0000 (0.0000, 0.0148) | 0.0458 | 0.0000 |
| row_uniform_repro | 1200:256 | 0.1016 (0.0703, 0.1447) | 0.2266 (0.1795, 0.2817) | 0.0458 | 0.1280 |
| row_uniform_repro | 256:768 | 0.0977 (0.0670, 0.1402) | 0.1797 (0.1375, 0.2313) | 0.0438 | 0.0913 |
| game_balanced_e1 | 384:256 | 0.0859 (0.0574, 0.1267) | 0.1055 (0.0735, 0.1491) | 0.0303 | 0.0175 |
| game_balanced_e1 | 768:256 | 0.0938 (0.0638, 0.1357) | 0.1797 (0.1375, 0.2313) | 0.0332 | 0.0736 |
| game_balanced_e1 | 768:768 | 0.0977 (0.0670, 0.1402) | 0.0000 (0.0000, 0.0148) | 0.0979 | 0.0000 |
| game_balanced_e1 | 1200:1200 | 0.0742 (0.0480, 0.1130) | 0.0000 (0.0000, 0.0148) | 0.0371 | 0.0000 |
| game_balanced_e1 | 1200:256 | 0.0742 (0.0480, 0.1130) | 0.2266 (0.1795, 0.2817) | 0.0371 | 0.1280 |
| game_balanced_e1 | 256:768 | 0.0898 (0.0606, 0.1312) | 0.1797 (0.1375, 0.2313) | 0.0314 | 0.0913 |
| game_balanced_e2 | 384:256 | 0.1055 (0.0735, 0.1491) | 0.1055 (0.0735, 0.1491) | 0.0535 | 0.0175 |
| game_balanced_e2 | 768:256 | 0.1328 (0.0966, 0.1799) | 0.1797 (0.1375, 0.2313) | 0.0592 | 0.0736 |
| game_balanced_e2 | 768:768 | 0.1016 (0.0703, 0.1447) | 0.0000 (0.0000, 0.0148) | 0.1081 | 0.0000 |
| game_balanced_e2 | 1200:1200 | 0.1016 (0.0703, 0.1447) | 0.0000 (0.0000, 0.0148) | 0.0543 | 0.0000 |
| game_balanced_e2 | 1200:256 | 0.1016 (0.0703, 0.1447) | 0.2266 (0.1795, 0.2817) | 0.0543 | 0.1280 |
| game_balanced_e2 | 256:768 | 0.1016 (0.0703, 0.1447) | 0.1797 (0.1375, 0.2313) | 0.0542 | 0.0913 |
| game_seat_balanced_e1 | 384:256 | 0.0312 (0.0159, 0.0604) | 0.1055 (0.0735, 0.1491) | 0.0180 | 0.0175 |
| game_seat_balanced_e1 | 768:256 | 0.0547 (0.0329, 0.0897) | 0.1797 (0.1375, 0.2313) | 0.0230 | 0.0736 |
| game_seat_balanced_e1 | 768:768 | 0.0547 (0.0329, 0.0897) | 0.0000 (0.0000, 0.0148) | 0.0135 | 0.0000 |
| game_seat_balanced_e1 | 1200:1200 | 0.0508 (0.0299, 0.0849) | 0.0000 (0.0000, 0.0148) | 0.0286 | 0.0000 |
| game_seat_balanced_e1 | 1200:256 | 0.0508 (0.0299, 0.0849) | 0.2266 (0.1795, 0.2817) | 0.0286 | 0.1280 |
| game_seat_balanced_e1 | 256:768 | 0.0508 (0.0299, 0.0849) | 0.1797 (0.1375, 0.2313) | 0.0171 | 0.0913 |
| game_seat_balanced_e2 | 384:256 | 0.0664 (0.0419, 0.1038) | 0.1055 (0.0735, 0.1491) | 0.0328 | 0.0175 |
| game_seat_balanced_e2 | 768:256 | 0.1016 (0.0703, 0.1447) | 0.1797 (0.1375, 0.2313) | 0.0515 | 0.0736 |
| game_seat_balanced_e2 | 768:768 | 0.1016 (0.0703, 0.1447) | 0.0000 (0.0000, 0.0148) | 0.0570 | 0.0000 |
| game_seat_balanced_e2 | 1200:1200 | 0.1055 (0.0735, 0.1491) | 0.0000 (0.0000, 0.0148) | 0.0627 | 0.0000 |
| game_seat_balanced_e2 | 1200:256 | 0.1055 (0.0735, 0.1491) | 0.2266 (0.1795, 0.2817) | 0.0627 | 0.1280 |
| game_seat_balanced_e2 | 256:768 | 0.0820 (0.0543, 0.1221) | 0.1797 (0.1375, 0.2313) | 0.0322 | 0.0913 |

## Realized Sampling

| lane | samples | game min/max | game CV | player/outcome strata |
|---|---:|---:|---:|---|
| row_uniform_repro | 69694 | 28/126 | 0.2610 | player_0:draw=2738, player_0:loss=14210, player_0:win=19656, player_1:draw=2656, player_1:loss=16284, player_1:win=14150 |
| game_balanced_e1 | 34847 | 40/41 | 0.0056 | player_0:draw=1496, player_0:loss=6437, player_0:win=10505, player_1:draw=1426, player_1:loss=8521, player_1:win=6462 |
| game_balanced_e2 | 69694 | 80/81 | 0.0039 | player_0:draw=2982, player_0:loss=12798, player_0:win=20965, player_1:draw=2863, player_1:loss=17083, player_1:win=13003 |
| game_seat_balanced_e1 | 34847 | 8/189 | 0.9234 | player_0:draw=5807, player_0:loss=5808, player_0:win=5808, player_1:draw=5808, player_1:loss=5808, player_1:win=5808 |
| game_seat_balanced_e2 | 69694 | 29/355 | 0.9161 | player_0:draw=11616, player_0:loss=11616, player_0:win=11615, player_1:draw=11615, player_1:loss=11616, player_1:win=11616 |

## Staged Strength

### Medium

| lane | budget | candidate-current DS |
|---|---|---:|
| row_uniform_repro | 384:256 | -0.1719 |
| row_uniform_repro | 768:256 | +0.3125 |
| row_uniform_repro | 768:768 | -0.7578 |
| row_uniform_repro | 1200:1200 | +0.0352 |
| row_uniform_repro | 1200:256 | -0.1680 |
| row_uniform_repro | 256:768 | +0.0000 |
| game_seat_balanced_e1 | 384:256 | -0.6562 |
| game_seat_balanced_e1 | 768:256 | +0.0352 |
| game_seat_balanced_e1 | 768:768 | -0.5898 |
| game_seat_balanced_e1 | 1200:1200 | -0.1406 |
| game_seat_balanced_e1 | 1200:256 | +0.0312 |
| game_seat_balanced_e1 | 256:768 | +0.1602 |
- fixed-large and held-out: not reached

## Stop Reasons

- pr155_value_training_not_reproducible
- no balanced lane passed medium carry criteria
