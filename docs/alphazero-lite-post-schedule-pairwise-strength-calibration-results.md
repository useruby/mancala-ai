# AlphaZero-Lite Post-Schedule Pairwise Strength Calibration Results

**Date**: 2026-06-30

**Classification**: `pairwise_still_too_weak`

## Input Hashes And Row Counts

| Input | Path | SHA256 | Rows |
|---|---|---|---|
| current_weights | model-artifact/current/weights.json | 8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a | n/a |
| current_artifact | model-artifact/current/weights.json | 8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a | n/a |
| init_checkpoint | /tmp/azlite_balanced_opening_puct_replay/balanced_w8s4_policy_head/checkpoint_epoch1.npz | 18dacb5ef38602a77abf3927dc8748ac167c2ca91237385ec010ab27a123adc9 | n/a |
| pr138_pairwise_targets | /tmp/azlite_post_schedule_pairwise_puct_update/validated_pairwise_targets.jsonl | 10648467a2cfb690933dcffaf17309b2ecdc8de5c095773e7fb7e2fa1bddb871 | 1130 |
| pr138_behavior_anchors | /tmp/azlite_post_schedule_pairwise_puct_update/behavior_anchor_rows.jsonl | 7ecc9a70c0a46bd3c726fb8aab570dd68c80eaf20ebc1539e5a4f517236418f8 | 8000 |
| pr138_probe_rows | /tmp/azlite_post_schedule_pairwise_puct_update/probe_rows.jsonl | 0e6b6c95c9a8ddd9e6713f41c2a70662381c4524131fec79c60b89207a4c4c5a | 7130 |
| local_pairwise_targets | /tmp/azlite_post_schedule_pairwise_strength_calibration/validated_pairwise_targets.jsonl | 10648467a2cfb690933dcffaf17309b2ecdc8de5c095773e7fb7e2fa1bddb871 | 1130 |
| local_behavior_anchors | /tmp/azlite_post_schedule_pairwise_strength_calibration/behavior_anchor_rows.jsonl | 7ecc9a70c0a46bd3c726fb8aab570dd68c80eaf20ebc1539e5a4f517236418f8 | 8000 |
| local_probe_rows | /tmp/azlite_post_schedule_pairwise_strength_calibration/probe_rows.jsonl | 0e6b6c95c9a8ddd9e6713f41c2a70662381c4524131fec79c60b89207a4c4c5a | 7130 |
| medium_suite | /tmp/azlite_opening_suite/medium_eval.jsonl | 57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04 | 128 |
| fixed_large_suite | /tmp/azlite_opening_suite/large_eval.jsonl | ebc86f053a8a4c12f3e937b5acfe37df7b0d2f1f0002969dcac077a9f99d65a4 | 384 |
| heldout_suites:heldout_seed43_large | /tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed43_large.jsonl | 5e0ed96ba56f99318c32a139692309759659da16a0a98c590e35f7e2496cade9 | 384 |
| heldout_suites:heldout_seed44_large | /tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed44_large.jsonl | 323f7abec32fb00d3b7b6b153ddd5692435755a3bc5340d8aa5aec32ca639620 | 384 |
| heldout_suites:heldout_seed45_large | /tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed45_large.jsonl | ca72c8b7fe1adf7229f81183321475033f8548dcdeb1c71fd16c85c35ce89cda | 384 |
| heldout_suites:heldout_seed46_large | /tmp/azlite_pr123_weighted_candidate_preflight/suites/heldout_seed46_large.jsonl | 7d8ef94fed48ec58ed11aab9818954915ba5158d8cd90291fb8b6970489b0f04 | 384 |
| heldout_suites:heldout_seed47_large | /tmp/azlite_pr123_weighted_candidate_preflight/suites/heldout_seed47_large.jsonl | 2b689b7444e14c7880833570059775184d823d5af0f31cf77ea2f0779dd3d03b | 384 |
| heldout_suites:heldout_seed48_large | /tmp/azlite_pr123_weighted_candidate_preflight/suites/heldout_seed48_large.jsonl | b3d6a64a3ae86fb46b8859e5b75899dd9703ad57bcd9de330e15ee674a56c262 | 384 |

- Pairwise target rows: `1130` expected `1130` match=`True`
- Behavior anchor rows: `8000` expected `8000` match=`True`
- Probe composition: `{"broad_anchor": 4000, "pairwise": 1130, "stability": 2000}` expected `{"broad_anchor": 4000, "pairwise": 1130, "stability": 2000}` match=`True`
- Dataset source: `reused_pr138`

## Promoted Search Schedule Confirmation

- Schedule manifest: `{"default_c_puct": 1.25, "overrides": {"768:768": 0.9}}`
- Root policy mode: `deterministic`
- Tactical root bias: `0.0`

## Pairwise/Anchor Weight Table

| Candidate | Pairwise w | Anchor w | Margin | Epochs | Source |
|---|---|---|---|---|---|
| current_ref | None | None | n/a | 0 | checked_in_current |
| pr138_best_ref | 1 | 4 | +0.1000 | 2 | reused_pairwise_margin010_kl4_e2 |
| pairwise_w4_kl2_margin010_e1 | 4 | 2 | +0.1000 | 1 | trained_strength_calibration |
| pairwise_w4_kl2_margin010_e2 | 4 | 2 | +0.1000 | 2 | trained_strength_calibration |
| pairwise_w8_kl2_margin010_e1 | 8 | 2 | +0.1000 | 1 | trained_strength_calibration |
| pairwise_w8_kl2_margin010_e2 | 8 | 2 | +0.1000 | 2 | trained_strength_calibration |
| pairwise_w8_kl1_margin010_e1 | 8 | 1 | +0.1000 | 1 | trained_strength_calibration |
| pairwise_w8_kl1_margin010_e2 | 8 | 1 | +0.1000 | 2 | trained_strength_calibration |
| pairwise_w16_kl2_margin010_e1 | 16 | 2 | +0.1000 | 1 | trained_strength_calibration |

## Training Loss Table

| Candidate | Pairwise loss | Behavior loss | Total loss | Validation loss | Checkpoint SHA256 | Artifact weights SHA256 | Delta norm |
|---|---|---|---|---|---|---|---|
| current_ref | n/a | n/a | n/a | n/a | n/a | 8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a | n/a |
| pr138_best_ref | +1.4099 | +1.3957 | +6.9926 | +7.0640 | 6482ce437799307fb641fbf1c3fbc38ad32ded09a6b66b724ee4c47be165b8c8 | 911a1415e2b4ee5fcd399b55cd1cf376c580613664ebc19fee88809fb6d1f4be | +0.0033 |
| pairwise_w4_kl2_margin010_e1 | +1.4123 | +1.3983 | +8.4459 | +8.7131 | 005e46ba15eb0ad5fdaa4ca224ce6fbf685470422c01219fc0742c1bd37426cf | 7fd68efaa7f4a5f156f4aef4b70c704532a2671e4c6fa46e861a1d66351a05d9 | +0.0017 |
| pairwise_w4_kl2_margin010_e2 | +1.4098 | +1.3957 | +8.4306 | +8.7051 | 176d919ad289b09a7fc857df6deef8ac83c1f71472074e3b8fad39fa7d6a7206 | 4833d8eeac3a09925bdc8dac3f044c3034f29fadcf5d0c9b6c2a4f5a0fbb1ab1 | +0.0034 |
| pairwise_w8_kl2_margin010_e1 | +1.4123 | +1.3983 | +14.0952 | +14.6332 | 3bd6031b38fff0f0a3715025e963020f6ca5bcf56720e67e45fb200c0cfe0ea6 | 1e029b211f1b1f754a7347067c6a905acd8a6a73236321a7037688df379256e3 | +0.0017 |
| pairwise_w8_kl2_margin010_e2 | +1.4098 | +1.3957 | +14.0697 | +14.6172 | 61b3d51acf058dcf8a4819051a1aa222a71418f36f711e569acd58e3008265ff | 3147d45a8b28ee3a16cbff9c7d227af3d7b70d9109fe53f46bb62800988ca7af | +0.0034 |
| pairwise_w8_kl1_margin010_e1 | +1.4123 | +1.3983 | +12.6970 | +13.2367 | 04a5ef715557c2208394bec782ef26a63fb6a8816d8d62f520a439b428f59844 | e21eb38fadc507a3789fe208193d7d0d0eaadb05ff1a2f10c8b0ab862d9019be | +0.0017 |
| pairwise_w8_kl1_margin010_e2 | +1.4098 | +1.3957 | +12.6740 | +13.2207 | 1a4d666d27e8db88cef36dda07816c6e6591488203982bf80b8be8cf6bf5f10e | 0118765e6577906fbff11bf4c5cbe2291bc311244a7f14b728c51a9ede911956 | +0.0034 |
| pairwise_w16_kl2_margin010_e1 | +1.4123 | +1.3983 | +25.3939 | +26.4734 | 2016d38190fd0597e606cb97a00251db6bccc6e2fea3e01896981a40101fa9e3 | bf87c797463a9b740e38ed9b0958d5af4a1075d13dd4f6b2b6edb9a7ed66c809 | +0.0017 |

## Probe Metrics Table

| Candidate | Success gain | Success rate | Margin improve | Stability preserve | Broad changed | Anchor KL | Max ctx drift | 384:256 P0 drift | Aborted |
|---|---|---|---|---|---|---|---|---|---|
| current_ref | +0.0000 | +0.0000 | +0.0000 | +1.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | False |
| pr138_best_ref | +0.0053 | +0.0053 | +0.0053 | +0.9995 | +0.0120 | +0.0000 | +0.0150 | +0.0089 | True |
| pairwise_w4_kl2_margin010_e1 | +0.0035 | +0.0035 | +0.0028 | +0.9995 | +0.0060 | +0.0000 | +0.0060 | +0.0060 | True |
| pairwise_w4_kl2_margin010_e2 | +0.0062 | +0.0062 | +0.0056 | +0.9995 | +0.0105 | +0.0000 | +0.0150 | +0.0060 | True |
| pairwise_w8_kl2_margin010_e1 | +0.0035 | +0.0035 | +0.0028 | +0.9995 | +0.0060 | +0.0000 | +0.0060 | +0.0060 | True |
| pairwise_w8_kl2_margin010_e2 | +0.0062 | +0.0062 | +0.0056 | +0.9995 | +0.0105 | +0.0000 | +0.0150 | +0.0060 | True |
| pairwise_w8_kl1_margin010_e1 | +0.0035 | +0.0035 | +0.0028 | +0.9995 | +0.0060 | +0.0000 | +0.0060 | +0.0060 | True |
| pairwise_w8_kl1_margin010_e2 | +0.0062 | +0.0062 | +0.0056 | +0.9995 | +0.0105 | +0.0000 | +0.0150 | +0.0060 | True |
| pairwise_w16_kl2_margin010_e1 | +0.0035 | +0.0035 | +0.0028 | +0.9995 | +0.0060 | +0.0000 | +0.0060 | +0.0060 | True |

## Aborted-Candidate Table

| Candidate | Reasons |
|---|---|
| pr138_best_ref | pairwise success gain < +0.03 |
| pairwise_w4_kl2_margin010_e1 | pairwise success gain < +0.03 |
| pairwise_w4_kl2_margin010_e2 | pairwise success gain < +0.03 |
| pairwise_w8_kl2_margin010_e1 | pairwise success gain < +0.03 |
| pairwise_w8_kl2_margin010_e2 | pairwise success gain < +0.03 |
| pairwise_w8_kl1_margin010_e1 | pairwise success gain < +0.03 |
| pairwise_w8_kl1_margin010_e2 | pairwise success gain < +0.03 |
| pairwise_w16_kl2_margin010_e1 | pairwise success gain < +0.03 |

## Evaluated-Candidate Table

| Candidate | Evaluated | Gate ran | Held-out delta 384:256 |
|---|---|---|---|
| current_ref | True | True | +0.0000 |

## Fixed Large DS Table

| Candidate | 384:256 | 768:256 | 768:768 | 1200:1200 | 1200:256 | 256:768 |
|---|---|---|---|---|---|---|
| current_ref | -0.3099 | -0.4049 | +0.6849 | +0.2812 | -0.1159 | -0.4049 |

## Held-Out Mean/Worst-Suite Table

| Candidate | Held-out mean 384:256 | Delta vs current | Held-out worst-suite 384:256 |
|---|---|---|---|
| current_ref | -0.2993 | +0.0000 | -0.3255 |

## Bootstrap CI For Candidate Minus Current

| Comparison | Mean | Lower 95% | Upper 95% |
|---|---|---|---|

## P0/P1 Split For 384:256

| Candidate | Mean P0 | Mean P1 | Gap |
|---|---|---|---|
| current_ref | +0.6157 | +0.9240 | +0.3084 |

## Duplicate Trajectory Count

| Candidate | Mean duplicates |
|---|---|
| current_ref | +1536.0000 |

## Gate Classification

- current_ref: `high_search_breakthrough`
