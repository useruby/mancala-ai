# AlphaZero-Lite PR123 Weighted Candidate Preflight Results

**Date**: 2026-06-20
**Classification**: `needs_balanced_replay_followup`
**Schema**: `azlite_pr123_weighted_candidate_preflight_v1`

## Candidate Manifest And Hashes

| Candidate | Artifact path | Artifact weights SHA256 | Checkpoint SHA256 | Parent promoted-current hash | Source experiment doc |
|---|---|---|---|---|---|
| promoted_current_ref | model-artifact/current | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | n/a | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | docs/alphazero-lite-promoted-current-opening-puct-disagreement-weight-ablation-results.md |
| disagreement_w8_policy_head_e1 | /tmp/azlite_promoted_current_opening_puct_disagreement_weight_ablation/disagreement_w8_policy_head_e1/artifact_disagreement_w8_policy_head_e1 | 2f1e1420c1cb02517faa702d4aa4a97f6d51e1bd0c87f782abad1183a76ce9ad | cd9ef83902516283d680fec0d3986cea832bf467042aaabd814ecd5026ec1e0e | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | docs/alphazero-lite-promoted-current-opening-puct-disagreement-weight-ablation-results.md |
| disagreement_w4_policy_head_e2 | /tmp/azlite_promoted_current_opening_puct_disagreement_weight_ablation/disagreement_w4_policy_head_e2/artifact_disagreement_w4_policy_head_e2 | 65b70a676c87593a22d46333272f2fb6e98d0cdedcd8dbc36237ba8a75184177 | 04ec8f9420459359b21b7f2d3b5ab5c6f42bc054711c3e13d9538dc8456ab616 | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | docs/alphazero-lite-promoted-current-opening-puct-disagreement-weight-ablation-results.md |
| disagreement_w16_policy_head_e2 | /tmp/azlite_promoted_current_opening_puct_disagreement_weight_ablation/disagreement_w16_policy_head_e2/artifact_disagreement_w16_policy_head_e2 | 51a191bbf417969f522024cc46d7fa4cb23e8f714d9a7d5e4c6eac40462c13be | c4b74cf357e67798b24b3ed49a593034b6c30ea9619ef5fd0bc3d36a7bcfac9f | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | docs/alphazero-lite-promoted-current-opening-puct-disagreement-weight-ablation-results.md |
| disagreement_w16_policy_head_e1 | /tmp/azlite_promoted_current_opening_puct_disagreement_weight_ablation/disagreement_w16_policy_head_e1/artifact_disagreement_w16_policy_head_e1 | ccae3a7db0e2154104fee2c1489c3dc0dfbd02bc329c8a90980ae9493f913d27 | dc0843cd6f0fc4278e9a2ae04c79e50847ac9e374d0af00ae0e69154efc49865 | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | docs/alphazero-lite-promoted-current-opening-puct-disagreement-weight-ablation-results.md |

## Suite Hashes

| Suite | Path | SHA256 | Rows |
|---|---|---|---|
| fixed_large | /tmp/azlite_opening_suite/large_eval.jsonl | ebc86f053a8a4c12f3e937b5acfe37df7b0d2f1f0002969dcac077a9f99d65a4 | 384 |
| medium | /tmp/azlite_opening_suite/medium_eval.jsonl | 57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04 | 128 |
| heldout_seed43_large | /tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed43_large.jsonl | 5e0ed96ba56f99318c32a139692309759659da16a0a98c590e35f7e2496cade9 | 384 |
| heldout_seed44_large | /tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed44_large.jsonl | 323f7abec32fb00d3b7b6b153ddd5692435755a3bc5340d8aa5aec32ca639620 | 384 |
| heldout_seed45_large | /tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed45_large.jsonl | ca72c8b7fe1adf7229f81183321475033f8548dcdeb1c71fd16c85c35ce89cda | 384 |
| heldout_seed46_large | /tmp/azlite_pr123_weighted_candidate_preflight/suites/heldout_seed46_large.jsonl | 7d8ef94fed48ec58ed11aab9818954915ba5158d8cd90291fb8b6970489b0f04 | 384 |
| heldout_seed47_large | /tmp/azlite_pr123_weighted_candidate_preflight/suites/heldout_seed47_large.jsonl | 2b689b7444e14c7880833570059775184d823d5af0f31cf77ea2f0779dd3d03b | 384 |
| heldout_seed48_large | /tmp/azlite_pr123_weighted_candidate_preflight/suites/heldout_seed48_large.jsonl | b3d6a64a3ae86fb46b8859e5b75899dd9703ad57bcd9de330e15ee674a56c262 | 384 |

## Fixed-Suite Table

| Suite | Candidate | Candidate SHA256 | Current SHA256 | 384:256 DS | 384:256 P0 / P1 | Duplicate trajectories | Delta vs promoted_current_ref |
|---|---|---|---|---|---|---|---|
| fixed_large | promoted_current_ref | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.3932 | 0.4115 / 0.8047 | 1536 | +0.0000 |
| fixed_large | disagreement_w8_policy_head_e1 | 2f1e1420c1cb02517faa702d4aa4a97f6d51e1bd0c87f782abad1183a76ce9ad | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.1562 | 0.5208 / 0.6771 | 1536 | +0.2370 |
| fixed_large | disagreement_w4_policy_head_e2 | 65b70a676c87593a22d46333272f2fb6e98d0cdedcd8dbc36237ba8a75184177 | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.1979 | 0.5430 / 0.7409 | 1536 | +0.1953 |
| fixed_large | disagreement_w16_policy_head_e2 | 51a191bbf417969f522024cc46d7fa4cb23e8f714d9a7d5e4c6eac40462c13be | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.3424 | 0.4622 / 0.8047 | 1536 | +0.0508 |
| fixed_large | disagreement_w16_policy_head_e1 | ccae3a7db0e2154104fee2c1489c3dc0dfbd02bc329c8a90980ae9493f913d27 | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.2604 | 0.4167 / 0.6771 | 1536 | +0.1328 |

## Held-Out-Suite Table

| Suite | Candidate | Candidate SHA256 | Current SHA256 | 384:256 DS | 384:256 P0 / P1 | Duplicate trajectories | Delta vs promoted_current_ref |
|---|---|---|---|---|---|---|---|
| heldout_seed43_large | promoted_current_ref | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.3932 | 0.4193 / 0.8125 | 1536 | +0.0000 |
| heldout_seed43_large | disagreement_w8_policy_head_e1 | 2f1e1420c1cb02517faa702d4aa4a97f6d51e1bd0c87f782abad1183a76ce9ad | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.1315 | 0.5352 / 0.6667 | 1536 | +0.2617 |
| heldout_seed43_large | disagreement_w4_policy_head_e2 | 65b70a676c87593a22d46333272f2fb6e98d0cdedcd8dbc36237ba8a75184177 | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.1784 | 0.5612 / 0.7396 | 1536 | +0.2148 |
| heldout_seed43_large | disagreement_w16_policy_head_e2 | 51a191bbf417969f522024cc46d7fa4cb23e8f714d9a7d5e4c6eac40462c13be | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.3294 | 0.4831 / 0.8125 | 1536 | +0.0638 |
| heldout_seed43_large | disagreement_w16_policy_head_e1 | ccae3a7db0e2154104fee2c1489c3dc0dfbd02bc329c8a90980ae9493f913d27 | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.2565 | 0.4102 / 0.6667 | 1536 | +0.1367 |
| heldout_seed44_large | promoted_current_ref | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.4089 | 0.4062 / 0.8151 | 1536 | +0.0000 |
| heldout_seed44_large | disagreement_w8_policy_head_e1 | 2f1e1420c1cb02517faa702d4aa4a97f6d51e1bd0c87f782abad1183a76ce9ad | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.1367 | 0.5326 / 0.6693 | 1536 | +0.2721 |
| heldout_seed44_large | disagreement_w4_policy_head_e2 | 65b70a676c87593a22d46333272f2fb6e98d0cdedcd8dbc36237ba8a75184177 | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.1875 | 0.5547 / 0.7422 | 1536 | +0.2214 |
| heldout_seed44_large | disagreement_w16_policy_head_e2 | 51a191bbf417969f522024cc46d7fa4cb23e8f714d9a7d5e4c6eac40462c13be | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.3451 | 0.4701 / 0.8151 | 1536 | +0.0638 |
| heldout_seed44_large | disagreement_w16_policy_head_e1 | ccae3a7db0e2154104fee2c1489c3dc0dfbd02bc329c8a90980ae9493f913d27 | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.2643 | 0.4049 / 0.6693 | 1536 | +0.1445 |
| heldout_seed45_large | promoted_current_ref | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.4349 | 0.3932 / 0.8281 | 1536 | +0.0000 |
| heldout_seed45_large | disagreement_w8_policy_head_e1 | 2f1e1420c1cb02517faa702d4aa4a97f6d51e1bd0c87f782abad1183a76ce9ad | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.1654 | 0.5195 / 0.6849 | 1536 | +0.2695 |
| heldout_seed45_large | disagreement_w4_policy_head_e2 | 65b70a676c87593a22d46333272f2fb6e98d0cdedcd8dbc36237ba8a75184177 | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.2161 | 0.5404 / 0.7565 | 1536 | +0.2188 |
| heldout_seed45_large | disagreement_w16_policy_head_e2 | 51a191bbf417969f522024cc46d7fa4cb23e8f714d9a7d5e4c6eac40462c13be | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.3711 | 0.4570 / 0.8281 | 1536 | +0.0638 |
| heldout_seed45_large | disagreement_w16_policy_head_e1 | ccae3a7db0e2154104fee2c1489c3dc0dfbd02bc329c8a90980ae9493f913d27 | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.2826 | 0.4023 / 0.6849 | 1536 | +0.1523 |
| heldout_seed46_large | promoted_current_ref | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.3854 | 0.4167 / 0.8021 | 1536 | +0.0000 |
| heldout_seed46_large | disagreement_w8_policy_head_e1 | 2f1e1420c1cb02517faa702d4aa4a97f6d51e1bd0c87f782abad1183a76ce9ad | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.1367 | 0.5299 / 0.6667 | 1536 | +0.2487 |
| heldout_seed46_large | disagreement_w4_policy_head_e2 | 65b70a676c87593a22d46333272f2fb6e98d0cdedcd8dbc36237ba8a75184177 | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.1810 | 0.5534 / 0.7344 | 1536 | +0.2044 |
| heldout_seed46_large | disagreement_w16_policy_head_e2 | 51a191bbf417969f522024cc46d7fa4cb23e8f714d9a7d5e4c6eac40462c13be | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.3424 | 0.4596 / 0.8021 | 1536 | +0.0430 |
| heldout_seed46_large | disagreement_w16_policy_head_e1 | ccae3a7db0e2154104fee2c1489c3dc0dfbd02bc329c8a90980ae9493f913d27 | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.2513 | 0.4154 / 0.6667 | 1536 | +0.1341 |
| heldout_seed47_large | promoted_current_ref | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.3802 | 0.4193 / 0.7995 | 1536 | +0.0000 |
| heldout_seed47_large | disagreement_w8_policy_head_e1 | 2f1e1420c1cb02517faa702d4aa4a97f6d51e1bd0c87f782abad1183a76ce9ad | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.1055 | 0.5482 / 0.6536 | 1536 | +0.2747 |
| heldout_seed47_large | disagreement_w4_policy_head_e2 | 65b70a676c87593a22d46333272f2fb6e98d0cdedcd8dbc36237ba8a75184177 | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.1576 | 0.5690 / 0.7266 | 1536 | +0.2227 |
| heldout_seed47_large | disagreement_w16_policy_head_e2 | 51a191bbf417969f522024cc46d7fa4cb23e8f714d9a7d5e4c6eac40462c13be | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.3333 | 0.4661 / 0.7995 | 1536 | +0.0469 |
| heldout_seed47_large | disagreement_w16_policy_head_e1 | ccae3a7db0e2154104fee2c1489c3dc0dfbd02bc329c8a90980ae9493f913d27 | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.2383 | 0.4154 / 0.6536 | 1536 | +0.1419 |
| heldout_seed48_large | promoted_current_ref | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.3932 | 0.4141 / 0.8073 | 1536 | +0.0000 |
| heldout_seed48_large | disagreement_w8_policy_head_e1 | 2f1e1420c1cb02517faa702d4aa4a97f6d51e1bd0c87f782abad1183a76ce9ad | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.1354 | 0.5312 / 0.6667 | 1536 | +0.2578 |
| heldout_seed48_large | disagreement_w4_policy_head_e2 | 65b70a676c87593a22d46333272f2fb6e98d0cdedcd8dbc36237ba8a75184177 | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.1810 | 0.5560 / 0.7370 | 1536 | +0.2122 |
| heldout_seed48_large | disagreement_w16_policy_head_e2 | 51a191bbf417969f522024cc46d7fa4cb23e8f714d9a7d5e4c6eac40462c13be | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.3516 | 0.4557 / 0.8073 | 1536 | +0.0417 |
| heldout_seed48_large | disagreement_w16_policy_head_e1 | ccae3a7db0e2154104fee2c1489c3dc0dfbd02bc329c8a90980ae9493f913d27 | 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece | -0.2656 | 0.4010 / 0.6667 | 1536 | +0.1276 |

## Aggregate Robustness Table

| Candidate | Mean DS 384:256 | Mean delta 384:256 | Worst-suite delta 384:256 | Stddev DS 384:256 | Mean delta 768:768 | Mean delta 1200:1200 | Mean delta 1200:256 |
|---|---|---|---|---|---|---|---|
| promoted_current_ref | -0.3984 | +0.0000 | +0.0000 | 0.0184 | +0.0000 | +0.0000 | +0.0000 |
| disagreement_w8_policy_head_e1 | -0.1382 | +0.2602 | +0.2370 | 0.0191 | -0.4001 | +0.1153 | +0.0000 |
| disagreement_w4_policy_head_e2 | -0.1856 | +0.2128 | +0.1953 | 0.0181 | -0.2320 | +0.1682 | +0.0000 |
| disagreement_w16_policy_head_e2 | -0.3451 | +0.0534 | +0.0417 | 0.0136 | -0.3845 | +0.2543 | +0.0000 |
| disagreement_w16_policy_head_e1 | -0.2599 | +0.1386 | +0.1276 | 0.0137 | -0.2824 | +0.1944 | +0.0841 |

## Bootstrap CI Table

| Comparison | Mean | Lower 95% | Upper 95% | Openings |
|---|---|---|---|---|
| disagreement_w8_policy_head_e1_minus_promoted_current_ref_384_256 | +0.2602 | +0.2325 | +0.2889 | 2688 |
| disagreement_w8_policy_head_e1_minus_promoted_current_ref_768_768 | -0.4001 | -0.4284 | -0.3718 | 2688 |
| disagreement_w8_policy_head_e1_minus_promoted_current_ref_1200_1200 | +0.1153 | +0.1062 | +0.1244 | 2688 |
| disagreement_w8_policy_head_e1_minus_promoted_current_ref_1200_256 | +0.0000 | +0.0000 | +0.0000 | 2688 |
| disagreement_w4_policy_head_e2_minus_promoted_current_ref_384_256 | +0.2128 | +0.1925 | +0.2334 | 2688 |
| disagreement_w4_policy_head_e2_minus_promoted_current_ref_768_768 | -0.2320 | -0.2476 | -0.2163 | 2688 |
| disagreement_w4_policy_head_e2_minus_promoted_current_ref_1200_1200 | +0.1682 | +0.1540 | +0.1823 | 2688 |
| disagreement_w4_policy_head_e2_minus_promoted_current_ref_1200_256 | +0.0000 | +0.0000 | +0.0000 | 2688 |
| disagreement_w16_policy_head_e2_minus_promoted_current_ref_384_256 | +0.0534 | +0.0329 | +0.0737 | 2688 |
| disagreement_w16_policy_head_e2_minus_promoted_current_ref_768_768 | -0.3845 | -0.4133 | -0.3564 | 2688 |
| disagreement_w16_policy_head_e2_minus_promoted_current_ref_1200_1200 | +0.2543 | +0.2180 | +0.2891 | 2688 |
| disagreement_w16_policy_head_e2_minus_promoted_current_ref_1200_256 | +0.0000 | +0.0000 | +0.0000 | 2688 |
| disagreement_w16_policy_head_e1_minus_promoted_current_ref_384_256 | +0.1386 | +0.1217 | +0.1555 | 2688 |
| disagreement_w16_policy_head_e1_minus_promoted_current_ref_768_768 | -0.2824 | -0.3155 | -0.2491 | 2688 |
| disagreement_w16_policy_head_e1_minus_promoted_current_ref_1200_1200 | +0.1944 | +0.1853 | +0.2035 | 2688 |
| disagreement_w16_policy_head_e1_minus_promoted_current_ref_1200_256 | +0.0841 | +0.0770 | +0.0911 | 2688 |
| w8_e1_minus_w4_e2_384_256 | +0.0474 | +0.0394 | +0.0554 | 2688 |
| w8_e1_minus_w4_e2_768_768 | -0.1682 | -0.1823 | -0.1540 | 2688 |
| w8_e1_minus_w16_e2_384_256 | +0.2068 | +0.1789 | +0.2351 | 2688 |

## Gate Classification Table

| Candidate | Classification | High-search preserved | 384:256 disadvantaged | 1200:1200 disadvantaged |
|---|---|---|---|---|
| promoted_current_ref | high_search_breakthrough | yes | +0.0000 | +1.0000 |
| disagreement_w8_policy_head_e1 | high_search_breakthrough | yes | +0.0000 | +1.0000 |
| disagreement_w4_policy_head_e2 | high_search_breakthrough | yes | +0.0000 | +1.0000 |
| disagreement_w16_policy_head_e2 | high_search_breakthrough | yes | +0.0000 | +1.0000 |
| disagreement_w16_policy_head_e1 | high_search_breakthrough | yes | +0.0000 | +1.0000 |

## 768:768 Regression Analysis

| Candidate | Mean DS 768:768 | Mean delta vs ref | Worst-suite delta vs ref | Lower 95% | Upper 95% | Assessment |
|---|---|---|---|---|---|---|
| disagreement_w8_policy_head_e1 | -0.5895 | -0.4001 | -0.4219 | -0.4284 | -0.3718 | unacceptable |
| disagreement_w4_policy_head_e2 | -0.4213 | -0.2320 | -0.2448 | -0.2476 | -0.2163 | unacceptable |
| disagreement_w16_policy_head_e2 | -0.5738 | -0.3845 | -0.4102 | -0.4133 | -0.3564 | unacceptable |
| disagreement_w16_policy_head_e1 | -0.4717 | -0.2824 | -0.3073 | -0.3155 | -0.2491 | unacceptable |

## Opening-Level Win Tie Loss Counts

| Comparison | Candidate better | Reference better | Tie |
|---|---|---|---|
| w8_e1 vs promoted_current_ref | 511 | 123 | 2054 |
| w4_e2 vs promoted_current_ref | 511 | 123 | 2054 |

## Final Classification And Recommendation

- Classification: `needs_balanced_replay_followup`
- Recommendation: `needs_balanced_replay_followup`
- Rationale: Weighted replay improved the primary 384:256 target, but every viable lane still carried unacceptable equal-budget or seat-balance risk.
