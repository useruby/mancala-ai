# AlphaZero-Lite Balanced Current Search Calibration Results

**Date**: 2026-06-27

**Classification**: `runtime_config_candidate`

## Inputs

- Current artifact: `model-artifact/current/weights.json`
- Current artifact weights SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- Expected SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- Medium suite: `/tmp/azlite_opening_suite/medium_eval.jsonl`
- Fixed large suite: `/tmp/azlite_opening_suite/large_eval.jsonl`
- Held-out suites:
- `/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed43_large.jsonl`
- `/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed44_large.jsonl`
- `/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed45_large.jsonl`
- `/tmp/azlite_pr123_weighted_candidate_preflight/suites/heldout_seed46_large.jsonl`
- `/tmp/azlite_pr123_weighted_candidate_preflight/suites/heldout_seed47_large.jsonl`
- `/tmp/azlite_pr123_weighted_candidate_preflight/suites/heldout_seed48_large.jsonl`
- Budget pairs: `384:256,768:256,768:768,1200:1200,1200:256,256:768`
- Games per opening: `2`
- Workers: `24`
- Seed: `42`

## Supported Controls

| Control | Supported | Scope | Notes |
|---|---|---|---|
| search modes | full | classic_only, policy_only, value_only | arena.evaluate_artifact_position has an ablation_mode helper, but the active arena CLI and game loop do not expose or pass ablation_mode, so opening-suite evaluation cannot run search-mode lanes beyond full. |
| c_puct | yes | global_only | arena CLI exposes a single --c-puct applied to both challenger and current. |
| root-prior transform | yes | challenger_only_current_only_or_both | arena CLI exposes --root-prior-transform, --challenger-root-prior-transform, and --current-root-prior-transform. In the active path, the generic transform falls back only to challenger unless current-specific is also provided. |
| value trust | yes | global_only | arena CLI passes value-trust schedule into live PUCT search and emits value_trust_summary telemetry when configured. Multipliers must be > 0. |
| tactical_root_bias | yes | global_only | arena CLI exposes a single --tactical-root-bias applied to both sides. |
| search_profile_hash | arena yes / benchmark no | reporting | arena report notes include search_profile and search_profile_hash, but run_opening_suite_seat_benchmark metrics.json does not preserve them. |
| custom gate search profile | yes | supported | script/ai/seat_aware_promotion_gate can pass deterministic search controls through to arena. |

## Lane Search Profiles

| Lane | Supported | Search profile | Unsupported reason |
|---|---|---|---|
| default_full_ref | True | {"c_puct": 1.25, "root_policy_mode": "deterministic", "search_mode": "full", "tactical_root_bias": 0.1} |  |
| policy_only_default | False | {"c_puct": 1.25, "root_policy_mode": "deterministic", "search_mode": "full", "tactical_root_bias": 0.1} | active arena CLI does not expose ablation search modes beyond full |
| value_only_default | False | {"c_puct": 1.25, "root_policy_mode": "deterministic", "search_mode": "full", "tactical_root_bias": 0.1} | active arena CLI does not expose ablation search modes beyond full |
| classic_only_default | False | {"c_puct": 1.25, "root_policy_mode": "deterministic", "search_mode": "full", "tactical_root_bias": 0.1} | active arena CLI does not expose ablation search modes beyond full |
| full_default_no_tactical_bias | True | {"c_puct": 1.25, "root_policy_mode": "deterministic", "search_mode": "full", "tactical_root_bias": 0.0} |  |
| full_value_trust_half | True | {"c_puct": 1.25, "root_policy_mode": "deterministic", "search_mode": "full", "tactical_root_bias": 0.1, "value_trust_schedule": {"enabled": true, "late": 0.5, "midgame": 0.5, "opening": 0.5}} |  |
| full_value_trust_zero | False | {"c_puct": 1.25, "root_policy_mode": "deterministic", "search_mode": "full", "tactical_root_bias": 0.1} | active arena CLI does not expose value disable / neutral value mode, and value-trust multipliers must be finite numbers > 0 |
| full_cpuct_0_75 | True | {"c_puct": 0.75, "root_policy_mode": "deterministic", "search_mode": "full", "tactical_root_bias": 0.1} |  |
| full_cpuct_1_00 | True | {"c_puct": 1.0, "root_policy_mode": "deterministic", "search_mode": "full", "tactical_root_bias": 0.1} |  |
| full_cpuct_1_50 | True | {"c_puct": 1.5, "root_policy_mode": "deterministic", "search_mode": "full", "tactical_root_bias": 0.1} |  |
| full_cpuct_2_00 | True | {"c_puct": 2.0, "root_policy_mode": "deterministic", "search_mode": "full", "tactical_root_bias": 0.1} |  |
| full_seed4_extra_turn_damp | True | {"c_puct": 1.25, "challenger_root_prior_transform": "seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5", "current_root_prior_transform": "seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5", "root_policy_mode": "deterministic", "search_mode": "full", "tactical_root_bias": 0.1} |  |

## Search Profile Hashes

Medium-suite arena hashes by lane and budget pair.

| Lane | Budget | search_profile_hash |
|---|---|---|
| default_full_ref | 384:256 | 9ca513f6be4cb4082846d2207f1838059385e6c01dcdfb3f123f065f6801c2d7 |
| default_full_ref | 768:256 | 69a7c114f5fa80c12cc8f4e7114d9ba5290070d5ac9025d179bdc87fd6ff6d2c |
| default_full_ref | 768:768 | dcb735d2782f7983ff9fd37765deffa291c1e960667296681150d808788c5a56 |
| default_full_ref | 1200:1200 | 0bd2d63ed211aa10d77dbe38baa7e5882f8ffdaaeafaa75400689c31bc7c7078 |
| default_full_ref | 1200:256 | e9506758c7af6de8027a9c21513256364235cbaa88b3b5650609570f00ccc8f8 |
| default_full_ref | 256:768 | 191a0e34bfe0672a49cb16b6d813c724daadb7f5afa0bd6434cb5531f30a7fc2 |
| full_default_no_tactical_bias | 384:256 | bef9390c802eb626f3481789f3982f195dac6c70ef6676cdf2ddd82e36c67aca |
| full_default_no_tactical_bias | 768:256 | 38484cf12cc8360fb400ec1a37d9d98a84de195890fd8b0cf4a35e3e3bb74790 |
| full_default_no_tactical_bias | 768:768 | de13545426faafe02396b419d3ab40c59a587538d0bd48d31599725266894436 |
| full_default_no_tactical_bias | 1200:1200 | 37167bcd344771a1dea0ef06d57398a625354201a0be656202734109e6c5020d |
| full_default_no_tactical_bias | 1200:256 | f14a2006941c7c7fe2a58916a41cf4d30cc36a826443a837953c36be631807c8 |
| full_default_no_tactical_bias | 256:768 | 6fcc482cd706bb901b43eab74010fa2266a4653cace683300e02fc3f5bcd496e |
| full_value_trust_half | 384:256 | 330910890fc86af9c8e4c7d72a5d5603776f8322288a4b3f5b479e8846e8e6ab |
| full_value_trust_half | 768:256 | d2649be6082864e65ae90875cfb0b91c33006e201c2ee239fb78a61a3a27778d |
| full_value_trust_half | 768:768 | c2eb28fad86042ec32a71d7c1d2c63992264b55bf0bde28e040b0392ddbb54f7 |
| full_value_trust_half | 1200:1200 | 6a889cc82c82488c58f9bcf6cbf3074d841e1ed3b5ae4a78a7971a67576a510b |
| full_value_trust_half | 1200:256 | be7d4cbb349bdd658a3aeb467e4fa061f147bcbd274484bcecde9433bd7277b5 |
| full_value_trust_half | 256:768 | 3809a7489c4b52dfcf23d375995906251e3a563e7c9ea5e7f0cd4d1a891edf07 |
| full_cpuct_0_75 | 384:256 | d3b9ae0f9477186b2004bc00bd4de7135a53cca2c4150b759128b0243c71a45c |
| full_cpuct_0_75 | 768:256 | 80b93050ee727023b63cd581e6309cf0d73df45982e0d1aa9bb148c2bf50f9f5 |
| full_cpuct_0_75 | 768:768 | a314bd4dc3bb135907ccb518a15a673270cf22bd342dcf2d7b079f96f5d1078d |
| full_cpuct_0_75 | 1200:1200 | 5ef961f1f61842715824d0dfe9c153cc30c26c2f210f07108aff849587a8164d |
| full_cpuct_0_75 | 1200:256 | 73b03c2d3f4f873e8dcf396660a24a667e1cda7d4be282d28b009e68ca225c63 |
| full_cpuct_0_75 | 256:768 | 2515f1b8b75330340243f8ba2e07d9f0d47d12e9524e70114ee0d5b5540ec469 |
| full_cpuct_1_00 | 384:256 | 74356768ed7edc2efe964bcaa835bf1bb896d0f8ccd5ca5c968b1805c344a5dc |
| full_cpuct_1_00 | 768:256 | f9e22cd4fa630c2a68f54317c7a8f6724cda2da055faa4960f9eba15f48d2523 |
| full_cpuct_1_00 | 768:768 | 14c773ac7d272bf613e03014371b4033a46c6b36da1a447c9bbfcaa1e4ab3113 |
| full_cpuct_1_00 | 1200:1200 | 9df60210e7bc4d57347bfdbd347c2c9e1ca4330b4216289da00025b1eb923860 |
| full_cpuct_1_00 | 1200:256 | 443be2042640326df21c4e32b19248b8698f9dbf52c80f4fb1d1796a0815a39a |
| full_cpuct_1_00 | 256:768 | a731ef08341074db3525665baff7e13021f1a598b676e864a328faf3a01796af |
| full_cpuct_1_50 | 384:256 | 432a01efa9f4999d28c8bf851bdbc0fa08738fb4b388c22fffa3e5cf1c8cfe2f |
| full_cpuct_1_50 | 768:256 | eff75b6c54a921cdb6328147a8fb8318a5384d1fd750f05cfa02784ddb95337b |
| full_cpuct_1_50 | 768:768 | 45ec76bcced7d2b46a6e8604873345e0f0bd9772296ec1e5cf2fc5d75121a8d8 |
| full_cpuct_1_50 | 1200:1200 | 8819eee84a39994498e389d9affbc735b24d1b7b146ea888caa11dff1abe8997 |
| full_cpuct_1_50 | 1200:256 | 27b90b99bb9aca85a7d5fc9062264b93c4a72d38ccf77d24d937d8a01e67f97f |
| full_cpuct_1_50 | 256:768 | ae1670d2fac82ea5d91b14f9c32ca698823ddf0a370876208bdb7c20b7d70c8c |
| full_cpuct_2_00 | 384:256 | dd29906fc957446ba82a1fdd7715ed77708806a52911b75c604b95fa04da886e |
| full_cpuct_2_00 | 768:256 | 705fab49d31fde91bc5497873410c73230bb7ac88d9f72e9888c0f338d66779b |
| full_cpuct_2_00 | 768:768 | 8cd5a8de16cd8311fcbe40ac668ed08b1907f890da61138a3f4cf4915975d48e |
| full_cpuct_2_00 | 1200:1200 | 1b3bd8e1529992494c9ee2b9c3bad5cc4ef4e52785ff8c1abd0858b046696960 |
| full_cpuct_2_00 | 1200:256 | 0fa3d93f1fdf58c6a35ff7447c512e4176c3cd3b52704d090cfd3882419a9b8d |
| full_cpuct_2_00 | 256:768 | 981fb691c68ea91fa25a0249d5b6206942adc06ee829a7e5ca3638c996755d00 |
| full_seed4_extra_turn_damp | 384:256 | 1e0b2dc59debba189da9d7f006778da948c50ee8b79554dfcd667f92cd9e9acd |
| full_seed4_extra_turn_damp | 768:256 | 16ada3fd89da248f08d8d5b0f1388071179b06062d81fbd276c2205edec7c435 |
| full_seed4_extra_turn_damp | 768:768 | e2cb9964d29fc6fcff1dec0f094345cd63dd3ffcb7dd7b0af15c32e0c8a95e71 |
| full_seed4_extra_turn_damp | 1200:1200 | 4be501131c591b1558c8098ee692cb17ff7aae892af207cb1aad61fd659014fa |
| full_seed4_extra_turn_damp | 1200:256 | 80458111a4948408ab65c3e64d08fae6e26c1bfc8f0cbe90604898cd8f90e527 |
| full_seed4_extra_turn_damp | 256:768 | 7eb45de9482aa4351fea5c4e97a9b76c3fc383ad782c97af443bb8a0c394289b |

## Value-Trust Telemetry

| Lane | Budget | Telemetry |
|---|---|---|
| full_value_trust_half | 384:256 | {"effective_multiplier": 0.5, "enabled": true, "phase_bucket": "opening", "schedule": {"late": 0.5, "midgame": 0.5, "opening": 0.5}} |

## Root-Prior Transform Telemetry

| Lane | Budget | Challenger transform | Current transform | Telemetry |
|---|---|---|---|---|
| full_seed4_extra_turn_damp | 384:256 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | {"challenger": {"activation_count": 0, "activation_rate": 0.0, "average_mass_shift": null, "legal_move_context_counts": {"legal_move_count": 0, "no_extra_turn_capture_count": 0, "no_extra_turn_noncapture_seed5_count": 0, "seed4_extra_turn_count": 0}, "root_states_evaluated": 5151, "top_changed_move_feature_patterns": []}, "current": {"activation_count": 0, "activation_rate": 0.0, "average_mass_shift": null, "legal_move_context_counts": {"legal_move_count": 0, "no_extra_turn_capture_count": 0, "no_extra_turn_noncapture_seed5_count": 0, "seed4_extra_turn_count": 0}, "root_states_evaluated": 4527, "top_changed_move_feature_patterns": []}} |

## Fixed Large DS Table

| Lane | 384:256 | 768:256 | 768:768 | 1200:1200 | 1200:256 | 256:768 |
|---|---|---|---|---|---|---|
| default_full_ref | -0.5456 | -0.4648 | -0.2943 | -0.1380 | -0.1797 | -0.4648 |
| full_default_no_tactical_bias | -0.3099 | -0.4049 | -0.1146 | +0.2812 | -0.1159 | -0.4049 |
| full_cpuct_1_00 | -0.3372 | -0.1615 | -0.1120 | -0.2031 | -0.1549 | -0.1615 |
| full_cpuct_1_50 | -0.1901 | -0.2188 | -0.5755 | -0.0990 | -0.1471 | -0.2188 |
| full_cpuct_0_75 | -0.1706 | -0.2057 | -0.4245 | -0.4271 | +0.0000 | -0.2057 |
| full_cpuct_2_00 | -0.0482 | -0.3268 | -0.4401 | -0.6797 | -0.2253 | -0.3268 |
| full_seed4_extra_turn_damp | -0.5456 | -0.4648 | -0.2943 | -0.1380 | -0.1797 | -0.4648 |

## Held-Out Mean And Worst-Suite DS Table

| Lane | Mean 384:256 | Worst 384:256 | Mean 768:768 | Mean 1200:1200 |
|---|---|---|---|---|
| default_full_ref | -0.5488 | -0.5664 | -0.3207 | -0.1706 |
| full_default_no_tactical_bias | -0.3084 | -0.3255 | -0.1484 | +0.2721 |
| full_cpuct_2_00 | -0.0375 | -0.0599 | -0.4449 | -0.6992 |
| full_cpuct_1_00 | -0.3390 | -0.3659 | -0.1141 | -0.2122 |
| full_cpuct_0_75 | -0.1738 | -0.1862 | -0.4149 | -0.4306 |
| full_cpuct_1_50 | -0.1940 | -0.2096 | -0.5933 | -0.1354 |

## Bootstrap CI

| Comparison | Mean | Lower 95% | Upper 95% |
|---|---|---|---|
| default_full_ref_minus_default_384_256 | +0.0000 | +0.0000 | +0.0000 |
| default_full_ref_minus_default_768_768 | +0.0000 | +0.0000 | +0.0000 |
| default_full_ref_minus_default_1200_1200 | +0.0000 | +0.0000 | +0.0000 |
| default_full_ref_minus_default_1200_256 | +0.0000 | +0.0000 | +0.0000 |
| full_default_no_tactical_bias_minus_default_384_256 | +0.2405 | +0.2194 | +0.2622 |
| full_default_no_tactical_bias_minus_default_768_768 | +0.1723 | +0.1567 | +0.1875 |
| full_default_no_tactical_bias_minus_default_1200_1200 | +0.4427 | +0.4089 | +0.4774 |
| full_default_no_tactical_bias_minus_default_1200_256 | +0.0740 | +0.0543 | +0.0933 |
| full_cpuct_2_00_minus_default_384_256 | +0.5113 | +0.4798 | +0.5434 |
| full_cpuct_2_00_minus_default_768_768 | -0.1241 | -0.1749 | -0.0734 |
| full_cpuct_2_00_minus_default_1200_1200 | -0.5286 | -0.5707 | -0.4848 |
| full_cpuct_2_00_minus_default_1200_256 | -0.0358 | -0.0675 | -0.0046 |
| full_cpuct_1_00_minus_default_384_256 | +0.2099 | +0.1923 | +0.2272 |
| full_cpuct_1_00_minus_default_768_768 | +0.2066 | +0.1476 | +0.2656 |
| full_cpuct_1_00_minus_default_1200_1200 | -0.0417 | -0.0885 | +0.0048 |
| full_cpuct_1_00_minus_default_1200_256 | +0.0428 | +0.0178 | +0.0675 |
| full_cpuct_0_75_minus_default_384_256 | +0.3750 | +0.3516 | +0.3993 |
| full_cpuct_0_75_minus_default_768_768 | -0.0942 | -0.1463 | -0.0421 |
| full_cpuct_0_75_minus_default_1200_1200 | -0.2600 | -0.3129 | -0.2066 |
| full_cpuct_0_75_minus_default_1200_256 | +0.1847 | +0.1641 | +0.2051 |
| full_cpuct_1_50_minus_default_384_256 | +0.3548 | +0.3270 | +0.3826 |
| full_cpuct_1_50_minus_default_768_768 | -0.2726 | -0.3073 | -0.2378 |
| full_cpuct_1_50_minus_default_1200_1200 | +0.0352 | +0.0017 | +0.0673 |
| full_cpuct_1_50_minus_default_1200_256 | +0.0391 | +0.0167 | +0.0614 |

## P0/P1 Split For 384:256

| Lane | Mean P0 | Mean P1 | Gap | Mean duplicates |
|---|---|---|---|---|
| default_full_ref | +0.4117 | +0.9605 | +0.5488 | +768.0000 |
| full_default_no_tactical_bias | +0.6157 | +0.9240 | +0.3084 | +768.0000 |
| full_cpuct_2_00 | +0.5851 | +0.6226 | +0.0375 | +768.0000 |
| full_cpuct_1_00 | +0.2535 | +0.5924 | +0.3390 | +768.0000 |
| full_cpuct_0_75 | +0.5768 | +0.7507 | +0.1738 | +768.0000 |
| full_cpuct_1_50 | +0.5137 | +0.7077 | +0.1940 | +768.0000 |

## Runtime Cost

Held-out 384:256 aggregate runtime cost.

| Lane | Mean move latency | P95 move latency | Avg sims/game |
|---|---|---|---|
| default_full_ref | +50.78 | +119.35 | +384.00 |
| full_default_no_tactical_bias | +49.54 | +119.21 | +384.00 |
| full_cpuct_2_00 | +50.39 | +119.09 | +384.00 |
| full_cpuct_1_00 | +50.68 | +118.79 | +384.00 |
| full_cpuct_0_75 | +50.40 | +119.79 | +384.00 |
| full_cpuct_1_50 | +49.23 | +119.19 | +384.00 |

## Gate Classification

- default_full_ref: `high_search_breakthrough`
- full_default_no_tactical_bias: `high_search_breakthrough`
