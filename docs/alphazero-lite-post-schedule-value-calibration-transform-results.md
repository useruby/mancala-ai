# AlphaZero-Lite Post-Schedule Value Calibration Transform Results

**Date**: 2026-07-03

**Classification**: `value_calibration_not_enough`

## Artifact Hash

- Current artifact: `model-artifact/current/weights.json`
- Current weights SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- Expected SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`

## Promoted Search Schedule Confirmation

- Runtime profile: `{"artifact": "model-artifact/current", "c_puct_schedule": {"default_c_puct": 1.25, "overrides": {"768:768": 0.9}}, "default_c_puct": 1.25, "root_policy_mode": "deterministic", "root_prior_transform": null, "search_mode": "full", "tactical_root_bias": 0.0}`

## Calibration-State Split And Hashes

- Calibration states: `/tmp/azlite_post_schedule_value_trust/value_calibration_states.jsonl` (`db730254dff1e77b178fcd060180ab261221816bca4d625fa0e56c6f11f35c91`)
- Calibration audit: `/tmp/azlite_post_schedule_value_trust/value_calibration_audit.json` (`ed72f5fd04e8656b67a710da664edc84479e6ea396741aad2a121e4050a3a41e`)
- Split: `{"group_count": 242, "group_hash": "bf7d5d7dd4b8db421b49603a94f4aec2a0f065853014c7e0b1af83b23a2d9560", "train_group_count": 169, "train_rows": 2884, "validation_group_count": 73, "validation_rows": 1212}`

## Transform Definitions And Parameters

| Transform | Supported | Improves val | Hash | Parameters |
|---|---|---|---|---|
| identity_ref | True | True | b4cd027da13708e3e07064b28da5f283e609afae41f9cb0acfca2b7c6d8ab4c7 | null |
| global_affine_tanh | True | True | 5281dab19b4ed143fb8af8c68fcb1df89e75d5de6f31f2016636e5cd3fdeefe0 | {"kind": "affine_tanh", "name": "global_affine_tanh", "phase_params": {"late": {"a": 1.39375, "b": 0.028333333333333377}, "midgame": {"a": 1.39375, "b": 0.028333333333333377}, "opening": {"a": 1.39375, "b": 0.028333333333333377}}, "version": "v1"} |
| opening_only_affine_tanh | True | True | 61adf30fabba6d58018dadcb59c6c5be2314ced6066fd95310fc27cad9c2a8b0 | {"kind": "affine_tanh", "name": "opening_only_affine_tanh", "phase_params": {"late": {"a": 1.0, "b": 0.0}, "midgame": {"a": 1.0, "b": 0.0}, "opening": {"a": 0.80875, "b": 0.027777777777777818}}, "version": "v1"} |
| conservative_opening_affine | True | True | 1091dad35b137c57ad430b56215aa4e1a49945f289b26429d8dab2479490ec23 | {"kind": "affine_tanh", "name": "conservative_opening_affine", "phase_params": {"late": {"a": 1.0, "b": 0.0}, "midgame": {"a": 1.0, "b": 0.0}, "opening": {"a": 0.9043749999999999, "b": 0.013888888888888909}}, "version": "v1"} |
| phase_affine_tanh | True | True | 1ae1a711e828265c421e5993b83c0b5b4b72ce0fe888b84a3b80c1c0f147f9a9 | {"kind": "affine_tanh", "name": "phase_affine_tanh", "phase_params": {"late": {"a": 1.7612499999999998, "b": 0.061111111111111144}, "midgame": {"a": 1.2325000000000002, "b": 0.04444444444444448}, "opening": {"a": 0.80875, "b": 0.027777777777777818}}, "version": "v1"} |
| phase_isotonic | True | True | 3b21ed7e51ccca8d09e3958321b8b435531281a898b4b8131818a2f4f5486902 | {"kind": "phase_isotonic", "name": "phase_isotonic", "phase_params": {"late": {"x": [-0.9913129210472107, -0.9879546165466309, -0.9838248491287231, -0.9831523299217224, -0.9820927381515503, -0.9819305539131165, -0.9610354900360107, -0.9568532705307007, -0.9542031288146973, -0.9428415894508362, -0.9426090121269226, -0.9423000812530518, -0.9304467439651489, -0.9144079089164734, -0.9134495854377747, -0.8704751133918762, -0.8627068400382996, -0.8555155992507935, -0.8434730172157288, -0.8393871188163757, -0.8306111693382263, -0.8099042177200317, -0.809389591217041, -0.8053398132324219, -0.8015608191490173, -0.7954935431480408, -0.791814923286438, -0.7819676995277405, -0.7652497887611389, -0.7595316767692566, -0.7549076676368713, -0.7528631091117859, -0.7516051530838013, -0.7498390078544617, -0.7151772379875183, -0.7137274742126465, -0.701789379119873, -0.3368953466415405, -0.3282385468482971, -0.30939140915870667, -0.2964131236076355, -0.25069940090179443, -0.24821878969669342, -0.0807357057929039, -0.0790247768163681, 0.03951758146286011, 0.041446033865213394, 0.06881280243396759, 0.07088456302881241, 0.10061212629079819, 0.11134325712919235, 0.22404322028160095, 0.232347309589386, 0.3228607177734375, 0.32721200585365295, 0.4140045940876007, 0.41608235239982605, 0.5681630373001099, 0.5730977058410645, 0.5823288559913635, 0.5881001949310303, 0.588186502456665, 0.5936993360519409, 0.5985649228096008, 0.602311372756958, 0.6068746447563171, 0.6073506474494934, 0.6113377213478088, 0.6419795155525208, 0.6521137952804565, 0.6561431288719177, 0.6574408411979675, 0.6647218465805054, 0.6763153076171875, 0.6766116619110107, 0.6795382499694824, 0.7057814002037048, 0.7317469120025635, 0.7376582026481628, 0.7485122084617615, 0.7515832781791687, 0.7590010166168213, 0.7883986234664917, 0.7903361320495605, 0.8006386160850525, 0.8121381402015686, 0.8225274682044983, 0.8616219758987427, 0.8773255348205566, 0.8844926953315735, 0.9485307931900024], "y": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.7894736842105263, -0.7894736842105263, -0.5, -0.5, -0.4, -0.4, -0.3557692307692308, -0.3557692307692308, -0.075, -0.075, 0.2, 0.2, 0.325, 0.325, 0.42105263157894735, 0.42105263157894735, 0.7916666666666666, 0.7916666666666666, 0.8571428571428571, 0.8571428571428571, 0.9130434782608695, 0.9130434782608695, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}, "midgame": {"x": [-0.8922008872032166, -0.8659066557884216, -0.824082612991333, -0.7591102719306946, -0.7128892540931702, -0.709248960018158, -0.6550464630126953, -0.6387799978256226, -0.24060121178627014, -0.21728309988975525, -0.2151310294866562, -0.21194209158420563, -0.08631666749715805, -0.07659479230642319, -0.025660129263997078, -0.017327575013041496, 0.029138265177607536, 0.029450464993715286, 0.0869712084531784, 0.09087926149368286, 0.09096033126115799, 0.09527666866779327, 0.12679187953472137, 0.12709805369377136, 0.2612857222557068, 0.27158865332603455, 0.2884441614151001, 0.290904700756073, 0.4060482978820801, 0.40738117694854736, 0.6390782594680786, 0.6918280124664307], "y": [-1.0, -1.0, -1.0, -1.0, -1.0, -0.5, -0.5, -0.4375, -0.4375, -0.375, -0.375, -0.23684210526315788, -0.23684210526315788, -0.058823529411764705, -0.058823529411764705, -0.045454545454545456, -0.045454545454545456, 0.044642857142857144, 0.044642857142857144, 0.125, 0.125, 0.28846153846153844, 0.28846153846153844, 0.3068181818181818, 0.3068181818181818, 0.4166666666666667, 0.4166666666666667, 0.5234375, 0.5234375, 0.8088235294117647, 0.8088235294117647, 1.0]}, "opening": {"x": [-0.6216900944709778, -0.4354524612426758, -0.4154627323150635, 0.14335238933563232, 0.1445661187171936, 0.17055173218250275, 0.17229889333248138, 0.22637629508972168, 0.23511114716529846, 0.2728944420814514, 0.2815372049808502, 0.3476254343986511, 0.36979731917381287], "y": [-0.5625, -0.5625, 0.010810810810810811, 0.010810810810810811, 0.057692307692307696, 0.057692307692307696, 0.26136363636363635, 0.26136363636363635, 0.4166666666666667, 0.4166666666666667, 0.6666666666666666, 0.6666666666666666, 1.0]}}, "version": "v1"} |
| opening_sign_bias_probe | True | True | f544eb9328a5985e2cbea2f36dbc4360f08712b5f59e37ad51e328b4911a5894 | {"kind": "affine_tanh", "name": "opening_sign_bias_probe", "phase_params": {"late": {"a": 1.0, "b": 0.0}, "midgame": {"a": 1.0, "b": 0.0}, "opening": {"a": 0.8083333333333332, "b": 0.027777777777777818}}, "version": "v1"} |

## Lane Flow

| Stage | Lanes |
|---|---|
| requested_runtime_tokens | identity, global_affine_tanh, opening_only_affine_tanh, phase_affine_tanh, phase_isotonic, conservative_opening_affine |
| validation_supported_runtime_lanes | identity_ref, global_affine_tanh, opening_only_affine_tanh, phase_affine_tanh, phase_isotonic, conservative_opening_affine |
| root_sensitivity_retained_lanes | identity_ref |
| root_sensitivity_filtered_lanes | global_affine_tanh, opening_only_affine_tanh, phase_affine_tanh, phase_isotonic, conservative_opening_affine |
| medium_carried_lanes | identity_ref |
| fixed_large_carried_lanes | identity_ref |
| heldout_carried_lanes | identity_ref |

## Validation Calibration Table

| Transform | Slice | Count | MSE | MAE | Sign acc | Spearman |
|---|---|---|---|---|---|---|
| identity_ref | overall | 1212 | +0.6802 | +0.7496 | +0.6246 | +0.5072 |
| identity_ref | opening | 412 | +0.8942 | +0.8947 | +0.4199 | -0.0116 |
| identity_ref | mid | 432 | +0.7283 | +0.8030 | +0.6667 | +0.5247 |
| identity_ref | late | 368 | +0.3842 | +0.5246 | +0.8043 | +0.7965 |
| global_affine_tanh | overall | 1212 | +0.6581 | +0.7134 | +0.6238 | +0.5072 |
| global_affine_tanh | opening | 412 | +0.9017 | +0.8958 | +0.4417 | -0.0116 |
| global_affine_tanh | mid | 432 | +0.7008 | +0.7655 | +0.6528 | +0.5247 |
| global_affine_tanh | late | 368 | +0.3352 | +0.4479 | +0.7935 | +0.7965 |
| opening_only_affine_tanh | overall | 1212 | +0.6782 | +0.7487 | +0.6295 | +0.5134 |
| opening_only_affine_tanh | opening | 412 | +0.8882 | +0.8920 | +0.4345 | -0.0116 |
| opening_only_affine_tanh | mid | 432 | +0.7283 | +0.8030 | +0.6667 | +0.5247 |
| opening_only_affine_tanh | late | 368 | +0.3842 | +0.5246 | +0.8043 | +0.7965 |
| conservative_opening_affine | overall | 1212 | +0.6791 | +0.7491 | +0.6304 | +0.5101 |
| conservative_opening_affine | opening | 412 | +0.8910 | +0.8933 | +0.4369 | -0.0116 |
| conservative_opening_affine | mid | 432 | +0.7283 | +0.8030 | +0.6667 | +0.5247 |
| conservative_opening_affine | late | 368 | +0.3842 | +0.5246 | +0.8043 | +0.7965 |
| phase_affine_tanh | overall | 1212 | +0.6520 | +0.7028 | +0.6196 | +0.5077 |
| phase_affine_tanh | opening | 412 | +0.8882 | +0.8920 | +0.4345 | -0.0116 |
| phase_affine_tanh | mid | 432 | +0.7121 | +0.7796 | +0.6481 | +0.5247 |
| phase_affine_tanh | late | 368 | +0.3170 | +0.4008 | +0.7935 | +0.7965 |
| phase_isotonic | overall | 1212 | +0.6548 | +0.6815 | +0.6609 | +0.5263 |
| phase_isotonic | opening | 412 | +0.8885 | +0.8795 | +0.4927 | +0.0630 |
| phase_isotonic | mid | 432 | +0.7092 | +0.7651 | +0.6898 | +0.5015 |
| phase_isotonic | late | 368 | +0.3292 | +0.3616 | +0.8152 | +0.7929 |
| opening_sign_bias_probe | overall | 1212 | +0.6782 | +0.7487 | +0.6295 | +0.5135 |
| opening_sign_bias_probe | opening | 412 | +0.8882 | +0.8920 | +0.4345 | -0.0116 |
| opening_sign_bias_probe | mid | 432 | +0.7283 | +0.8030 | +0.6667 | +0.5247 |
| opening_sign_bias_probe | late | 368 | +0.3842 | +0.5246 | +0.8043 | +0.7965 |

## Root Sensitivity Prefilter

- Thresholds: `{"min_mean_abs_value_delta": 0.01, "min_move_change_rate": 0.01}`

| Lane | Retain | Reason | Max move change rate | Max mean abs value delta |
|---|---|---|---|---|
| identity_ref | True | reference_lane | n/a | n/a |
| global_affine_tanh | False | below_root_sensitivity_threshold | +0.0000 | +0.0000 |
| opening_only_affine_tanh | False | below_root_sensitivity_threshold | +0.0000 | +0.0000 |
| phase_affine_tanh | False | below_root_sensitivity_threshold | +0.0000 | +0.0000 |
| phase_isotonic | False | below_root_sensitivity_threshold | +0.0000 | +0.0000 |
| conservative_opening_affine | False | below_root_sensitivity_threshold | +0.0000 | +0.0000 |

## Medium DS Table

| Lane | 384:256 | 768:256 | 768:768 | 1200:1200 | 1200:256 | 256:768 |
|---|---|---|---|---|---|---|
| identity_ref | -0.2344 | -0.3750 | +0.6484 | +0.3906 | -0.1367 | -0.3750 |

## Fixed Large DS Table

| Lane | 384:256 | 768:256 | 768:768 | 1200:1200 | 1200:256 | 256:768 |
|---|---|---|---|---|---|---|
| identity_ref | -0.3099 | -0.4049 | +0.6849 | +0.2812 | -0.1159 | -0.4049 |

## Held-Out Mean/Worst-Suite DS Table

| Lane | Held-out mean 384:256 | Worst-suite 384:256 | Held-out mean 1200:1200 | Delta 384:256 | Delta 1200:1200 | Classification |
|---|---|---|---|---|---|---|
| identity_ref | -0.3084 | -0.3255 | +0.2721 | +0.0000 | +0.0000 | reference |

## Bootstrap CI

| Comparison | Mean | Lower 95% | Upper 95% |
|---|---|---|---|

## P0/P1 Split For 384:256

| Lane | Mean P0 | Mean P1 | Gap |
|---|---|---|---|
| identity_ref | +0.6133 | +0.9232 | -0.3099 |

## Duplicate Trajectory Count

| Lane | Mean duplicates |
|---|---|
| identity_ref | 1536 |

## Runtime Cost

| Lane | Mean move latency | P95 move latency | Relative slowdown |
|---|---|---|---|
| identity_ref | +50.6650 | +120.3400 | +0.000 |

## Gate Classification

| Lane | Classification |
|---|---|
| identity_ref | high_search_breakthrough |

## Runtime Sensitivity Diagnostic

| Budget | Lane | Move changes | Move change rate | Mean abs value delta | Max abs value delta |
|---|---|---|---|---|---|
| 384:256 | global_affine_tanh | 0 | +0.0000 | +0.0000 | +0.0000 |
| 384:256 | opening_only_affine_tanh | 0 | +0.0000 | +0.0000 | +0.0000 |
| 384:256 | phase_affine_tanh | 0 | +0.0000 | +0.0000 | +0.0000 |
| 384:256 | phase_isotonic | 0 | +0.0000 | +0.0000 | +0.0000 |
| 384:256 | conservative_opening_affine | 0 | +0.0000 | +0.0000 | +0.0000 |
| 768:768 | global_affine_tanh | 0 | +0.0000 | +0.0000 | +0.0000 |
| 768:768 | opening_only_affine_tanh | 0 | +0.0000 | +0.0000 | +0.0000 |
| 768:768 | phase_affine_tanh | 0 | +0.0000 | +0.0000 | +0.0000 |
| 768:768 | phase_isotonic | 0 | +0.0000 | +0.0000 | +0.0000 |
| 768:768 | conservative_opening_affine | 0 | +0.0000 | +0.0000 | +0.0000 |

