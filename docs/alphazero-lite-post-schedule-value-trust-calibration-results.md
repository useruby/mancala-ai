# AlphaZero-Lite Post-Schedule Value-Trust Calibration Results

**Date**: 2026-07-01

**Classification**: `value_trust_not_enough`

## Artifact Hash

- Current artifact: `model-artifact/current/weights.json`
- Current weights SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- Expected SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`

## Promoted Search Schedule Confirmation

- Runtime profile: `{"artifact": "model-artifact/current", "c_puct_schedule": {"default_c_puct": 1.25, "overrides": {"768:768": 0.9}}, "default_c_puct": 1.25, "root_policy_mode": "deterministic", "root_prior_transform": null, "search_mode": "full", "tactical_root_bias": 0.0}`

## Supported And Unsupported Value Controls

- Controls: `{"search_modes": {"notes": "The active opening-suite benchmark path does not expose search-mode ablations beyond full.", "supported": ["full"], "unsupported": ["policy_only", "value_only", "classic_only"]}, "value_disable_or_zero": {"notes": "The active path does not expose a disabled-value runtime lane through the opening-suite benchmark, and value-trust multipliers must stay finite and > 0.", "supported": false}, "value_trust": {"notes": "The active arena and gate paths support global value-trust schedules with opening/midgame/late multipliers > 0 and emit value_trust_summary telemetry.", "scope": "global_opening_midgame_late_schedule", "supported": true}}`

## Value Calibration Audit Table

| Slice | Count | MSE | MAE | Sign acc | Spearman | Overconf rate |
|---|---|---|---|---|---|---|
| overall | 4096 | +0.6649 | +0.7329 | +0.6079 | +0.4898 | +0.0000 |
| opening | 1360 | +0.8740 | +0.8787 | +0.4449 | +0.0440 | +0.0000 |
| mid | 1368 | +0.7325 | +0.7924 | +0.6096 | +0.4208 | +0.0000 |
| late | 1368 | +0.3894 | +0.5285 | +0.7683 | +0.7870 | +0.0000 |

## Overconfident-Wrong-State Table

| State | Suite | Budget | Phase | Seat | Raw value | Outcome | Final margin |
|---|---|---|---|---|---|---|---|

## Value-Trust Lane Definitions

| Lane | Description | Schedule |
|---|---|---|
| default_value_trust_1_00 | Promoted runtime default with no configured value-trust override. | null |
| value_trust_all_0_75 | Uniform value trust 0.75 in opening, midgame, and late. | {"enabled": true, "late": 0.75, "midgame": 0.75, "opening": 0.75} |
| value_trust_all_0_50 | Uniform value trust 0.50 in opening, midgame, and late. | {"enabled": true, "late": 0.5, "midgame": 0.5, "opening": 0.5} |
| value_trust_all_0_25 | Uniform value trust 0.25 in opening, midgame, and late. | {"enabled": true, "late": 0.25, "midgame": 0.25, "opening": 0.25} |
| value_trust_opening_0_50_mid_1_00_late_1_00 | Opening-only trust reduction to 0.50. | {"enabled": true, "late": 1.0, "midgame": 1.0, "opening": 0.5} |
| value_trust_opening_0_75_mid_1_00_late_1_00 | Opening-only trust reduction to 0.75. | {"enabled": true, "late": 1.0, "midgame": 1.0, "opening": 0.75} |
| value_trust_opening_1_00_mid_0_50_late_1_00 | Midgame-only trust reduction to 0.50. | {"enabled": true, "late": 1.0, "midgame": 0.5, "opening": 1.0} |
| value_trust_opening_1_00_mid_1_00_late_0_50 | Late-only trust reduction to 0.50. | {"enabled": true, "late": 0.5, "midgame": 1.0, "opening": 1.0} |
| value_trust_all_1_25 | Uniform value trust 1.25 in opening, midgame, and late. | {"enabled": true, "late": 1.25, "midgame": 1.25, "opening": 1.25} |
| value_trust_opening_1_25_mid_1_00_late_1_00 | Opening-only trust increase to 1.25. | {"enabled": true, "late": 1.0, "midgame": 1.0, "opening": 1.25} |

## Effective Search Profile Hashes

| Lane | Budget | Hash |
|---|---|---|
| default_value_trust_1_00 | 384:256 | bef9390c802eb626f3481789f3982f195dac6c70ef6676cdf2ddd82e36c67aca |
| default_value_trust_1_00 | 768:256 | 38484cf12cc8360fb400ec1a37d9d98a84de195890fd8b0cf4a35e3e3bb74790 |
| default_value_trust_1_00 | 768:768 | ce678807806fda0f938a18715fc02c9f40690b0d1a5a7834ddc7c69a3a8ac5cf |
| default_value_trust_1_00 | 1200:1200 | 37167bcd344771a1dea0ef06d57398a625354201a0be656202734109e6c5020d |
| default_value_trust_1_00 | 1200:256 | f14a2006941c7c7fe2a58916a41cf4d30cc36a826443a837953c36be631807c8 |
| default_value_trust_1_00 | 256:768 | 6fcc482cd706bb901b43eab74010fa2266a4653cace683300e02fc3f5bcd496e |
| value_trust_all_0_25 | 384:256 | 3dd8c3cf430794c65193477bb50dd30574873d581a007b77ed25ebb56d9449b6 |
| value_trust_all_0_25 | 768:256 | 51f92684413837400705c11528b6a25022db12e82a06bfc121e59895b3ed298d |
| value_trust_all_0_25 | 768:768 | b6fe883f891e521051a7eb8f3eb282aee352f41b53b15633f478a3f7cd13b76d |
| value_trust_all_0_25 | 1200:1200 | 266fde186425e3447b52edd4e6fbc73c6782330f14b2ee698fbc440acb98f473 |
| value_trust_all_0_25 | 1200:256 | 2cd8cdf1deac124e21f97d7c4d845bf5eb1b75ace9a4146f28642f1567d5602b |
| value_trust_all_0_25 | 256:768 | 34f39775f0c30c6013ab80c23d5faf7b4f217c0883fa2814d8354cc5903c2cd4 |
| value_trust_opening_0_50_mid_1_00_late_1_00 | 384:256 | 31c2b51918cb71a61489db67d2dd07e832ceb88ff86cea913b8af431d7fda213 |
| value_trust_opening_0_50_mid_1_00_late_1_00 | 768:256 | 39a52c0a6af076bd3b0f84900422f9de6b4fcc395ba93df1f11025d6a8e4285f |
| value_trust_opening_0_50_mid_1_00_late_1_00 | 768:768 | e4b0536dbc0b306fc2bf8f5748f1a1f783d8d4ac48601b6174fd78010a938248 |
| value_trust_opening_0_50_mid_1_00_late_1_00 | 1200:1200 | 9f190dedee20df211320113949f173f235b2d4b4c6da24fff5086b5fc7fcb48f |
| value_trust_opening_0_50_mid_1_00_late_1_00 | 1200:256 | 6a43c80e629a0f0d07d2f1209fd33abdcc50e1ccb94995ab338e59bbb7fe66dc |
| value_trust_opening_0_50_mid_1_00_late_1_00 | 256:768 | b00da028defe5e4c0b8213f1447a8a81eec01d7a6525888bb7207d77b0e926fe |

## Value-Trust Telemetry

| Lane | Budget | Telemetry |
|---|---|---|
| value_trust_all_0_25 | 384:256 | {"effective_multiplier": 0.25, "enabled": true, "phase_bucket": "opening", "schedule": {"late": 0.25, "midgame": 0.25, "opening": 0.25}} |
| value_trust_all_0_25 | 768:256 | {"effective_multiplier": 0.25, "enabled": true, "phase_bucket": "opening", "schedule": {"late": 0.25, "midgame": 0.25, "opening": 0.25}} |
| value_trust_all_0_25 | 768:768 | {"effective_multiplier": 0.25, "enabled": true, "phase_bucket": "opening", "schedule": {"late": 0.25, "midgame": 0.25, "opening": 0.25}} |
| value_trust_all_0_25 | 1200:1200 | {"effective_multiplier": 0.25, "enabled": true, "phase_bucket": "opening", "schedule": {"late": 0.25, "midgame": 0.25, "opening": 0.25}} |
| value_trust_all_0_25 | 1200:256 | {"effective_multiplier": 0.25, "enabled": true, "phase_bucket": "opening", "schedule": {"late": 0.25, "midgame": 0.25, "opening": 0.25}} |
| value_trust_all_0_25 | 256:768 | {"effective_multiplier": 0.25, "enabled": true, "phase_bucket": "opening", "schedule": {"late": 0.25, "midgame": 0.25, "opening": 0.25}} |
| value_trust_opening_0_50_mid_1_00_late_1_00 | 384:256 | {"effective_multiplier": 0.5, "enabled": true, "phase_bucket": "opening", "schedule": {"late": 1.0, "midgame": 1.0, "opening": 0.5}} |
| value_trust_opening_0_50_mid_1_00_late_1_00 | 768:256 | {"effective_multiplier": 0.5, "enabled": true, "phase_bucket": "opening", "schedule": {"late": 1.0, "midgame": 1.0, "opening": 0.5}} |
| value_trust_opening_0_50_mid_1_00_late_1_00 | 768:768 | {"effective_multiplier": 0.5, "enabled": true, "phase_bucket": "opening", "schedule": {"late": 1.0, "midgame": 1.0, "opening": 0.5}} |
| value_trust_opening_0_50_mid_1_00_late_1_00 | 1200:1200 | {"effective_multiplier": 0.5, "enabled": true, "phase_bucket": "opening", "schedule": {"late": 1.0, "midgame": 1.0, "opening": 0.5}} |
| value_trust_opening_0_50_mid_1_00_late_1_00 | 1200:256 | {"effective_multiplier": 0.5, "enabled": true, "phase_bucket": "opening", "schedule": {"late": 1.0, "midgame": 1.0, "opening": 0.5}} |
| value_trust_opening_0_50_mid_1_00_late_1_00 | 256:768 | {"effective_multiplier": 0.5, "enabled": true, "phase_bucket": "opening", "schedule": {"late": 1.0, "midgame": 1.0, "opening": 0.5}} |

## Medium DS Table

| Lane | 384:256 | 768:256 | 768:768 | 1200:1200 | 1200:256 | 256:768 |
|---|---|---|---|---|---|---|
| default_value_trust_1_00 | -0.2344 | -0.3750 | +0.6484 | +0.3906 | -0.1367 | -0.3750 |
| value_trust_all_0_75 | -0.2930 | -0.0430 | -0.5781 | -0.6562 | +0.0742 | -0.0430 |
| value_trust_all_0_50 | +0.0195 | -0.0273 | -0.7656 | -0.5781 | -0.1289 | -0.0273 |
| value_trust_all_0_25 | -0.0898 | -0.1289 | -0.2656 | -0.6094 | +0.1289 | -0.1289 |
| value_trust_opening_0_50_mid_1_00_late_1_00 | -0.2539 | +0.0000 | -0.6719 | -0.3750 | +0.0469 | +0.0000 |
| value_trust_opening_0_75_mid_1_00_late_1_00 | -0.2930 | -0.4258 | -0.4141 | -0.6562 | -0.2734 | -0.4258 |
| value_trust_opening_1_00_mid_0_50_late_1_00 | -0.3789 | -0.5000 | +0.3594 | -0.0312 | -0.0234 | -0.5000 |
| value_trust_opening_1_00_mid_1_00_late_0_50 | -0.2188 | -0.3750 | +0.6094 | +0.3906 | -0.1172 | -0.3750 |
| value_trust_all_1_25 | -0.4609 | +0.1055 | +0.0391 | -0.0781 | -0.1641 | +0.1055 |
| value_trust_opening_1_25_mid_1_00_late_1_00 | -0.5000 | -0.1289 | -0.1875 | -0.0391 | -0.0273 | -0.1289 |

## Fixed Large DS Table

| Lane | 384:256 | 768:256 | 768:768 | 1200:1200 | 1200:256 | 256:768 |
|---|---|---|---|---|---|---|
| default_value_trust_1_00 | -0.3099 | -0.4049 | +0.6849 | +0.2812 | -0.1159 | -0.4049 |
| value_trust_all_0_25 | -0.0911 | -0.2161 | -0.1823 | -0.5938 | +0.1328 | -0.2161 |
| value_trust_opening_0_50_mid_1_00_late_1_00 | -0.0469 | +0.0234 | -0.5833 | -0.2604 | +0.0365 | +0.0234 |

## Held-Out Mean/Worst-Suite DS Table

| Lane | Held-out mean 384:256 | Delta vs default | Worst-suite 384:256 | Held-out mean 768:768 | Held-out mean 1200:1200 |
|---|---|---|---|---|---|
| default_value_trust_1_00 | -0.3084 | +0.0000 | -0.3255 | +0.6766 | +0.2721 |
| value_trust_all_0_25 | -0.0840 | +0.2244 | -0.1276 | -0.1649 | -0.5738 |
| value_trust_opening_0_50_mid_1_00_late_1_00 | -0.0163 | +0.2921 | -0.0495 | -0.5894 | -0.2726 |

## Bootstrap CI

| Comparison | Mean | Lower 95% | Upper 95% |
|---|---|---|---|
| value_trust_all_0_25 minus default_value_trust_1_00 @ 384:256 | +0.2244 | +0.1921 | +0.2567 |
| value_trust_all_0_25 minus default_value_trust_1_00 @ 768:768 | -0.8416 | -0.8754 | -0.8073 |
| value_trust_all_0_25 minus default_value_trust_1_00 @ 1200:1200 | -0.8459 | -0.8898 | -0.8016 |
| value_trust_all_0_25 minus default_value_trust_1_00 @ 1200:256 | +0.2533 | +0.2331 | +0.2732 |
| value_trust_opening_0_50_mid_1_00_late_1_00 minus default_value_trust_1_00 @ 384:256 | +0.2921 | +0.2444 | +0.3388 |
| value_trust_opening_0_50_mid_1_00_late_1_00 minus default_value_trust_1_00 @ 768:768 | -1.2661 | -1.3108 | -1.2218 |
| value_trust_opening_0_50_mid_1_00_late_1_00 minus default_value_trust_1_00 @ 1200:1200 | -0.5447 | -0.6042 | -0.4857 |
| value_trust_opening_0_50_mid_1_00_late_1_00 minus default_value_trust_1_00 @ 1200:256 | +0.1411 | +0.1311 | +0.1513 |

## P0/P1 Split For 384:256

| Lane | Mean P0 | Mean P1 | Gap |
|---|---|---|---|
| default_value_trust_1_00 | +0.6133 | +0.9232 | -0.3099 |
| value_trust_all_0_25 | +0.5234 | +0.6146 | -0.0911 |
| value_trust_opening_0_50_mid_1_00_late_1_00 | +0.5807 | +0.6276 | -0.0469 |

## Duplicate Trajectory Count

| Lane | Mean duplicates |
|---|---|
| default_value_trust_1_00 | 1536 |
| value_trust_all_0_25 | 1536 |
| value_trust_opening_0_50_mid_1_00_late_1_00 | 1536 |

## Runtime Cost

| Lane | Mean move latency | P95 move latency | Relative slowdown |
|---|---|---|---|
| default_value_trust_1_00 | +49.7300 | +119.3900 | +0.000 |
| value_trust_all_0_25 | +50.3350 | +120.4850 | +0.012 |
| value_trust_opening_0_50_mid_1_00_late_1_00 | +50.7350 | +119.9200 | +0.020 |

## Gate Classification

| Lane | Classification |
|---|---|
| default_value_trust_1_00 | high_search_breakthrough |

## Decision Summary

- The refreshed audit now includes traced continuation states instead of only suite-root opening states.
- Calibration is materially phase-dependent: opening remains the weakest slice (`sign_accuracy=+0.4449`, `spearman=+0.0440`), while mid (`+0.6096`, `+0.4208`) and late (`+0.7683`, `+0.7870`) are notably stronger.
- This supports an opening-skewed value calibration problem rather than a uniform value failure across all phases.
- However, the value-trust reduction lanes that improved held-out `384:256` still caused severe robustness regressions at `768:768` and `1200:1200`.
- Because no lane satisfied the held-out improvement plus robustness constraints, the final classification remains `value_trust_not_enough` rather than a shippable runtime candidate.
