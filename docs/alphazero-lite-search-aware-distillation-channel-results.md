# AlphaZero-Lite Search-Aware Distillation Channel Audit

**Classification:** `search_aware_top1_misalignment_not_confirmed`

**Question:** does the harmful step's policy channel or value channel drive deterministic-search move flips, and do those flips move toward or away from a stronger D1200 reference?

- policy/value grand-mean gradient norm ratio: 5.89

## Probe (training) alignment

| Channel | Move change | Reference alignment | Flips toward | Flips away | Net delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| joint | 0.0186 | 0.8604 | 0.0068 | 0.0088 | -0.0020 |
| policy | 0.0215 | 0.8564 | 0.0059 | 0.0117 | -0.0059 |
| value | 0.0195 | 0.8564 | 0.0059 | 0.0117 | -0.0059 |

## Validation (held-out) alignment

| Channel | Move change | Reference alignment | Flips toward | Flips away | Net delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| joint | 0.0293 | 0.8418 | 0.0137 | 0.0117 | +0.0020 |
| policy | 0.0215 | 0.8359 | 0.0078 | 0.0117 | -0.0039 |
| value | 0.0215 | 0.8398 | 0.0098 | 0.0098 | +0.0000 |

## Classification evidence

| Signal | Value |
| --- | ---: |
| joint_probe_move_change | 0.0186 |
| policy_probe_move_change | 0.0215 |
| value_probe_move_change | 0.0195 |
| policy_probe_net_reference_delta | -0.0059 |
| value_probe_net_reference_delta | -0.0059 |
| validation_joint_move_change | 0.0293 |
| validation_policy_move_change | 0.0215 |
| validation_value_move_change | 0.0215 |

## Next action

`the harmful step flips deterministic moves roughly balanced toward and away from a stronger D1200 reference; investigate the value/Q-value channel's effect on PUCT root rankings or treat distillation as saturated`

Full evidence: `docs/data/alphazero-lite-search-aware-distillation-channel-summary.json`.
