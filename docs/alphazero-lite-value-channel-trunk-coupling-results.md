# AlphaZero-Lite Value-Channel Trunk-Coupling Audit

**Classification:** `value_channel_harm_from_trunk_coupling`

**Question:** does the value channel's harmful effect come from the value head (Q-value change) or from the shared trunk coupling into the policy prior?

## Probe (training) decomposition

| Variant | Top-1 change | Q-ranking change | Root-value delta |
| --- | ---: | ---: | ---: |
| value full | 0.0195 | 0.1270 | +0.0002 |
| value head only | 0.0020 | 0.0322 | -0.0002 |
| value trunk only | 0.0205 | 0.1221 | +0.0002 |
| policy full | 0.0215 | 0.1162 | +0.0002 |
| policy head only | 0.0029 | 0.0205 | -0.0000 |
| policy trunk only | 0.0215 | 0.1211 | +0.0002 |
| joint full | 0.0186 | 0.1289 | +0.0002 |

Both channels' private-head-only changes are inert (top-1 flips ~0.2-0.3%), while their trunk-only components reproduce the full move and Q-ranking changes. The harm flows through the shared trunk, where each channel's gradient leaks into the other channel's head.

## Classification evidence

| Signal | Value |
| --- | ---: |
| value_head_top1_change | 0.0020 |
| value_head_q_ranking_change | 0.0322 |
| value_trunk_top1_change | 0.0205 |
| value_trunk_q_ranking_change | 0.1221 |
| value_full_top1_change | 0.0195 |
| policy_trunk_top1_change | 0.0215 |
| policy_head_top1_change | 0.0029 |

## Next action

`decouple the value head from the shared trunk (freeze the trunk for the value channel, or use a separate value representation) in a separate experiment; do not change the policy target`

Full evidence: `docs/data/alphazero-lite-value-channel-trunk-coupling-summary.json`.
