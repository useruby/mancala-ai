# AlphaZero-Lite Value-Trunk Decoupling Ablation

**Classification:** `value_trunk_decoupling_partial`

- deterministic reproduction: `True`
- trunk relative L2 from current: {'baseline_joint': 0.002054898999631405, 'heads_only': 0.0, 'value_detached_trunk': 0.002002815483137965}

## Frozen-probe search effect (384 sims)

| Lane | Top-1 change | Q-ranking change |
| --- | ---: | ---: |
| baseline joint (harmful) | 0.1230 | 0.3818 |
| heads only (trunk frozen) | 0.0342 | 0.2070 |
| value detached trunk | 0.1104 | 0.3643 |

Value-channel decoupling barely changes the trunk perturbation (relative L2 0.00206 -> 0.00200) because the policy gradient dominates the trunk (5.9x the value gradient). Only freezing the trunk entirely (heads-only) is safe (3.4% top-1 change).

## Classification evidence

| Signal | Value |
| --- | ---: |
| decoupled_trunk_delta | 0.0020 |
| joint_trunk_delta | 0.0021 |
| decoupled_top1_change | 0.1104 |
| decoupled_q_ranking_change | 0.3643 |
| joint_top1_change | 0.1230 |
| joint_q_ranking_change | 0.3818 |
| heads_top1_change | 0.0342 |

## Next action

`value-channel decoupling reduces harm only marginally because the policy channel dominates the trunk coupling; the shared trunk itself is the root cause — test a separate (dual) value representation or frozen-trunk (heads-only) distillation`

Full evidence: `docs/data/alphazero-lite-value-trunk-decoupling-summary.json`.
