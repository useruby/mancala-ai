# Post-Divergence Q/Visit Amplification

**Classification:** `inconclusive`

**Recommended follow-up:** Use the frozen audit results to choose a causal post-divergence intervention without changing search behavior.

Unvisited root children use value sum and Q of zero, matching `PUCT.root_summary()`.

The exact PR #221 4,096-root manifest and ordering were reused with deterministic 1,200-simulation searches (`c_puct=1.25`, zero FPU, no root noise). The JSON summary records all artifact, replay, and manifest hashes.

## Outcomes

- Actual first divergence before simulation 384: 4,027 roots.
- Amplified at 1,200 simulations: 40 roots.
- Washed out: 3,022 roots with an actual divergence, agreeing final move, and final visit JS in the bottom 75%.

## Primary Comparison

```json
{
  "auprc": {
    "estimate": 0.028589137291113127,
    "lower_95": -0.0005075067881241278,
    "samples": 10000,
    "upper_95": 0.08901240838060184
  },
  "spearman_final_visit_js": {
    "estimate": -0.037676794028689224,
    "lower_95": -0.07244263947979211,
    "samples": 10000,
    "upper_95": -0.003993412181447622
  }
}
```

## Lead/Lag

Dominant ordering: `path_divergence -> backup_value_difference -> q_ranking_difference -> visit_leader_difference -> root_move_difference`

| Event | Mean first relative sim | Mean affected fraction | Mean longest run | Disappears again |
| --- | ---: | ---: | ---: | ---: |
| Path divergence | 1.00 | 0.7220 | 169.09 | 1.0000 |
| Backup value difference | 9.99 | 0.5511 | 144.02 | 0.9633 |
| Q-ranking difference | 93.01 | 0.0529 | 24.66 | 0.7320 |
| Visit-leader difference | 262.26 | 0.0093 | 7.71 | 0.3576 |
| Deterministic root move | 258.10 | 0.0093 | 7.74 | 0.3568 |

```json
{
  "backup_value_difference": {
    "disappears_again_rate": 0.963302752293578,
    "mean_first_relative_simulation": 9.986357786357786,
    "mean_fraction_remaining": 0.5510769927393028,
    "mean_longest_consecutive_run": 144.0190924869824
  },
  "path_divergence": {
    "disappears_again_rate": 1.0,
    "mean_first_relative_simulation": 1.0,
    "mean_fraction_remaining": 0.7219986426487794,
    "mean_longest_consecutive_run": 169.09471857178278
  },
  "q_ranking_difference": {
    "disappears_again_rate": 0.7319613191172825,
    "mean_first_relative_simulation": 93.01285956006768,
    "mean_fraction_remaining": 0.05294236947604806,
    "mean_longest_consecutive_run": 24.655343416811306
  },
  "root_move_difference": {
    "disappears_again_rate": 0.3568063476320357,
    "mean_first_relative_simulation": 258.0971743625086,
    "mean_fraction_remaining": 0.009306357707128649,
    "mean_longest_consecutive_run": 7.737416315397967
  },
  "visit_leader_difference": {
    "disappears_again_rate": 0.35755021076121996,
    "mean_first_relative_simulation": 262.2572214580468,
    "mean_fraction_remaining": 0.009261143732046184,
    "mean_longest_consecutive_run": 7.707413835854203
  }
}
```

## Predictors At 1200

| Predictor | AUROC move | AUPRC move | Top 10% | Top 25% | Spearman final JS |
| --- | ---: | ---: | ---: | ---: | ---: |
| policy_l1 | 0.5687 | 0.01073 | 0.00489 | 0.00781 | 0.2858 |
| max_action_delta | 0.5700 | 0.01091 | 0.00733 | 0.01074 | 0.2564 |
| max_pressure_ratio | 0.3947 | 0.00770 | 0.00244 | 0.00684 | -0.1274 |
| minimum_parent_puct_margin | 0.5601 | 0.01144 | 0.01222 | 0.00879 | 0.1561 |
| backup_gap_auc_32 | 0.6146 | 0.01550 | 0.02200 | 0.01562 | 0.0855 |
| q_divergence_auc_32 (primary) | 0.6114 | 0.03886 | 0.01222 | 0.01465 | 0.2484 |
| visit_js_auc_32 | 0.5926 | 0.04050 | 0.01222 | 0.00977 | 0.3206 |
| path_divergence_fraction_32 | 0.5932 | 0.01771 | 0.02200 | 0.01562 | 0.1290 |
| backup_gap_auc_64 | 0.5845 | 0.01506 | 0.01956 | 0.01465 | 0.1340 |
| q_divergence_auc_64 | 0.5884 | 0.03831 | 0.01711 | 0.01270 | 0.2738 |
| visit_js_auc_64 | 0.5873 | 0.04098 | 0.01711 | 0.01172 | 0.3500 |
| path_divergence_fraction_64 | 0.5618 | 0.02142 | 0.01956 | 0.01367 | 0.1732 |

Final-JS predictor quartiles and the conditional first-divergence-before-384 analysis are retained in the JSON summary.

## Search Budget

Among 39 roots agreeing at 384 but differing at 1,200, Q divergence was visible by simulation 384 in 87.18% and visit divergence in 64.10%. No root first diverged after 384. Mean root-Q L1 at 384 was 0.01315 and mean visit JS was 0.000336.

## Held-Out PR #220 States

| State hash | Q AUC32 pct | Backup AUC32 pct | Path pct | Visit-JS AUC32 pct | Policy L1 pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| `6d7a71e6007e3943a223024aa1515659887ecb275806402f645a5b6367f942fe` | 35.72 | 15.43 | 16.02 | 65.80 | 33.25 |
| `cd6293ed266fb1db26224cb9208d2494bd9f35be6d37526c55b2a64437e98d8c` | 99.56 | 96.83 | 100.00 | 93.97 | 48.32 |

## Validation

```json
{
  "continuous_1200_prefix_equals_standalone_384": true,
  "first_divergence_alignment": true,
  "manifest_hash_and_order": true,
  "pr221_artifact_invariants": true,
  "replay_hash": true,
  "root_trajectory_matches_puct_summary": true
}
```
