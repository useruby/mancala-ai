# AlphaZero-Lite Frozen-Trunk Policy-Prior Localization Results

**Classification:** `distributed_prior_compounding`

- candidate checkpoint state hash matches PR #200 step 46: `True`
- candidate trunk identical to incumbent: `True`
- candidate value stack identical to incumbent: `True`
- candidate value outputs equal incumbent on frozen probe: `True`
- candidate_all reproduces PR #200 arena: `True`
- incumbent_all search-equivalent to incumbent: `True`
- candidate weights sha256: `68d8313f57b7303c1e9617c774328741e2900b87ac40359ab2a9d7001c19b23f`
- incumbent weights sha256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`

## Findings

The harmful policy prior is distributed throughout the tree: shallow substitutions are insufficient and only substituting the incumbent prior at every node (incumbent_all) rescues the deficit.

## Recovery fractions (384:256, paired effect / P0 effect)

| Intervention | Paired effect | Paired recovery | P0 effect | P0 recovery | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| candidate_all (baseline) | -0.1895 | n/a | -0.3672 | n/a | [-0.2324, -0.1465] |
| incumbent_root | -0.1895 | 0.000 | -0.3672 | 0.000 | [-0.2324, -0.1465] |
| incumbent_depth1 | -0.1348 | 0.289 | -0.2578 | 0.298 | [-0.1758, -0.0957] |
| incumbent_depth2 | -0.0879 | 0.536 | -0.1641 | 0.553 | [-0.1230, -0.0547] |
| incumbent_all | +0.0000 | 1.000 | +0.0000 | 1.000 | [+0.0000, +0.0000] |

## Canonical arena (candidate versus frozen incumbent)

| Intervention | Context | Paired effect | 95% CI | P0 | P1 | W-D-L | Div. rate |
| --- | --- | ---: | --- | ---: | ---: | --- | ---: |
| candidate_all | 384:256 | -0.1895 | [-0.2324, -0.1465] | -0.3672 | -0.0117 | 302-38-172 | 0.0000 |
| candidate_all | 1200:1200 | -0.0605 | [-0.0801, -0.0430] | -0.1211 | +0.0000 | 184-82-246 | 0.0000 |
| incumbent_root | 384:256 | -0.1895 | [-0.2324, -0.1465] | -0.3672 | -0.0117 | 302-38-172 | 0.1289 |
| incumbent_root | 1200:1200 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | +0.0000 | 184-144-184 | 0.2461 |
| incumbent_depth1 | 384:256 | -0.1348 | [-0.1758, -0.0957] | -0.2578 | -0.0117 | 330-38-144 | 0.1953 |
| incumbent_depth1 | 1200:1200 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | +0.0000 | 184-144-184 | 0.2461 |
| incumbent_depth2 | 384:256 | -0.0879 | [-0.1230, -0.0547] | -0.1641 | -0.0117 | 354-38-120 | 0.2695 |
| incumbent_depth2 | 1200:1200 | +0.0645 | [+0.0469, +0.0840] | +0.0078 | +0.1211 | 246-86-180 | 0.4219 |
| incumbent_all | 384:256 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | +0.0000 | 402-32-78 | 0.3320 |
| incumbent_all | 1200:1200 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | +0.0000 | 184-144-184 | 0.3086 |

## PUCT probe (384:256, 256 states, versus candidate_all)

| Intervention | Selected-move change | Mean visit JS | Mean root-value delta | Mean Q-rank change |
| --- | ---: | ---: | ---: | ---: |
| candidate_all | 0.0000 | 0.000000 | +0.000000 | +0.0000 |
| incumbent_root | 0.0078 | 0.000577 | -0.000208 | +0.0078 |
| incumbent_depth1 | 0.0078 | 0.001182 | +0.000094 | -0.0117 |
| incumbent_depth2 | 0.0195 | 0.001296 | +0.000216 | +0.0039 |
| incumbent_all | 0.0273 | 0.001572 | -0.000091 | -0.0273 |

## Per-depth override telemetry (probe, affected nodes / candidate-vs-incumbent legal-policy L1/JS)

| Intervention | Depth | Expanded | Affected | Affected frac | Mean legal L1 | Mean legal JS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| incumbent_root | 0 | 256 | 256 | 1.0000 | 0.019738 | 0.000085 |
| incumbent_root | 1 | 951 | 0 | 0.0000 | 0.000000 | 0.000000 |
| incumbent_root | 2 | 3001 | 0 | 0.0000 | 0.000000 | 0.000000 |
| incumbent_root | 3 | 7053 | 0 | 0.0000 | 0.000000 | 0.000000 |
| incumbent_root | 4 | 65260 | 0 | 0.0000 | 0.000000 | 0.000000 |
| incumbent_root | overall | 76521 | 256 | 0.0033 | - | - |
| incumbent_depth1 | 0 | 256 | 256 | 1.0000 | 0.019738 | 0.000085 |
| incumbent_depth1 | 1 | 952 | 952 | 1.0000 | 0.020183 | 0.000086 |
| incumbent_depth1 | 2 | 3007 | 0 | 0.0000 | 0.000000 | 0.000000 |
| incumbent_depth1 | 3 | 7076 | 0 | 0.0000 | 0.000000 | 0.000000 |
| incumbent_depth1 | 4 | 65261 | 0 | 0.0000 | 0.000000 | 0.000000 |
| incumbent_depth1 | overall | 76552 | 1208 | 0.0158 | - | - |
| incumbent_depth2 | 0 | 256 | 256 | 1.0000 | 0.019738 | 0.000085 |
| incumbent_depth2 | 1 | 952 | 952 | 1.0000 | 0.020183 | 0.000086 |
| incumbent_depth2 | 2 | 3006 | 3006 | 1.0000 | 0.021159 | 0.000090 |
| incumbent_depth2 | 3 | 7056 | 0 | 0.0000 | 0.000000 | 0.000000 |
| incumbent_depth2 | 4 | 65292 | 0 | 0.0000 | 0.000000 | 0.000000 |
| incumbent_depth2 | overall | 76562 | 4214 | 0.0550 | - | - |
| incumbent_all | 0 | 256 | 256 | 1.0000 | 0.019738 | 0.000085 |
| incumbent_all | 1 | 952 | 952 | 1.0000 | 0.020183 | 0.000086 |
| incumbent_all | 2 | 3005 | 3005 | 1.0000 | 0.021172 | 0.000090 |
| incumbent_all | 3 | 7068 | 7068 | 1.0000 | 0.021236 | 0.000090 |
| incumbent_all | 4 | 65304 | 65304 | 1.0000 | 0.019678 | 0.000084 |
| incumbent_all | overall | 76585 | 76585 | 1.0000 | - | - |

## Classification evidence

```json
{
  "candidate_all_ci": {
    "lower_95": -0.232421875,
    "samples": 10000,
    "unique_openings": 128,
    "upper_95": -0.146484375
  },
  "candidate_all_p0_effect": -0.3671875,
  "candidate_all_paired_effect": -0.189453125,
  "incumbent_all_paired_effect": 0.0,
  "incumbent_all_search_equivalent": true,
  "primary_context": "384:256",
  "recovery": {
    "incumbent_all": {
      "ci_lower_95": 0.0,
      "ci_upper_95": 0.0,
      "p0_effect": 0.0,
      "p0_recovery_fraction": 1.0,
      "paired_effect": 0.0,
      "paired_recovery_fraction": 1.0
    },
    "incumbent_depth1": {
      "ci_lower_95": -0.17578125,
      "ci_upper_95": -0.095703125,
      "p0_effect": -0.2578125,
      "p0_recovery_fraction": 0.2978723404255319,
      "paired_effect": -0.134765625,
      "paired_recovery_fraction": 0.28865979381443296
    },
    "incumbent_depth2": {
      "ci_lower_95": -0.123046875,
      "ci_upper_95": -0.0546875,
      "p0_effect": -0.1640625,
      "p0_recovery_fraction": 0.5531914893617021,
      "paired_effect": -0.087890625,
      "paired_recovery_fraction": 0.5360824742268041
    },
    "incumbent_root": {
      "ci_lower_95": -0.232421875,
      "ci_upper_95": -0.146484375,
      "p0_effect": -0.3671875,
      "p0_recovery_fraction": 0.0,
      "paired_effect": -0.189453125,
      "paired_recovery_fraction": 0.0
    }
  },
  "recovery_threshold": 0.7
}
```

## Recommended next experiment (not implemented here)

`change policy-target construction or apply same-state incumbent-prior constraints throughout the replay (a global policy trust region on legal-policy divergence from the incumbent) rather than only reducing the learning rate`

## Exact commands

```bash
python ml/alphazero_lite/run_frozen_trunk_policy_prior_localization.py \
  --pr191-workdir /tmp/azlite_shared_trunk_learning \
  --candidate-snapshot /tmp/azlite_frozen_trunk_distillation/policy_head/snapshots/step_0046.pt \
  --candidate-artifact /tmp/azlite_frozen_trunk_distillation/policy_head/snapshot_artifacts/step_0046/artifact \
  --incumbent-snapshot /tmp/azlite_frozen_trunk_distillation/heads_only/snapshots/step_0000.pt \
  --incumbent /home/alex/Mancala/ai/model-artifact/current \
  --workdir /tmp/azlite_frozen_trunk_policy_prior_localization --arena-workers 24
```

Full evidence: `docs/data/alphazero-lite-frozen-trunk-policy-prior-localization-summary.json`.
