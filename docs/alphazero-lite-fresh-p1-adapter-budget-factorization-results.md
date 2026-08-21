# PR #214 Adapter Budget Factorization

**Classification:** `budget_robust_adapter_regression`

**Recommended follow-up:** Return to search-target/policy alignment rather than attributing the regression to one budget side.

## Five-Context Matched Arena

| Context | A16 raw | P1/P1 raw | Effect | 95% CI | Seat A | Seat B | W/D/L |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| 384:256 | 0.8086 | 0.8184 | -0.0098 | [-0.0195, -0.0020] | -0.0195 | +0.0000 | 390/48/74 |
| 384:384 | 0.4902 | 0.5000 | -0.0098 | [-0.0195, -0.0020] | -0.0195 | +0.0000 | 232/38/242 |
| 1200:256 | 0.9082 | 0.9082 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | +0.0000 | 432/66/14 |
| 384:1200 | 0.0410 | 0.0547 | -0.0137 | [-0.0293, +0.0020] | +0.0195 | -0.0469 | 14/14/484 |
| 1200:1200 | 0.4805 | 0.5000 | -0.0195 | [-0.0391, -0.0039] | -0.0391 | +0.0000 | 180/132/200 |

The seat columns are treatment effects. The reported PR #214 high/high loss remains entirely Seat A; the new 384:1200 loss is instead concentrated in Seat B.

## Opening-Paired Factorial Contrasts

| Contrast | Effect | 95% CI |
| --- | ---: | --- |
| candidate_search_increment_low_parent | +0.0098 | [+0.0020, +0.0195] |
| parent_search_increment_low_candidate | -0.0039 | [-0.0254, +0.0195] |
| equalization_384 | +0.0000 | [+0.0000, +0.0000] |
| candidate_search_increment_high_parent | -0.0059 | [-0.0371, +0.0215] |
| parent_search_increment_high_candidate | -0.0195 | [-0.0391, -0.0039] |
| high_high_interaction | -0.0156 | [-0.0566, +0.0176] |

## Opening Attribution

```json
{
  "effect_correlation": {
    "1200:1200": {
      "1200:1200": 1.0,
      "1200:256": null,
      "384:1200": -0.590177257578397,
      "384:256": 1.0,
      "384:384": 1.0
    },
    "1200:256": {
      "1200:1200": null,
      "1200:256": null,
      "384:1200": null,
      "384:256": null,
      "384:384": null
    },
    "384:1200": {
      "1200:1200": -0.590177257578397,
      "1200:256": null,
      "384:1200": 0.9999999999999999,
      "384:256": -0.590177257578397,
      "384:384": -0.590177257578397
    },
    "384:256": {
      "1200:1200": 1.0,
      "1200:256": null,
      "384:1200": -0.590177257578397,
      "384:256": 1.0,
      "384:384": 1.0
    },
    "384:384": {
      "1200:1200": 1.0,
      "1200:256": null,
      "384:1200": -0.590177257578397,
      "384:256": 1.0,
      "384:384": 1.0
    }
  },
  "fraction_openings_challenger_search_worsens_a16": 0.0,
  "fraction_openings_parent_search_worsens_a16": 0.09375,
  "safe_neutral_384_256_to_negative_1200_256": 0.0,
  "safe_neutral_384_256_to_negative_384_1200": 0.09375
}
```

## Artifact And Contract Hashes

```json
{
  "a16_artifact_weights": "74e160734554dba9ecf4ecf3ac13b3e44526c4f6e7e5e52d9fe8bde0054eb789",
  "a16_state": "0b322e0996a4902cb8737ff32a429bfd45803ee4a55d713e2960ea9e9faf5068",
  "arena_configuration": "1dddb19bad4d9a06b07813544df63e6a05b72f1d779482c59fe16e85bccee1df",
  "p1_artifact_weights": "77969733ece5ced92d3a143a0fe9d82863ca3ec4faa477470ff5826ac22e4e12",
  "p1_checkpoint": "e4a9c95302c17b63405107d26b4cd4f90755d2b21980846ac6e9f0baabf7efe9",
  "replay": "892827d8ee67a66e6324a2aaec7011df1a21625fc3f6bcd87cab39ce655d2a88",
  "suite": "ff21c42946ed32f525c5ab95b71c4d90a4cfe2ccc351406dd29cae52da4b1837"
}
```

## Invariants

```json
{
  "a16_state_hash": true,
  "fit_fraction_reproduced": true,
  "inherited_parameters_byte_identical": true,
  "mean_legal_l1_reproduced": true,
  "only_adapter_parameters_differ": true,
  "p1_checkpoint_hash": true,
  "replay_hash": true
}
```

## Deterministic Probe

```json
{
  "agree_384_diverge_1200_opening_indexes": [
    10,
    13
  ],
  "by_budget": {
    "1200": {
      "a16_nonterminal_leaf_count_mean": 1198.5078125,
      "a16_terminal_leaf_count_mean": 1.4921875,
      "expanded_node_count": "not exposed by PUCT telemetry",
      "max_depth": "not exposed by PUCT telemetry",
      "mean_depth": "not exposed by PUCT telemetry",
      "p1_nonterminal_leaf_count_mean": 1198.484375,
      "p1_terminal_leaf_count_mean": 1.515625,
      "q_ranking_change_rate": 0.1015625,
      "root_value_delta_mean": 1.9595927925972007e-05,
      "selected_move_change_rate": 0.0234375,
      "top1_top2_visit_margin_delta_mean": -2.8828125,
      "visit_distribution_js_mean": 8.480217498434503e-05
    },
    "384": {
      "a16_nonterminal_leaf_count_mean": 383.90625,
      "a16_terminal_leaf_count_mean": 0.09375,
      "expanded_node_count": "not exposed by PUCT telemetry",
      "max_depth": "not exposed by PUCT telemetry",
      "mean_depth": "not exposed by PUCT telemetry",
      "p1_nonterminal_leaf_count_mean": 383.90625,
      "p1_terminal_leaf_count_mean": 0.09375,
      "q_ranking_change_rate": 0.109375,
      "root_value_delta_mean": -0.0001280095259037499,
      "selected_move_change_rate": 0.03125,
      "top1_top2_visit_margin_delta_mean": -0.3984375,
      "visit_distribution_js_mean": 4.681168985254762e-05
    }
  }
}
```
