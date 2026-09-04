# Generation-3 Turn-Completion PUCT Results

**Classification:** `turn_completion_no_search_gain`

The result records every qualification gate, provenance hash, overlap audit, and whether incomplete budget-boundary extensions occurred.

```json
{
  "classification": "turn_completion_no_search_gain",
  "aggregate_metrics": {
    "full": {
      "baseline": {
        "mean_regret": 0.30383680555555553,
        "best_reference_action_agreement": 0.5520833333333334,
        "top_two_agreement": 0.7395833333333334,
        "catastrophic_miss_rate": 0.2916666666666667,
        "p95_runtime_seconds": 0.11141255949041788
      },
      "candidate": {
        "mean_regret": 0.367421875,
        "best_reference_action_agreement": 0.5,
        "top_two_agreement": 0.7291666666666666,
        "catastrophic_miss_rate": 0.3020833333333333,
        "p95_runtime_seconds": 0.10957587597658838
      },
      "paired_hierarchical_bootstrap": {
        "mean": 0.06358506944444445,
        "lower_95": -0.013805338541666665,
        "upper_95": 0.1507700737847222,
        "samples": 10000,
        "seed": 275001
      }
    },
    "root_extra_turn": {
      "baseline": {
        "mean_regret": 0.3965625,
        "best_reference_action_agreement": 0.609375,
        "top_two_agreement": 0.78125,
        "catastrophic_miss_rate": 0.34375,
        "p95_runtime_seconds": 0.12058594552800059
      },
      "candidate": {
        "mean_regret": 0.50609375,
        "best_reference_action_agreement": 0.515625,
        "top_two_agreement": 0.75,
        "catastrophic_miss_rate": 0.40625,
        "p95_runtime_seconds": 0.11862532019731588
      },
      "paired_hierarchical_bootstrap": {
        "mean": 0.10953125000000001,
        "lower_95": -0.005961914062499997,
        "upper_95": 0.2358743489583333,
        "samples": 10000,
        "seed": 275001
      }
    },
    "no_root_extra_turn": {
      "baseline": {
        "mean_regret": 0.11838541666666667,
        "best_reference_action_agreement": 0.4375,
        "top_two_agreement": 0.65625,
        "catastrophic_miss_rate": 0.1875,
        "p95_runtime_seconds": 0.10123615697375499
      },
      "candidate": {
        "mean_regret": 0.09007812500000001,
        "best_reference_action_agreement": 0.46875,
        "top_two_agreement": 0.6875,
        "catastrophic_miss_rate": 0.09375,
        "p95_runtime_seconds": 0.09207897368469276
      },
      "paired_hierarchical_bootstrap": {
        "mean": -0.028307291666666668,
        "lower_95": -0.06815104166666669,
        "upper_95": 0.0015625000000000014,
        "samples": 10000,
        "seed": 275001
      }
    }
  },
  "invariants": {
    "budget_ok": true,
    "invariants_ok": true,
    "artifact_unchanged": true,
    "registry_unchanged": true
  },
  "elapsed_seconds": 75.81282289500814
}
```
