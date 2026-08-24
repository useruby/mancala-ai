# Shadow-Target Policy Distillation

**Classification:** `calibrated_distillation_still_unsafe`

**Recommended follow-up:** Test online or iteratively refreshed search-target generation rather than a static cache.

## Calibration

| Lane | Batch | Primary norm | Auxiliary norm | Raw ratio | Cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| shadow_sensitive | 1 | 0.00797855854 | 0.151517496 | 18.990585 | 0.537341 |
| shadow_sensitive | 2 | 0.00768181076 | 0.154770046 | 20.147599 | -0.059520 |
| shadow_sensitive | 3 | 0.00594586786 | 0.141192466 | 23.746318 | 0.250189 |
| shadow_sensitive | 4 | 0.00741512422 | 0.158143967 | 21.327218 | 0.288924 |
| parent_sensitive | 1 | 0.00797855854 | 0.152631208 | 19.130173 | 0.450308 |
| parent_sensitive | 2 | 0.00768181076 | 0.146332815 | 19.049261 | -0.188516 |
| parent_sensitive | 3 | 0.00594586786 | 0.148605138 | 24.993011 | 0.334539 |
| parent_sensitive | 4 | 0.00741512422 | 0.153937951 | 20.759996 | 0.109327 |
| shadow_random25 | 1 | 0.00797855854 | 0.131089017 | 16.430163 | 0.831667 |
| shadow_random25 | 2 | 0.00768181076 | 0.122949421 | 16.005266 | -0.201659 |
| shadow_random25 | 3 | 0.00594586786 | 0.143919453 | 24.204953 | 0.442190 |
| shadow_random25 | 4 | 0.00741512422 | 0.107852019 | 14.544870 | 0.069594 |

```json
{
  "acceptance_max_weighted_ratio": 0.49999999999999994,
  "behavior_loss_weight": 0.020005593131945318,
  "max_configured_weight": 0.25,
  "max_raw_ratio": 24.99301053971703,
  "target_weighted_ratio": 0.5
}
```

## Gradient Geometry

Shadow-sensitive auxiliary gradients are more aligned with the primary objective than the parent anchor by median cosine (+0.047624); shadow-vs-parent auxiliary cosine is 0.815130 median.

```json
{
  "per_lane_cosine": {
    "parent_sensitive": {
      "max": 0.45030812997448805,
      "median": 0.22193296607346708,
      "min": -0.18851596591917402
    },
    "shadow_random25": {
      "max": 0.8316674904004969,
      "median": 0.25589181029160024,
      "min": -0.2016586646663036
    },
    "shadow_sensitive": {
      "max": 0.537341364578625,
      "median": 0.269556606470853,
      "min": -0.05951967578814945
    }
  },
  "shadow_sensitive_vs_parent_sensitive": [
    0.8032302614797536,
    0.8125874145428521,
    0.8176732815569077,
    0.8299836462409691
  ]
}
```

## Checkpoint Metrics

| Lane | Step | CE(search) | CE(P1) | CE(beta095) | Fit | L1 | JS | Top-1 | Adapter norm | Movement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_continue | 1 | 1.100517 | 0.947238 | 0.954902 | 0.3462 | 0.001860 | 7.57e-07 | 0.0020 | 0.002631 | 0.002631 |
| baseline_continue | 4 | 1.100463 | 0.947239 | 0.954900 | 0.4034 | 0.002157 | 1.02e-06 | 0.0022 | 0.003057 | 0.003057 |
| baseline_continue | 16 | 1.100276 | 0.947243 | 0.954895 | 0.6013 | 0.003115 | 2.11e-06 | 0.0028 | 0.004595 | 0.004595 |
| shadow_sensitive | 1 | 1.100516 | 0.947238 | 0.954902 | 0.3468 | 0.001862 | 7.59e-07 | 0.0020 | 0.002636 | 0.002636 |
| shadow_sensitive | 4 | 1.100457 | 0.947239 | 0.954900 | 0.4088 | 0.002185 | 1.04e-06 | 0.0022 | 0.003102 | 0.003102 |
| shadow_sensitive | 16 | 1.100235 | 0.947244 | 0.954894 | 0.6444 | 0.003328 | 2.42e-06 | 0.0030 | 0.004958 | 0.004958 |
| parent_sensitive | 1 | 1.100516 | 0.947238 | 0.954902 | 0.3467 | 0.001862 | 7.58e-07 | 0.0020 | 0.002636 | 0.002636 |
| parent_sensitive | 4 | 1.100458 | 0.947239 | 0.954900 | 0.4086 | 0.002182 | 1.04e-06 | 0.0022 | 0.003101 | 0.003101 |
| parent_sensitive | 16 | 1.100237 | 0.947244 | 0.954894 | 0.6428 | 0.003315 | 2.40e-06 | 0.0030 | 0.004948 | 0.004948 |
| shadow_random25 | 1 | 1.100516 | 0.947238 | 0.954902 | 0.3468 | 0.001863 | 7.59e-07 | 0.0020 | 0.002635 | 0.002635 |
| shadow_random25 | 4 | 1.100457 | 0.947239 | 0.954900 | 0.4089 | 0.002189 | 1.05e-06 | 0.0022 | 0.003098 | 0.003098 |
| shadow_random25 | 16 | 1.100234 | 0.947245 | 0.954894 | 0.6452 | 0.003364 | 2.47e-06 | 0.0029 | 0.004935 | 0.004935 |

Anchor CE and cached ordinary-search anchor metrics are in the JSON summary for every checkpoint.

## Evaluation

```json
{
  "arena_matrix": {
    "baseline_continue:1": {
      "1200:1200": {
        "opening_bootstrap_ci": {
          "lower_95": -0.0390625,
          "samples": 10000,
          "unique_openings": 128,
          "upper_95": -0.00390625
        },
        "paired_candidate_effect": -0.01953125,
        "safe": false,
        "seat_a_effect": -0.0390625,
        "seat_b_effect": 0.0,
        "win_draw_loss": {
          "draws": 132,
          "losses": 200,
          "wins": 180
        }
      },
      "384:256": {
        "opening_bootstrap_ci": {
          "lower_95": -0.01953125,
          "samples": 10000,
          "unique_openings": 128,
          "upper_95": -0.001953125
        },
        "paired_candidate_effect": -0.009765625,
        "safe": true,
        "seat_a_effect": -0.01953125,
        "seat_b_effect": 0.0,
        "win_draw_loss": {
          "draws": 48,
          "losses": 74,
          "wins": 390
        }
      }
    },
    "baseline_continue:16": {
      "1200:1200": {
        "opening_bootstrap_ci": {
          "lower_95": -0.0390625,
          "samples": 10000,
          "unique_openings": 128,
          "upper_95": -0.00390625
        },
        "paired_candidate_effect": -0.01953125,
        "safe": false,
        "seat_a_effect": -0.0390625,
        "seat_b_effect": 0.0,
        "win_draw_loss": {
          "draws": 132,
          "losses": 200,
          "wins": 180
        }
      },
      "384:256": {
        "opening_bootstrap_ci": {
          "lower_95": -0.0390625,
          "samples": 10000,
          "unique_openings": 128,
          "upper_95": -0.00390625
        },
        "paired_candidate_effect": -0.01953125,
        "safe": false,
        "seat_a_effect": -0.0390625,
        "seat_b_effect": 0.0,
        "win_draw_loss": {
          "draws": 38,
          "losses": 84,
          "wins": 390
        }
      }
    },
    "baseline_continue:4": {
      "1200:1200": {
        "opening_bootstrap_ci": {
          "lower_95": -0.0390625,
          "samples": 10000,
          "unique_openings": 128,
          "upper_95": -0.00390625
        },
        "paired_candidate_effect": -0.01953125,
        "safe": false,
        "seat_a_effect": -0.0390625,
        "seat_b_effect": 0.0,
        "win_draw_loss": {
          "draws": 132,
          "losses": 200,
          "wins": 180
        }
      },
      "384:256": {
        "opening_bootstrap_ci": {
          "lower_95": -0.01953125,
          "samples": 10000,
          "unique_openings": 128,
          "upper_95": -0.001953125
        },
        "paired_candidate_effect": -0.009765625,
        "safe": true,
        "seat_a_effect": -0.01953125,
        "seat_b_effect": 0.0,
        "win_draw_loss": {
          "draws": 48,
          "losses": 74,
          "wins": 390
        }
      }
    },
    "parent_sensitive:1": {
      "1200:1200": {
        "opening_bootstrap_ci": {
          "lower_95": -0.0390625,
          "samples": 10000,
          "unique_openings": 128,
          "upper_95": -0.00390625
        },
        "paired_candidate_effect": -0.01953125,
        "safe": false,
        "seat_a_effect": -0.0390625,
        "seat_b_effect": 0.0,
        "win_draw_loss": {
          "draws": 132,
          "losses": 200,
          "wins": 180
        }
      },
      "384:256": {
        "opening_bootstrap_ci": {
          "lower_95": -0.01953125,
          "samples": 10000,
          "unique_openings": 128,
          "upper_95": -0.001953125
        },
        "paired_candidate_effect": -0.009765625,
        "safe": true,
        "seat_a_effect": -0.01953125,
        "seat_b_effect": 0.0,
        "win_draw_loss": {
          "draws": 48,
          "losses": 74,
          "wins": 390
        }
      }
    },
    "parent_sensitive:16": {
      "1200:1200": {
        "opening_bootstrap_ci": {
          "lower_95": -0.0390625,
          "samples": 10000,
          "unique_openings": 128,
          "upper_95": -0.00390625
        },
        "paired_candidate_effect": -0.01953125,
        "safe": false,
        "seat_a_effect": -0.0390625,
        "seat_b_effect": 0.0,
        "win_draw_loss": {
          "draws": 132,
          "losses": 200,
          "wins": 180
        }
      },
      "384:256": {
        "opening_bootstrap_ci": {
          "lower_95": -0.0390625,
          "samples": 10000,
          "unique_openings": 128,
          "upper_95": -0.00390625
        },
        "paired_candidate_effect": -0.01953125,
        "safe": false,
        "seat_a_effect": -0.0390625,
        "seat_b_effect": 0.0,
        "win_draw_loss": {
          "draws": 38,
          "losses": 84,
          "wins": 390
        }
      }
    },
    "parent_sensitive:4": {
      "1200:1200": {
        "opening_bootstrap_ci": {
          "lower_95": -0.0390625,
          "samples": 10000,
          "unique_openings": 128,
          "upper_95": -0.00390625
        },
        "paired_candidate_effect": -0.01953125,
        "safe": false,
        "seat_a_effect": -0.0390625,
        "seat_b_effect": 0.0,
        "win_draw_loss": {
          "draws": 132,
          "losses": 200,
          "wins": 180
        }
      },
      "384:256": {
        "opening_bootstrap_ci": {
          "lower_95": -0.01953125,
          "samples": 10000,
          "unique_openings": 128,
          "upper_95": -0.001953125
        },
        "paired_candidate_effect": -0.009765625,
        "safe": true,
        "seat_a_effect": -0.01953125,
        "seat_b_effect": 0.0,
        "win_draw_loss": {
          "draws": 48,
          "losses": 74,
          "wins": 390
        }
      }
    },
    "shadow_random25:1": {
      "1200:1200": {
        "opening_bootstrap_ci": {
          "lower_95": -0.0390625,
          "samples": 10000,
          "unique_openings": 128,
          "upper_95": -0.00390625
        },
        "paired_candidate_effect": -0.01953125,
        "safe": false,
        "seat_a_effect": -0.0390625,
        "seat_b_effect": 0.0,
        "win_draw_loss": {
          "draws": 132,
          "losses": 200,
          "wins": 180
        }
      },
      "384:256": {
        "opening_bootstrap_ci": {
          "lower_95": -0.01953125,
          "samples": 10000,
          "unique_openings": 128,
          "upper_95": -0.001953125
        },
        "paired_candidate_effect": -0.009765625,
        "safe": true,
        "seat_a_effect": -0.01953125,
        "seat_b_effect": 0.0,
        "win_draw_loss": {
          "draws": 48,
          "losses": 74,
          "wins": 390
        }
      }
    },
    "shadow_random25:16": {
      "1200:1200": {
        "opening_bootstrap_ci": {
          "lower_95": -0.0390625,
          "samples": 10000,
          "unique_openings": 128,
          "upper_95": -0.00390625
        },
        "paired_candidate_effect": -0.01953125,
        "safe": false,
        "seat_a_effect": -0.0390625,
        "seat_b_effect": 0.0,
        "win_draw_loss": {
          "draws": 132,
          "losses": 200,
          "wins": 180
        }
      },
      "384:256": {
        "opening_bootstrap_ci": {
          "lower_95": -0.0390625,
          "samples": 10000,
          "unique_openings": 128,
          "upper_95": -0.00390625
        },
        "paired_candidate_effect": -0.01953125,
        "safe": false,
        "seat_a_effect": -0.0390625,
        "seat_b_effect": 0.0,
        "win_draw_loss": {
          "draws": 38,
          "losses": 84,
          "wins": 390
        }
      }
    },
    "shadow_random25:4": {
      "1200:1200": {
        "opening_bootstrap_ci": {
          "lower_95": -0.0390625,
          "samples": 10000,
          "unique_openings": 128,
          "upper_95": -0.00390625
        },
        "paired_candidate_effect": -0.01953125,
        "safe": false,
        "seat_a_effect": -0.0390625,
        "seat_b_effect": 0.0,
        "win_draw_loss": {
          "draws": 132,
          "losses": 200,
          "wins": 180
        }
      },
      "384:256": {
        "opening_bootstrap_ci": {
          "lower_95": -0.01953125,
          "samples": 10000,
          "unique_openings": 128,
          "upper_95": -0.001953125
        },
        "paired_candidate_effect": -0.009765625,
        "safe": true,
        "seat_a_effect": -0.01953125,
        "seat_b_effect": 0.0,
        "win_draw_loss": {
          "draws": 48,
          "losses": 74,
          "wins": 390
        }
      }
    },
    "shadow_sensitive:1": {
      "1200:1200": {
        "opening_bootstrap_ci": {
          "lower_95": -0.0390625,
          "samples": 10000,
          "unique_openings": 128,
          "upper_95": -0.00390625
        },
        "paired_candidate_effect": -0.01953125,
        "safe": false,
        "seat_a_effect": -0.0390625,
        "seat_b_effect": 0.0,
        "win_draw_loss": {
          "draws": 132,
          "losses": 200,
          "wins": 180
        }
      },
      "384:256": {
        "opening_bootstrap_ci": {
          "lower_95": -0.01953125,
          "samples": 10000,
          "unique_openings": 128,
          "upper_95": -0.001953125
        },
        "paired_candidate_effect": -0.009765625,
        "safe": true,
        "seat_a_effect": -0.01953125,
        "seat_b_effect": 0.0,
        "win_draw_loss": {
          "draws": 48,
          "losses": 74,
          "wins": 390
        }
      }
    },
    "shadow_sensitive:16": {
      "1200:1200": {
        "opening_bootstrap_ci": {
          "lower_95": -0.0390625,
          "samples": 10000,
          "unique_openings": 128,
          "upper_95": -0.00390625
        },
        "paired_candidate_effect": -0.01953125,
        "safe": false,
        "seat_a_effect": -0.0390625,
        "seat_b_effect": 0.0,
        "win_draw_loss": {
          "draws": 132,
          "losses": 200,
          "wins": 180
        }
      },
      "384:256": {
        "opening_bootstrap_ci": {
          "lower_95": -0.0390625,
          "samples": 10000,
          "unique_openings": 128,
          "upper_95": -0.00390625
        },
        "paired_candidate_effect": -0.01953125,
        "safe": false,
        "seat_a_effect": -0.0390625,
        "seat_b_effect": 0.0,
        "win_draw_loss": {
          "draws": 38,
          "losses": 84,
          "wins": 390
        }
      }
    },
    "shadow_sensitive:4": {
      "1200:1200": {
        "opening_bootstrap_ci": {
          "lower_95": -0.0390625,
          "samples": 10000,
          "unique_openings": 128,
          "upper_95": -0.00390625
        },
        "paired_candidate_effect": -0.01953125,
        "safe": false,
        "seat_a_effect": -0.0390625,
        "seat_b_effect": 0.0,
        "win_draw_loss": {
          "draws": 132,
          "losses": 200,
          "wins": 180
        }
      },
      "384:256": {
        "opening_bootstrap_ci": {
          "lower_95": -0.01953125,
          "samples": 10000,
          "unique_openings": 128,
          "upper_95": -0.001953125
        },
        "paired_candidate_effect": -0.009765625,
        "safe": true,
        "seat_a_effect": -0.01953125,
        "seat_b_effect": 0.0,
        "win_draw_loss": {
          "draws": 48,
          "losses": 74,
          "wins": 390
        }
      }
    }
  },
  "frozen_diagnostics": {
    "baseline_continue:1": {
      "new_divergence_rate": 0.0,
      "rescue_rate": 0.075
    },
    "baseline_continue:16": {
      "new_divergence_rate": 0.0,
      "rescue_rate": 0.3
    },
    "baseline_continue:4": {
      "new_divergence_rate": 0.0,
      "rescue_rate": 0.125
    },
    "parent_sensitive:1": {
      "new_divergence_rate": 0.0,
      "rescue_rate": 0.075
    },
    "parent_sensitive:16": {
      "new_divergence_rate": 0.0,
      "rescue_rate": 0.3
    },
    "parent_sensitive:4": {
      "new_divergence_rate": 0.0,
      "rescue_rate": 0.175
    },
    "shadow_random25:1": {
      "new_divergence_rate": 0.0,
      "rescue_rate": 0.075
    },
    "shadow_random25:16": {
      "new_divergence_rate": 0.0,
      "rescue_rate": 0.3
    },
    "shadow_random25:4": {
      "new_divergence_rate": 0.0,
      "rescue_rate": 0.15
    },
    "shadow_sensitive:1": {
      "new_divergence_rate": 0.0,
      "rescue_rate": 0.075
    },
    "shadow_sensitive:16": {
      "new_divergence_rate": 0.0,
      "rescue_rate": 0.3
    },
    "shadow_sensitive:4": {
      "new_divergence_rate": 0.0,
      "rescue_rate": 0.175
    }
  },
  "invariants": {
    "artifact_hashes": true,
    "calibration_acceptance": true,
    "inherited_parameters_byte_identical": true,
    "selector_manifest_hash": true,
    "target_cache_hash": true
  },
  "p0_gate": {}
}
```
