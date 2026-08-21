# PR #214 Search-Margin Sensitivity

**Classification:** `margin_explains_first_flip_not_final_instability`

**Recommended follow-up:** Model post-divergence Q/visit amplification rather than another policy constraint.

## Predictor Comparison

| Predictor | AUPRC move @384 | AUPRC move @1200 | Spearman JS @384 | Spearman JS @1200 |
| --- | ---: | ---: | ---: | ---: |
| max_pressure_ratio | 0.007274463878330963 | 0.007695613274295689 | -0.09153871029087478 | -0.12735927517892298 |
| max_flip_excess | 0.00590903094737688 | 0.008849062163701535 | -0.05266207154744299 | -0.09620709071205985 |
| policy_l1 | 0.012223371545548745 | 0.010725447636572884 | 0.24236906690635532 | 0.2858204965080816 |
| max_action_delta | 0.010801455337412894 | 0.01091266595278725 | 0.20870322163809354 | 0.25643145170539017 |
| minimum_parent_puct_margin | 0.006405105653731659 | 0.011437306680896223 | 0.1318280906243423 | 0.15607618232321957 |

## Held-Out PR #220 States

| State hash | Pressure percentile | Flip-excess percentile | L1 percentile | Max-delta percentile |
| --- | ---: | ---: | ---: | ---: |
| `6d7a71e6007e3943a223024aa1515659887ecb275806402f645a5b6367f942fe` | 26.20 | 28.76 | 33.25 | 20.41 |
| `cd6293ed266fb1db26224cb9208d2494bd9f35be6d37526c55b2a64437e98d8c` | 3.37 | 67.48 | 48.32 | 57.13 |

## Validation

```json
{
  "bootstrap": {
    "1200": {
      "auprc_pressure_minus_l1": {
        "estimate": -0.0030672450765013622,
        "lower_95": -0.005652344215149043,
        "samples": 10000,
        "upper_95": -0.0008599533189922481
      },
      "spearman_pressure_minus_l1": {
        "estimate": -0.4132117757381271,
        "lower_95": -0.4524443873783652,
        "samples": 10000,
        "upper_95": -0.3734911433693063
      }
    },
    "384": {
      "auprc_pressure_minus_l1": {
        "estimate": -0.005306852246420452,
        "lower_95": -0.014107110625186459,
        "samples": 10000,
        "upper_95": 0.0015372525403247659
      },
      "spearman_pressure_minus_l1": {
        "estimate": -0.33382945774215467,
        "lower_95": -0.37224139749959906,
        "samples": 10000,
        "upper_95": -0.29459056551622936
      }
    }
  },
  "invariants": {
    "a16_state_hash": true,
    "continuous_1200_prefix_equals_standalone_384": true,
    "p1_checkpoint_hash": true,
    "pr220_parent_commit": true,
    "pre_divergence_trace_statistics": true,
    "replay_hash": true,
    "sample_state_round_trip": true
  },
  "search_budget": {
    "cross_before_384": {
      "count": 4027,
      "mean_visit_js_1200": 0.0005037269866198285,
      "mean_visit_js_384": 0.00021141707503171173,
      "root_move_difference_rate_1200": 0.009932952570151478,
      "root_move_difference_rate_384": 0.00645641917059846
    },
    "first_cross_after_384": {
      "count": 6,
      "mean_visit_js_1200": 8.222978752960347e-07,
      "mean_visit_js_384": 0.0,
      "root_move_difference_rate_1200": 0.0,
      "root_move_difference_rate_384": 0.0
    },
    "never_cross_by_1200": {
      "count": 63,
      "mean_visit_js_1200": 0.0,
      "mean_visit_js_384": 0.0,
      "root_move_difference_rate_1200": 0.0,
      "root_move_difference_rate_384": 0.0
    }
  }
}
```
