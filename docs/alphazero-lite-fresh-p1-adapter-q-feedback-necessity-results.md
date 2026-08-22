# Q-Feedback Necessity for Post-Divergence Amplification

**Classification:** `q_feedback_necessary_for_amplification`

**Recommended follow-up:** Add a diagnostic synchronized-backup or Q-feedback intervention on the frozen amplified roots to localize which backup differences cause persistence.

## Frozen Full Outcome

The frozen PR #222 full-amplified set contains 40 roots.

## Primary Causal Metric

```json
{
  "estimate": 0.0,
  "lower_95": 0.0,
  "samples": 10000,
  "upper_95": 0.0
}
```

## Root Identity Overlap

```json
{
  "full_count": 40,
  "full_only_count": 40,
  "intersection_count": 0,
  "jaccard": 0.0,
  "policy_only_count": 8,
  "policy_only_only_count": 8
}
```

## First-Divergence Timing

```json
{
  "same_action_pair_rate": 0.287109375,
  "same_depth_rate": 0.489013671875,
  "same_first_simulation_rate": 0.02587890625,
  "simulation_delta": {
    "estimate": -9.1859621869576,
    "lower_95": -10.642970493429209,
    "samples": 10000,
    "upper_95": -7.804599553682123
  }
}
```

## Visit-JS Causal Effect

```json
{
  "all_roots": {
    "1200": {
      "mean": {
        "estimate": 0.0004953035769276557,
        "lower_95": 0.00026995874373094705,
        "samples": 10000,
        "upper_95": 0.0007775974346785356
      },
      "median": {
        "estimate": 5.154782325465169e-07,
        "lower_95": 3.4824402062968887e-07,
        "samples": 10000,
        "upper_95": 7.55279473606084e-07
      }
    },
    "384": {
      "mean": {
        "estimate": 0.0002050549090881966,
        "lower_95": 9.166690662763141e-05,
        "samples": 10000,
        "upper_95": 0.0003739837404401578
      },
      "median": {
        "estimate": 0.0,
        "lower_95": 0.0,
        "samples": 10000,
        "upper_95": 0.0
      }
    }
  },
  "full_amplified": {
    "1200": {
      "mean": {
        "estimate": 0.035656932117437035,
        "lower_95": 0.015400156002327426,
        "samples": 10000,
        "upper_95": 0.06016598574173308
      },
      "median": {
        "estimate": 0.001544143678336522,
        "lower_95": 0.0002625455151896517,
        "samples": 10000,
        "upper_95": 0.003830314049528929
      }
    },
    "384": {
      "mean": {
        "estimate": 0.007262678991455457,
        "lower_95": 7.529726857719759e-05,
        "samples": 10000,
        "upper_95": 0.021370054806692004
      },
      "median": {
        "estimate": 1.2832746342275774e-05,
        "lower_95": 0.0,
        "samples": 10000,
        "upper_95": 4.742551911161158e-05
      }
    }
  },
  "washed_out": {
    "1200": {
      "mean": {
        "estimate": 1.0901154721057207e-06,
        "lower_95": 8.306864708245822e-07,
        "samples": 10000,
        "upper_95": 1.286066000652037e-06
      },
      "median": {
        "estimate": 0.0,
        "lower_95": 0.0,
        "samples": 10000,
        "upper_95": 0.0
      }
    },
    "384": {
      "mean": {
        "estimate": 4.276776516962606e-05,
        "lower_95": 3.1630703839836986e-05,
        "samples": 10000,
        "upper_95": 5.613858380657204e-05
      },
      "median": {
        "estimate": 0.0,
        "lower_95": 0.0,
        "samples": 10000,
        "upper_95": 0.0
      }
    }
  }
}
```

## Pure Visit Reinforcement

Policy-only retained roots: 8.

## Held-Out PR #220 States

```json
[
  {
    "full": {
      "at_1200": {
        "a16_root_move": 3,
        "a16_top1_top2_visit_margin": 25.0,
        "a16_visit_leader": 3,
        "p1_root_move": 1,
        "p1_top1_top2_visit_margin": 4.0,
        "p1_visit_leader": 1,
        "root_move_difference": true,
        "visit_js": 8.479123807419363e-05,
        "visit_l1": 30.0,
        "visit_leader_difference": true
      },
      "at_384": {
        "a16_root_move": 1,
        "a16_top1_top2_visit_margin": 16.0,
        "a16_visit_leader": 1,
        "p1_root_move": 1,
        "p1_top1_top2_visit_margin": 28.0,
        "p1_visit_leader": 1,
        "root_move_difference": false,
        "visit_js": 0.00014535455691206713,
        "visit_l1": 12.0,
        "visit_leader_difference": false
      },
      "first_divergence": {
        "action_pair": [
          5,
          3
        ],
        "depth": 0,
        "simulation": 28,
        "state_hash": "6d7a71e6007e3943a223024aa1515659887ecb275806402f645a5b6367f942fe"
      }
    },
    "policy_only": {
      "at_1200": {
        "a16_root_move": 1,
        "a16_top1_top2_visit_margin": 195.0,
        "a16_visit_leader": 1,
        "p1_root_move": 1,
        "p1_top1_top2_visit_margin": 195.0,
        "p1_visit_leader": 1,
        "root_move_difference": false,
        "visit_js": 0.0,
        "visit_l1": 0.0,
        "visit_leader_difference": false
      },
      "at_384": {
        "a16_root_move": 1,
        "a16_top1_top2_visit_margin": 62.0,
        "a16_visit_leader": 1,
        "p1_root_move": 1,
        "p1_top1_top2_visit_margin": 63.0,
        "p1_visit_leader": 1,
        "root_move_difference": false,
        "visit_js": 1.4502454158506488e-05,
        "visit_l1": 2.0,
        "visit_leader_difference": false
      },
      "first_divergence": {
        "action_pair": [
          5,
          2
        ],
        "depth": 2,
        "simulation": 33,
        "state_hash": "a7c1aed93fab2f2e72be7fe7fa0a8b1b6594d97f0cf032bbd5b13f704b30ee39"
      }
    },
    "removing_q_rescues_p1_root_move": true,
    "state_hash": "6d7a71e6007e3943a223024aa1515659887ecb275806402f645a5b6367f942fe"
  },
  {
    "full": {
      "at_1200": {
        "a16_root_move": 2,
        "a16_top1_top2_visit_margin": 175.0,
        "a16_visit_leader": 2,
        "p1_root_move": 0,
        "p1_top1_top2_visit_margin": 31.0,
        "p1_visit_leader": 0,
        "root_move_difference": true,
        "visit_js": 0.006530710209454418,
        "visit_l1": 208.0,
        "visit_leader_difference": true
      },
      "at_384": {
        "a16_root_move": 2,
        "a16_top1_top2_visit_margin": 49.0,
        "a16_visit_leader": 2,
        "p1_root_move": 2,
        "p1_top1_top2_visit_margin": 49.0,
        "p1_visit_leader": 2,
        "root_move_difference": false,
        "visit_js": 0.0,
        "visit_l1": 0.0,
        "visit_leader_difference": false
      },
      "first_divergence": {
        "action_pair": [
          5,
          2
        ],
        "depth": 1,
        "simulation": 50,
        "state_hash": "f5fd1f34118f5fedfc1c0dbca563a31e9c24e6b69fea7bd89a6e7b372352443b"
      }
    },
    "policy_only": {
      "at_1200": {
        "a16_root_move": 1,
        "a16_top1_top2_visit_margin": 28.0,
        "a16_visit_leader": 1,
        "p1_root_move": 1,
        "p1_top1_top2_visit_margin": 27.0,
        "p1_visit_leader": 1,
        "root_move_difference": false,
        "visit_js": 8.06828632312972e-07,
        "visit_l1": 2.0,
        "visit_leader_difference": false
      },
      "at_384": {
        "a16_root_move": 1,
        "a16_top1_top2_visit_margin": 9.0,
        "a16_visit_leader": 1,
        "p1_root_move": 1,
        "p1_top1_top2_visit_margin": 9.0,
        "p1_visit_leader": 1,
        "root_move_difference": false,
        "visit_js": 0.0,
        "visit_l1": 0.0,
        "visit_leader_difference": false
      },
      "first_divergence": {
        "action_pair": [
          4,
          0
        ],
        "depth": 1,
        "simulation": 48,
        "state_hash": "a1cc4794c63478d78db813edc55fd1f2d6be1910200dafcb2cebad2ccb8662d7"
      }
    },
    "removing_q_rescues_p1_root_move": true,
    "state_hash": "cd6293ed266fb1db26224cb9208d2494bd9f35be6d37526c55b2a64437e98d8c"
  }
]
```

## Invariants

```json
{
  "artifact_hashes": true,
  "full_amplified_1200_reproduces": true,
  "full_first_divergences_reproduce": true,
  "full_visit_js_reproduces": true,
  "no_first_divergence_invariant_failure": true,
  "policy_only_no_q_rank_divergence": true,
  "policy_only_prior_l1_matches_full": true,
  "policy_only_root_priors_match_full": true,
  "policy_only_zero_q_and_backups": true,
  "pr221_manifest_hash_and_order": true,
  "replay_hash": true,
  "trajectory_matches_puct_summary": true,
  "value_only_flat_identity": true
}
```
