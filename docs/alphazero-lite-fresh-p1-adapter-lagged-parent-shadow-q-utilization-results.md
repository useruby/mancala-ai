# Lagged-Parent Shadow-Q Policy Utilization

**Classification:** `candidate_policy_survival_budget_dependent`

**Recommended follow-up:** Test a prespecified partial parent-Q usage rule at both budgets, with candidate-policy collapse at 1200 as the explicit failure mode.

## Three-Way Move Agreement

| Population | Budget | All agree | Parent preserved | Candidate survives | Third way |
| --- | ---: | ---: | ---: | ---: | ---: |
| replay | 384 | 4064 | 22 | 4 | 6 |
| replay | 1200 | 4056 | 39 | 1 | 0 |
| canonical_treatment | 384 | 9282 | 50 | 0 | 0 |
| canonical_treatment | 1200 | 10152 | 34 | 0 | 10 |

## Visit Distributions

| Population | Budget | Pair | Mean JS | P95 JS | Mean L1 | P95 L1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| replay | 384 | a16_p1 | 0.000208 | 0.000288 | 0.008331 | 0.036458 |
| replay | 384 | shadow_p1 | 0.000009 | 0.000039 | 0.002041 | 0.005208 |
| replay | 384 | shadow_a16 | 0.000213 | 0.000289 | 0.008910 | 0.036458 |
| replay | 1200 | a16_p1 | 0.000495 | 0.000296 | 0.011031 | 0.036667 |
| replay | 1200 | shadow_p1 | 0.000003 | 0.000014 | 0.001394 | 0.003333 |
| replay | 1200 | shadow_a16 | 0.000497 | 0.000295 | 0.011228 | 0.036667 |
| canonical_treatment | 384 | a16_p1 | 0.000355 | 0.000263 | 0.009783 | 0.031250 |
| canonical_treatment | 384 | shadow_p1 | 0.000009 | 0.000040 | 0.001912 | 0.005208 |
| canonical_treatment | 384 | shadow_a16 | 0.000359 | 0.000237 | 0.010183 | 0.031250 |
| canonical_treatment | 1200 | a16_p1 | 0.000085 | 0.000289 | 0.007827 | 0.035000 |
| canonical_treatment | 1200 | shadow_p1 | 0.000004 | 0.000007 | 0.001331 | 0.003333 |
| canonical_treatment | 1200 | shadow_a16 | 0.000084 | 0.000250 | 0.008025 | 0.036667 |

## Canonical Trajectories

```json
{
  "1200:1200": {
    "differing_moves_per_game": {
      "count": 512,
      "mean": 0.37109375,
      "p50": 0.0,
      "p90": 0.0,
      "p95": 0.0,
      "p99": 19.0
    },
    "final_outcome_agreement_rate": 1.0,
    "first_divergence_ply": {
      "count": 10,
      "mean": 20.0,
      "p50": 20.0,
      "p90": 20.0,
      "p95": 20.0,
      "p99": 20.0
    },
    "games": 512,
    "identical_complete_trajectory_rate": 0.98046875,
    "reconvergence": "not assessed after divergence; full state reconstruction is deliberately limited to shared pre-move states"
  },
  "384:256": {
    "differing_moves_per_game": {
      "count": 512,
      "mean": 0.0,
      "p50": 0.0,
      "p90": 0.0,
      "p95": 0.0,
      "p99": 0.0
    },
    "final_outcome_agreement_rate": 1.0,
    "first_divergence_ply": {
      "count": 0,
      "mean": null,
      "p50": null,
      "p90": null,
      "p95": null,
      "p99": null
    },
    "games": 512,
    "identical_complete_trajectory_rate": 1.0,
    "reconvergence": "not assessed after divergence; full state reconstruction is deliberately limited to shared pre-move states"
  }
}
```

## Frozen 40

```json
{
  "rescue_decomposition": {
    "equal_a16": 1,
    "equal_p1": 39
  },
  "rescued_roots": 39,
  "unrescued_root": [
    {
      "a16_move": 5,
      "no_future_information": true,
      "p1_move": 3,
      "primary": true,
      "self_identity": true,
      "shadow_move": 5,
      "state_hash": "362958a9d30519f98e27a71256c89f1acba0a61e3d93e609adb59609bff61674"
    }
  ]
}
```

## 384 To 1200 Transitions

```json
{
  "candidate_survives_384_to_parent_preserved_1200": 0,
  "paired_roots": 4096,
  "parent_preserved_384_to_candidate_survives_1200": 0,
  "shadow_differs_parent_384_to_equals_parent_1200": 10
}
```
