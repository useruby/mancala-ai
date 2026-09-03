# PR #268 Gumbel Root Preflight

**Classification:** `gumbel_root_regresses_subsets`

This is a diagnostic-only, non-training corpus; no arena or sealed suite was consumed.

```json
{
  "classification": "gumbel_root_regresses_subsets",
  "frozen_model": {
    "artifact": "8eaa5765786468df9fade11727449af2263a5d0d6da0fa2138040ef2b8629d34",
    "snapshot": "f38a3a9db15a095870c0294150fe81e4719cf77102c4b3de316fcae5e1b9c3ff",
    "weights": "74e160734554dba9ecf4ecf3ac13b3e44526c4f6e7e5e52d9fe8bde0054eb789"
  },
  "guardrails": {
    "arena_run": false,
    "consumed_suite_registry_modified": false,
    "diagnostic_only": true,
    "not_training_eligible": true,
    "replay_created": false
  },
  "invariants": {
    "a16_unchanged": true,
    "budget_ok": true,
    "perspective_and_legal_selection_ok": true,
    "repeated_small_subset_deterministic": true
  },
  "paired_hierarchical_bootstrap": {
    "lower_95": 0.044857911426341254,
    "mean": 0.08078185655042201,
    "upper_95": 0.11984561240471639
  },
  "preregistration": {
    "candidate_set": "all legal root actions (maximum six)",
    "catastrophic_regret_threshold": 0.25,
    "continuation_simulations": 2400,
    "evaluation_budget": 384,
    "gumbel_scale": 1.0,
    "seeds": [
      267101,
      267102,
      267103
    ]
  }
}
```
