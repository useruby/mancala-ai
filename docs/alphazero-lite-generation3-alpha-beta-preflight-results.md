# Generation-3 Alpha-Beta Preflight Results

**Classification:** `alpha_beta_regresses_subsets`

This diagnostic used a fresh 64-state standard-start corpus only. It did not train, generate replay/self-play, run an arena, consume a suite, mutate an artifact, mutate the registry, or change the A16 closeout ledger.

```json
{
  "aggregate_metrics": {
    "artifact_value": {
      "alpha_beta": {
        "catastrophic_miss_rate": 0.203125,
        "exact_best_agreement": 0.484375,
        "mean_regret": 0.2580989583333333,
        "mean_runtime_seconds": 0.024863304346581572,
        "top_two_agreement": 0.828125
      },
      "ordinary_puct": {
        "catastrophic_miss_rate": 0.28125,
        "exact_best_agreement": 0.421875,
        "mean_regret": 0.30701171875,
        "mean_runtime_seconds": 0.04241172567283987,
        "top_two_agreement": 0.796875
      },
      "paired_hierarchical_bootstrap": {
        "lower_95": -0.16614746093750002,
        "mean": -0.048912760416666666,
        "samples": 10000,
        "seed": 270001,
        "upper_95": 0.06502636718749999
      }
    },
    "heuristic_value": {
      "alpha_beta": {
        "catastrophic_miss_rate": 0.234375,
        "exact_best_agreement": 0.578125,
        "mean_regret": 0.24579427083333333,
        "mean_runtime_seconds": 0.007194192981842207,
        "top_two_agreement": 0.84375
      },
      "ordinary_puct": {
        "catastrophic_miss_rate": 0.28125,
        "exact_best_agreement": 0.421875,
        "mean_regret": 0.30701171875,
        "mean_runtime_seconds": 0.04241172567283987,
        "top_two_agreement": 0.796875
      },
      "paired_hierarchical_bootstrap": {
        "lower_95": -0.18916080729166665,
        "mean": -0.06121744791666666,
        "samples": 10000,
        "seed": 270001,
        "upper_95": 0.06038297526041658
      }
    }
  },
  "classification": "alpha_beta_regresses_subsets",
  "determinism": {
    "checked": 24,
    "passed": true,
    "results": [
      {
        "equal_excluding_runtime": true,
        "lane": "artifact_value",
        "state_hash": "03be0097927b4f3ca454e4b19f23b766d9d41fc6548d2e9c55bee31847e81234"
      },
      {
        "equal_excluding_runtime": true,
        "lane": "heuristic_value",
        "state_hash": "03be0097927b4f3ca454e4b19f23b766d9d41fc6548d2e9c55bee31847e81234"
      },
      {
        "equal_excluding_runtime": true,
        "lane": "artifact_value",
        "state_hash": "05e83b74e8862dc395561d16f367bc45e1363e2e5091cc186a039090b51abac6"
      },
      {
        "equal_excluding_runtime": true,
        "lane": "heuristic_value",
        "state_hash": "05e83b74e8862dc395561d16f367bc45e1363e2e5091cc186a039090b51abac6"
      },
      {
        "equal_excluding_runtime": true,
        "lane": "artifact_value",
        "state_hash": "06c879407b80f7098f947040b85d66946c6a812dbbfc962cda2b1d95523f3b4b"
      },
      {
        "equal_excluding_runtime": true,
        "lane": "heuristic_value",
        "state_hash": "06c879407b80f7098f947040b85d66946c6a812dbbfc962cda2b1d95523f3b4b"
      },
      {
        "equal_excluding_runtime": true,
        "lane": "artifact_value",
        "state_hash": "090e78338a1e609ee89f9652cb265d239806ea6c52006e144ac6cd6cd0e3fbc2"
      },
      {
        "equal_excluding_runtime": true,
        "lane": "heuristic_value",
        "state_hash": "090e78338a1e609ee89f9652cb265d239806ea6c52006e144ac6cd6cd0e3fbc2"
      },
      {
        "equal_excluding_runtime": true,
        "lane": "artifact_value",
        "state_hash": "0d178666b789c488d7122dc34f295da246ad2244b6e48a17ae3e8a7296e56968"
      },
      {
        "equal_excluding_runtime": true,
        "lane": "heuristic_value",
        "state_hash": "0d178666b789c488d7122dc34f295da246ad2244b6e48a17ae3e8a7296e56968"
      },
      {
        "equal_excluding_runtime": true,
        "lane": "artifact_value",
        "state_hash": "10d1b78ed6b6d233df68c269c2d6089f4242fbe249f38d4dd43027da2b878e28"
      },
      {
        "equal_excluding_runtime": true,
        "lane": "heuristic_value",
        "state_hash": "10d1b78ed6b6d233df68c269c2d6089f4242fbe249f38d4dd43027da2b878e28"
      },
      {
        "equal_excluding_runtime": true,
        "lane": "artifact_value",
        "state_hash": "1a31416137f9268608bf8acac0d5926b8e756db48f95b2637330b16ca5ddd574"
      },
      {
        "equal_excluding_runtime": true,
        "lane": "heuristic_value",
        "state_hash": "1a31416137f9268608bf8acac0d5926b8e756db48f95b2637330b16ca5ddd574"
      },
      {
        "equal_excluding_runtime": true,
        "lane": "artifact_value",
        "state_hash": "1bd0b2221655d6cdab578222f34ece17f547c57c17f1444d89d72eebac12b350"
      },
      {
        "equal_excluding_runtime": true,
        "lane": "heuristic_value",
        "state_hash": "1bd0b2221655d6cdab578222f34ece17f547c57c17f1444d89d72eebac12b350"
      },
      {
        "equal_excluding_runtime": true,
        "lane": "artifact_value",
        "state_hash": "1c6954c4352a48f2b094eadf20012fbe99904cc6b49acd24e0d7d61537606acf"
      },
      {
        "equal_excluding_runtime": true,
        "lane": "heuristic_value",
        "state_hash": "1c6954c4352a48f2b094eadf20012fbe99904cc6b49acd24e0d7d61537606acf"
      },
      {
        "equal_excluding_runtime": true,
        "lane": "artifact_value",
        "state_hash": "1ccfd36a5adfa106d3d3dcfa36a434ac075c52a3b45075f1274402b18ff45df4"
      },
      {
        "equal_excluding_runtime": true,
        "lane": "heuristic_value",
        "state_hash": "1ccfd36a5adfa106d3d3dcfa36a434ac075c52a3b45075f1274402b18ff45df4"
      },
      {
        "equal_excluding_runtime": true,
        "lane": "artifact_value",
        "state_hash": "1dc98f8ed45375d70f99a96893d050caa9145495c481d3364b973c673b7ae214"
      },
      {
        "equal_excluding_runtime": true,
        "lane": "heuristic_value",
        "state_hash": "1dc98f8ed45375d70f99a96893d050caa9145495c481d3364b973c673b7ae214"
      },
      {
        "equal_excluding_runtime": true,
        "lane": "artifact_value",
        "state_hash": "1fe78743f3ba614b9219e2f6a27d3f4f2a318c3fc3e7c349859f021cd9fd3ee5"
      },
      {
        "equal_excluding_runtime": true,
        "lane": "heuristic_value",
        "state_hash": "1fe78743f3ba614b9219e2f6a27d3f4f2a318c3fc3e7c349859f021cd9fd3ee5"
      }
    ]
  },
  "guardrails": {
    "diagnostic_only": true,
    "no_arena": true,
    "no_artifact_mutation": true,
    "no_closeout_ledger_mutation": true,
    "no_existing_suite_consumed": true,
    "no_self_play_or_replay": true,
    "no_suite_registry_mutation": true,
    "no_training": true,
    "not_training_eligible": true
  },
  "invariants": {
    "after_protected_hashes": {
      "closeout_ledger": "6e6c1f0b2611efd0efb4d1c5e8a81b16a9d4425d694e76bc33cc1a2874be388d",
      "closeout_markdown": "2dab39dfb6c5aa0a0510fa0832ef0a1192c32c84cdfd40ecad190a2a30b265c8",
      "closeout_validator": "493ff77445a30fd56f18b6eaa3967fc5d85d737dbb80bbbfe558e3d4ca5c364a",
      "metadata": "eb87375b9aba39b270078a81fa3cff120240333d7b2b7993d1de072009be02d6",
      "weights": "8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a"
    },
    "before_protected_hashes": {
      "closeout_ledger": "6e6c1f0b2611efd0efb4d1c5e8a81b16a9d4425d694e76bc33cc1a2874be388d",
      "closeout_markdown": "2dab39dfb6c5aa0a0510fa0832ef0a1192c32c84cdfd40ecad190a2a30b265c8",
      "closeout_validator": "493ff77445a30fd56f18b6eaa3967fc5d85d737dbb80bbbfe558e3d4ca5c364a",
      "metadata": "eb87375b9aba39b270078a81fa3cff120240333d7b2b7993d1de072009be02d6",
      "weights": "8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a"
    },
    "budget_ok": true,
    "legal_selection_ok": true,
    "protected_hashes_unchanged": true
  },
  "lane_classifications": {
    "artifact_value": "regresses_subsets",
    "heuristic_value": "regresses_subsets"
  },
  "search_telemetry_totals": {
    "artifact_value": {
      "budget_utilization_mean": 1.0,
      "completed_depth_mean": 5.078125,
      "leaf_calls": 24576,
      "nodes": 44129
    },
    "heuristic_value": {
      "budget_utilization_mean": 1.0,
      "completed_depth_mean": 5.359375,
      "leaf_calls": 24576,
      "nodes": 47370
    }
  }
}
```

The artifact-value lane directly tests whether the frozen value head is compatible with alpha-beta. The heuristic lane isolates the search mechanism. Both lanes improved aggregate regret versus PUCT, but their paired confidence intervals crossed zero and preregistered subsets increased catastrophic misses. Therefore this run cannot distinguish a value-head incompatibility from a search-mechanism limitation as a qualified gain: both fail the fixed diagnostic gate. No training is recommended or permitted.
