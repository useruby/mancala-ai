# Generation-3 Implicit-Minimax PUCT Results

**Classification:** `implicit_minimax_no_search_gain`

The candidate improved mean regret and catastrophic misses but does not qualify because the preregistered paired-bootstrap 95% upper bound is not below zero. This diagnostic lane is closed; no variant or follow-up run was started.

```json
{
  "aggregate_metrics": {
    "baseline": {
      "mean_regret": 0.28186197916666667,
      "exact_best_agreement": 0.546875,
      "top_two_agreement": 0.828125,
      "catastrophic_miss_rate": 0.265625,
      "p95_runtime_seconds": 0.11368163633160293
    },
    "candidate": {
      "mean_regret": 0.24072265625,
      "exact_best_agreement": 0.609375,
      "top_two_agreement": 0.84375,
      "catastrophic_miss_rate": 0.234375,
      "p95_runtime_seconds": 0.08219485895242541
    },
    "paired_hierarchical_bootstrap": {
      "mean": -0.041139322916666665,
      "lower_95": -0.10035237630208334,
      "upper_95": 0.008608398437499972,
      "samples": 10000,
      "seed": 274001
    }
  },
  "invariants": {
    "budget_ok": true,
    "legal_selection_ok": true,
    "artifact_unchanged": true
  },
  "determinism": {
    "checked": 36,
    "results": [
      {
        "state_hash": "031ef519035c91078377d8bc83e9defc4e6e453053c53fe6e244d2c596329a04",
        "seed": 274101,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "031ef519035c91078377d8bc83e9defc4e6e453053c53fe6e244d2c596329a04",
        "seed": 274102,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "031ef519035c91078377d8bc83e9defc4e6e453053c53fe6e244d2c596329a04",
        "seed": 274103,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "06b4926edf8a0f923868563fa784d2f917b7c9f460a5af27f6dae1111e1e6a12",
        "seed": 274101,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "06b4926edf8a0f923868563fa784d2f917b7c9f460a5af27f6dae1111e1e6a12",
        "seed": 274102,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "06b4926edf8a0f923868563fa784d2f917b7c9f460a5af27f6dae1111e1e6a12",
        "seed": 274103,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "0a7a91d36031e0aa033de552bf84a53c0c4ca043e803af1a659b1106a4d2e7f3",
        "seed": 274101,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "0a7a91d36031e0aa033de552bf84a53c0c4ca043e803af1a659b1106a4d2e7f3",
        "seed": 274102,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "0a7a91d36031e0aa033de552bf84a53c0c4ca043e803af1a659b1106a4d2e7f3",
        "seed": 274103,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "1adbbba87838f595f47584007d5783c34e6f00055bed02278429ddf4cf8ebf65",
        "seed": 274101,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "1adbbba87838f595f47584007d5783c34e6f00055bed02278429ddf4cf8ebf65",
        "seed": 274102,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "1adbbba87838f595f47584007d5783c34e6f00055bed02278429ddf4cf8ebf65",
        "seed": 274103,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "1d8ee1c4af97d8302a1c616829dd42ea65884ccde5eaba2c485e87a6e902e283",
        "seed": 274101,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "1d8ee1c4af97d8302a1c616829dd42ea65884ccde5eaba2c485e87a6e902e283",
        "seed": 274102,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "1d8ee1c4af97d8302a1c616829dd42ea65884ccde5eaba2c485e87a6e902e283",
        "seed": 274103,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "224ea7810bac8e2f68891a2f1e3d28ea9f92602134e31ef8f4c775bc71330320",
        "seed": 274101,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "224ea7810bac8e2f68891a2f1e3d28ea9f92602134e31ef8f4c775bc71330320",
        "seed": 274102,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "224ea7810bac8e2f68891a2f1e3d28ea9f92602134e31ef8f4c775bc71330320",
        "seed": 274103,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "25509f089aa5d7160461718bcd352cbc3d9691a03cc54558f2105ec896281ae1",
        "seed": 274101,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "25509f089aa5d7160461718bcd352cbc3d9691a03cc54558f2105ec896281ae1",
        "seed": 274102,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "25509f089aa5d7160461718bcd352cbc3d9691a03cc54558f2105ec896281ae1",
        "seed": 274103,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "28b2166510bf59cf9e40e8442937380ced144731c7cdc7d485f26282a29baea8",
        "seed": 274101,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "28b2166510bf59cf9e40e8442937380ced144731c7cdc7d485f26282a29baea8",
        "seed": 274102,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "28b2166510bf59cf9e40e8442937380ced144731c7cdc7d485f26282a29baea8",
        "seed": 274103,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "2f647c152249c7677740fb6be4134fa75702b7e3129f6340f915c4f3b343d7aa",
        "seed": 274101,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "2f647c152249c7677740fb6be4134fa75702b7e3129f6340f915c4f3b343d7aa",
        "seed": 274102,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "2f647c152249c7677740fb6be4134fa75702b7e3129f6340f915c4f3b343d7aa",
        "seed": 274103,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "356a52472801c8721c9edc99562f0a6c4a5df098db00ce5c158052e7379825fb",
        "seed": 274101,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "356a52472801c8721c9edc99562f0a6c4a5df098db00ce5c158052e7379825fb",
        "seed": 274102,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "356a52472801c8721c9edc99562f0a6c4a5df098db00ce5c158052e7379825fb",
        "seed": 274103,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "3735d0e58c7df70e139ec24d83d2a13eb150abf8ae8f8e7777306588b5e67754",
        "seed": 274101,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "3735d0e58c7df70e139ec24d83d2a13eb150abf8ae8f8e7777306588b5e67754",
        "seed": 274102,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "3735d0e58c7df70e139ec24d83d2a13eb150abf8ae8f8e7777306588b5e67754",
        "seed": 274103,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "3a5459d040af635ca90ac740443f77630e79e941776255661a5a0ff3f212733e",
        "seed": 274101,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "3a5459d040af635ca90ac740443f77630e79e941776255661a5a0ff3f212733e",
        "seed": 274102,
        "equal_excluding_runtime": true
      },
      {
        "state_hash": "3a5459d040af635ca90ac740443f77630e79e941776255661a5a0ff3f212733e",
        "seed": 274103,
        "equal_excluding_runtime": true
      }
    ],
    "passed": true
  },
  "corpus_identity_audit": {
    "reachable_from_standard_start_by_legal_prefixes": true,
    "minimum_legal_actions": 2,
    "canonical_state_deduplicated": true,
    "canonical_hash_sorted": true,
    "registered_evaluation_suites_loaded": false,
    "registered_evaluation_suites_consumed": false
  },
  "elapsed_seconds": 45.5294129899703
}
```

## Reproduction

```bash
PYTHONPATH=. .venv/bin/pytest -q ml/alphazero_lite/test_implicit_minimax_puct.py
PYTHONPATH=. .venv/bin/python ml/alphazero_lite/run_generation3_implicit_minimax_puct.py
```
