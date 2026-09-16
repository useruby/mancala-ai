# R61 Early-Trunk Replay Provenance Audit

Inherited #319 classification: `cluster_representation_drift_early_trunk_primary` at A0.

## Baseline SHA Parity

```json
{
  "T61": {
    "expected_sha256": "50736a6063cd0558329c433ad7a6c60b741c83ceee75d36e21e703749af8c9de",
    "reference_sha256": "50736a6063cd0558329c433ad7a6c60b741c83ceee75d36e21e703749af8c9de",
    "instrumented_sha256": "50736a6063cd0558329c433ad7a6c60b741c83ceee75d36e21e703749af8c9de",
    "reproduced": true
  },
  "T63": {
    "expected_sha256": "bab60465a9bc93ac569ed28b356ef2abde2c1cbb0987a4a474e26117f9a57324",
    "reference_sha256": "bab60465a9bc93ac569ed28b356ef2abde2c1cbb0987a4a474e26117f9a57324",
    "instrumented_sha256": "bab60465a9bc93ac569ed28b356ef2abde2c1cbb0987a4a474e26117f9a57324",
    "reproduced": true
  }
}
```

## Formation Window

Steps 1-108 are pre-registered as formation. Per-step ordered batch provenance, compact-row source mapping, A0 pre/post vectors and movement, fixed-downstream margin effects, input gradients, and probe alignments are in the JSON artifact.

## Ranking

```json
{
  "T61": {
    "worst_1": [
      3
    ],
    "worst_5": [
      3,
      85,
      4,
      101,
      95
    ],
    "worst_10": [
      3,
      85,
      4,
      101,
      95,
      84,
      41,
      74,
      81,
      94
    ],
    "worst_20": [
      3,
      85,
      4,
      101,
      95,
      84,
      41,
      74,
      81,
      94,
      99,
      75,
      63,
      73,
      69,
      19,
      102,
      96,
      70,
      72
    ],
    "best_20": [
      1,
      2,
      24,
      6,
      16,
      7,
      9,
      44,
      23,
      39,
      45,
      46,
      91,
      28,
      10,
      22,
      88,
      61,
      89,
      92
    ],
    "negative_concentration": {
      "1": 0.15372423816174766,
      "5": 0.36047051452008877,
      "10": 0.5187849770981461,
      "20": 0.736320485469551
    }
  },
  "T63": {
    "worst_1": [
      3
    ],
    "worst_5": [
      3,
      4,
      61,
      101,
      28
    ],
    "worst_10": [
      3,
      4,
      61,
      101,
      28,
      100,
      11,
      27,
      50,
      49
    ],
    "worst_20": [
      3,
      4,
      61,
      101,
      28,
      100,
      11,
      27,
      50,
      49,
      17,
      99,
      18,
      62,
      102,
      29,
      48,
      45,
      60,
      34
    ],
    "best_20": [
      13,
      6,
      79,
      52,
      9,
      40,
      39,
      15,
      53,
      7,
      5,
      80,
      105,
      14,
      10,
      38,
      24,
      76,
      78,
      2
    ],
    "negative_concentration": {
      "1": 0.08567428636552434,
      "5": 0.2969564305863724,
      "10": 0.475350619869136,
      "20": 0.7195049714118111
    }
  }
}
```

## Classification

`early_trunk_provenance_heterogeneous`

Exactly one next experiment: restrict analysis to the earliest pre-step-108 window in which cumulative T61-vs-T63 A0 damage first becomes material, and repeat the 2x2 batch-state counterfactual there only.
