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

## Content And Source Attribution

The JSON artifact contains raw-count composition comparisons, exposure-normalized source effects, replay-row/state recurrence, and all eight pre-registered family summaries.

```json
{
  "selected_family": null,
  "counterfactual": {
    "steps": [
      {
        "t61_step": 3,
        "content_stable_harm": false,
        "t61_state_dependent_harm": true,
        "t63_batch_protective": false,
        "t61_cluster_a0_effect": -0.06374144852161408,
        "t63_recipient_t61_batch_effect": 0.0658799409866333,
        "replacement_improvement": 0.0034185945987701416,
        "t61_control_a0_effect": 0.1872037649154663
      },
      {
        "t61_step": 85,
        "content_stable_harm": true,
        "t61_state_dependent_harm": false,
        "t63_batch_protective": true,
        "t61_cluster_a0_effect": -0.0738977313041687,
        "t63_recipient_t61_batch_effect": -0.0247730553150177,
        "replacement_improvement": 0.02355750203132629,
        "t61_control_a0_effect": 0.04481637477874756
      },
      {
        "t61_step": 4,
        "content_stable_harm": true,
        "t61_state_dependent_harm": false,
        "t63_batch_protective": false,
        "t61_cluster_a0_effect": -0.07187562882900238,
        "t63_recipient_t61_batch_effect": -0.13840622156858445,
        "replacement_improvement": -0.03378225564956665,
        "t61_control_a0_effect": 0.009890735149383545
      },
      {
        "t61_step": 101,
        "content_stable_harm": true,
        "t61_state_dependent_harm": false,
        "t63_batch_protective": false,
        "t61_cluster_a0_effect": -0.05722641870379448,
        "t63_recipient_t61_batch_effect": -0.10373406410217285,
        "replacement_improvement": -0.01101866364479065,
        "t61_control_a0_effect": 0.01819649338722229
      },
      {
        "t61_step": 95,
        "content_stable_harm": false,
        "t61_state_dependent_harm": false,
        "t63_batch_protective": false,
        "t61_cluster_a0_effect": 0.018003815412521364,
        "t63_recipient_t61_batch_effect": -0.005843168497085572,
        "replacement_improvement": -0.024291348457336426,
        "t61_control_a0_effect": 0.07960081100463867
      },
      {
        "t61_step": 84,
        "content_stable_harm": true,
        "t61_state_dependent_harm": false,
        "t63_batch_protective": false,
        "t61_cluster_a0_effect": -0.02441921830177307,
        "t63_recipient_t61_batch_effect": -0.006976038217544556,
        "replacement_improvement": -0.024516403675079346,
        "t61_control_a0_effect": 0.036763012409210205
      },
      {
        "t61_step": 41,
        "content_stable_harm": false,
        "t61_state_dependent_harm": true,
        "t63_batch_protective": false,
        "t61_cluster_a0_effect": -0.01316385269165039,
        "t63_recipient_t61_batch_effect": 0.017440134286880495,
        "replacement_improvement": -0.003561025857925417,
        "t61_control_a0_effect": 0.03926116228103638
      },
      {
        "t61_step": 74,
        "content_stable_harm": true,
        "t61_state_dependent_harm": false,
        "t63_batch_protective": true,
        "t61_cluster_a0_effect": -0.03724954724311828,
        "t63_recipient_t61_batch_effect": -0.006539070606231689,
        "replacement_improvement": 0.022157514095306394,
        "t61_control_a0_effect": 0.014115899801254272
      },
      {
        "t61_step": 81,
        "content_stable_harm": false,
        "t61_state_dependent_harm": false,
        "t63_batch_protective": false,
        "t61_cluster_a0_effect": 0.038923409581184384,
        "t63_recipient_t61_batch_effect": 0.014367318153381348,
        "replacement_improvement": -0.006253796815872188,
        "t61_control_a0_effect": 0.08652713894844055
      },
      {
        "t61_step": 94,
        "content_stable_harm": false,
        "t61_state_dependent_harm": false,
        "t63_batch_protective": false,
        "t61_cluster_a0_effect": 0.021748217940330505,
        "t63_recipient_t61_batch_effect": 0.0016411006450653075,
        "replacement_improvement": -0.02184736132621765,
        "t61_control_a0_effect": 0.06761026382446289
      }
    ],
    "content_stable_fraction": 0.5,
    "state_dependent_fraction": 0.2,
    "protective_replacement_fraction": 0.2,
    "cluster_harm_stronger_than_controls": true
  }
}
```

## Classification

`early_trunk_provenance_heterogeneous`

Exactly one next experiment: restrict analysis to the earliest pre-step-108 window in which cumulative T61-vs-T63 A0 damage first becomes material, and repeat the 2x2 batch-state counterfactual there only.
