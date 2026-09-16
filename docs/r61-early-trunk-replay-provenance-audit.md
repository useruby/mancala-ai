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

Steps 1-108 are pre-registered as formation. The immutable compact-row map contains source artifact/SHA/local JSONL row, canonical state hash, effective replay weight, sharpened policy target, and value target.

Unique compact rows: 61036; effective weighted replay exposures: 61041.

## A0 Movement And Contexts

A0 before/after vectors, L2/normalized L2/cosine/support flips, and fixed pre/post-downstream margin effects are recorded per frozen state and step in the machine artifact.

```json
{
  "windows": {
    "T61": {
      "formation": {
        "steps": 108,
        "cumulative_cluster_a0_harm": 1.2607320036739111,
        "cumulative_cluster_a0_help": 1.8499011054635048,
        "cumulative_cluster_a0_net": 0.5891691017895937,
        "cumulative_control_a0_effect": 0.41160255298018456,
        "cumulative_anchor_margin_effect": 0.14314516633749008,
        "cumulative_cluster_specific_pre_context": 0.17756654880940917,
        "cumulative_cluster_specific_post_context": 0.11955590434372426,
        "pre_post_sign_agree": true
      },
      "maintenance": {
        "steps": 324,
        "cumulative_cluster_a0_harm": 3.209655299782753,
        "cumulative_cluster_a0_help": 4.508255995996296,
        "cumulative_cluster_a0_net": 1.2986006962135435,
        "cumulative_control_a0_effect": 0.18053158465772867,
        "cumulative_anchor_margin_effect": 0.6836930960416794,
        "cumulative_cluster_specific_pre_context": 1.1180691115558148,
        "cumulative_cluster_specific_post_context": -0.8397585747763514,
        "pre_post_sign_agree": false
      }
    },
    "T63": {
      "formation": {
        "steps": 108,
        "cumulative_cluster_a0_harm": 1.6055294282734394,
        "cumulative_cluster_a0_help": 2.2274957201443613,
        "cumulative_cluster_a0_net": 0.6219662918709219,
        "cumulative_control_a0_effect": 0.36295647267252207,
        "cumulative_anchor_margin_effect": 2.9294926449656487,
        "cumulative_cluster_specific_pre_context": 0.25900981919839977,
        "cumulative_cluster_specific_post_context": -0.09042038517072797,
        "pre_post_sign_agree": false
      },
      "maintenance": {
        "steps": 324,
        "cumulative_cluster_a0_harm": 3.8899956747889517,
        "cumulative_cluster_a0_help": 4.588663784042001,
        "cumulative_cluster_a0_net": 0.6986681092530489,
        "cumulative_control_a0_effect": 0.7854665210470557,
        "cumulative_anchor_margin_effect": 2.6709553375840187,
        "cumulative_cluster_specific_pre_context": -0.08679841179400677,
        "cumulative_cluster_specific_post_context": -0.9219848960638046,
        "pre_post_sign_agree": true
      }
    }
  },
  "movement": {
    "T61": {
      "anchor": {
        "l2": 0.0053556452389953105,
        "normalized_l2": 0.004159402634724285,
        "cosine": 0.9999905301740876,
        "support_flips": 0.11342592592592593,
        "activated": 0.05555555555555555,
        "deactivated": 0.05787037037037037
      },
      "controls": {
        "l2": 0.004713894863487911,
        "normalized_l2": 0.003592920226057888,
        "cosine": 0.9999927922531411,
        "support_flips": 0.004629629629629629,
        "activated": 0.0023148148148148147,
        "deactivated": 0.0023148148148148147
      },
      "cluster": {
        "l2": 0.005896579907426645,
        "normalized_l2": 0.004331592118358929,
        "cosine": 0.9999896961505766,
        "support_flips": 0.14953703703703702,
        "activated": 0.07407407407407407,
        "deactivated": 0.07546296296296297
      }
    },
    "T63": {
      "anchor": {
        "l2": 0.00528747125860752,
        "normalized_l2": 0.004099733856424724,
        "cosine": 0.9999906893957544,
        "support_flips": 0.1111111111111111,
        "activated": 0.05555555555555555,
        "deactivated": 0.05555555555555555
      },
      "controls": {
        "l2": 0.004648770308078922,
        "normalized_l2": 0.003549353570970534,
        "cosine": 0.9999930134250058,
        "support_flips": 0.004629629629629629,
        "activated": 0.0023148148148148147,
        "deactivated": 0.0023148148148148147
      },
      "cluster": {
        "l2": 0.005824836275725695,
        "normalized_l2": 0.004283609357123539,
        "cosine": 0.9999897413231709,
        "support_flips": 0.16018518518518518,
        "activated": 0.07916666666666666,
        "deactivated": 0.08101851851851852
      }
    }
  }
}
```

## Harmful And Protective Ranking

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

Worst-20, best-20, T61-formation, and T63-formation composition tables retain raw counts, fractions, and the eight pre-registered family dimensions in the machine artifact.

```json
{
  "source_accounting": {
    "T61": [
      {
        "source": "dynamic",
        "source_artifact": "/home/alex/Mancala/ai/.tmp/internal-cluster-learning-dynamics/versions/seed61/internal-cluster-learning-dynamics-seed61-iter1/self_play.jsonl",
        "source_sha256": "6218a63b4817a58ce41df9c701c3816739976d26a1629aa32b158f3d0650aa87",
        "raw_compact_rows": 61009,
        "effective_weighted_rows": 61009,
        "expected_exposure_fraction": 0.9994757621926247,
        "observed_formation_exposures": 54908,
        "observed_formation_fraction": 0.9994721226131751,
        "negative_a0_movement_share": 0.9995854356607221,
        "positive_a0_movement_share": 0.9995264736393527,
        "negative_a0_effect_per_1000_exposures": 0.029718087805988905,
        "positive_a0_effect_per_1000_exposures": 0.032948695632755445
      },
      {
        "source": "fixed:family_leave_one_out_without_opening_extra_turn_overbias",
        "source_artifact": "/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl",
        "source_sha256": "735e66b10fb58438a215b11870ee4d7c32aca065e173663d46e7c41a55dd5a71",
        "raw_compact_rows": 22,
        "effective_weighted_rows": 22,
        "expected_exposure_fraction": 0.0003604134925705673,
        "observed_formation_exposures": 21,
        "observed_formation_fraction": 0.0003822560387352786,
        "negative_a0_movement_share": 0.00025097860745150566,
        "positive_a0_movement_share": 0.0003915956858179043,
        "negative_a0_effect_per_1000_exposures": 0.019509852115463997,
        "positive_a0_effect_per_1000_exposures": 0.03375189442608861
      },
      {
        "source": "fixed:guard_safe_controls_only",
        "source_artifact": "/tmp/azlite_guard_safe_opening_replay/guard_safe_controls_only.jsonl",
        "source_sha256": "ffac24dc8fb773fddd4dc14f397317d7c5e0a973631ca71adb216b5e68b444a5",
        "raw_compact_rows": 5,
        "effective_weighted_rows": 10,
        "expected_exposure_fraction": 0.00016382431480480334,
        "observed_formation_exposures": 8,
        "observed_formation_fraction": 0.00014562134808962995,
        "negative_a0_movement_share": 0.00016358573182641823,
        "positive_a0_movement_share": 8.193067482936394e-05,
        "negative_a0_effect_per_1000_exposures": 0.033380435706931166,
        "positive_a0_effect_per_1000_exposures": 0.01853685680544004
      }
    ],
    "T63": [
      {
        "source": "dynamic",
        "source_artifact": "/home/alex/Mancala/ai/.tmp/internal-cluster-learning-dynamics/versions/seed61/internal-cluster-learning-dynamics-seed61-iter1/self_play.jsonl",
        "source_sha256": "6218a63b4817a58ce41df9c701c3816739976d26a1629aa32b158f3d0650aa87",
        "raw_compact_rows": 61009,
        "effective_weighted_rows": 61009,
        "expected_exposure_fraction": 0.9994757621926247,
        "observed_formation_exposures": 54910,
        "observed_formation_fraction": 0.9995085279501975,
        "negative_a0_movement_share": 0.9995634096260112,
        "positive_a0_movement_share": 0.9996518537012239,
        "negative_a0_effect_per_1000_exposures": 0.023722971833854488,
        "positive_a0_effect_per_1000_exposures": 0.028440416852478743
      },
      {
        "source": "fixed:family_leave_one_out_without_opening_extra_turn_overbias",
        "source_artifact": "/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl",
        "source_sha256": "735e66b10fb58438a215b11870ee4d7c32aca065e173663d46e7c41a55dd5a71",
        "raw_compact_rows": 22,
        "effective_weighted_rows": 22,
        "expected_exposure_fraction": 0.0003604134925705673,
        "observed_formation_exposures": 19,
        "observed_formation_fraction": 0.0003458507017128711,
        "negative_a0_movement_share": 0.0003406158300285252,
        "positive_a0_movement_share": 0.0001895630820973122,
        "negative_a0_effect_per_1000_exposures": 0.02336261294610602,
        "positive_a0_effect_per_1000_exposures": 0.015586147645463873
      },
      {
        "source": "fixed:guard_safe_controls_only",
        "source_artifact": "/tmp/azlite_guard_safe_opening_replay/guard_safe_controls_only.jsonl",
        "source_sha256": "ffac24dc8fb773fddd4dc14f397317d7c5e0a973631ca71adb216b5e68b444a5",
        "raw_compact_rows": 5,
        "effective_weighted_rows": 10,
        "expected_exposure_fraction": 0.00016382431480480334,
        "observed_formation_exposures": 8,
        "observed_formation_fraction": 0.00014562134808962995,
        "negative_a0_movement_share": 9.597454396023954e-05,
        "positive_a0_movement_share": 0.00015858321667880547,
        "negative_a0_effect_per_1000_exposures": 0.01563422138133319,
        "positive_a0_effect_per_1000_exposures": 0.030967479688115418
      }
    ]
  },
  "recurrence_support_ge_3": {
    "rows_support_ge_3": 0,
    "states_support_ge_3": 0
  },
  "input_layer_gradient_response": {
    "t61_worst_20": {
      "preclip_norm": 2.13566135764122,
      "postclip_norm": 0.593408893754212,
      "update_norm": 0.011159521713852882,
      "cosine_update_to_negative_raw_gradient": 0.2724772930145264,
      "cluster_probe_alignment": 0.012206782633438707,
      "control_probe_alignment": 0.09925680602900684
    },
    "t61_best_20": {
      "preclip_norm": 2.2794180512428284,
      "postclip_norm": 0.5963291302392154,
      "update_norm": 0.014821566874161363,
      "cosine_update_to_negative_raw_gradient": 0.2920960128307343,
      "cluster_probe_alignment": 0.11680572144687176,
      "control_probe_alignment": 0.02722362233325839
    },
    "t61_neutral_20": {
      "preclip_norm": 2.0310528457164763,
      "postclip_norm": 0.5703180556020956,
      "update_norm": 0.009484659042209386,
      "cosine_update_to_negative_raw_gradient": 0.2542816173285246,
      "cluster_probe_alignment": -0.005665471637621522,
      "control_probe_alignment": 0.039725751243531705
    },
    "t63_formation": {
      "preclip_norm": 2.1703286987763866,
      "postclip_norm": 0.6098155027899923,
      "update_norm": 0.011105957402226826,
      "cosine_update_to_negative_raw_gradient": 0.26737335061921774,
      "cluster_probe_alignment": 0.052445553300289986,
      "control_probe_alignment": 0.03268750974745705
    }
  },
  "selected_family": null
}
```

## Probe And Counterfactuals

Cluster and matched-control probe alignments are included for worst, best, neutral, and T63 formation batches. The finite-step test verifies that descending a positive-alignment probe gradient lowers its loss. Historical T61/T63 clone cells reproduced their next tensors within tolerance.

```json
{
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
```

## Formation Timing And Control Safety

No deterministic content family qualified, so no family timing claim or replay-content intervention is made. Counterfactual rows retain matched-control A0 effects and the classification requires cluster-specific harm.

## Classification

`early_trunk_provenance_heterogeneous`

Exactly one next experiment: restrict analysis to the earliest pre-step-108 window in which cumulative T61-vs-T63 A0 damage first becomes material, and repeat the 2x2 batch-state counterfactual there only.
