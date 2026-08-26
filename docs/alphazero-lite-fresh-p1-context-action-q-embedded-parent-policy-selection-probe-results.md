# Embedded-Parent-Policy Selection-Aligned Context Action-Q Probe

**Classification:** `parent_policy_context_not_informative`

**Recommended follow-up:** test a fixed-length recent root backup/selection trajectory representation.

```json
{
  "classification": "parent_policy_context_not_informative",
  "correction_magnitude": {
    "centered_scale_by_simulation_window": {
      "1_384": {
        "mean_centered_a16_q_std": 0.10582625865936279,
        "mean_centered_correction_std": 1.507831335067749
      },
      "385_1200": {
        "mean_centered_a16_q_std": 0.10464580357074738,
        "mean_centered_correction_std": 1.55545175075531
      }
    },
    "exact_p1_minus_a16_delta_q": {
      "max_abs": 0.6372873783111572,
      "mean": -1.7430746083846316e-05,
      "p50": 0.0,
      "p90": 0.0013983696699142456,
      "p95": 0.0036381289828568697,
      "p99": 0.013890653848648071,
      "std": 0.007775260601192713
    },
    "predicted_delta_q": {
      "max_abs": 21.90676498413086,
      "mean": -0.2533155083656311,
      "p50": -0.0419723279774189,
      "p90": 9.130023956298828,
      "p95": 12.425284385681152,
      "p99": 15.309982299804688,
      "std": 7.040801048278809
    }
  },
  "embedded_parent_policy_features": true,
  "exclusions": {
    "canonical_opening": 1,
    "canonical_suite_sha256": "ff21c42946ed32f525c5ab95b71c4d90a4cfe2ccc351406dd29cae52da4b1837",
    "duplicate": 0,
    "eligible_roots": 4015,
    "frozen_amplified": 40,
    "held_out": 0,
    "manifest_roots": 4096,
    "not_fresh_replay": 0,
    "washed_controls": 40
  },
  "features_unchanged": false,
  "frozen_40_40_offline": {
    "amplified": {
      "exact_flip_action_recall": 0.21030435445353388,
      "exact_flip_detection_rate": 0.4198465525125756,
      "exact_parent_flip_count": 19681,
      "exact_parent_score_regret_captured": 0.015242203138768673,
      "false_flip_rate": 0.4400225996680674,
      "nonflip_preservation_rate": 0.5599774003319327,
      "overall_exact_parent_action_agreement": 0.41660416666666666
    },
    "washed": {
      "exact_flip_action_recall": 0.37965167272271644,
      "exact_flip_detection_rate": 0.5619596541786743,
      "exact_parent_flip_count": 7981,
      "exact_parent_score_regret_captured": 0.25218576192855835,
      "false_flip_rate": 0.21537269796846498,
      "nonflip_preservation_rate": 0.784627302031535,
      "overall_exact_parent_action_agreement": 0.7172916666666667
    }
  },
  "guardrails": {
    "arena_run": false,
    "loss": "masked_cross_entropy_only",
    "p1_runtime": false,
    "puct_changed": false
  },
  "hashes": {
    "a16_weights": "74e160734554dba9ecf4ecf3ac13b3e44526c4f6e7e5e52d9fe8bde0054eb789",
    "aligned_root_context_cache": "8cfa271f6aabcd0f812312e5243c8889c6e0c697b6aa4c1e52a53932e8377aa9",
    "p1_weights": "77969733ece5ced92d3a143a0fe9d82863ca3ec4faa477470ff5826ac22e4e12",
    "probe_checkpoint": "b4a40835e13c43ec54fbb5ab4cba7d26fe967dc64d2e0ceafa102857442380b4",
    "replay": "892827d8ee67a66e6324a2aaec7011df1a21625fc3f6bcd87cab39ce655d2a88",
    "split_manifest": "e6e9dfd9ccefea4d3ce792dbbdb583525e84c5a6bbf0f922c4328408c52ad0f3"
  },
  "historical_mse_context_probe": {
    "exact_flip_action_recall": 0.178392212594167,
    "exact_flip_detection_rate": 0.3126784299422736,
    "exact_parent_flip_count": 159026,
    "exact_parent_score_regret_captured": 0.10154946893453598,
    "false_flip_rate": 0.1267428477678871,
    "nonflip_preservation_rate": 0.8732571522321129,
    "overall_exact_parent_action_agreement": 0.7585813615608136
  },
  "historical_pr238_instantaneous_selection_ce": {
    "exact_flip_action_recall": 0.35334473608089245,
    "exact_flip_detection_rate": 0.5217322953479305,
    "exact_parent_flip_count": 159026,
    "exact_parent_score_regret_captured": 0.23206011950969696,
    "false_flip_rate": 0.1828209710977486,
    "nonflip_preservation_rate": 0.8171790289022514,
    "overall_exact_parent_action_agreement": 0.7406309672063097
  },
  "historical_pr238_temporal_selection_ce": {
    "exact_flip_action_recall": 0.345081936287148,
    "exact_flip_detection_rate": 0.5050809301623634,
    "exact_parent_flip_count": 159026,
    "exact_parent_score_regret_captured": 0.21622265875339508,
    "false_flip_rate": 0.17126081628290252,
    "nonflip_preservation_rate": 0.8287391837170975,
    "overall_exact_parent_action_agreement": 0.7489196762141968
  },
  "invariants": {
    "adapter_logits_equal_combined_minus_base": true,
    "base_logits_equal_p1_forward": true,
    "combined_logits_equal_a16_forward": true,
    "exact_correction_parent_q_counterfactual": true,
    "pre_simulation_p1_evidence_only": true,
    "puct_move_tie_order": true,
    "unvisited_fpu_preserved": true,
    "value_output_unchanged": true,
    "zero_correction_ordinary_a16_winner": true
  },
  "live_frozen_root": "not_run_validation_gate_failed",
  "optimization": {
    "batch_size_root_decisions": 256,
    "best_validation_selection_ce": 0.6582212179080239,
    "lr": 0.001,
    "seed": 239,
    "steps": 2000,
    "train_decisions": 3854400,
    "validation_decisions": 963600,
    "validation_history": [
      {
        "step": 50,
        "validation_selection_ce": 1.0096989170452924
      },
      {
        "step": 100,
        "validation_selection_ce": 0.8387696037118293
      },
      {
        "step": 150,
        "validation_selection_ce": 0.7568802686844094
      },
      {
        "step": 200,
        "validation_selection_ce": 0.732331473040551
      },
      {
        "step": 250,
        "validation_selection_ce": 0.7216795585382129
      },
      {
        "step": 300,
        "validation_selection_ce": 0.7187981029999806
      },
      {
        "step": 350,
        "validation_selection_ce": 0.7168993598428095
      },
      {
        "step": 400,
        "validation_selection_ce": 0.7140150957752828
      },
      {
        "step": 450,
        "validation_selection_ce": 0.7121346345349633
      },
      {
        "step": 500,
        "validation_selection_ce": 0.7095075091230963
      },
      {
        "step": 550,
        "validation_selection_ce": 0.7080796644845014
      },
      {
        "step": 600,
        "validation_selection_ce": 0.7048508071741062
      },
      {
        "step": 650,
        "validation_selection_ce": 0.7038559422348485
      },
      {
        "step": 700,
        "validation_selection_ce": 0.7020456987874247
      },
      {
        "step": 750,
        "validation_selection_ce": 0.7027332647631603
      },
      {
        "step": 800,
        "validation_selection_ce": 0.6994461676509314
      },
      {
        "step": 850,
        "validation_selection_ce": 0.697354295476436
      },
      {
        "step": 900,
        "validation_selection_ce": 0.69589366388697
      },
      {
        "step": 950,
        "validation_selection_ce": 0.6941937316666504
      },
      {
        "step": 1000,
        "validation_selection_ce": 0.6913893810888205
      },
      {
        "step": 1050,
        "validation_selection_ce": 0.6896614481988117
      },
      {
        "step": 1100,
        "validation_selection_ce": 0.6894472370963217
      },
      {
        "step": 1150,
        "validation_selection_ce": 0.6866551079914351
      },
      {
        "step": 1200,
        "validation_selection_ce": 0.6834288874301934
      },
      {
        "step": 1250,
        "validation_selection_ce": 0.6830342681618734
      },
      {
        "step": 1300,
        "validation_selection_ce": 0.6806282189774187
      },
      {
        "step": 1350,
        "validation_selection_ce": 0.6797825263416485
      },
      {
        "step": 1400,
        "validation_selection_ce": 0.6779873721834138
      },
      {
        "step": 1450,
        "validation_selection_ce": 0.6771804159694116
      },
      {
        "step": 1500,
        "validation_selection_ce": 0.6733169752020418
      },
      {
        "step": 1550,
        "validation_selection_ce": 0.6733127189563388
      },
      {
        "step": 1600,
        "validation_selection_ce": 0.6741564614567929
      },
      {
        "step": 1650,
        "validation_selection_ce": 0.6696512635113442
      },
      {
        "step": 1700,
        "validation_selection_ce": 0.6667165011496601
      },
      {
        "step": 1750,
        "validation_selection_ce": 0.6688636292694713
      },
      {
        "step": 1800,
        "validation_selection_ce": 0.6641794863297465
      },
      {
        "step": 1850,
        "validation_selection_ce": 0.6624193867215876
      },
      {
        "step": 1900,
        "validation_selection_ce": 0.6598619890984978
      },
      {
        "step": 1950,
        "validation_selection_ce": 0.6594926628090624
      },
      {
        "step": 2000,
        "validation_selection_ce": 0.6582212179080239
      }
    ],
    "weight_decay": 0.0
  },
  "parent_policy_feature_ablations": {
    "full": {
      "exact_flip_action_recall": 0.3541307710688818,
      "exact_flip_detection_rate": 0.5207198822834002,
      "exact_parent_flip_count": 159026,
      "exact_parent_score_regret_captured": 0.23126959800720215,
      "false_flip_rate": 0.18363133782597996,
      "nonflip_preservation_rate": 0.81636866217402,
      "overall_exact_parent_action_agreement": 0.7400840597758406
    },
    "no_candidate": {
      "exact_flip_action_recall": 0.354118194509074,
      "exact_flip_detection_rate": 0.5207010174436885,
      "exact_parent_flip_count": 159026,
      "exact_parent_score_regret_captured": 0.23126931488513947,
      "false_flip_rate": 0.18363506650724482,
      "nonflip_preservation_rate": 0.8163649334927552,
      "overall_exact_parent_action_agreement": 0.7400788709007887
    },
    "no_shift": {
      "exact_flip_action_recall": 0.3541307710688818,
      "exact_flip_detection_rate": 0.5207198822834002,
      "exact_parent_flip_count": 159026,
      "exact_parent_score_regret_captured": 0.23126959800720215,
      "false_flip_rate": 0.1836276091447151,
      "nonflip_preservation_rate": 0.8163723908552849,
      "overall_exact_parent_action_agreement": 0.7400871731008717
    }
  },
  "policy_drift_quartiles": {
    "q1": {
      "exact_parent_flip_rate": 0.08682025736820258,
      "flip_action_recall": 0.3931149892421707,
      "nonflip_preservation": 0.8933472736777508,
      "policy_l1_range": [
        0.0,
        0.0008888617157936096
      ],
      "regret_capture": 0.16762210428714752
    },
    "q2": {
      "exact_parent_flip_rate": 0.17312162723121627,
      "flip_action_recall": 0.31821124565399833,
      "nonflip_preservation": 0.8093325635683627,
      "policy_l1_range": [
        0.0008888617157936096,
        0.0017390847206115723
      ],
      "regret_capture": 0.2299330085515976
    },
    "q3": {
      "exact_parent_flip_rate": 0.18740556247405563,
      "flip_action_recall": 0.3611172639879502,
      "nonflip_preservation": 0.7769445324233477,
      "policy_l1_range": [
        0.0017390847206115723,
        0.002334192395210266
      ],
      "regret_capture": 0.27169063687324524
    },
    "q4": {
      "exact_parent_flip_rate": 0.21278538812785389,
      "flip_action_recall": 0.3612953570035115,
      "nonflip_preservation": 0.7751581944737397,
      "policy_l1_range": [
        0.002334192395210266,
        0.004697352647781372
      ],
      "regret_capture": 0.24546273052692413
    }
  },
  "recommended_follow_up": "test a fixed-length recent root backup/selection trajectory representation.",
  "regret_quartiles_exact_parent_flips": {
    "q1": {
      "exact_parent_action_recall": 0.42626455718489825,
      "flip_detection": 0.529038911386674,
      "regret_capture": 0.4138469994068146,
      "regret_range": [
        0.0,
        0.0005592876113951206
      ]
    },
    "q2": {
      "exact_parent_action_recall": 0.39082425736348314,
      "flip_detection": 0.5367608219936112,
      "regret_capture": 0.3893084228038788,
      "regret_range": [
        0.0005592876113951206,
        0.0018936246633529663
      ]
    },
    "q3": {
      "exact_parent_action_recall": 0.3454824429017004,
      "flip_detection": 0.5397424288157763,
      "regret_capture": 0.3377959728240967,
      "regret_range": [
        0.0018936246633529663,
        0.005772685632109642
      ]
    },
    "q4": {
      "exact_parent_action_recall": 0.2539490894456183,
      "flip_detection": 0.4773367542006238,
      "regret_capture": 0.2054515779018402,
      "regret_range": [
        0.005772685632109642,
        0.49935781955718994
      ]
    }
  },
  "schema": "azlite_context_action_q_embedded_parent_policy_selection_probe_v1",
  "selection_ce_minus_historical_mse": {
    "exact_flip_action_recall": 0.1757385584747148,
    "exact_parent_score_regret_captured": 0.12972012907266617,
    "nonflip_preservation_rate": -0.05688849005809282,
    "overall_exact_parent_action_agreement": -0.018497301784972997
  },
  "simulation_windows": {
    "1_384": {
      "exact_flip_action_recall": 0.3801599624996338,
      "exact_flip_detection_rate": 0.5443705504936571,
      "exact_parent_flip_count": 34133,
      "exact_parent_score_regret_captured": 0.27526864409446716,
      "false_flip_rate": 0.22963033196095092,
      "nonflip_preservation_rate": 0.770369668039049,
      "overall_exact_parent_action_agreement": 0.7271754358655044
    },
    "385_1200": {
      "exact_flip_action_recall": 0.34701704659188265,
      "exact_flip_detection_rate": 0.5142562033100334,
      "exact_parent_flip_count": 124893,
      "exact_parent_score_regret_captured": 0.20784960687160492,
      "false_flip_rate": 0.1598476492160911,
      "nonflip_preservation_rate": 0.8401523507839089,
      "overall_exact_parent_action_agreement": 0.7461587063218812
    }
  },
  "temporal_features": [],
  "validation_gate_passed": false,
  "validation_selection_consequence": {
    "exact_flip_action_recall": 0.3541307710688818,
    "exact_flip_detection_rate": 0.5207198822834002,
    "exact_parent_flip_count": 159026,
    "exact_parent_score_regret_captured": 0.23126959800720215,
    "false_flip_rate": 0.18363133782597996,
    "nonflip_preservation_rate": 0.81636866217402,
    "overall_exact_parent_action_agreement": 0.7400840597758406
  }
}
```
