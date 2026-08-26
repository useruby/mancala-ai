# Causal Root Backup Trajectory Selection Probe

**Classification:** `recent_root_trajectory_not_informative`

```json
{
  "classification": "recent_root_trajectory_not_informative",
  "correction_magnitude": {
    "centered_scale_by_simulation_window": {
      "1_384": {
        "mean_centered_a16_q_std": 0.10582625865936279,
        "mean_centered_correction_std": 1.17551851272583
      },
      "385_1200": {
        "mean_centered_a16_q_std": 0.10464580357074738,
        "mean_centered_correction_std": 1.2716089487075806
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
      "max_abs": 10.96557903289795,
      "mean": 1.8995587825775146,
      "p50": 1.8212240934371948,
      "p90": 4.605389595031738,
      "p95": 5.310159206390381,
      "p99": 6.420958995819092,
      "std": 2.016488790512085
    }
  },
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
  "frozen_40_40_offline": {
    "amplified": {
      "exact_flip_action_recall": 0.16183120776383314,
      "exact_flip_detection_rate": 0.29297291804278236,
      "exact_parent_flip_count": 19681,
      "exact_parent_score_regret_captured": 0.020029697567224503,
      "false_flip_rate": 0.22087644337723789,
      "nonflip_preservation_rate": 0.7791235566227621,
      "overall_exact_parent_action_agreement": 0.5260208333333334
    },
    "washed": {
      "exact_flip_action_recall": 0.3352963287808545,
      "exact_flip_detection_rate": 0.4963037213381782,
      "exact_parent_flip_count": 7981,
      "exact_parent_score_regret_captured": 0.23630556464195251,
      "false_flip_rate": 0.16384717259301831,
      "nonflip_preservation_rate": 0.8361528274069817,
      "overall_exact_parent_action_agreement": 0.752875
    }
  },
  "guardrails": {
    "arena_run": false,
    "history_length_tuned": false,
    "loss": "masked_cross_entropy_only",
    "p1_runtime": false
  },
  "hashes": {
    "a16_weights": "74e160734554dba9ecf4ecf3ac13b3e44526c4f6e7e5e52d9fe8bde0054eb789",
    "aligned_root_context_cache": "8cfa271f6aabcd0f812312e5243c8889c6e0c697b6aa4c1e52a53932e8377aa9",
    "p1_weights": "77969733ece5ced92d3a143a0fe9d82863ca3ec4faa477470ff5826ac22e4e12",
    "replay": "892827d8ee67a66e6324a2aaec7011df1a21625fc3f6bcd87cab39ce655d2a88",
    "root_backup_history_cache": "f183e49ec6877d264c6b49f38693e710e64f43ac03e7feba6eb4f70cf5b86d1b",
    "split_manifest": "e6e9dfd9ccefea4d3ce792dbbdb583525e84c5a6bbf0f922c4328408c52ad0f3"
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
  "historical_pr239_embedded_parent_policy_selection_ce": {
    "exact_flip_action_recall": 0.3541307710688818,
    "exact_flip_detection_rate": 0.5207198822834002,
    "exact_parent_flip_count": 159026,
    "exact_parent_score_regret_captured": 0.23126959800720215,
    "false_flip_rate": 0.18363133782597996,
    "nonflip_preservation_rate": 0.81636866217402,
    "overall_exact_parent_action_agreement": 0.7400840597758406
  },
  "history": {
    "event_size": 7,
    "lanes": {
      "trajectory_actions_only": "selected_action_one_hot_plus_zero_value",
      "trajectory_actions_values": "selected_action_one_hot_plus_root_value"
    },
    "left_padding": "seven_zeros",
    "length": 32,
    "order": "oldest_to_newest"
  },
  "history_dependence_diagnostics": {
    "normal": {
      "exact_flip_action_recall": 0.3025102813376429,
      "exact_flip_detection_rate": 0.4238237772439727,
      "exact_parent_flip_count": 159026,
      "exact_parent_score_regret_captured": 0.18200014531612396,
      "false_flip_rate": 0.13664498231362188,
      "nonflip_preservation_rate": 0.8633550176863781,
      "overall_exact_parent_action_agreement": 0.7707970112079701
    },
    "permuted": {
      "exact_flip_action_recall": 0.3368757310125388,
      "exact_flip_detection_rate": 0.48846100637631584,
      "exact_parent_flip_count": 159026,
      "exact_parent_score_regret_captured": 0.21725991368293762,
      "false_flip_rate": 0.16693678890941044,
      "nonflip_preservation_rate": 0.8330632110905896,
      "overall_exact_parent_action_agreement": 0.751175799086758
    },
    "reverse": {
      "exact_flip_action_recall": 0.34456629733502697,
      "exact_flip_detection_rate": 0.5179467508457737,
      "exact_parent_flip_count": 159026,
      "exact_parent_score_regret_captured": 0.2300221025943756,
      "false_flip_rate": 0.18482824451200264,
      "nonflip_preservation_rate": 0.8151717554879974,
      "overall_exact_parent_action_agreement": 0.7375062266500623
    },
    "zero": {
      "exact_flip_action_recall": 0.35313093456415934,
      "exact_flip_detection_rate": 0.5249456063788311,
      "exact_parent_flip_count": 159026,
      "exact_parent_score_regret_captured": 0.24955080449581146,
      "false_flip_rate": 0.20265631253309205,
      "nonflip_preservation_rate": 0.797343687466908,
      "overall_exact_parent_action_agreement": 0.7240338314653383
    }
  },
  "invariants": {
    "completed_backups_only": true,
    "ordinary_a16_context_cache_reproduced": true,
    "p1_runtime": false,
    "selection_ce_unchanged": true,
    "simulation_t_backup_excluded": true,
    "unvisited_fpu_preserved": true
  },
  "live_frozen_root": "not_run_validation_gate_failed",
  "optimization": {
    "batch_size_root_decisions": 256,
    "lanes": {
      "trajectory_actions_only": {
        "best_validation_selection_ce": 0.6134031209157877,
        "checkpoint_sha256": "4733b6414a113b337cf51cd619d6c0d3ddc3521429f6b1be991ae0ef305300f4",
        "metrics": {
          "exact_flip_action_recall": 0.30449108950737613,
          "exact_flip_detection_rate": 0.4293826166790336,
          "exact_parent_flip_count": 159026,
          "exact_parent_score_regret_captured": 0.18307262659072876,
          "false_flip_rate": 0.13905495330448162,
          "nonflip_preservation_rate": 0.8609450466955184,
          "overall_exact_parent_action_agreement": 0.7691116645911167
        },
        "seed": 241,
        "validation_history": [
          {
            "step": 50,
            "validation_selection_ce": 0.9587608545793443
          },
          {
            "step": 100,
            "validation_selection_ce": 0.7466063087498216
          },
          {
            "step": 150,
            "validation_selection_ce": 0.6776526550318305
          },
          {
            "step": 200,
            "validation_selection_ce": 0.6581252916208586
          },
          {
            "step": 250,
            "validation_selection_ce": 0.6515097893397805
          },
          {
            "step": 300,
            "validation_selection_ce": 0.6476193299378632
          },
          {
            "step": 350,
            "validation_selection_ce": 0.6419589781929418
          },
          {
            "step": 400,
            "validation_selection_ce": 0.6418917113193687
          },
          {
            "step": 450,
            "validation_selection_ce": 0.6389687794914578
          },
          {
            "step": 500,
            "validation_selection_ce": 0.6389531393910206
          },
          {
            "step": 550,
            "validation_selection_ce": 0.6380551224736665
          },
          {
            "step": 600,
            "validation_selection_ce": 0.6356139852952145
          },
          {
            "step": 650,
            "validation_selection_ce": 0.6349132444414176
          },
          {
            "step": 700,
            "validation_selection_ce": 0.6350435277702107
          },
          {
            "step": 750,
            "validation_selection_ce": 0.6330609113123962
          },
          {
            "step": 800,
            "validation_selection_ce": 0.6321391274217453
          },
          {
            "step": 850,
            "validation_selection_ce": 0.6320084227502173
          },
          {
            "step": 900,
            "validation_selection_ce": 0.6300568267961615
          },
          {
            "step": 950,
            "validation_selection_ce": 0.6309019508852707
          },
          {
            "step": 1000,
            "validation_selection_ce": 0.6265124856292484
          },
          {
            "step": 1050,
            "validation_selection_ce": 0.6260305888744844
          },
          {
            "step": 1100,
            "validation_selection_ce": 0.6278110550705849
          },
          {
            "step": 1150,
            "validation_selection_ce": 0.6272013232340798
          },
          {
            "step": 1200,
            "validation_selection_ce": 0.6252548120228667
          },
          {
            "step": 1250,
            "validation_selection_ce": 0.6229735831533443
          },
          {
            "step": 1300,
            "validation_selection_ce": 0.6233052660291647
          },
          {
            "step": 1350,
            "validation_selection_ce": 0.6234047044347856
          },
          {
            "step": 1400,
            "validation_selection_ce": 0.6198855647826996
          },
          {
            "step": 1450,
            "validation_selection_ce": 0.62067814163075
          },
          {
            "step": 1500,
            "validation_selection_ce": 0.6177985718736217
          },
          {
            "step": 1550,
            "validation_selection_ce": 0.6189491055980665
          },
          {
            "step": 1600,
            "validation_selection_ce": 0.6181292149475275
          },
          {
            "step": 1650,
            "validation_selection_ce": 0.6174415032880441
          },
          {
            "step": 1700,
            "validation_selection_ce": 0.6177306475229608
          },
          {
            "step": 1750,
            "validation_selection_ce": 0.6177087199653318
          },
          {
            "step": 1800,
            "validation_selection_ce": 0.6157365785490284
          },
          {
            "step": 1850,
            "validation_selection_ce": 0.6152189441774336
          },
          {
            "step": 1900,
            "validation_selection_ce": 0.6153992292088263
          },
          {
            "step": 1950,
            "validation_selection_ce": 0.6135591380751544
          },
          {
            "step": 2000,
            "validation_selection_ce": 0.6134031209157877
          }
        ]
      },
      "trajectory_actions_values": {
        "best_validation_selection_ce": 0.6105830562555132,
        "checkpoint_sha256": "6899d8cb42b8350afd4c49a8852133258eed900d6f851b16fc9dea4212305488",
        "metrics": {
          "exact_flip_action_recall": 0.3025102813376429,
          "exact_flip_detection_rate": 0.4238237772439727,
          "exact_parent_flip_count": 159026,
          "exact_parent_score_regret_captured": 0.18200014531612396,
          "false_flip_rate": 0.13664498231362188,
          "nonflip_preservation_rate": 0.8633550176863781,
          "overall_exact_parent_action_agreement": 0.7707970112079701
        },
        "seed": 240,
        "validation_history": [
          {
            "step": 50,
            "validation_selection_ce": 0.957631573960117
          },
          {
            "step": 100,
            "validation_selection_ce": 0.753918814779862
          },
          {
            "step": 150,
            "validation_selection_ce": 0.6807572403049112
          },
          {
            "step": 200,
            "validation_selection_ce": 0.6583535985760591
          },
          {
            "step": 250,
            "validation_selection_ce": 0.6510122238545558
          },
          {
            "step": 300,
            "validation_selection_ce": 0.6447330663292277
          },
          {
            "step": 350,
            "validation_selection_ce": 0.6426694616157022
          },
          {
            "step": 400,
            "validation_selection_ce": 0.6411498658594723
          },
          {
            "step": 450,
            "validation_selection_ce": 0.6400119143969392
          },
          {
            "step": 500,
            "validation_selection_ce": 0.6377017464010287
          },
          {
            "step": 550,
            "validation_selection_ce": 0.6372510710669138
          },
          {
            "step": 600,
            "validation_selection_ce": 0.6358617614264704
          },
          {
            "step": 650,
            "validation_selection_ce": 0.6346951394023713
          },
          {
            "step": 700,
            "validation_selection_ce": 0.633806299547676
          },
          {
            "step": 750,
            "validation_selection_ce": 0.6305947459194362
          },
          {
            "step": 800,
            "validation_selection_ce": 0.6311436062722149
          },
          {
            "step": 850,
            "validation_selection_ce": 0.6292415744627893
          },
          {
            "step": 900,
            "validation_selection_ce": 0.6286424035578657
          },
          {
            "step": 950,
            "validation_selection_ce": 0.6276815885842316
          },
          {
            "step": 1000,
            "validation_selection_ce": 0.627530440809124
          },
          {
            "step": 1050,
            "validation_selection_ce": 0.6265763343820536
          },
          {
            "step": 1100,
            "validation_selection_ce": 0.6272068389474529
          },
          {
            "step": 1150,
            "validation_selection_ce": 0.6252753858617749
          },
          {
            "step": 1200,
            "validation_selection_ce": 0.6239115019624487
          },
          {
            "step": 1250,
            "validation_selection_ce": 0.6222482035040311
          },
          {
            "step": 1300,
            "validation_selection_ce": 0.623077533447975
          },
          {
            "step": 1350,
            "validation_selection_ce": 0.6222372179359952
          },
          {
            "step": 1400,
            "validation_selection_ce": 0.6198854206191301
          },
          {
            "step": 1450,
            "validation_selection_ce": 0.6190215400641961
          },
          {
            "step": 1500,
            "validation_selection_ce": 0.6191277422495233
          },
          {
            "step": 1550,
            "validation_selection_ce": 0.6193056299598348
          },
          {
            "step": 1600,
            "validation_selection_ce": 0.6176046946752413
          },
          {
            "step": 1650,
            "validation_selection_ce": 0.6172191589786186
          },
          {
            "step": 1700,
            "validation_selection_ce": 0.6158606169856203
          },
          {
            "step": 1750,
            "validation_selection_ce": 0.6157516919077062
          },
          {
            "step": 1800,
            "validation_selection_ce": 0.6135213408701743
          },
          {
            "step": 1850,
            "validation_selection_ce": 0.6153057144310723
          },
          {
            "step": 1900,
            "validation_selection_ce": 0.6129326749036005
          },
          {
            "step": 1950,
            "validation_selection_ce": 0.6130513506580952
          },
          {
            "step": 2000,
            "validation_selection_ce": 0.6105830562555132
          }
        ]
      }
    },
    "lr": 0.001,
    "steps": 2000,
    "weight_decay": 0.0
  },
  "regret_quartiles_exact_parent_flips": {
    "q1": {
      "exact_parent_action_recall": 0.38018462157607463,
      "flip_detection": 0.4705837965641271,
      "regret_capture": 0.3712990880012512,
      "regret_range": [
        0.0,
        0.0005592876113951206
      ]
    },
    "q2": {
      "exact_parent_action_recall": 0.3440651960661016,
      "flip_detection": 0.4613527177603944,
      "regret_capture": 0.34211790561676025,
      "regret_range": [
        0.0005592876113951206,
        0.0018936246633529663
      ]
    },
    "q3": {
      "exact_parent_action_recall": 0.28614548747358887,
      "flip_detection": 0.42695442197404165,
      "regret_capture": 0.27879267930984497,
      "regret_range": [
        0.0018936246633529663,
        0.005772685632109642
      ]
    },
    "q4": {
      "exact_parent_action_recall": 0.1996428212093772,
      "flip_detection": 0.3364020525203743,
      "regret_capture": 0.15747448801994324,
      "regret_range": [
        0.005772685632109642,
        0.49935781955718994
      ]
    }
  },
  "schema": "azlite_context_action_q_trajectory_selection_probe_v1",
  "simulation_windows": {
    "1_384": {
      "exact_flip_action_recall": 0.34295256789617085,
      "exact_flip_detection_rate": 0.48434066738932996,
      "exact_parent_flip_count": 34133,
      "exact_parent_score_regret_captured": 0.24224884808063507,
      "false_flip_rate": 0.20258260733209588,
      "nonflip_preservation_rate": 0.7974173926679041,
      "overall_exact_parent_action_agreement": 0.7471104452054794
    },
    "385_1200": {
      "exact_flip_action_recall": 0.29145748760939366,
      "exact_flip_detection_rate": 0.4072846356481148,
      "exact_parent_flip_count": 124893,
      "exact_parent_score_regret_captured": 0.14993074536323547,
      "false_flip_rate": 0.10255206418342432,
      "nonflip_preservation_rate": 0.8974479358165757,
      "overall_exact_parent_action_agreement": 0.7819436305032599
    }
  },
  "validation_gate_passed": false
}
```
