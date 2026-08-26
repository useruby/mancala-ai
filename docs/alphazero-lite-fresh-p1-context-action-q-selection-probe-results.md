# Selection-Aligned Context Action-Q Probe

**Classification:** `selection_loss_improves_but_insufficient`

**Recommended follow-up:** add temporal root-trajectory features while KEEPING this same selection loss.

```json
{
  "classification": "selection_loss_improves_but_insufficient",
  "correction_magnitude": {
    "centered_scale_by_simulation_window": {
      "1_384": {
        "mean_centered_a16_q_std": 0.10582625865936279,
        "mean_centered_correction_std": 1.5597201585769653
      },
      "385_1200": {
        "mean_centered_a16_q_std": 0.10464580357074738,
        "mean_centered_correction_std": 1.6180825233459473
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
      "max_abs": 22.449644088745117,
      "mean": 0.46609199047088623,
      "p50": 0.8967658281326294,
      "p90": 9.951699256896973,
      "p95": 13.173319816589355,
      "p99": 15.406989097595215,
      "std": 7.154633522033691
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
  "features_unchanged": true,
  "frozen_40_40_offline": {
    "amplified": {
      "exact_flip_action_recall": 0.19699202276307098,
      "exact_flip_detection_rate": 0.4109039174838677,
      "exact_parent_flip_count": 19681,
      "exact_parent_score_regret_captured": 0.015103967860341072,
      "false_flip_rate": 0.4483915392492673,
      "nonflip_preservation_rate": 0.5516084607507328,
      "overall_exact_parent_action_agreement": 0.40620833333333334
    },
    "washed": {
      "exact_flip_action_recall": 0.3794010775592031,
      "exact_flip_detection_rate": 0.5630873324144844,
      "exact_parent_flip_count": 7981,
      "exact_parent_score_regret_captured": 0.24861198663711548,
      "false_flip_rate": 0.2135235763012569,
      "nonflip_preservation_rate": 0.786476423698743,
      "overall_exact_parent_action_agreement": 0.7187916666666667
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
    "probe_checkpoint": "aaaaac96973348a6f8386fa3c877a3c8cd491be89f24e5db6376418e5b073236",
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
  "invariants": {
    "exact_correction_parent_q_counterfactual": true,
    "pre_simulation_p1_evidence_only": true,
    "puct_move_tie_order": true,
    "unvisited_fpu_preserved": true,
    "zero_correction_ordinary_a16_winner": true
  },
  "live_frozen_root": "not_run_validation_gate_failed",
  "optimization": {
    "batch_size_root_decisions": 256,
    "best_validation_selection_ce": 0.6591775108581265,
    "lr": 0.001,
    "seed": 238,
    "steps": 2000,
    "train_decisions": 3854400,
    "validation_decisions": 963600,
    "validation_history": [
      {
        "step": 50,
        "validation_selection_ce": 0.9884072413589016
      },
      {
        "step": 100,
        "validation_selection_ce": 0.8237242935326348
      },
      {
        "step": 150,
        "validation_selection_ce": 0.7551998069069602
      },
      {
        "step": 200,
        "validation_selection_ce": 0.7310371975960104
      },
      {
        "step": 250,
        "validation_selection_ce": 0.722045368148658
      },
      {
        "step": 300,
        "validation_selection_ce": 0.7193139448280699
      },
      {
        "step": 350,
        "validation_selection_ce": 0.7156379111271371
      },
      {
        "step": 400,
        "validation_selection_ce": 0.713939944243916
      },
      {
        "step": 450,
        "validation_selection_ce": 0.711302036461171
      },
      {
        "step": 500,
        "validation_selection_ce": 0.7090460025190367
      },
      {
        "step": 550,
        "validation_selection_ce": 0.7076757488195309
      },
      {
        "step": 600,
        "validation_selection_ce": 0.7055147409656821
      },
      {
        "step": 650,
        "validation_selection_ce": 0.7038363714602143
      },
      {
        "step": 700,
        "validation_selection_ce": 0.7039939480690899
      },
      {
        "step": 750,
        "validation_selection_ce": 0.7017364035765132
      },
      {
        "step": 800,
        "validation_selection_ce": 0.700239321659986
      },
      {
        "step": 850,
        "validation_selection_ce": 0.697386104445974
      },
      {
        "step": 900,
        "validation_selection_ce": 0.6950976382612242
      },
      {
        "step": 950,
        "validation_selection_ce": 0.6940401014404803
      },
      {
        "step": 1000,
        "validation_selection_ce": 0.6919523841664721
      },
      {
        "step": 1050,
        "validation_selection_ce": 0.6903471359943344
      },
      {
        "step": 1100,
        "validation_selection_ce": 0.6884158374601105
      },
      {
        "step": 1150,
        "validation_selection_ce": 0.6877026526806215
      },
      {
        "step": 1200,
        "validation_selection_ce": 0.686909755074963
      },
      {
        "step": 1250,
        "validation_selection_ce": 0.6843711602831342
      },
      {
        "step": 1300,
        "validation_selection_ce": 0.6818377791959351
      },
      {
        "step": 1350,
        "validation_selection_ce": 0.6804015918191709
      },
      {
        "step": 1400,
        "validation_selection_ce": 0.6835222620981865
      },
      {
        "step": 1450,
        "validation_selection_ce": 0.6763728575114641
      },
      {
        "step": 1500,
        "validation_selection_ce": 0.676532519551519
      },
      {
        "step": 1550,
        "validation_selection_ce": 0.6737747249389496
      },
      {
        "step": 1600,
        "validation_selection_ce": 0.6745893499454357
      },
      {
        "step": 1650,
        "validation_selection_ce": 0.669539443000613
      },
      {
        "step": 1700,
        "validation_selection_ce": 0.6687924808741309
      },
      {
        "step": 1750,
        "validation_selection_ce": 0.6662503075827301
      },
      {
        "step": 1800,
        "validation_selection_ce": 0.6658444258205719
      },
      {
        "step": 1850,
        "validation_selection_ce": 0.6634165516905193
      },
      {
        "step": 1900,
        "validation_selection_ce": 0.6632254039721649
      },
      {
        "step": 1950,
        "validation_selection_ce": 0.6608482217056318
      },
      {
        "step": 2000,
        "validation_selection_ce": 0.6591775108581265
      }
    ],
    "weight_decay": 0.0
  },
  "recommended_follow_up": "add temporal root-trajectory features while KEEPING this same selection loss.",
  "regret_quartiles_exact_parent_flips": {
    "q1": {
      "exact_parent_action_recall": 0.423975652086425,
      "flip_detection": 0.5275800487964384,
      "regret_capture": 0.41141456365585327,
      "regret_range": [
        0.0,
        0.0005592876113951206
      ]
    },
    "q2": {
      "exact_parent_action_recall": 0.38954146439620696,
      "flip_detection": 0.5366350579772116,
      "regret_capture": 0.3879317045211792,
      "regret_range": [
        0.0005592876113951206,
        0.0018936246633529663
      ]
    },
    "q3": {
      "exact_parent_action_recall": 0.3444259985914076,
      "flip_detection": 0.5409246403058657,
      "regret_capture": 0.3371095061302185,
      "regret_range": [
        0.0018936246633529663,
        0.005772685632109642
      ]
    },
    "q4": {
      "exact_parent_action_recall": 0.25543314216722,
      "flip_detection": 0.48178891236542915,
      "regret_capture": 0.20660614967346191,
      "regret_range": [
        0.005772685632109642,
        0.49935781955718994
      ]
    }
  },
  "schema": "azlite_context_action_q_selection_probe_v1",
  "selection_ce_minus_historical_mse": {
    "exact_flip_action_recall": 0.17495252348672546,
    "exact_parent_score_regret_captured": 0.13051065057516098,
    "nonflip_preservation_rate": -0.05607812332986151,
    "overall_exact_parent_action_agreement": -0.01795039435450385
  },
  "simulation_windows": {
    "1_384": {
      "exact_flip_action_recall": 0.3777868924501216,
      "exact_flip_detection_rate": 0.5424662350218263,
      "exact_parent_flip_count": 34133,
      "exact_parent_score_regret_captured": 0.27555838227272034,
      "false_flip_rate": 0.22938600169937168,
      "nonflip_preservation_rate": 0.7706139983006284,
      "overall_exact_parent_action_agreement": 0.7271300332088003
    },
    "385_1200": {
      "exact_flip_action_recall": 0.3466647450217386,
      "exact_flip_detection_rate": 0.516065752283955,
      "exact_parent_flip_count": 124893,
      "exact_parent_score_regret_captured": 0.20890669524669647,
      "false_flip_rate": 0.1587446144563547,
      "nonflip_preservation_rate": 0.8412553855436453,
      "overall_exact_parent_action_agreement": 0.74698434791102
    }
  },
  "validation_gate_passed": false,
  "validation_selection_consequence": {
    "exact_flip_action_recall": 0.35334473608089245,
    "exact_flip_detection_rate": 0.5217322953479305,
    "exact_parent_flip_count": 159026,
    "exact_parent_score_regret_captured": 0.23206011950969696,
    "false_flip_rate": 0.1828209710977486,
    "nonflip_preservation_rate": 0.8171790289022514,
    "overall_exact_parent_action_agreement": 0.7406309672063097
  }
}
```
