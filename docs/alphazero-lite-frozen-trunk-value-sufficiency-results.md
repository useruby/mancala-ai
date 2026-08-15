# Frozen-Trunk Value Sufficiency Audit

**Classification:** `value_head_calibration_bottleneck`

```json
{
  "causal_puct_check": null,
  "classification": "value_head_calibration_bottleneck",
  "concordance_gain_bootstrap_95": {
    "lower": -0.026150235793482135,
    "upper": 0.03204767990635248
  },
  "current_weights_sha256": "8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a",
  "evaluation_domain": {
    "excluded_from_probe_training": true,
    "state_count": 240
  },
  "manifest": {
    "current_weights_sha256": "8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a",
    "schema": "azlite_frozen_trunk_value_sufficiency_v1",
    "source_provenance": {
      "additional_selfplay": {
        "path": "/tmp/azlite_denoised_puct_convergence/additional_standard_start_selfplay_states.jsonl",
        "sha256": "63df5bf39601cc2c44bd47a92d310e02aef6232c04bdf4bed525576200a5751c"
      },
      "generated_opening_family_diagnostic": {
        "families_requested": 1024,
        "generation": "seeded disjoint standard-start opening prefixes"
      },
      "opening_family_diagnostic": {
        "path": "/tmp/azlite_opening_suite/large_eval.jsonl",
        "sha256": "ebc86f053a8a4c12f3e937b5acfe37df7b0d2f1f0002969dcac077a9f99d65a4"
      },
      "standard_start_selfplay": {
        "path": "/tmp/azlite_distribution_aligned_selfplay/pilot_standard_replay.jsonl",
        "sha256": "485ea65c8ca1e9a3416bbccbf97f7e212a9f6082da759fd4c5cc8452c612385b"
      }
    },
    "split_counts": {
      "test": 881,
      "train": 2411,
      "validation": 804
    },
    "split_unit": "source_game/opening_family",
    "state_count": 4096
  },
  "schema": "azlite_frozen_trunk_value_sufficiency_v1",
  "target_stability": {
    "D768_D1200_normalized_margin_mae": 0.09954833984374999,
    "D768_D1200_outcome_agreement": 0.80859375
  },
  "test": {
    "affine_current_value": {
      "calibration": {
        "margin_prediction_deciles": [
          {
            "bucket": 1,
            "mean_prediction": -0.25615579301586155,
            "mean_target": -0.22064393939393936,
            "n": 88
          },
          {
            "bucket": 2,
            "mean_prediction": -0.11452385669061815,
            "mean_target": -0.11032196969696968,
            "n": 88
          },
          {
            "bucket": 3,
            "mean_prediction": -0.02988881108995479,
            "mean_target": -0.030898876404494378,
            "n": 89
          },
          {
            "bucket": 4,
            "mean_prediction": -0.0008519821703017812,
            "mean_target": -0.0009578544061302692,
            "n": 87
          },
          {
            "bucket": 5,
            "mean_prediction": 0.01942955694442919,
            "mean_target": -0.009469696969696963,
            "n": 88
          },
          {
            "bucket": 6,
            "mean_prediction": 0.038269068293285316,
            "mean_target": 0.01685393258426966,
            "n": 89
          },
          {
            "bucket": 7,
            "mean_prediction": 0.05905664478765267,
            "mean_target": 0.04501915708812261,
            "n": 87
          },
          {
            "bucket": 8,
            "mean_prediction": 0.09078503570838818,
            "mean_target": 0.11363636363636363,
            "n": 88
          },
          {
            "bucket": 9,
            "mean_prediction": 0.13797685659395328,
            "mean_target": 0.1396780303030303,
            "n": 88
          },
          {
            "bucket": 10,
            "mean_prediction": 0.23667551773316894,
            "mean_target": 0.20739700374531833,
            "n": 89
          }
        ],
        "outcome_value_buckets": [
          {
            "bucket": 1,
            "mean_prediction": -0.25615579301586155,
            "mean_target": -0.8636363636363636,
            "n": 88
          },
          {
            "bucket": 2,
            "mean_prediction": -0.11452385669061815,
            "mean_target": -0.45454545454545453,
            "n": 88
          },
          {
            "bucket": 3,
            "mean_prediction": -0.02988881108995479,
            "mean_target": -0.20224719101123595,
            "n": 89
          },
          {
            "bucket": 4,
            "mean_prediction": -0.0008519821703017812,
            "mean_target": 0.034482758620689655,
            "n": 87
          },
          {
            "bucket": 5,
            "mean_prediction": 0.01942955694442919,
            "mean_target": -0.07954545454545454,
            "n": 88
          },
          {
            "bucket": 6,
            "mean_prediction": 0.038269068293285316,
            "mean_target": 0.011235955056179775,
            "n": 89
          },
          {
            "bucket": 7,
            "mean_prediction": 0.05905664478765267,
            "mean_target": 0.26436781609195403,
            "n": 87
          },
          {
            "bucket": 8,
            "mean_prediction": 0.09078503570838818,
            "mean_target": 0.4659090909090909,
            "n": 88
          },
          {
            "bucket": 9,
            "mean_prediction": 0.13797685659395328,
            "mean_target": 0.48863636363636365,
            "n": 88
          },
          {
            "bucket": 10,
            "mean_prediction": 0.23667551773316894,
            "mean_target": 0.8876404494382022,
            "n": 89
          }
        ]
      },
      "margin": {
        "mae": 0.16488158637563927,
        "pairwise_concordance": 0.679076424005343,
        "pearson": 0.47581678425617085,
        "rmse": 0.21580220343849196,
        "spearman": 0.5033187395275931
      },
      "n": 881,
      "outcome": {
        "brier_style_squared_error": 0.7601146615549077,
        "mae": 0.8159605286868097,
        "sign_accuracy": 0.6118047673098751
      }
    },
    "current_value": {
      "calibration": {
        "margin_prediction_deciles": [
          {
            "bucket": 1,
            "mean_prediction": -0.6653687584806572,
            "mean_target": -0.22064393939393936,
            "n": 88
          },
          {
            "bucket": 2,
            "mean_prediction": -0.31478074260733346,
            "mean_target": -0.11032196969696968,
            "n": 88
          },
          {
            "bucket": 3,
            "mean_prediction": -0.10527973965312658,
            "mean_target": -0.030898876404494378,
            "n": 89
          },
          {
            "bucket": 4,
            "mean_prediction": -0.03340354772691412,
            "mean_target": -0.0009578544061302692,
            "n": 87
          },
          {
            "bucket": 5,
            "mean_prediction": 0.01680027452940439,
            "mean_target": -0.009469696969696963,
            "n": 88
          },
          {
            "bucket": 6,
            "mean_prediction": 0.06343457944105181,
            "mean_target": 0.01685393258426966,
            "n": 89
          },
          {
            "bucket": 7,
            "mean_prediction": 0.11522607166658748,
            "mean_target": 0.041193181818181816,
            "n": 88
          },
          {
            "bucket": 8,
            "mean_prediction": 0.19399359822273254,
            "mean_target": 0.11829501915708812,
            "n": 87
          },
          {
            "bucket": 9,
            "mean_prediction": 0.31024583242833614,
            "mean_target": 0.1396780303030303,
            "n": 88
          },
          {
            "bucket": 10,
            "mean_prediction": 0.5545591467551971,
            "mean_target": 0.20739700374531833,
            "n": 89
          }
        ],
        "outcome_value_buckets": [
          {
            "bucket": 1,
            "mean_prediction": -0.6653687584806572,
            "mean_target": -0.8636363636363636,
            "n": 88
          },
          {
            "bucket": 2,
            "mean_prediction": -0.31478074260733346,
            "mean_target": -0.45454545454545453,
            "n": 88
          },
          {
            "bucket": 3,
            "mean_prediction": -0.10527973965312658,
            "mean_target": -0.20224719101123595,
            "n": 89
          },
          {
            "bucket": 4,
            "mean_prediction": -0.03340354772691412,
            "mean_target": 0.034482758620689655,
            "n": 87
          },
          {
            "bucket": 5,
            "mean_prediction": 0.01680027452940439,
            "mean_target": -0.07954545454545454,
            "n": 88
          },
          {
            "bucket": 6,
            "mean_prediction": 0.06343457944105181,
            "mean_target": 0.011235955056179775,
            "n": 89
          },
          {
            "bucket": 7,
            "mean_prediction": 0.11522607166658748,
            "mean_target": 0.25,
            "n": 88
          },
          {
            "bucket": 8,
            "mean_prediction": 0.19399359822273254,
            "mean_target": 0.4827586206896552,
            "n": 87
          },
          {
            "bucket": 9,
            "mean_prediction": 0.31024583242833614,
            "mean_target": 0.48863636363636365,
            "n": 88
          },
          {
            "bucket": 10,
            "mean_prediction": 0.5545591467551971,
            "mean_target": 0.8876404494382022,
            "n": 89
          }
        ]
      },
      "margin": {
        "mae": 0.24391000012966607,
        "pairwise_concordance": 0.679076424005343,
        "pearson": 0.47581678425617085,
        "rmse": 0.30022932965971516,
        "spearman": 0.5033187395275931
      },
      "n": 881,
      "outcome": {
        "brier_style_squared_error": 0.6665477008959544,
        "mae": 0.7393255253114567,
        "sign_accuracy": 0.6095346197502838
      }
    },
    "frozen_trunk_linear": {
      "calibration": {
        "margin_prediction_deciles": [
          {
            "bucket": 1,
            "mean_prediction": -0.25730628905917907,
            "mean_target": -0.212594696969697,
            "n": 88
          },
          {
            "bucket": 2,
            "mean_prediction": -0.125372742495366,
            "mean_target": -0.10890151515151515,
            "n": 88
          },
          {
            "bucket": 3,
            "mean_prediction": -0.07327191842613451,
            "mean_target": -0.0594569288389513,
            "n": 89
          },
          {
            "bucket": 4,
            "mean_prediction": -0.03232106776065232,
            "mean_target": -0.03879310344827586,
            "n": 87
          },
          {
            "bucket": 5,
            "mean_prediction": 0.006372358319762716,
            "mean_target": -0.012310606060606057,
            "n": 88
          },
          {
            "bucket": 6,
            "mean_prediction": 0.04327770151166933,
            "mean_target": 0.06694756554307117,
            "n": 89
          },
          {
            "bucket": 7,
            "mean_prediction": 0.07472692984014401,
            "mean_target": 0.038793103448275856,
            "n": 87
          },
          {
            "bucket": 8,
            "mean_prediction": 0.10618836085007781,
            "mean_target": 0.1074810606060606,
            "n": 88
          },
          {
            "bucket": 9,
            "mean_prediction": 0.14865135997178128,
            "mean_target": 0.15056818181818182,
            "n": 88
          },
          {
            "bucket": 10,
            "mean_prediction": 0.23992597086069278,
            "mean_target": 0.21769662921348318,
            "n": 89
          }
        ],
        "outcome_value_buckets": [
          {
            "bucket": 1,
            "mean_prediction": -0.25730628905917907,
            "mean_target": -0.7727272727272727,
            "n": 88
          },
          {
            "bucket": 2,
            "mean_prediction": -0.125372742495366,
            "mean_target": -0.3522727272727273,
            "n": 88
          },
          {
            "bucket": 3,
            "mean_prediction": -0.07327191842613451,
            "mean_target": -0.3258426966292135,
            "n": 89
          },
          {
            "bucket": 4,
            "mean_prediction": -0.03232106776065232,
            "mean_target": -0.13793103448275862,
            "n": 87
          },
          {
            "bucket": 5,
            "mean_prediction": 0.006372358319762716,
            "mean_target": 0.011363636363636364,
            "n": 88
          },
          {
            "bucket": 6,
            "mean_prediction": 0.04327770151166933,
            "mean_target": 0.24719101123595505,
            "n": 89
          },
          {
            "bucket": 7,
            "mean_prediction": 0.07472692984014401,
            "mean_target": 0.13793103448275862,
            "n": 87
          },
          {
            "bucket": 8,
            "mean_prediction": 0.10618836085007781,
            "mean_target": 0.38636363636363635,
            "n": 88
          },
          {
            "bucket": 9,
            "mean_prediction": 0.14865135997178128,
            "mean_target": 0.5681818181818182,
            "n": 88
          },
          {
            "bucket": 10,
            "mean_prediction": 0.23992597086069278,
            "mean_target": 0.7865168539325843,
            "n": 89
          }
        ]
      },
      "margin": {
        "mae": 0.1653246123551371,
        "pairwise_concordance": 0.6837760846156992,
        "pearson": 0.497661691000595,
        "rmse": 0.21304178910246038,
        "spearman": 0.5014021881583094
      },
      "n": 881,
      "outcome": {
        "brier_style_squared_error": 0.7636086917862469,
        "mae": 0.8179471074536685,
        "sign_accuracy": 0.6220204313280363
      }
    },
    "frozen_trunk_value_head": {
      "calibration": {
        "margin_prediction_deciles": [
          {
            "bucket": 1,
            "mean_prediction": -0.18345941603183746,
            "mean_target": -0.2315340909090909,
            "n": 88
          },
          {
            "bucket": 2,
            "mean_prediction": -0.08481792360544205,
            "mean_target": -0.12263257575757575,
            "n": 88
          },
          {
            "bucket": 3,
            "mean_prediction": -0.04083772003650665,
            "mean_target": -0.06601123595505617,
            "n": 89
          },
          {
            "bucket": 4,
            "mean_prediction": -0.007985546253621578,
            "mean_target": -0.01053639846743295,
            "n": 87
          },
          {
            "bucket": 5,
            "mean_prediction": 0.015848014503717422,
            "mean_target": 0.019412878787878788,
            "n": 88
          },
          {
            "bucket": 6,
            "mean_prediction": 0.041146568953990936,
            "mean_target": 0.04353932584269663,
            "n": 89
          },
          {
            "bucket": 7,
            "mean_prediction": 0.06119764223694801,
            "mean_target": 0.09138257575757577,
            "n": 88
          },
          {
            "bucket": 8,
            "mean_prediction": 0.07991534471511841,
            "mean_target": 0.10632183908045975,
            "n": 87
          },
          {
            "bucket": 9,
            "mean_prediction": 0.1106952354311943,
            "mean_target": 0.10132575757575757,
            "n": 88
          },
          {
            "bucket": 10,
            "mean_prediction": 0.16969698667526245,
            "mean_target": 0.21956928838951312,
            "n": 89
          }
        ],
        "outcome_value_buckets": [
          {
            "bucket": 1,
            "mean_prediction": -0.18345941603183746,
            "mean_target": -0.7840909090909091,
            "n": 88
          },
          {
            "bucket": 2,
            "mean_prediction": -0.08481792360544205,
            "mean_target": -0.4772727272727273,
            "n": 88
          },
          {
            "bucket": 3,
            "mean_prediction": -0.04083772003650665,
            "mean_target": -0.16853932584269662,
            "n": 89
          },
          {
            "bucket": 4,
            "mean_prediction": -0.007985546253621578,
            "mean_target": -0.09195402298850575,
            "n": 87
          },
          {
            "bucket": 5,
            "mean_prediction": 0.015848014503717422,
            "mean_target": 0.10227272727272728,
            "n": 88
          },
          {
            "bucket": 6,
            "mean_prediction": 0.041146568953990936,
            "mean_target": 0.16853932584269662,
            "n": 89
          },
          {
            "bucket": 7,
            "mean_prediction": 0.06119764223694801,
            "mean_target": 0.2727272727272727,
            "n": 88
          },
          {
            "bucket": 8,
            "mean_prediction": 0.07991534471511841,
            "mean_target": 0.3448275862068966,
            "n": 87
          },
          {
            "bucket": 9,
            "mean_prediction": 0.1106952354311943,
            "mean_target": 0.4659090909090909,
            "n": 88
          },
          {
            "bucket": 10,
            "mean_prediction": 0.16969698667526245,
            "mean_target": 0.7191011235955056,
            "n": 89
          }
        ]
      },
      "margin": {
        "mae": 0.16450659483206273,
        "pairwise_concordance": 0.6826638679515313,
        "pearson": 0.5077800772768575,
        "rmse": 0.21239038866417495,
        "spearman": 0.49817470378926515
      },
      "n": 881,
      "outcome": {
        "brier_style_squared_error": 0.7932677895732463,
        "mae": 0.8343835888820224,
        "sign_accuracy": 0.6106696935300795
      }
    },
    "raw_feature_ridge": {
      "calibration": {
        "margin_prediction_deciles": [
          {
            "bucket": 1,
            "mean_prediction": -0.051601499169518415,
            "mean_target": -0.05445075757575757,
            "n": 88
          },
          {
            "bucket": 2,
            "mean_prediction": -0.03343007969473495,
            "mean_target": -0.05492424242424243,
            "n": 88
          },
          {
            "bucket": 3,
            "mean_prediction": -0.020477273174698208,
            "mean_target": -0.14091760299625467,
            "n": 89
          },
          {
            "bucket": 4,
            "mean_prediction": -0.008590920555505832,
            "mean_target": -0.014367816091954021,
            "n": 87
          },
          {
            "bucket": 5,
            "mean_prediction": 0.009726706753569195,
            "mean_target": 0.001420454545454543,
            "n": 88
          },
          {
            "bucket": 6,
            "mean_prediction": 0.03137412203877473,
            "mean_target": 0.0028089887640449463,
            "n": 89
          },
          {
            "bucket": 7,
            "mean_prediction": 0.0481252498718596,
            "mean_target": 0.09232954545454546,
            "n": 88
          },
          {
            "bucket": 8,
            "mean_prediction": 0.06148556777630315,
            "mean_target": 0.11446360153256703,
            "n": 87
          },
          {
            "bucket": 9,
            "mean_prediction": 0.07533456155698544,
            "mean_target": 0.08806818181818182,
            "n": 88
          },
          {
            "bucket": 10,
            "mean_prediction": 0.09980108983143596,
            "mean_target": 0.11891385767790263,
            "n": 89
          }
        ],
        "outcome_value_buckets": [
          {
            "bucket": 1,
            "mean_prediction": -0.051601499169518415,
            "mean_target": -0.1590909090909091,
            "n": 88
          },
          {
            "bucket": 2,
            "mean_prediction": -0.03343007969473495,
            "mean_target": -0.1590909090909091,
            "n": 88
          },
          {
            "bucket": 3,
            "mean_prediction": -0.020477273174698208,
            "mean_target": -0.3707865168539326,
            "n": 89
          },
          {
            "bucket": 4,
            "mean_prediction": -0.008590920555505832,
            "mean_target": 0.08045977011494253,
            "n": 87
          },
          {
            "bucket": 5,
            "mean_prediction": 0.009726706753569195,
            "mean_target": -0.03409090909090909,
            "n": 88
          },
          {
            "bucket": 6,
            "mean_prediction": 0.03137412203877473,
            "mean_target": 0.0449438202247191,
            "n": 89
          },
          {
            "bucket": 7,
            "mean_prediction": 0.0481252498718596,
            "mean_target": 0.22727272727272727,
            "n": 88
          },
          {
            "bucket": 8,
            "mean_prediction": 0.06148556777630315,
            "mean_target": 0.3793103448275862,
            "n": 87
          },
          {
            "bucket": 9,
            "mean_prediction": 0.07533456155698544,
            "mean_target": 0.23863636363636365,
            "n": 88
          },
          {
            "bucket": 10,
            "mean_prediction": 0.09980108983143596,
            "mean_target": 0.3146067415730337,
            "n": 89
          }
        ]
      },
      "margin": {
        "mae": 0.18718970590007034,
        "pairwise_concordance": 0.6029550070195047,
        "pearson": 0.2965245352690721,
        "rmse": 0.23508986478567356,
        "spearman": 0.27629190980901697
      },
      "n": 881,
      "outcome": {
        "brier_style_squared_error": 0.8499135439652259,
        "mae": 0.8638513515813891,
        "sign_accuracy": 0.5232690124858116
      }
    },
    "slices": {
      "current_value_quartile": {
        "1": {
          "affine_current_value": {
            "margin": {
              "mae": 0.14735571214618556,
              "pairwise_concordance": 0.6536236964332661,
              "pearson": 0.4096423083767121,
              "rmse": 0.19362374869477422,
              "spearman": 0.3936805137016164
            },
            "n": 222,
            "outcome": {
              "brier_style_squared_error": 0.6634576320364903,
              "mae": 0.7605688947610502,
              "sign_accuracy": 0.7117117117117117
            }
          },
          "current_value": {
            "margin": {
              "mae": 0.3161489219144658,
              "pairwise_concordance": 0.6536236964332661,
              "pearson": 0.40964230837671234,
              "rmse": 0.3724751855594663,
              "spearman": 0.3936805137016164
            },
            "n": 222,
            "outcome": {
              "brier_style_squared_error": 0.48452760259179367,
              "mae": 0.6000911116197303,
              "sign_accuracy": 0.7117117117117117
            }
          },
          "frozen_trunk_linear": {
            "margin": {
              "mae": 0.14627658279584735,
              "pairwise_concordance": 0.6697923056699676,
              "pearson": 0.47140427922856487,
              "rmse": 0.1901017619119344,
              "spearman": 0.4638499871947512
            },
            "n": 222,
            "outcome": {
              "brier_style_squared_error": 0.6914625086500258,
              "mae": 0.7734974680223415,
              "sign_accuracy": 0.6981981981981982
            }
          },
          "frozen_trunk_value_head": {
            "margin": {
              "mae": 0.14634288795387293,
              "pairwise_concordance": 0.6774603452808693,
              "pearson": 0.46498022688997975,
              "rmse": 0.19424074325764168,
              "spearman": 0.47459433136035617
            },
            "n": 222,
            "outcome": {
              "brier_style_squared_error": 0.7413643862816256,
              "mae": 0.8016292000571444,
              "sign_accuracy": 0.6846846846846847
            }
          },
          "raw_feature_ridge": {
            "margin": {
              "mae": 0.19228680609279594,
              "pairwise_concordance": 0.602006835509596,
              "pearson": 0.35052373742134346,
              "rmse": 0.24179131596913428,
              "spearman": 0.2623880500430224
            },
            "n": 222,
            "outcome": {
              "brier_style_squared_error": 0.8461248676521379,
              "mae": 0.858284601809233,
              "sign_accuracy": 0.5585585585585585
            }
          }
        },
        "2": {
          "affine_current_value": {
            "margin": {
              "mae": 0.18452860492310577,
              "pairwise_concordance": 0.5073418104753254,
              "pearson": 0.03611057885086555,
              "rmse": 0.23346925213614445,
              "spearman": 0.022844666723157832
            },
            "n": 237,
            "outcome": {
              "brier_style_squared_error": 0.8703772393108854,
              "mae": 0.8717648918429372,
              "sign_accuracy": 0.43037974683544306
            }
          },
          "current_value": {
            "margin": {
              "mae": 0.18602949200545496,
              "pairwise_concordance": 0.5073418104753254,
              "pearson": 0.03611057885086555,
              "rmse": 0.23500838759655326,
              "spearman": 0.022844666723157832
            },
            "n": 237,
            "outcome": {
              "brier_style_squared_error": 0.8690238421512305,
              "mae": 0.8723482554756352,
              "sign_accuracy": 0.4219409282700422
            }
          },
          "frozen_trunk_linear": {
            "margin": {
              "mae": 0.1820835795236764,
              "pairwise_concordance": 0.5865879503481684,
              "pearson": 0.26339124851969326,
              "rmse": 0.22885055073561605,
              "spearman": 0.22746678589908395
            },
            "n": 237,
            "outcome": {
              "brier_style_squared_error": 0.8368797636140757,
              "mae": 0.8585426260642054,
              "sign_accuracy": 0.5274261603375527
            }
          },
          "frozen_trunk_value_head": {
            "margin": {
              "mae": 0.18355983625766695,
              "pairwise_concordance": 0.5717529518619436,
              "pearson": 0.2326470292145774,
              "rmse": 0.22856140257396043,
              "spearman": 0.20075307531294004
            },
            "n": 237,
            "outcome": {
              "brier_style_squared_error": 0.857602884438962,
              "mae": 0.8686239671827597,
              "sign_accuracy": 0.4767932489451477
            }
          },
          "raw_feature_ridge": {
            "margin": {
              "mae": 0.18771682234366222,
              "pairwise_concordance": 0.5552528004844081,
              "pearson": 0.16691803010471648,
              "rmse": 0.23347412653724778,
              "spearman": 0.1459013556076797
            },
            "n": 237,
            "outcome": {
              "brier_style_squared_error": 0.8709708355204459,
              "mae": 0.8744218774171085,
              "sign_accuracy": 0.4008438818565401
            }
          }
        },
        "3": {
          "affine_current_value": {
            "margin": {
              "mae": 0.18797763288679992,
              "pairwise_concordance": 0.5413464791455223,
              "pearson": 0.1267898041736899,
              "rmse": 0.2427823345218911,
              "spearman": 0.1184536092493189
            },
            "n": 210,
            "outcome": {
              "brier_style_squared_error": 0.8231856699508834,
              "mae": 0.845304178991909,
              "sign_accuracy": 0.5476190476190477
            }
          },
          "current_value": {
            "margin": {
              "mae": 0.198850975345288,
              "pairwise_concordance": 0.5413464791455223,
              "pearson": 0.12678980417368987,
              "rmse": 0.24974553793619397,
              "spearman": 0.1184536092493189
            },
            "n": 210,
            "outcome": {
              "brier_style_squared_error": 0.8008308078083781,
              "mae": 0.8378439290715115,
              "sign_accuracy": 0.5476190476190477
            }
          },
          "frozen_trunk_linear": {
            "margin": {
              "mae": 0.1903916068044116,
              "pairwise_concordance": 0.5825721328113672,
              "pearson": 0.24262371200550084,
              "rmse": 0.24245082669499915,
              "spearman": 0.2304729942045981
            },
            "n": 210,
            "outcome": {
              "brier_style_squared_error": 0.7979320470484056,
              "mae": 0.829676266574753,
              "sign_accuracy": 0.5285714285714286
            }
          },
          "frozen_trunk_value_head": {
            "margin": {
              "mae": 0.18346500197279506,
              "pairwise_concordance": 0.5903049635107052,
              "pearson": 0.28141342164307065,
              "rmse": 0.23464568367617877,
              "spearman": 0.24548985561706937
            },
            "n": 210,
            "outcome": {
              "brier_style_squared_error": 0.8088780694952309,
              "mae": 0.8369771244312038,
              "sign_accuracy": 0.5428571428571428
            }
          },
          "raw_feature_ridge": {
            "margin": {
              "mae": 0.18600038819125608,
              "pairwise_concordance": 0.581267217630854,
              "pearson": 0.24805431994279722,
              "rmse": 0.23825728166438073,
              "spearman": 0.22919405363905238
            },
            "n": 210,
            "outcome": {
              "brier_style_squared_error": 0.8289928900940037,
              "mae": 0.846603253206998,
              "sign_accuracy": 0.5238095238095238
            }
          }
        },
        "4": {
          "affine_current_value": {
            "margin": {
              "mae": 0.13839210956358936,
              "pairwise_concordance": 0.5845723112569994,
              "pearson": 0.18821681029176526,
              "rmse": 0.18694945199994373,
              "spearman": 0.2728928340678067
            },
            "n": 212,
            "outcome": {
              "brier_style_squared_error": 0.6755897458085257,
              "mae": 0.7825130857596664,
              "sign_accuracy": 0.7735849056603774
            }
          },
          "current_value": {
            "margin": {
              "mae": 0.2776035614217977,
              "pairwise_concordance": 0.5845723112569994,
              "pearson": 0.1882168102917654,
              "rmse": 0.32522801887443276,
              "spearman": 0.2728928340678067
            },
            "n": 212,
            "outcome": {
              "brier_style_squared_error": 0.49778432303941816,
              "mae": 0.6388292422975009,
              "sign_accuracy": 0.7735849056603774
            }
          },
          "frozen_trunk_linear": {
            "margin": {
              "mae": 0.14170536003849024,
              "pairwise_concordance": 0.6288858853060436,
              "pearson": 0.36622755307378446,
              "rmse": 0.18437019805532842,
              "spearman": 0.3569200232243861
            },
            "n": 212,
            "outcome": {
              "brier_style_squared_error": 0.7232469183333812,
              "mae": 0.8074921953198454,
              "sign_accuracy": 0.7405660377358491
            }
          },
          "frozen_trunk_value_head": {
            "margin": {
              "mae": 0.14344744016006353,
              "pairwise_concordance": 0.6410503958293107,
              "pearson": 0.42032394710212173,
              "rmse": 0.187113427227274,
              "spearman": 0.38168850355103884
            },
            "n": 212,
            "outcome": {
              "brier_style_squared_error": 0.7602346728937577,
              "mae": 0.8278357690542866,
              "sign_accuracy": 0.75
            }
          },
          "raw_feature_ridge": {
            "margin": {
              "mae": 0.18244099778183748,
              "pairwise_concordance": 0.6438984359915042,
              "pearson": 0.3444220588422541,
              "rmse": 0.2264551051427904,
              "spearman": 0.36371353399841055
            },
            "n": 212,
            "outcome": {
              "brier_style_squared_error": 0.8510637579080329,
              "mae": 0.874949014246367,
              "sign_accuracy": 0.6226415094339622
            }
          }
        }
      },
      "phase": {
        "late": {
          "affine_current_value": {
            "margin": {
              "mae": 0.09697509873035862,
              "pairwise_concordance": 0.8401058901389808,
              "pearson": 0.8188882778109476,
              "rmse": 0.11730633396803161,
              "spearman": 0.8159771447369453
            },
            "n": 183,
            "outcome": {
              "brier_style_squared_error": 0.5409063015295098,
              "mae": 0.6610694466803984,
              "sign_accuracy": 0.6666666666666666
            }
          },
          "current_value": {
            "margin": {
              "mae": 0.33228858301262515,
              "pairwise_concordance": 0.8401058901389808,
              "pearson": 0.8188882778109486,
              "rmse": 0.3780858755287131,
              "spearman": 0.8159771447369453
            },
            "n": 183,
            "outcome": {
              "brier_style_squared_error": 0.3529256879343417,
              "mae": 0.4987785910120071,
              "sign_accuracy": 0.6612021857923497
            }
          },
          "frozen_trunk_linear": {
            "margin": {
              "mae": 0.09877002662145329,
              "pairwise_concordance": 0.7810721376571806,
              "pearson": 0.7049604609459009,
              "rmse": 0.12262057154331803,
              "spearman": 0.707690741190385
            },
            "n": 183,
            "outcome": {
              "brier_style_squared_error": 0.6286916888379573,
              "mae": 0.7101861479791541,
              "sign_accuracy": 0.6284153005464481
            }
          },
          "frozen_trunk_value_head": {
            "margin": {
              "mae": 0.09147605005454931,
              "pairwise_concordance": 0.7755790866975513,
              "pearson": 0.7116616857944011,
              "rmse": 0.11876571796902136,
              "spearman": 0.6932710909902643
            },
            "n": 183,
            "outcome": {
              "brier_style_squared_error": 0.6564132660635754,
              "mae": 0.7223333458268577,
              "sign_accuracy": 0.6174863387978142
            }
          },
          "raw_feature_ridge": {
            "margin": {
              "mae": 0.1309896833187105,
              "pairwise_concordance": 0.6191264063534083,
              "pearson": 0.4098222832907293,
              "rmse": 0.15577589237090536,
              "spearman": 0.3351844164388712
            },
            "n": 183,
            "outcome": {
              "brier_style_squared_error": 0.752078929093212,
              "mae": 0.7706701671787882,
              "sign_accuracy": 0.5136612021857924
            }
          }
        },
        "midgame": {
          "affine_current_value": {
            "margin": {
              "mae": 0.11255849014903861,
              "pairwise_concordance": 0.8011584939578548,
              "pearson": 0.7778147231726188,
              "rmse": 0.14315515029475911,
              "spearman": 0.7786162551936222
            },
            "n": 146,
            "outcome": {
              "brier_style_squared_error": 0.6712598764700459,
              "mae": 0.77888557072836,
              "sign_accuracy": 0.7671232876712328
            }
          },
          "current_value": {
            "margin": {
              "mae": 0.22777918428655702,
              "pairwise_concordance": 0.8011584939578548,
              "pearson": 0.7778147231726182,
              "rmse": 0.27698304780038085,
              "spearman": 0.7786162551936222
            },
            "n": 146,
            "outcome": {
              "brier_style_squared_error": 0.45305040660002865,
              "mae": 0.6159056610011572,
              "sign_accuracy": 0.7808219178082192
            }
          },
          "frozen_trunk_linear": {
            "margin": {
              "mae": 0.13207525809333187,
              "pairwise_concordance": 0.7665035453909917,
              "pearson": 0.6814608722789519,
              "rmse": 0.16533680345308868,
              "spearman": 0.702759970308388
            },
            "n": 146,
            "outcome": {
              "brier_style_squared_error": 0.7227075217649884,
              "mae": 0.8055555846057302,
              "sign_accuracy": 0.726027397260274
            }
          },
          "frozen_trunk_value_head": {
            "margin": {
              "mae": 0.136922932715598,
              "pairwise_concordance": 0.7811844602017377,
              "pearson": 0.7293568411376962,
              "rmse": 0.1687715600333191,
              "spearman": 0.7245505287613393
            },
            "n": 146,
            "outcome": {
              "brier_style_squared_error": 0.7786361479231263,
              "mae": 0.8349536423575793,
              "sign_accuracy": 0.7328767123287672
            }
          },
          "raw_feature_ridge": {
            "margin": {
              "mae": 0.18814607457308147,
              "pairwise_concordance": 0.5892339958054529,
              "pearson": 0.2606758376768513,
              "rmse": 0.21834369740687146,
              "spearman": 0.24311839048325995
            },
            "n": 146,
            "outcome": {
              "brier_style_squared_error": 0.8837347250315545,
              "mae": 0.8905474463346916,
              "sign_accuracy": 0.4794520547945205
            }
          }
        },
        "opening": {
          "affine_current_value": {
            "margin": {
              "mae": 0.20123314305710677,
              "pairwise_concordance": 0.6098339410901019,
              "pearson": 0.3403632664932452,
              "rmse": 0.2536631364235703,
              "spearman": 0.3135031418334803
            },
            "n": 552,
            "outcome": {
              "brier_style_squared_error": 0.8562884450821499,
              "mae": 0.8771163472902643,
              "sign_accuracy": 0.552536231884058
            }
          },
          "current_value": {
            "margin": {
              "mae": 0.2188770625309567,
              "pairwise_concordance": 0.6098339410901019,
              "pearson": 0.3403632664932452,
              "rmse": 0.2760045980448119,
              "spearman": 0.3135031418334803
            },
            "n": 552,
            "outcome": {
              "brier_style_squared_error": 0.8269887033220057,
              "mae": 0.851715723076136,
              "sign_accuracy": 0.5471014492753623
            }
          },
          "frozen_trunk_linear": {
            "margin": {
              "mae": 0.1961831176295713,
              "pairwise_concordance": 0.642108238148743,
              "pearson": 0.41951573910867346,
              "rmse": 0.24540386799550187,
              "spearman": 0.4005061093703648
            },
            "n": 552,
            "outcome": {
              "brier_style_squared_error": 0.8191546743272629,
              "mae": 0.8569496761486596,
              "sign_accuracy": 0.592391304347826
            }
          },
          "frozen_trunk_value_head": {
            "margin": {
              "mae": 0.1960134867945425,
              "pairwise_concordance": 0.6409363819975322,
              "pearson": 0.43113600550349906,
              "rmse": 0.24451118021487273,
              "spearman": 0.4002452709117032
            },
            "n": 552,
            "outcome": {
              "brier_style_squared_error": 0.8425080024051073,
              "mae": 0.8713799053161958,
              "sign_accuracy": 0.5760869565217391
            }
          },
          "raw_feature_ridge": {
            "margin": {
              "mae": 0.20556828254160878,
              "pairwise_concordance": 0.6002729735505173,
              "pearson": 0.2939318143457531,
              "rmse": 0.2599098967194828,
              "spearman": 0.289217483197109
            },
            "n": 552,
            "outcome": {
              "brier_style_squared_error": 0.8734023883237306,
              "mae": 0.8876820162764866,
              "sign_accuracy": 0.5380434782608695
            }
          }
        }
      },
      "player": {
        "0": {
          "affine_current_value": {
            "margin": {
              "mae": 0.1583707706560578,
              "pairwise_concordance": 0.6661383731738125,
              "pearson": 0.42677588411345185,
              "rmse": 0.20776791457824534,
              "spearman": 0.4774683763509418
            },
            "n": 424,
            "outcome": {
              "brier_style_squared_error": 0.752660174935491,
              "mae": 0.8112199086991303,
              "sign_accuracy": 0.6367924528301887
            }
          },
          "current_value": {
            "margin": {
              "mae": 0.22985806853054938,
              "pairwise_concordance": 0.6661383731738125,
              "pearson": 0.4267758841134522,
              "rmse": 0.28522614083415465,
              "spearman": 0.4774683763509418
            },
            "n": 424,
            "outcome": {
              "brier_style_squared_error": 0.6542380038198464,
              "mae": 0.7340818716942488,
              "sign_accuracy": 0.6344339622641509
            }
          },
          "frozen_trunk_linear": {
            "margin": {
              "mae": 0.1599926686320773,
              "pairwise_concordance": 0.6557115279201124,
              "pearson": 0.41427178487127586,
              "rmse": 0.20557161367062057,
              "spearman": 0.4301932447005297
            },
            "n": 424,
            "outcome": {
              "brier_style_squared_error": 0.7526841106556857,
              "mae": 0.8138457353493712,
              "sign_accuracy": 0.6462264150943396
            }
          },
          "frozen_trunk_value_head": {
            "margin": {
              "mae": 0.16372429262621566,
              "pairwise_concordance": 0.6295431254319693,
              "pearson": 0.3715539297116425,
              "rmse": 0.20834723887719353,
              "spearman": 0.3590738423028786
            },
            "n": 424,
            "outcome": {
              "brier_style_squared_error": 0.7951694343422069,
              "mae": 0.8360117551514259,
              "sign_accuracy": 0.6108490566037735
            }
          },
          "raw_feature_ridge": {
            "margin": {
              "mae": 0.18085992329913306,
              "pairwise_concordance": 0.5278009485450083,
              "pearson": 0.08195016653229983,
              "rmse": 0.2241386767574194,
              "spearman": 0.038580143418266544
            },
            "n": 424,
            "outcome": {
              "brier_style_squared_error": 0.8362584662339085,
              "mae": 0.8577040960966609,
              "sign_accuracy": 0.5660377358490566
            }
          }
        },
        "1": {
          "affine_current_value": {
            "margin": {
              "mae": 0.17092225566470384,
              "pairwise_concordance": 0.6635899630184297,
              "pearson": 0.44120910222691606,
              "rmse": 0.222997620438945,
              "spearman": 0.4556622462101573
            },
            "n": 457,
            "outcome": {
              "brier_style_squared_error": 0.7670308592061827,
              "mae": 0.8203588281939782,
              "sign_accuracy": 0.5886214442013129
            }
          },
          "current_value": {
            "margin": {
              "mae": 0.2569472408255643,
              "pairwise_concordance": 0.6635899630184297,
              "pearson": 0.4412091022269158,
              "rmse": 0.31350769987555993,
              "spearman": 0.4556622462101573
            },
            "n": 457,
            "outcome": {
              "brier_style_squared_error": 0.6779685139381202,
              "mae": 0.7441905343567435,
              "sign_accuracy": 0.5864332603938731
            }
          },
          "frozen_trunk_linear": {
            "margin": {
              "mae": 0.17027153607193657,
              "pairwise_concordance": 0.6563057143148222,
              "pearson": 0.4295983894262031,
              "rmse": 0.2197455579554941,
              "spearman": 0.4341080573544552
            },
            "n": 457,
            "outcome": {
              "brier_style_squared_error": 0.7737444081962206,
              "mae": 0.8217523192090775,
              "sign_accuracy": 0.5995623632385121
            }
          },
          "frozen_trunk_value_head": {
            "margin": {
              "mae": 0.165232406944271,
              "pairwise_concordance": 0.6680216388031419,
              "pearson": 0.4556395997340292,
              "rmse": 0.21607393531931576,
              "spearman": 0.4522312333440791
            },
            "n": 457,
            "outcome": {
              "brier_style_squared_error": 0.7915034626978869,
              "mae": 0.8328729926058142,
              "sign_accuracy": 0.6105032822757112
            }
          },
          "raw_feature_ridge": {
            "margin": {
              "mae": 0.1930624144838721,
              "pairwise_concordance": 0.5113135079515471,
              "pearson": 0.02107091527630392,
              "rmse": 0.24481256325386214,
              "spearman": 0.0005848903167208726
            },
            "n": 457,
            "outcome": {
              "brier_style_squared_error": 0.8625825876371703,
              "mae": 0.8695547133440257,
              "sign_accuracy": 0.48358862144420134
            }
          }
        }
      },
      "source_domain": {
        "additional_selfplay": {
          "affine_current_value": {
            "margin": {
              "mae": 0.13213949321633073,
              "pairwise_concordance": 0.7383548067393458,
              "pearson": 0.6249349120827792,
              "rmse": 0.16466021990415772,
              "spearman": 0.6329787234042553
            },
            "n": 47,
            "outcome": {
              "brier_style_squared_error": 0.7645952230695913,
              "mae": 0.8366487217609926,
              "sign_accuracy": 0.6808510638297872
            }
          },
          "current_value": {
            "margin": {
              "mae": 0.20421882862132704,
              "pairwise_concordance": 0.7383548067393458,
              "pearson": 0.6249349120827792,
              "rmse": 0.264611832230453,
              "spearman": 0.6329787234042553
            },
            "n": 47,
            "outcome": {
              "brier_style_squared_error": 0.6091436717362594,
              "mae": 0.7165495482808415,
              "sign_accuracy": 0.6595744680851063
            }
          },
          "frozen_trunk_linear": {
            "margin": {
              "mae": 0.134316818430179,
              "pairwise_concordance": 0.7145688800792864,
              "pearson": 0.5495562466574352,
              "rmse": 0.18002342283317477,
              "spearman": 0.5316836262719704
            },
            "n": 47,
            "outcome": {
              "brier_style_squared_error": 0.7782199685444243,
              "mae": 0.8453333807660015,
              "sign_accuracy": 0.7446808510638298
            }
          },
          "frozen_trunk_value_head": {
            "margin": {
              "mae": 0.15612350954263018,
              "pairwise_concordance": 0.6422200198216056,
              "pearson": 0.3536671589697839,
              "rmse": 0.1976950852162106,
              "spearman": 0.3256244218316374
            },
            "n": 47,
            "outcome": {
              "brier_style_squared_error": 0.8661530628578602,
              "mae": 0.8929552481363606,
              "sign_accuracy": 0.574468085106383
            }
          },
          "raw_feature_ridge": {
            "margin": {
              "mae": 0.17450460509091206,
              "pairwise_concordance": 0.48265609514370666,
              "pearson": 0.022299856360258812,
              "rmse": 0.21710935279030105,
              "spearman": -0.07874653098982425
            },
            "n": 47,
            "outcome": {
              "brier_style_squared_error": 0.9324189551215292,
              "mae": 0.9260903264109341,
              "sign_accuracy": 0.40425531914893614
            }
          }
        },
        "generated_opening_family_diagnostic": {
          "affine_current_value": {
            "margin": {
              "mae": 0.18296602236264647,
              "pairwise_concordance": 0.5584480351087173,
              "pearson": 0.22087878980015274,
              "rmse": 0.22675761871241337,
              "spearman": 0.17125176616370352
            },
            "n": 147,
            "outcome": {
              "brier_style_squared_error": 0.8008303905616859,
              "mae": 0.8160077456516338,
              "sign_accuracy": 0.46258503401360546
            }
          },
          "current_value": {
            "margin": {
              "mae": 0.20304845748322722,
              "pairwise_concordance": 0.5584480351087173,
              "pearson": 0.2208787898001527,
              "rmse": 0.24740094602762727,
              "spearman": 0.17125176616370352
            },
            "n": 147,
            "outcome": {
              "brier_style_squared_error": 0.7950937921373107,
              "mae": 0.815733636960468,
              "sign_accuracy": 0.4489795918367347
            }
          },
          "frozen_trunk_linear": {
            "margin": {
              "mae": 0.18224962530032027,
              "pairwise_concordance": 0.5860762018751247,
              "pearson": 0.24920034657331963,
              "rmse": 0.23024393734465445,
              "spearman": 0.2299752925976018
            },
            "n": 147,
            "outcome": {
              "brier_style_squared_error": 0.7718083164390264,
              "mae": 0.8029591577671567,
              "sign_accuracy": 0.5238095238095238
            }
          },
          "frozen_trunk_value_head": {
            "margin": {
              "mae": 0.17729552675715948,
              "pairwise_concordance": 0.6016357470576501,
              "pearson": 0.29585735597654267,
              "rmse": 0.22292159669572006,
              "spearman": 0.28994174493196023
            },
            "n": 147,
            "outcome": {
              "brier_style_squared_error": 0.7803387336122948,
              "mae": 0.8072693929884767,
              "sign_accuracy": 0.5170068027210885
            }
          },
          "raw_feature_ridge": {
            "margin": {
              "mae": 0.1812898344878076,
              "pairwise_concordance": 0.5422900458807102,
              "pearson": 0.09397603545893139,
              "rmse": 0.23285209888730404,
              "spearman": 0.12384679899357003
            },
            "n": 147,
            "outcome": {
              "brier_style_squared_error": 0.8044848788284951,
              "mae": 0.8171861087274579,
              "sign_accuracy": 0.42857142857142855
            }
          }
        },
        "opening_family_diagnostic": {
          "affine_current_value": {
            "margin": {
              "mae": 0.1704909611261848,
              "pairwise_concordance": 0.549889135254989,
              "pearson": 0.07299580309684295,
              "rmse": 0.22807458328581462,
              "spearman": 0.1380725743496764
            },
            "n": 62,
            "outcome": {
              "brier_style_squared_error": 0.8766815938009723,
              "mae": 0.8908613013982043,
              "sign_accuracy": 0.5645161290322581
            }
          },
          "current_value": {
            "margin": {
              "mae": 0.18482467406244713,
              "pairwise_concordance": 0.549889135254989,
              "pearson": 0.07299580309684295,
              "rmse": 0.24318583339816538,
              "spearman": 0.1380725743496764
            },
            "n": 62,
            "outcome": {
              "brier_style_squared_error": 0.856375290819785,
              "mae": 0.8771480467116963,
              "sign_accuracy": 0.5483870967741935
            }
          },
          "frozen_trunk_linear": {
            "margin": {
              "mae": 0.1841736029761807,
              "pairwise_concordance": 0.5254988913525499,
              "pearson": 0.08318299523782645,
              "rmse": 0.23611003938289588,
              "spearman": 0.10679660547455366
            },
            "n": 62,
            "outcome": {
              "brier_style_squared_error": 0.8562493683981168,
              "mae": 0.883342042194348,
              "sign_accuracy": 0.5161290322580645
            }
          },
          "frozen_trunk_value_head": {
            "margin": {
              "mae": 0.18189289032666872,
              "pairwise_concordance": 0.5144124168514412,
              "pearson": 0.06974777544122666,
              "rmse": 0.2321983870147364,
              "spearman": 0.046712497796580287
            },
            "n": 62,
            "outcome": {
              "brier_style_squared_error": 0.8771458018180136,
              "mae": 0.8938926305779586,
              "sign_accuracy": 0.532258064516129
            }
          },
          "raw_feature_ridge": {
            "margin": {
              "mae": 0.1755223423224432,
              "pairwise_concordance": 0.49889135254988914,
              "pearson": 0.07645027903319464,
              "rmse": 0.22755643194760033,
              "spearman": 0.005615572511394828
            },
            "n": 62,
            "outcome": {
              "brier_style_squared_error": 0.8859420155604564,
              "mae": 0.8968353624748109,
              "sign_accuracy": 0.5
            }
          }
        },
        "standard_start_selfplay": {
          "affine_current_value": {
            "margin": {
              "mae": 0.16253388246182107,
              "pairwise_concordance": 0.7047659740175451,
              "pearson": 0.5197787004438451,
              "rmse": 0.21529762724092405,
              "spearman": 0.5748551486851807
            },
            "n": 625,
            "outcome": {
              "brier_style_squared_error": 0.7386379441877997,
              "mae": 0.8069635144845343,
              "sign_accuracy": 0.6464
            }
          },
          "current_value": {
            "margin": {
              "mae": 0.26236667540340375,
              "pairwise_concordance": 0.7047659740175451,
              "pearson": 0.5197787004438452,
              "rmse": 0.31863773779294136,
              "spearman": 0.5748551486851807
            },
            "n": 625,
            "outcome": {
              "brier_style_squared_error": 0.6217995463083524,
              "mae": 0.7093950968014077,
              "sign_accuracy": 0.6496
            }
          },
          "frozen_trunk_linear": {
            "margin": {
              "mae": 0.16180581554397933,
              "pairwise_concordance": 0.7131437472281416,
              "pearson": 0.5573686569664388,
              "rmse": 0.20866586176180993,
              "spearman": 0.5787920373556157
            },
            "n": 625,
            "outcome": {
              "brier_style_squared_error": 0.7513914169358009,
              "mae": 0.8129256479405731,
              "sign_accuracy": 0.6464
            }
          },
          "frozen_trunk_value_head": {
            "margin": {
              "mae": 0.16040432554398043,
              "pairwise_concordance": 0.71570737828161,
              "pearson": 0.5759290941073933,
              "rmse": 0.20883251490407373,
              "spearman": 0.5852725485377244
            },
            "n": 625,
            "outcome": {
              "brier_style_squared_error": 0.7825070321695782,
              "mae": 0.8304529620440211,
              "sign_accuracy": 0.6432
            }
          },
          "raw_feature_ridge": {
            "margin": {
              "mae": 0.19068867770398384,
              "pairwise_concordance": 0.6286899520806516,
              "pearson": 0.3715114149935337,
              "rmse": 0.23763566718818624,
              "spearman": 0.34770769230769233
            },
            "n": 625,
            "outcome": {
              "brier_style_squared_error": 0.8508199347041842,
              "mae": 0.8668746319128245,
              "sign_accuracy": 0.5568
            }
          }
        }
      }
    }
  },
  "validation": {
    "affine_current_value": {
      "calibration": {
        "margin_prediction_deciles": [
          {
            "bucket": 1,
            "mean_prediction": -0.27923088089962667,
            "mean_target": -0.24948559670781892,
            "n": 81
          },
          {
            "bucket": 2,
            "mean_prediction": -0.09779794687521239,
            "mean_target": -0.07552083333333334,
            "n": 80
          },
          {
            "bucket": 3,
            "mean_prediction": -0.022014738057052518,
            "mean_target": 0.016666666666666673,
            "n": 80
          },
          {
            "bucket": 4,
            "mean_prediction": 0.005032449155818741,
            "mean_target": -0.03703703703703704,
            "n": 81
          },
          {
            "bucket": 5,
            "mean_prediction": 0.023444921041817075,
            "mean_target": -0.028125000000000004,
            "n": 80
          },
          {
            "bucket": 6,
            "mean_prediction": 0.04345002057030463,
            "mean_target": 0.007291666666666663,
            "n": 80
          },
          {
            "bucket": 7,
            "mean_prediction": 0.06734860445754809,
            "mean_target": 0.06944444444444445,
            "n": 81
          },
          {
            "bucket": 8,
            "mean_prediction": 0.10263532816779977,
            "mean_target": 0.1171875,
            "n": 80
          },
          {
            "bucket": 9,
            "mean_prediction": 0.15282280041498814,
            "mean_target": 0.2015625,
            "n": 80
          },
          {
            "bucket": 10,
            "mean_prediction": 0.24996946622514343,
            "mean_target": 0.2618312757201646,
            "n": 81
          }
        ],
        "outcome_value_buckets": [
          {
            "bucket": 1,
            "mean_prediction": -0.27923088089962667,
            "mean_target": -0.8765432098765432,
            "n": 81
          },
          {
            "bucket": 2,
            "mean_prediction": -0.09779794687521239,
            "mean_target": -0.3875,
            "n": 80
          },
          {
            "bucket": 3,
            "mean_prediction": -0.022014738057052518,
            "mean_target": 0.075,
            "n": 80
          },
          {
            "bucket": 4,
            "mean_prediction": 0.005032449155818741,
            "mean_target": -0.12345679012345678,
            "n": 81
          },
          {
            "bucket": 5,
            "mean_prediction": 0.023444921041817075,
            "mean_target": -0.1375,
            "n": 80
          },
          {
            "bucket": 6,
            "mean_prediction": 0.04345002057030463,
            "mean_target": 0.0625,
            "n": 80
          },
          {
            "bucket": 7,
            "mean_prediction": 0.06734860445754809,
            "mean_target": 0.2716049382716049,
            "n": 81
          },
          {
            "bucket": 8,
            "mean_prediction": 0.10263532816779977,
            "mean_target": 0.4125,
            "n": 80
          },
          {
            "bucket": 9,
            "mean_prediction": 0.15282280041498814,
            "mean_target": 0.7,
            "n": 80
          },
          {
            "bucket": 10,
            "mean_prediction": 0.24996946622514343,
            "mean_target": 0.8641975308641975,
            "n": 81
          }
        ]
      },
      "margin": {
        "mae": 0.17074500502645176,
        "pairwise_concordance": 0.6894539037223438,
        "pearson": 0.516396374352768,
        "rmse": 0.22689469143804183,
        "spearman": 0.5292736598560073
      },
      "n": 804,
      "outcome": {
        "brier_style_squared_error": 0.7467757856493245,
        "mae": 0.8061461270585293,
        "sign_accuracy": 0.595771144278607
      }
    },
    "current_value": {
      "calibration": {
        "margin_prediction_deciles": [
          {
            "bucket": 1,
            "mean_prediction": -0.7224875798931828,
            "mean_target": -0.24948559670781892,
            "n": 81
          },
          {
            "bucket": 2,
            "mean_prediction": -0.2733783323317766,
            "mean_target": -0.07552083333333334,
            "n": 80
          },
          {
            "bucket": 3,
            "mean_prediction": -0.08578868620097638,
            "mean_target": 0.016666666666666673,
            "n": 80
          },
          {
            "bucket": 4,
            "mean_prediction": -0.018837545477040685,
            "mean_target": -0.03703703703703704,
            "n": 81
          },
          {
            "bucket": 5,
            "mean_prediction": 0.02673968910239637,
            "mean_target": -0.028125000000000004,
            "n": 80
          },
          {
            "bucket": 6,
            "mean_prediction": 0.07625922779552638,
            "mean_target": 0.007291666666666663,
            "n": 80
          },
          {
            "bucket": 7,
            "mean_prediction": 0.13541648656497768,
            "mean_target": 0.06944444444444445,
            "n": 81
          },
          {
            "bucket": 8,
            "mean_prediction": 0.22276332918554545,
            "mean_target": 0.1171875,
            "n": 80
          },
          {
            "bucket": 9,
            "mean_prediction": 0.3469946768134832,
            "mean_target": 0.2015625,
            "n": 80
          },
          {
            "bucket": 10,
            "mean_prediction": 0.5874662660522225,
            "mean_target": 0.2618312757201646,
            "n": 81
          }
        ],
        "outcome_value_buckets": [
          {
            "bucket": 1,
            "mean_prediction": -0.7224875798931828,
            "mean_target": -0.8765432098765432,
            "n": 81
          },
          {
            "bucket": 2,
            "mean_prediction": -0.2733783323317766,
            "mean_target": -0.3875,
            "n": 80
          },
          {
            "bucket": 3,
            "mean_prediction": -0.08578868620097638,
            "mean_target": 0.075,
            "n": 80
          },
          {
            "bucket": 4,
            "mean_prediction": -0.018837545477040685,
            "mean_target": -0.12345679012345678,
            "n": 81
          },
          {
            "bucket": 5,
            "mean_prediction": 0.02673968910239637,
            "mean_target": -0.1375,
            "n": 80
          },
          {
            "bucket": 6,
            "mean_prediction": 0.07625922779552638,
            "mean_target": 0.0625,
            "n": 80
          },
          {
            "bucket": 7,
            "mean_prediction": 0.13541648656497768,
            "mean_target": 0.2716049382716049,
            "n": 81
          },
          {
            "bucket": 8,
            "mean_prediction": 0.22276332918554545,
            "mean_target": 0.4125,
            "n": 80
          },
          {
            "bucket": 9,
            "mean_prediction": 0.3469946768134832,
            "mean_target": 0.7,
            "n": 80
          },
          {
            "bucket": 10,
            "mean_prediction": 0.5874662660522225,
            "mean_target": 0.8641975308641975,
            "n": 81
          }
        ]
      },
      "margin": {
        "mae": 0.24861367238827822,
        "pairwise_concordance": 0.6894539037223438,
        "pearson": 0.5163963743527683,
        "rmse": 0.30794878009582,
        "spearman": 0.5292736598560073
      },
      "n": 804,
      "outcome": {
        "brier_style_squared_error": 0.6489075041019908,
        "mae": 0.7217773367199842,
        "sign_accuracy": 0.6044776119402985
      }
    },
    "frozen_trunk_linear": {
      "calibration": {
        "margin_prediction_deciles": [
          {
            "bucket": 1,
            "mean_prediction": -0.264703891397815,
            "mean_target": -0.2330246913580247,
            "n": 81
          },
          {
            "bucket": 2,
            "mean_prediction": -0.13779882517663516,
            "mean_target": -0.12604166666666666,
            "n": 80
          },
          {
            "bucket": 3,
            "mean_prediction": -0.07330939585210243,
            "mean_target": -0.10364583333333335,
            "n": 80
          },
          {
            "bucket": 4,
            "mean_prediction": -0.02209484433969068,
            "mean_target": 0.0015432098765432154,
            "n": 81
          },
          {
            "bucket": 5,
            "mean_prediction": 0.0137639545695874,
            "mean_target": 0.06979166666666667,
            "n": 80
          },
          {
            "bucket": 6,
            "mean_prediction": 0.04838159866347974,
            "mean_target": 0.06354166666666666,
            "n": 80
          },
          {
            "bucket": 7,
            "mean_prediction": 0.08198524211023073,
            "mean_target": 0.03600823045267489,
            "n": 81
          },
          {
            "bucket": 8,
            "mean_prediction": 0.11715738240783964,
            "mean_target": 0.134375,
            "n": 80
          },
          {
            "bucket": 9,
            "mean_prediction": 0.16457660489024648,
            "mean_target": 0.15104166666666669,
            "n": 80
          },
          {
            "bucket": 10,
            "mean_prediction": 0.26318864854548113,
            "mean_target": 0.2896090534979424,
            "n": 81
          }
        ],
        "outcome_value_buckets": [
          {
            "bucket": 1,
            "mean_prediction": -0.264703891397815,
            "mean_target": -0.8395061728395061,
            "n": 81
          },
          {
            "bucket": 2,
            "mean_prediction": -0.13779882517663516,
            "mean_target": -0.4625,
            "n": 80
          },
          {
            "bucket": 3,
            "mean_prediction": -0.07330939585210243,
            "mean_target": -0.375,
            "n": 80
          },
          {
            "bucket": 4,
            "mean_prediction": -0.02209484433969068,
            "mean_target": -0.037037037037037035,
            "n": 81
          },
          {
            "bucket": 5,
            "mean_prediction": 0.0137639545695874,
            "mean_target": 0.1625,
            "n": 80
          },
          {
            "bucket": 6,
            "mean_prediction": 0.04838159866347974,
            "mean_target": 0.2375,
            "n": 80
          },
          {
            "bucket": 7,
            "mean_prediction": 0.08198524211023073,
            "mean_target": 0.2345679012345679,
            "n": 81
          },
          {
            "bucket": 8,
            "mean_prediction": 0.11715738240783964,
            "mean_target": 0.5,
            "n": 80
          },
          {
            "bucket": 9,
            "mean_prediction": 0.16457660489024648,
            "mean_target": 0.6125,
            "n": 80
          },
          {
            "bucket": 10,
            "mean_prediction": 0.26318864854548113,
            "mean_target": 0.8271604938271605,
            "n": 81
          }
        ]
      },
      "margin": {
        "mae": 0.17109991915350645,
        "pairwise_concordance": 0.7030883362201379,
        "pearson": 0.542390158381023,
        "rmse": 0.22280090425534216,
        "spearman": 0.5479803014582957
      },
      "n": 804,
      "outcome": {
        "brier_style_squared_error": 0.7386123899555448,
        "mae": 0.802358772196875,
        "sign_accuracy": 0.6492537313432836
      }
    },
    "frozen_trunk_value_head": {
      "calibration": {
        "margin_prediction_deciles": [
          {
            "bucket": 1,
            "mean_prediction": -0.1785896271467209,
            "mean_target": -0.22633744855967078,
            "n": 81
          },
          {
            "bucket": 2,
            "mean_prediction": -0.07720345258712769,
            "mean_target": -0.1140625,
            "n": 80
          },
          {
            "bucket": 3,
            "mean_prediction": -0.037232983857393265,
            "mean_target": -0.017187500000000005,
            "n": 80
          },
          {
            "bucket": 4,
            "mean_prediction": -0.003938885405659676,
            "mean_target": -0.023148148148148147,
            "n": 81
          },
          {
            "bucket": 5,
            "mean_prediction": 0.02634691633284092,
            "mean_target": 0.07395833333333332,
            "n": 80
          },
          {
            "bucket": 6,
            "mean_prediction": 0.04795852303504944,
            "mean_target": 0.042708333333333334,
            "n": 80
          },
          {
            "bucket": 7,
            "mean_prediction": 0.06767430901527405,
            "mean_target": 0.03189300411522633,
            "n": 81
          },
          {
            "bucket": 8,
            "mean_prediction": 0.0870356559753418,
            "mean_target": 0.0765625,
            "n": 80
          },
          {
            "bucket": 9,
            "mean_prediction": 0.11383261531591415,
            "mean_target": 0.1984375,
            "n": 80
          },
          {
            "bucket": 10,
            "mean_prediction": 0.1753552109003067,
            "mean_target": 0.24125514403292178,
            "n": 81
          }
        ],
        "outcome_value_buckets": [
          {
            "bucket": 1,
            "mean_prediction": -0.1785896271467209,
            "mean_target": -0.7530864197530864,
            "n": 81
          },
          {
            "bucket": 2,
            "mean_prediction": -0.07720345258712769,
            "mean_target": -0.5,
            "n": 80
          },
          {
            "bucket": 3,
            "mean_prediction": -0.037232983857393265,
            "mean_target": 0.0,
            "n": 80
          },
          {
            "bucket": 4,
            "mean_prediction": -0.003938885405659676,
            "mean_target": -0.1728395061728395,
            "n": 81
          },
          {
            "bucket": 5,
            "mean_prediction": 0.02634691633284092,
            "mean_target": 0.2875,
            "n": 80
          },
          {
            "bucket": 6,
            "mean_prediction": 0.04795852303504944,
            "mean_target": 0.125,
            "n": 80
          },
          {
            "bucket": 7,
            "mean_prediction": 0.06767430901527405,
            "mean_target": 0.14814814814814814,
            "n": 81
          },
          {
            "bucket": 8,
            "mean_prediction": 0.0870356559753418,
            "mean_target": 0.3,
            "n": 80
          },
          {
            "bucket": 9,
            "mean_prediction": 0.11383261531591415,
            "mean_target": 0.6625,
            "n": 80
          },
          {
            "bucket": 10,
            "mean_prediction": 0.1753552109003067,
            "mean_target": 0.7654320987654321,
            "n": 81
          }
        ]
      },
      "margin": {
        "mae": 0.17930510209726208,
        "pairwise_concordance": 0.6748030981404621,
        "pearson": 0.4892162255938182,
        "rmse": 0.23305313263126803,
        "spearman": 0.47799764972389047
      },
      "n": 804,
      "outcome": {
        "brier_style_squared_error": 0.7837527647544628,
        "mae": 0.8282763421288182,
        "sign_accuracy": 0.6169154228855721
      }
    },
    "raw_feature_ridge": {
      "calibration": {
        "margin_prediction_deciles": [
          {
            "bucket": 1,
            "mean_prediction": -0.049204320054173946,
            "mean_target": 0.004629629629629627,
            "n": 81
          },
          {
            "bucket": 2,
            "mean_prediction": -0.028210786517394993,
            "mean_target": -0.05937499999999999,
            "n": 80
          },
          {
            "bucket": 3,
            "mean_prediction": -0.013865378192835532,
            "mean_target": -0.02395833333333333,
            "n": 80
          },
          {
            "bucket": 4,
            "mean_prediction": 0.0012335101915513927,
            "mean_target": 0.005144032921810703,
            "n": 81
          },
          {
            "bucket": 5,
            "mean_prediction": 0.01799139133890676,
            "mean_target": -0.03125,
            "n": 80
          },
          {
            "bucket": 6,
            "mean_prediction": 0.03757993736279238,
            "mean_target": -0.029687500000000006,
            "n": 80
          },
          {
            "bucket": 7,
            "mean_prediction": 0.05070768274697186,
            "mean_target": 0.07150205761316873,
            "n": 81
          },
          {
            "bucket": 8,
            "mean_prediction": 0.06282941348075556,
            "mean_target": 0.146875,
            "n": 80
          },
          {
            "bucket": 9,
            "mean_prediction": 0.07523264423372564,
            "mean_target": 0.09166666666666666,
            "n": 80
          },
          {
            "bucket": 10,
            "mean_prediction": 0.09750461915398038,
            "mean_target": 0.10648148148148148,
            "n": 81
          }
        ],
        "outcome_value_buckets": [
          {
            "bucket": 1,
            "mean_prediction": -0.049204320054173946,
            "mean_target": -0.037037037037037035,
            "n": 81
          },
          {
            "bucket": 2,
            "mean_prediction": -0.028210786517394993,
            "mean_target": -0.2625,
            "n": 80
          },
          {
            "bucket": 3,
            "mean_prediction": -0.013865378192835532,
            "mean_target": -0.1375,
            "n": 80
          },
          {
            "bucket": 4,
            "mean_prediction": 0.0012335101915513927,
            "mean_target": 0.0,
            "n": 81
          },
          {
            "bucket": 5,
            "mean_prediction": 0.01799139133890676,
            "mean_target": -0.125,
            "n": 80
          },
          {
            "bucket": 6,
            "mean_prediction": 0.03757993736279238,
            "mean_target": -0.0125,
            "n": 80
          },
          {
            "bucket": 7,
            "mean_prediction": 0.05070768274697186,
            "mean_target": 0.2839506172839506,
            "n": 81
          },
          {
            "bucket": 8,
            "mean_prediction": 0.06282941348075556,
            "mean_target": 0.4375,
            "n": 80
          },
          {
            "bucket": 9,
            "mean_prediction": 0.07523264423372564,
            "mean_target": 0.3625,
            "n": 80
          },
          {
            "bucket": 10,
            "mean_prediction": 0.09750461915398038,
            "mean_target": 0.345679012345679,
            "n": 81
          }
        ]
      },
      "margin": {
        "mae": 0.20208203802185715,
        "pairwise_concordance": 0.5693715480898068,
        "pearson": 0.19247841774687197,
        "rmse": 0.26002003321914235,
        "spearman": 0.18683450548899955
      },
      "n": 804,
      "outcome": {
        "brier_style_squared_error": 0.8450422764486188,
        "mae": 0.8588166349372836,
        "sign_accuracy": 0.5223880597014925
      }
    },
    "slices": {
      "current_value_quartile": {
        "1": {
          "affine_current_value": {
            "margin": {
              "mae": 0.1432979135402473,
              "pairwise_concordance": 0.7065668674346529,
              "pearson": 0.5123366269606376,
              "rmse": 0.19204818800697804,
              "spearman": 0.5307269484306758
            },
            "n": 192,
            "outcome": {
              "brier_style_squared_error": 0.6820584739935412,
              "mae": 0.780707751949396,
              "sign_accuracy": 0.703125
            }
          },
          "current_value": {
            "margin": {
              "mae": 0.3343820874579251,
              "pairwise_concordance": 0.7065668674346529,
              "pearson": 0.5123366269606383,
              "rmse": 0.39111454910326615,
              "spearman": 0.5307269484306758
            },
            "n": 192,
            "outcome": {
              "brier_style_squared_error": 0.5109190255972723,
              "mae": 0.60582837377054,
              "sign_accuracy": 0.703125
            }
          },
          "frozen_trunk_linear": {
            "margin": {
              "mae": 0.15222145906467535,
              "pairwise_concordance": 0.6730600549675457,
              "pearson": 0.4502066884936353,
              "rmse": 0.20134524823398808,
              "spearman": 0.45724371320836615
            },
            "n": 192,
            "outcome": {
              "brier_style_squared_error": 0.6973695027328137,
              "mae": 0.7897449918191745,
              "sign_accuracy": 0.7291666666666666
            }
          },
          "frozen_trunk_value_head": {
            "margin": {
              "mae": 0.1551347198548885,
              "pairwise_concordance": 0.6535875095023683,
              "pearson": 0.4348904244071544,
              "rmse": 0.20622949626666495,
              "spearman": 0.4075309253180696
            },
            "n": 192,
            "outcome": {
              "brier_style_squared_error": 0.7546113524807484,
              "mae": 0.8240103457810619,
              "sign_accuracy": 0.7447916666666666
            }
          },
          "raw_feature_ridge": {
            "margin": {
              "mae": 0.20009633006574382,
              "pairwise_concordance": 0.6185603181100521,
              "pearson": 0.3288101104221456,
              "rmse": 0.25698991347851585,
              "spearman": 0.30171513441662373
            },
            "n": 192,
            "outcome": {
              "brier_style_squared_error": 0.873197867804366,
              "mae": 0.8860852738861832,
              "sign_accuracy": 0.5625
            }
          }
        },
        "2": {
          "affine_current_value": {
            "margin": {
              "mae": 0.19559867044948628,
              "pairwise_concordance": 0.4874586549062844,
              "pearson": -0.045901989012093074,
              "rmse": 0.2560176406964737,
              "spearman": -0.03150668619535836
            },
            "n": 216,
            "outcome": {
              "brier_style_squared_error": 0.8180319842630424,
              "mae": 0.8188625593383753,
              "sign_accuracy": 0.35648148148148145
            }
          },
          "current_value": {
            "margin": {
              "mae": 0.1993217664707634,
              "pairwise_concordance": 0.4874586549062844,
              "pearson": -0.04590198901209304,
              "rmse": 0.25841309408959573,
              "spearman": -0.03150668619535836
            },
            "n": 216,
            "outcome": {
              "brier_style_squared_error": 0.8184732930095762,
              "mae": 0.822159971875711,
              "sign_accuracy": 0.3888888888888889
            }
          },
          "frozen_trunk_linear": {
            "margin": {
              "mae": 0.18797189746117718,
              "pairwise_concordance": 0.6150771775082691,
              "pearson": 0.2918429494302168,
              "rmse": 0.24551356077319691,
              "spearman": 0.31983114826325626
            },
            "n": 216,
            "outcome": {
              "brier_style_squared_error": 0.7661131063379742,
              "mae": 0.7995378333725471,
              "sign_accuracy": 0.5462962962962963
            }
          },
          "frozen_trunk_value_head": {
            "margin": {
              "mae": 0.1929618556784174,
              "pairwise_concordance": 0.5748346196251378,
              "pearson": 0.22779696313627126,
              "rmse": 0.2507403870792846,
              "spearman": 0.20026435180223623
            },
            "n": 216,
            "outcome": {
              "brier_style_squared_error": 0.7952274730759493,
              "mae": 0.8122445659178926,
              "sign_accuracy": 0.4675925925925926
            }
          },
          "raw_feature_ridge": {
            "margin": {
              "mae": 0.199667463149369,
              "pairwise_concordance": 0.5369808893789049,
              "pearson": 0.11519969715063208,
              "rmse": 0.25750828294320866,
              "spearman": 0.09665154383834054
            },
            "n": 216,
            "outcome": {
              "brier_style_squared_error": 0.8150548770722035,
              "mae": 0.821195205850089,
              "sign_accuracy": 0.41203703703703703
            }
          }
        },
        "3": {
          "affine_current_value": {
            "margin": {
              "mae": 0.19168328512738683,
              "pairwise_concordance": 0.5638270657457151,
              "pearson": 0.1832852371384407,
              "rmse": 0.2533272122930731,
              "spearman": 0.17974064699904752
            },
            "n": 182,
            "outcome": {
              "brier_style_squared_error": 0.8619067459087318,
              "mae": 0.8802548668991506,
              "sign_accuracy": 0.554945054945055
            }
          },
          "current_value": {
            "margin": {
              "mae": 0.20063372566313534,
              "pairwise_concordance": 0.5638270657457151,
              "pearson": 0.18328523713844075,
              "rmse": 0.26079197047443414,
              "spearman": 0.17974064699904752
            },
            "n": 182,
            "outcome": {
              "brier_style_squared_error": 0.8403452672144366,
              "mae": 0.8691542667265122,
              "sign_accuracy": 0.554945054945055
            }
          },
          "frozen_trunk_linear": {
            "margin": {
              "mae": 0.19086635163830395,
              "pairwise_concordance": 0.5918393451010489,
              "pearson": 0.30813074281733854,
              "rmse": 0.2444713051706137,
              "spearman": 0.2485252271503517
            },
            "n": 182,
            "outcome": {
              "brier_style_squared_error": 0.8422274031310034,
              "mae": 0.8685095289243737,
              "sign_accuracy": 0.5824175824175825
            }
          },
          "frozen_trunk_value_head": {
            "margin": {
              "mae": 0.1935019945729423,
              "pairwise_concordance": 0.5612049117421335,
              "pearson": 0.2387811668817414,
              "rmse": 0.24860210795323037,
              "spearman": 0.16796635119250825
            },
            "n": 182,
            "outcome": {
              "brier_style_squared_error": 0.860720957878497,
              "mae": 0.8793064997743196,
              "sign_accuracy": 0.5274725274725275
            }
          },
          "raw_feature_ridge": {
            "margin": {
              "mae": 0.19265427356052678,
              "pairwise_concordance": 0.5492453312867741,
              "pearson": 0.13627654588831511,
              "rmse": 0.2544884845970439,
              "spearman": 0.1332107798007626
            },
            "n": 182,
            "outcome": {
              "brier_style_squared_error": 0.8696136205400264,
              "mae": 0.8828034643453454,
              "sign_accuracy": 0.5494505494505495
            }
          }
        },
        "4": {
          "affine_current_value": {
            "margin": {
              "mae": 0.15247716790311355,
              "pairwise_concordance": 0.6075925489271399,
              "pearson": 0.29611914805267403,
              "rmse": 0.19896844073042116,
              "spearman": 0.32638036434210566
            },
            "n": 214,
            "outcome": {
              "brier_style_squared_error": 0.6350026556032271,
              "mae": 0.7531070055515845,
              "sign_accuracy": 0.7757009345794392
            }
          },
          "current_value": {
            "margin": {
              "mae": 0.26222052420503994,
              "pairwise_concordance": 0.6075925489271399,
              "pearson": 0.2961191480526738,
              "rmse": 0.30626456433790844,
              "spearman": 0.32638036434210566
            },
            "n": 214,
            "outcome": {
              "brier_style_squared_error": 0.4387481797206934,
              "mae": 0.5991462639698358,
              "sign_accuracy": 0.7757009345794392
            }
          },
          "frozen_trunk_linear": {
            "margin": {
              "mae": 0.15419723836082194,
              "pairwise_concordance": 0.6682386229662816,
              "pearson": 0.4587065937911296,
              "rmse": 0.19610492956388986,
              "spearman": 0.45902635985441004
            },
            "n": 214,
            "outcome": {
              "brier_style_squared_error": 0.6597364423397789,
              "mae": 0.7602640567490646,
              "sign_accuracy": 0.7383177570093458
            }
          },
          "frozen_trunk_value_head": {
            "margin": {
              "mae": 0.17513229923012355,
              "pairwise_concordance": 0.6155152086771988,
              "pearson": 0.33639787902325136,
              "rmse": 0.2233366480404643,
              "spearman": 0.3271848434663621
            },
            "n": 214,
            "outcome": {
              "brier_style_squared_error": 0.7328574517195925,
              "mae": 0.8048859041327802,
              "sign_accuracy": 0.7289719626168224
            }
          },
          "raw_feature_ridge": {
            "margin": {
              "mae": 0.21431875405920917,
              "pairwise_concordance": 0.5684036783777411,
              "pearson": 0.19611032171997028,
              "rmse": 0.26971924690548577,
              "spearman": 0.17281913607268476
            },
            "n": 214,
            "outcome": {
              "brier_style_squared_error": 0.8291517161699553,
              "mae": 0.8519243314437228,
              "sign_accuracy": 0.5747663551401869
            }
          }
        }
      },
      "phase": {
        "late": {
          "affine_current_value": {
            "margin": {
              "mae": 0.10223366133455027,
              "pairwise_concordance": 0.8605341246290801,
              "pearson": 0.8589272396870444,
              "rmse": 0.12919517696177624,
              "spearman": 0.8716371391076113
            },
            "n": 127,
            "outcome": {
              "brier_style_squared_error": 0.5386683739873045,
              "mae": 0.6884506432479647,
              "sign_accuracy": 0.7952755905511811
            }
          },
          "current_value": {
            "margin": {
              "mae": 0.35918909977642405,
              "pairwise_concordance": 0.8605341246290801,
              "pearson": 0.8589272396870448,
              "rmse": 0.40462677573992245,
              "spearman": 0.8716371391076113
            },
            "n": 127,
            "outcome": {
              "brier_style_squared_error": 0.2732563847632053,
              "mae": 0.42869816686752743,
              "sign_accuracy": 0.7952755905511811
            }
          },
          "frozen_trunk_linear": {
            "margin": {
              "mae": 0.10640080326584535,
              "pairwise_concordance": 0.847450768815754,
              "pearson": 0.8353236060067071,
              "rmse": 0.136391377501252,
              "spearman": 0.8459880014998125
            },
            "n": 127,
            "outcome": {
              "brier_style_squared_error": 0.6199503476030302,
              "mae": 0.7361040292298237,
              "sign_accuracy": 0.7874015748031497
            }
          },
          "frozen_trunk_value_head": {
            "margin": {
              "mae": 0.1293022379974573,
              "pairwise_concordance": 0.8044240625843,
              "pearson": 0.7624540270380509,
              "rmse": 0.16717463068167548,
              "spearman": 0.7863704536932883
            },
            "n": 127,
            "outcome": {
              "brier_style_squared_error": 0.6883028900734256,
              "mae": 0.7796693164250249,
              "sign_accuracy": 0.7952755905511811
            }
          },
          "raw_feature_ridge": {
            "margin": {
              "mae": 0.1939852419011189,
              "pairwise_concordance": 0.6093876449959535,
              "pearson": 0.3082765499211138,
              "rmse": 0.23853584923187898,
              "spearman": 0.30310273715785524
            },
            "n": 127,
            "outcome": {
              "brier_style_squared_error": 0.8395768141114022,
              "mae": 0.8572650787526029,
              "sign_accuracy": 0.5905511811023622
            }
          }
        },
        "midgame": {
          "affine_current_value": {
            "margin": {
              "mae": 0.12215226176762936,
              "pairwise_concordance": 0.8068325601012231,
              "pearson": 0.750452664935225,
              "rmse": 0.1563915152516769,
              "spearman": 0.7859076526426814
            },
            "n": 160,
            "outcome": {
              "brier_style_squared_error": 0.6266213881501639,
              "mae": 0.7290292367346226,
              "sign_accuracy": 0.7
            }
          },
          "current_value": {
            "margin": {
              "mae": 0.23826123350493916,
              "pairwise_concordance": 0.8068325601012231,
              "pearson": 0.7504526649352251,
              "rmse": 0.3006538838500516,
              "spearman": 0.7859076526426814
            },
            "n": 160,
            "outcome": {
              "brier_style_squared_error": 0.45397212409354754,
              "mae": 0.5941456514585297,
              "sign_accuracy": 0.725
            }
          },
          "frozen_trunk_linear": {
            "margin": {
              "mae": 0.1319865137234474,
              "pairwise_concordance": 0.786756642766765,
              "pearson": 0.7019732198859533,
              "rmse": 0.16861735690575902,
              "spearman": 0.7476268604242353
            },
            "n": 160,
            "outcome": {
              "brier_style_squared_error": 0.6338180781832353,
              "mae": 0.7284857293543117,
              "sign_accuracy": 0.7
            }
          },
          "frozen_trunk_value_head": {
            "margin": {
              "mae": 0.14789354384738546,
              "pairwise_concordance": 0.7434837621256853,
              "pearson": 0.6032492957694585,
              "rmse": 0.1928530395647566,
              "spearman": 0.6561701628969882
            },
            "n": 160,
            "outcome": {
              "brier_style_squared_error": 0.7219270144314548,
              "mae": 0.777810759170461,
              "sign_accuracy": 0.64375
            }
          },
          "raw_feature_ridge": {
            "margin": {
              "mae": 0.18149047910504235,
              "pairwise_concordance": 0.5250948966680725,
              "pearson": 0.04209053232752264,
              "rmse": 0.23812141507405252,
              "spearman": 0.07633598968709715
            },
            "n": 160,
            "outcome": {
              "brier_style_squared_error": 0.8126950796346062,
              "mae": 0.8218044788695416,
              "sign_accuracy": 0.45625
            }
          }
        },
        "opening": {
          "affine_current_value": {
            "margin": {
              "mae": 0.2026130506169413,
              "pairwise_concordance": 0.6045541483994364,
              "pearson": 0.3349983801714387,
              "rmse": 0.2615152603552498,
              "spearman": 0.2988934718276628
            },
            "n": 517,
            "outcome": {
              "brier_style_squared_error": 0.8350820620147833,
              "mae": 0.8589237458124303,
              "sign_accuracy": 0.5145067698259188
            }
          },
          "current_value": {
            "margin": {
              "mae": 0.22465489278100495,
              "pairwise_concordance": 0.6045541483994364,
              "pearson": 0.33499838017143846,
              "rmse": 0.28157353641572047,
              "spearman": 0.2988934718276628
            },
            "n": 517,
            "outcome": {
              "brier_style_squared_error": 0.801513602665582,
              "mae": 0.833270807151502,
              "sign_accuracy": 0.5203094777562862
            }
          },
          "frozen_trunk_linear": {
            "margin": {
              "mae": 0.19909785452399464,
              "pairwise_concordance": 0.6420278790073122,
              "pearson": 0.4135220898802592,
              "rmse": 0.25264226141815394,
              "spearman": 0.3979978391160974
            },
            "n": 517,
            "outcome": {
              "brier_style_squared_error": 0.8001929881418868,
              "mae": 0.8414961787957641,
              "sign_accuracy": 0.5996131528046421
            }
          },
          "frozen_trunk_value_head": {
            "margin": {
              "mae": 0.20130938267880066,
              "pairwise_concordance": 0.6188242331698793,
              "pearson": 0.3802388892863022,
              "rmse": 0.25707845108968425,
              "spearman": 0.33622915859432434
            },
            "n": 517,
            "outcome": {
              "brier_style_squared_error": 0.8263335271068284,
              "mae": 0.8558345346582551,
              "sign_accuracy": 0.5647969052224371
            }
          },
          "raw_feature_ridge": {
            "margin": {
              "mae": 0.21044362899675875,
              "pairwise_concordance": 0.5677022251257389,
              "pearson": 0.20907853448813501,
              "rmse": 0.27132552645909924,
              "spearman": 0.19893027389316353
            },
            "n": 517,
            "outcome": {
              "brier_style_squared_error": 0.8563955940638383,
              "mae": 0.8706522105780826,
              "sign_accuracy": 0.5261121856866537
            }
          }
        }
      },
      "player": {
        "0": {
          "affine_current_value": {
            "margin": {
              "mae": 0.15997395220635927,
              "pairwise_concordance": 0.6951858976048062,
              "pearson": 0.5470340953353934,
              "rmse": 0.2124390737179071,
              "spearman": 0.5471939554099826
            },
            "n": 401,
            "outcome": {
              "brier_style_squared_error": 0.7486309496970082,
              "mae": 0.8123894016277973,
              "sign_accuracy": 0.6134663341645885
            }
          },
          "current_value": {
            "margin": {
              "mae": 0.238050618503612,
              "pairwise_concordance": 0.6951858976048062,
              "pearson": 0.5470340953353934,
              "rmse": 0.28981697186535993,
              "spearman": 0.5471939554099826
            },
            "n": 401,
            "outcome": {
              "brier_style_squared_error": 0.643951382020257,
              "mae": 0.7240036853082232,
              "sign_accuracy": 0.6259351620947631
            }
          },
          "frozen_trunk_linear": {
            "margin": {
              "mae": 0.16113381124043805,
              "pairwise_concordance": 0.7135253353007825,
              "pearson": 0.5633235443180498,
              "rmse": 0.20794151281782722,
              "spearman": 0.5705171772062382
            },
            "n": 401,
            "outcome": {
              "brier_style_squared_error": 0.7303149305488067,
              "mae": 0.8036938654936238,
              "sign_accuracy": 0.7032418952618454
            }
          },
          "frozen_trunk_value_head": {
            "margin": {
              "mae": 0.1697960926630807,
              "pairwise_concordance": 0.6883217833522174,
              "pearson": 0.5124569736614883,
              "rmse": 0.22142945751095733,
              "spearman": 0.5164307514795102
            },
            "n": 401,
            "outcome": {
              "brier_style_squared_error": 0.7854076304250269,
              "mae": 0.8351504828589573,
              "sign_accuracy": 0.6683291770573566
            }
          },
          "raw_feature_ridge": {
            "margin": {
              "mae": 0.19341831557513275,
              "pairwise_concordance": 0.5583515585887064,
              "pearson": 0.17794743987917666,
              "rmse": 0.2480391886393542,
              "spearman": 0.13698459076190125
            },
            "n": 401,
            "outcome": {
              "brier_style_squared_error": 0.8418449895942037,
              "mae": 0.8650294382973907,
              "sign_accuracy": 0.57356608478803
            }
          }
        },
        "1": {
          "affine_current_value": {
            "margin": {
              "mae": 0.18146260349011703,
              "pairwise_concordance": 0.6780002617458448,
              "pearson": 0.4645298886125894,
              "rmse": 0.24041739567315307,
              "spearman": 0.4925563813001425
            },
            "n": 403,
            "outcome": {
              "brier_style_squared_error": 0.7449298283711081,
              "mae": 0.7999338364821609,
              "sign_accuracy": 0.5781637717121588
            }
          },
          "current_value": {
            "margin": {
              "mae": 0.25912430416929844,
              "pairwise_concordance": 0.6780002617458448,
              "pearson": 0.4645298886125893,
              "rmse": 0.3249880546954947,
              "spearman": 0.4925563813001425
            },
            "n": 403,
            "outcome": {
              "brier_style_squared_error": 0.6538390300443611,
              "mae": 0.7195620370081137,
              "sign_accuracy": 0.5831265508684863
            }
          },
          "frozen_trunk_linear": {
            "margin": {
              "mae": 0.18101656747395414,
              "pairwise_concordance": 0.6773197225494045,
              "pearson": 0.4807278036899361,
              "rmse": 0.23666230900743734,
              "spearman": 0.47715064458558737
            },
            "n": 403,
            "outcome": {
              "brier_style_squared_error": 0.7468686709036885,
              "mae": 0.801030304673311,
              "sign_accuracy": 0.5955334987593052
            }
          },
          "frozen_trunk_value_head": {
            "margin": {
              "mae": 0.18876692041762616,
              "pairwise_concordance": 0.6530166208611439,
              "pearson": 0.4290533020281423,
              "rmse": 0.24407028381949428,
              "spearman": 0.4059125422930797
            },
            "n": 403,
            "outcome": {
              "brier_style_squared_error": 0.7821061118167549,
              "mae": 0.8214363162410122,
              "sign_accuracy": 0.56575682382134
            }
          },
          "raw_feature_ridge": {
            "margin": {
              "mae": 0.2107027643274067,
              "pairwise_concordance": 0.49896610391310037,
              "pearson": -0.01992079129555571,
              "rmse": 0.2714170000558265,
              "spearman": -0.03288602072310486
            },
            "n": 403,
            "outcome": {
              "brier_style_squared_error": 0.8482236958744759,
              "mae": 0.8526346643481942,
              "sign_accuracy": 0.47146401985111663
            }
          }
        }
      },
      "source_domain": {
        "additional_selfplay": {
          "affine_current_value": {
            "margin": {
              "mae": 0.14127367638275493,
              "pairwise_concordance": 0.7509529860228716,
              "pearson": 0.5958928307918729,
              "rmse": 0.20822295098231758,
              "spearman": 0.6919860627177703
            },
            "n": 41,
            "outcome": {
              "brier_style_squared_error": 0.7225435235659402,
              "mae": 0.8263520188900713,
              "sign_accuracy": 0.8536585365853658
            }
          },
          "current_value": {
            "margin": {
              "mae": 0.2502041514773618,
              "pairwise_concordance": 0.7509529860228716,
              "pearson": 0.5958928307918729,
              "rmse": 0.30326348429196354,
              "spearman": 0.6919860627177703
            },
            "n": 41,
            "outcome": {
              "brier_style_squared_error": 0.49450520564293055,
              "mae": 0.6515752141273058,
              "sign_accuracy": 0.8780487804878049
            }
          },
          "frozen_trunk_linear": {
            "margin": {
              "mae": 0.16730379095529532,
              "pairwise_concordance": 0.7191867852604829,
              "pearson": 0.5724642471335362,
              "rmse": 0.21764942653056238,
              "spearman": 0.6080139372822301
            },
            "n": 41,
            "outcome": {
              "brier_style_squared_error": 0.7267802165795967,
              "mae": 0.8278685877371151,
              "sign_accuracy": 0.8536585365853658
            }
          },
          "frozen_trunk_value_head": {
            "margin": {
              "mae": 0.1838930430154816,
              "pairwise_concordance": 0.7141041931385006,
              "pearson": 0.5165913588257585,
              "rmse": 0.22997594765077806,
              "spearman": 0.5656794425087109
            },
            "n": 41,
            "outcome": {
              "brier_style_squared_error": 0.8296435247507046,
              "mae": 0.8901346208791748,
              "sign_accuracy": 0.7804878048780488
            }
          },
          "raw_feature_ridge": {
            "margin": {
              "mae": 0.2256175337327448,
              "pairwise_concordance": 0.5527318932655655,
              "pearson": 0.15071917072879565,
              "rmse": 0.26339302030881917,
              "spearman": 0.14947735191637634
            },
            "n": 41,
            "outcome": {
              "brier_style_squared_error": 0.9339250495699153,
              "mae": 0.9454038988610413,
              "sign_accuracy": 0.5365853658536586
            }
          }
        },
        "generated_opening_family_diagnostic": {
          "affine_current_value": {
            "margin": {
              "mae": 0.20484317715634737,
              "pairwise_concordance": 0.5299276287782035,
              "pearson": 0.10788394543658815,
              "rmse": 0.2541534805874538,
              "spearman": 0.08799605694909479
            },
            "n": 158,
            "outcome": {
              "brier_style_squared_error": 0.842119799222019,
              "mae": 0.8545048657968075,
              "sign_accuracy": 0.4177215189873418
            }
          },
          "current_value": {
            "margin": {
              "mae": 0.22246948131438707,
              "pairwise_concordance": 0.5299276287782035,
              "pearson": 0.10788394543658808,
              "rmse": 0.27427062733670143,
              "spearman": 0.08799605694909479
            },
            "n": 158,
            "outcome": {
              "brier_style_squared_error": 0.8423760268481956,
              "mae": 0.8573033507718835,
              "sign_accuracy": 0.44936708860759494
            }
          },
          "frozen_trunk_linear": {
            "margin": {
              "mae": 0.2136794941267432,
              "pairwise_concordance": 0.5340996168582376,
              "pearson": 0.10768774181409091,
              "rmse": 0.26508062445210734,
              "spearman": 0.10658255230399219
            },
            "n": 158,
            "outcome": {
              "brier_style_squared_error": 0.84943809224572,
              "mae": 0.8592007629916397,
              "sign_accuracy": 0.5189873417721519
            }
          },
          "frozen_trunk_value_head": {
            "margin": {
              "mae": 0.20568435570730567,
              "pairwise_concordance": 0.5414218816517667,
              "pearson": 0.1260194078211449,
              "rmse": 0.2555955564020636,
              "spearman": 0.11833260060332329
            },
            "n": 158,
            "outcome": {
              "brier_style_squared_error": 0.8415795078339287,
              "mae": 0.8545419830070732,
              "sign_accuracy": 0.4936708860759494
            }
          },
          "raw_feature_ridge": {
            "margin": {
              "mae": 0.20223083877897183,
              "pairwise_concordance": 0.5217539378458919,
              "pearson": 0.051413859177869774,
              "rmse": 0.25553462471287636,
              "spearman": 0.06002351835146397
            },
            "n": 158,
            "outcome": {
              "brier_style_squared_error": 0.8438150969617778,
              "mae": 0.8519296843185945,
              "sign_accuracy": 0.46835443037974683
            }
          }
        },
        "opening_family_diagnostic": {
          "affine_current_value": {
            "margin": {
              "mae": 0.20916563914459596,
              "pairwise_concordance": 0.5396301188903567,
              "pearson": 0.08046850590951526,
              "rmse": 0.2763132269852093,
              "spearman": 0.1306066891366347
            },
            "n": 57,
            "outcome": {
              "brier_style_squared_error": 0.9184064923123388,
              "mae": 0.924473077354762,
              "sign_accuracy": 0.47368421052631576
            }
          },
          "current_value": {
            "margin": {
              "mae": 0.21774365355625575,
              "pairwise_concordance": 0.5396301188903567,
              "pearson": 0.08046850590951526,
              "rmse": 0.28756801127702314,
              "spearman": 0.1306066891366347
            },
            "n": 57,
            "outcome": {
              "brier_style_squared_error": 0.9270678285619939,
              "mae": 0.925911264322502,
              "sign_accuracy": 0.45614035087719296
            }
          },
          "frozen_trunk_linear": {
            "margin": {
              "mae": 0.19783037397447634,
              "pairwise_concordance": 0.5977542932628798,
              "pearson": 0.31284253945892826,
              "rmse": 0.2566522826967993,
              "spearman": 0.2950479647394348
            },
            "n": 57,
            "outcome": {
              "brier_style_squared_error": 0.8861982219429383,
              "mae": 0.904961788483587,
              "sign_accuracy": 0.5614035087719298
            }
          },
          "frozen_trunk_value_head": {
            "margin": {
              "mae": 0.21080048205821136,
              "pairwise_concordance": 0.4986789960369881,
              "pearson": -0.003675262594228721,
              "rmse": 0.27631437219464156,
              "spearman": -0.0070650764843142345
            },
            "n": 57,
            "outcome": {
              "brier_style_squared_error": 0.9090422298197224,
              "mae": 0.9197698785155489,
              "sign_accuracy": 0.5087719298245614
            }
          },
          "raw_feature_ridge": {
            "margin": {
              "mae": 0.2066389982799535,
              "pairwise_concordance": 0.5132100396301189,
              "pearson": 0.03088961570042361,
              "rmse": 0.2758626139222575,
              "spearman": 0.058465128338086605
            },
            "n": 57,
            "outcome": {
              "brier_style_squared_error": 0.9011627026360793,
              "mae": 0.91552299481034,
              "sign_accuracy": 0.631578947368421
            }
          }
        },
        "standard_start_selfplay": {
          "affine_current_value": {
            "margin": {
              "mae": 0.15912244505041864,
              "pairwise_concordance": 0.7347662152429638,
              "pearson": 0.5968317900370567,
              "rmse": 0.21382642383430206,
              "spearman": 0.6385845916062733
            },
            "n": 548,
            "outcome": {
              "brier_style_squared_error": 0.7032469869652755,
              "mae": 0.7783837941157803,
              "sign_accuracy": 0.6405109489051095
            }
          },
          "current_value": {
            "margin": {
              "mae": 0.2592435330095331,
              "pairwise_concordance": 0.7347662152429638,
              "pearson": 0.5968317900370566,
              "rmse": 0.31931577512399484,
              "spearman": 0.6385845916062733
            },
            "n": 548,
            "outcome": {
              "brier_style_squared_error": 0.5757456959791822,
              "mae": 0.6667217581301234,
              "sign_accuracy": 0.6441605839416058
            }
          },
          "frozen_trunk_linear": {
            "margin": {
              "mae": 0.15632698573299544,
              "pairwise_concordance": 0.751895838334769,
              "pearson": 0.6382633707075773,
              "rmse": 0.2053642755345656,
              "spearman": 0.6638258862198814
            },
            "n": 548,
            "outcome": {
              "brier_style_squared_error": 0.6921931668228527,
              "mae": 0.7733892303883618,
              "sign_accuracy": 0.6806569343065694
            }
          },
          "frozen_trunk_value_head": {
            "margin": {
              "mae": 0.16808016723264893,
              "pairwise_concordance": 0.722296753420982,
              "pearson": 0.5858023755934999,
              "rmse": 0.22125180415766127,
              "spearman": 0.5979056773097129
            },
            "n": 548,
            "outcome": {
              "brier_style_squared_error": 0.7506147244713948,
              "mae": 0.8065586555201091,
              "sign_accuracy": 0.6514598540145985
            }
          },
          "raw_feature_ridge": {
            "margin": {
              "mae": 0.19980427784214544,
              "pairwise_concordance": 0.5881123522517052,
              "pearson": 0.23914526005320733,
              "rmse": 0.259349196094249,
              "spearman": 0.23587942601330325
            },
            "n": 548,
            "outcome": {
              "brier_style_squared_error": 0.8329087661717255,
              "mae": 0.8484257551272735,
              "sign_accuracy": 0.5255474452554745
            }
          }
        }
      }
    }
  }
}
```
