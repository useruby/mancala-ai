# PUCT Visit Allocation Causal Audit

- Classification: `visit_allocation_not_primary`
- Next action: Retire the PR #180 visit-allocation label and proceed to the stronger-teacher decision.

## PR #180 Reproduction

```json
{
  "artifact_sha256": {
    "forced": "4976ecea8c651b95f781b75319ce9b77ddee4439e748a9e6a16125756cc7f27b",
    "moves": "9d1abddacb8a9018865be9710270494b7d3fc431eb6275765c675f35af788795",
    "teachers": "fec49963eeda1d16445b4c9fdc4eaa5e50ca404219179beeddf4890cc22703ba"
  },
  "recomputed_metrics_verified": true
}
```

## Q/Visit Disagreement Counts

```json
{
  "D1200": {
    "count": 233,
    "fraction": 0.3033854166666667
  },
  "D384": {
    "count": 274,
    "fraction": 0.3567708333333333
  },
  "D768": {
    "count": 264,
    "fraction": 0.34375
  },
  "intersections": {
    "D1200": 21,
    "D384": 48,
    "D384|D1200": 14,
    "D384|D768": 44,
    "D384|D768|D1200": 168,
    "D768": 22,
    "D768|D1200": 30
  }
}
```

## Missing-Q Forced Interventions

```json
{
  "completed_count": 278,
  "requested_task_count": 240
}
```

## Direct Paired Causal Results

```json
{
  "D1200": {
    "1200": {
      "n": 233,
      "normalized_store_margin_delta": {
        "lower": -0.02575554363376252,
        "mean": 0.0055436337625178805,
        "median": 0.0,
        "n": 233,
        "upper": 0.039341917024320466
      },
      "outcome_delta": {
        "lower": -0.09871244635193133,
        "mean": 0.012875536480686695,
        "median": 0.0,
        "n": 233,
        "upper": 0.12446351931330472
      },
      "q_better_fraction": 0.38197424892703863,
      "store_margin_delta": {
        "lower": -1.2362660944206008,
        "mean": 0.26609442060085836,
        "median": 0.0,
        "n": 233,
        "upper": 1.888412017167382
      },
      "tied_fraction": 0.1759656652360515,
      "visit_better_fraction": 0.44206008583690987
    },
    "768": {
      "n": 233,
      "normalized_store_margin_delta": {
        "lower": -0.0278969957081545,
        "mean": 0.0026824034334764013,
        "median": 0.0,
        "n": 233,
        "upper": 0.03379828326180258
      },
      "outcome_delta": {
        "lower": -0.12875536480686695,
        "mean": -0.008583690987124463,
        "median": 0.0,
        "n": 233,
        "upper": 0.1072961373390558
      },
      "q_better_fraction": 0.38197424892703863,
      "store_margin_delta": {
        "lower": -1.3390557939914163,
        "mean": 0.12875536480686695,
        "median": 0.0,
        "n": 233,
        "upper": 1.6223175965665235
      },
      "tied_fraction": 0.18025751072961374,
      "visit_better_fraction": 0.43776824034334766
    }
  },
  "D384": {
    "1200": {
      "n": 274,
      "normalized_store_margin_delta": {
        "lower": -0.030721563260340635,
        "mean": 0.0007603406326034072,
        "median": 0.0,
        "n": 274,
        "upper": 0.03254257907542578
      },
      "outcome_delta": {
        "lower": -0.11313868613138686,
        "mean": 0.0036496350364963502,
        "median": 0.0,
        "n": 274,
        "upper": 0.12043795620437957
      },
      "q_better_fraction": 0.40875912408759124,
      "store_margin_delta": {
        "lower": -1.47463503649635,
        "mean": 0.0364963503649635,
        "median": 0.0,
        "n": 274,
        "upper": 1.562043795620438
      },
      "tied_fraction": 0.14233576642335766,
      "visit_better_fraction": 0.4489051094890511
    },
    "768": {
      "n": 274,
      "normalized_store_margin_delta": {
        "lower": -0.03604014598540146,
        "mean": -0.004257907542579074,
        "median": 0.0,
        "n": 274,
        "upper": 0.028284671532846715
      },
      "outcome_delta": {
        "lower": -0.1678832116788321,
        "mean": -0.051094890510948905,
        "median": 0.0,
        "n": 274,
        "upper": 0.06569343065693431
      },
      "q_better_fraction": 0.38686131386861317,
      "store_margin_delta": {
        "lower": -1.72992700729927,
        "mean": -0.20437956204379562,
        "median": 0.0,
        "n": 274,
        "upper": 1.3576642335766422
      },
      "tied_fraction": 0.16423357664233576,
      "visit_better_fraction": 0.4489051094890511
    }
  },
  "D768": {
    "1200": {
      "n": 264,
      "normalized_store_margin_delta": {
        "lower": -0.019258996212121213,
        "mean": 0.011994949494949492,
        "median": 0.0,
        "n": 264,
        "upper": 0.043876262626262624
      },
      "outcome_delta": {
        "lower": -0.030303030303030304,
        "mean": 0.08333333333333333,
        "median": 0.0,
        "n": 264,
        "upper": 0.19696969696969696
      },
      "q_better_fraction": 0.42803030303030304,
      "store_margin_delta": {
        "lower": -0.924431818181818,
        "mean": 0.5757575757575758,
        "median": 0.0,
        "n": 264,
        "upper": 2.106060606060606
      },
      "tied_fraction": 0.16287878787878787,
      "visit_better_fraction": 0.4090909090909091
    },
    "768": {
      "n": 264,
      "normalized_store_margin_delta": {
        "lower": -0.03377525252525253,
        "mean": -0.0017361111111111108,
        "median": 0.0,
        "n": 264,
        "upper": 0.030303030303030304
      },
      "outcome_delta": {
        "lower": -0.13257575757575757,
        "mean": -0.011363636363636364,
        "median": 0.0,
        "n": 264,
        "upper": 0.10606060606060606
      },
      "q_better_fraction": 0.38636363636363635,
      "store_margin_delta": {
        "lower": -1.621212121212121,
        "mean": -0.08333333333333333,
        "median": 0.0,
        "n": 264,
        "upper": 1.4545454545454546
      },
      "tied_fraction": 0.17803030303030304,
      "visit_better_fraction": 0.4356060606060606
    }
  }
}
```

## Paired Concordance And Regret Differences

```json
{
  "D1200": {
    "1200": {
      "q_minus_visit_concordance": {
        "lower": -0.03045257729828042,
        "mean": -0.004877645502645504,
        "median": 0.0,
        "n": 768,
        "upper": 0.020488384589947083
      },
      "visit_minus_q_regret": {
        "lower": -0.008192274305555556,
        "mean": 0.0016818576388888888,
        "median": 0.0,
        "n": 768,
        "upper": 0.011338975694444446
      }
    },
    "768": {
      "q_minus_visit_concordance": {
        "lower": -0.021820023148148147,
        "mean": 0.00423900462962963,
        "median": 0.0,
        "n": 768,
        "upper": 0.03007884837962962
      },
      "visit_minus_q_regret": {
        "lower": -0.008517795138888888,
        "mean": 0.0008138020833333339,
        "median": 0.0,
        "n": 768,
        "upper": 0.01025390625
      }
    }
  },
  "D384": {
    "1200": {
      "q_minus_visit_concordance": {
        "lower": -0.025531167328042326,
        "mean": 0.0028955853174603154,
        "median": 0.0,
        "n": 768,
        "upper": 0.030826926256613752
      },
      "visit_minus_q_regret": {
        "lower": -0.010904947916666666,
        "mean": 0.0004340277777777771,
        "median": 0.0,
        "n": 768,
        "upper": 0.011881510416666666
      }
    },
    "768": {
      "q_minus_visit_concordance": {
        "lower": -0.0228734447337963,
        "mean": 0.004448784722222221,
        "median": 0.0,
        "n": 768,
        "upper": 0.03279803240740741
      },
      "visit_minus_q_regret": {
        "lower": -0.012749565972222222,
        "mean": -0.001302083333333333,
        "median": 0.0,
        "n": 768,
        "upper": 0.01058078342013887
      }
    }
  },
  "D768": {
    "1200": {
      "q_minus_visit_concordance": {
        "lower": -0.021933593749999994,
        "mean": 0.005309606481481482,
        "median": 0.0,
        "n": 768,
        "upper": 0.031524884259259256
      },
      "visit_minus_q_regret": {
        "lower": -0.006998697916666668,
        "mean": 0.004014756944444444,
        "median": 0.0,
        "n": 768,
        "upper": 0.014811197916666666
      }
    },
    "768": {
      "q_minus_visit_concordance": {
        "lower": -0.020891746238425924,
        "mean": 0.006503182870370374,
        "median": 0.0,
        "n": 768,
        "upper": 0.034114764178240733
      },
      "visit_minus_q_regret": {
        "lower": -0.011503092447916667,
        "mean": -0.0007052951388888887,
        "median": 0.0,
        "n": 768,
        "upper": 0.010470920138888888
      }
    }
  }
}
```

## Prior-Pressure Attribution

```json
{
  "D1200": {
    "1200": {
      "prior_does_not_favor_visit": {
        "n": 2,
        "normalized_store_margin_delta": {
          "lower": 0.0,
          "mean": 0.14583333333333331,
          "median": 0.14583333333333331,
          "n": 2,
          "upper": 0.29166666666666663
        },
        "outcome_delta": {
          "lower": 0.0,
          "mean": 0.0,
          "median": 0.0,
          "n": 2,
          "upper": 0.0
        },
        "q_better_fraction": 0.5,
        "store_margin_delta": {
          "lower": 0.0,
          "mean": 7.0,
          "median": 7.0,
          "n": 2,
          "upper": 14.0
        },
        "tied_fraction": 0.5,
        "visit_better_fraction": 0.0
      },
      "prior_favors_visit": {
        "n": 231,
        "normalized_store_margin_delta": {
          "lower": -0.0283234126984127,
          "mean": 0.004329004329004329,
          "median": 0.0,
          "n": 231,
          "upper": 0.03841991341991341
        },
        "outcome_delta": {
          "lower": -0.1038961038961039,
          "mean": 0.012987012987012988,
          "median": 0.0,
          "n": 231,
          "upper": 0.12987012987012986
        },
        "q_better_fraction": 0.38095238095238093,
        "store_margin_delta": {
          "lower": -1.3595238095238094,
          "mean": 0.2077922077922078,
          "median": 0.0,
          "n": 231,
          "upper": 1.844155844155844
        },
        "tied_fraction": 0.17316017316017315,
        "visit_better_fraction": 0.4458874458874459
      },
      "visit_higher_prior_fraction": 0.9914163090128756
    },
    "768": {
      "prior_does_not_favor_visit": {
        "n": 2,
        "normalized_store_margin_delta": {
          "lower": 0.0,
          "mean": 0.2916666666666667,
          "median": 0.2916666666666667,
          "n": 2,
          "upper": 0.5833333333333334
        },
        "outcome_delta": {
          "lower": 0.0,
          "mean": 1.0,
          "median": 1.0,
          "n": 2,
          "upper": 2.0
        },
        "q_better_fraction": 0.5,
        "store_margin_delta": {
          "lower": 0.0,
          "mean": 14.0,
          "median": 14.0,
          "n": 2,
          "upper": 28.0
        },
        "tied_fraction": 0.5,
        "visit_better_fraction": 0.0
      },
      "prior_favors_visit": {
        "n": 231,
        "normalized_store_margin_delta": {
          "lower": -0.030848665223665226,
          "mean": 0.00018037518037518262,
          "median": 0.0,
          "n": 231,
          "upper": 0.03210678210678211
        },
        "outcome_delta": {
          "lower": -0.1341991341991342,
          "mean": -0.017316017316017316,
          "median": 0.0,
          "n": 231,
          "upper": 0.1038961038961039
        },
        "q_better_fraction": 0.38095238095238093,
        "store_margin_delta": {
          "lower": -1.4807359307359307,
          "mean": 0.008658008658008658,
          "median": 0.0,
          "n": 231,
          "upper": 1.5411255411255411
        },
        "tied_fraction": 0.1774891774891775,
        "visit_better_fraction": 0.44155844155844154
      },
      "visit_higher_prior_fraction": 0.9914163090128756
    }
  },
  "D384": {
    "1200": {
      "prior_does_not_favor_visit": {
        "n": 0,
        "normalized_store_margin_delta": {
          "lower": 0.0,
          "mean": 0.0,
          "median": 0.0,
          "n": 0,
          "upper": 0.0
        },
        "outcome_delta": {
          "lower": 0.0,
          "mean": 0.0,
          "median": 0.0,
          "n": 0,
          "upper": 0.0
        },
        "q_better_fraction": 0.0,
        "store_margin_delta": {
          "lower": 0.0,
          "mean": 0.0,
          "median": 0.0,
          "n": 0,
          "upper": 0.0
        },
        "tied_fraction": 0.0,
        "visit_better_fraction": 0.0
      },
      "prior_favors_visit": {
        "n": 274,
        "normalized_store_margin_delta": {
          "lower": -0.031177767639902678,
          "mean": 0.0007603406326034072,
          "median": 0.0,
          "n": 274,
          "upper": 0.032394312652068075
        },
        "outcome_delta": {
          "lower": -0.11678832116788321,
          "mean": 0.0036496350364963502,
          "median": 0.0,
          "n": 274,
          "upper": 0.12043795620437957
        },
        "q_better_fraction": 0.40875912408759124,
        "store_margin_delta": {
          "lower": -1.4965328467153283,
          "mean": 0.0364963503649635,
          "median": 0.0,
          "n": 274,
          "upper": 1.5549270072992674
        },
        "tied_fraction": 0.14233576642335766,
        "visit_better_fraction": 0.4489051094890511
      },
      "visit_higher_prior_fraction": 1.0
    },
    "768": {
      "prior_does_not_favor_visit": {
        "n": 0,
        "normalized_store_margin_delta": {
          "lower": 0.0,
          "mean": 0.0,
          "median": 0.0,
          "n": 0,
          "upper": 0.0
        },
        "outcome_delta": {
          "lower": 0.0,
          "mean": 0.0,
          "median": 0.0,
          "n": 0,
          "upper": 0.0
        },
        "q_better_fraction": 0.0,
        "store_margin_delta": {
          "lower": 0.0,
          "mean": 0.0,
          "median": 0.0,
          "n": 0,
          "upper": 0.0
        },
        "tied_fraction": 0.0,
        "visit_better_fraction": 0.0
      },
      "prior_favors_visit": {
        "n": 274,
        "normalized_store_margin_delta": {
          "lower": -0.0364963503649635,
          "mean": -0.004257907542579074,
          "median": 0.0,
          "n": 274,
          "upper": 0.028588807785888078
        },
        "outcome_delta": {
          "lower": -0.1678832116788321,
          "mean": -0.051094890510948905,
          "median": 0.0,
          "n": 274,
          "upper": 0.06934306569343066
        },
        "q_better_fraction": 0.38686131386861317,
        "store_margin_delta": {
          "lower": -1.7518248175182483,
          "mean": -0.20437956204379562,
          "median": 0.0,
          "n": 274,
          "upper": 1.3722627737226278
        },
        "tied_fraction": 0.16423357664233576,
        "visit_better_fraction": 0.4489051094890511
      },
      "visit_higher_prior_fraction": 1.0
    }
  },
  "D768": {
    "1200": {
      "prior_does_not_favor_visit": {
        "n": 2,
        "normalized_store_margin_delta": {
          "lower": 0.0,
          "mean": 0.0625,
          "median": 0.0625,
          "n": 2,
          "upper": 0.125
        },
        "outcome_delta": {
          "lower": 0.0,
          "mean": 1.0,
          "median": 1.0,
          "n": 2,
          "upper": 2.0
        },
        "q_better_fraction": 0.5,
        "store_margin_delta": {
          "lower": 0.0,
          "mean": 3.0,
          "median": 3.0,
          "n": 2,
          "upper": 6.0
        },
        "tied_fraction": 0.5,
        "visit_better_fraction": 0.0
      },
      "prior_favors_visit": {
        "n": 262,
        "normalized_store_margin_delta": {
          "lower": -0.020201176844783716,
          "mean": 0.011609414758269722,
          "median": 0.0,
          "n": 262,
          "upper": 0.04373409669211196
        },
        "outcome_delta": {
          "lower": -0.03816793893129771,
          "mean": 0.07633587786259542,
          "median": 0.0,
          "n": 262,
          "upper": 0.19083969465648856
        },
        "q_better_fraction": 0.42748091603053434,
        "store_margin_delta": {
          "lower": -0.9696564885496181,
          "mean": 0.5572519083969466,
          "median": 0.0,
          "n": 262,
          "upper": 2.099236641221374
        },
        "tied_fraction": 0.16030534351145037,
        "visit_better_fraction": 0.4122137404580153
      },
      "visit_higher_prior_fraction": 0.9924242424242424
    },
    "768": {
      "prior_does_not_favor_visit": {
        "n": 2,
        "normalized_store_margin_delta": {
          "lower": 0.0,
          "mean": 0.0625,
          "median": 0.0625,
          "n": 2,
          "upper": 0.125
        },
        "outcome_delta": {
          "lower": 0.0,
          "mean": 1.0,
          "median": 1.0,
          "n": 2,
          "upper": 2.0
        },
        "q_better_fraction": 0.5,
        "store_margin_delta": {
          "lower": 0.0,
          "mean": 3.0,
          "median": 3.0,
          "n": 2,
          "upper": 6.0
        },
        "tied_fraction": 0.5,
        "visit_better_fraction": 0.0
      },
      "prior_favors_visit": {
        "n": 262,
        "normalized_store_margin_delta": {
          "lower": -0.03435114503816794,
          "mean": -0.0022264631043256993,
          "median": 0.0,
          "n": 262,
          "upper": 0.030057251908396945
        },
        "outcome_delta": {
          "lower": -0.13740458015267176,
          "mean": -0.019083969465648856,
          "median": 0.0,
          "n": 262,
          "upper": 0.09923664122137404
        },
        "q_better_fraction": 0.38549618320610685,
        "store_margin_delta": {
          "lower": -1.6488549618320612,
          "mean": -0.10687022900763359,
          "median": 0.0,
          "n": 262,
          "upper": 1.4427480916030535
        },
        "tied_fraction": 0.17557251908396945,
        "visit_better_fraction": 0.4389312977099237
      },
      "visit_higher_prior_fraction": 0.9924242424242424
    }
  }
}
```

## Root PUCT Pressure Diagnostic

```json
{
  "D1200": {
    "1200": {
      "mean_prior_ratio_visit_over_q": 58.00257537291686,
      "mean_q_best_prior": 0.09245476878973627,
      "mean_q_best_q": -0.2556332020857799,
      "mean_q_best_u": 0.015489014148560496,
      "mean_q_margin": 0.024650317892761214,
      "mean_visit_best_prior": 0.5603808018302968,
      "mean_visit_best_q": -0.28028351997854106,
      "mean_visit_best_u": 0.035215640544291424,
      "mean_visit_margin": 0.3676788268955652,
      "n": 233
    },
    "768": {
      "mean_prior_ratio_visit_over_q": 58.00257537291686,
      "mean_q_best_prior": 0.09245476878973627,
      "mean_q_best_q": -0.2556332020857799,
      "mean_q_best_u": 0.015489014148560496,
      "mean_q_margin": 0.024650317892761214,
      "mean_visit_best_prior": 0.5603808018302968,
      "mean_visit_best_q": -0.28028351997854106,
      "mean_visit_best_u": 0.035215640544291424,
      "mean_visit_margin": 0.3676788268955652,
      "n": 233
    }
  },
  "D384": {
    "1200": {
      "mean_prior_ratio_visit_over_q": 49.54523004777818,
      "mean_q_best_prior": 0.08715384093550438,
      "mean_q_best_q": -0.16606765559185485,
      "mean_q_best_u": 0.02816352239241187,
      "mean_q_margin": 0.038030270429851826,
      "mean_visit_best_prior": 0.5689051678118697,
      "mean_visit_best_q": -0.20409792602170665,
      "mean_visit_best_u": 0.0617417404509847,
      "mean_visit_margin": 0.3874505778588808,
      "n": 274
    },
    "768": {
      "mean_prior_ratio_visit_over_q": 49.54523004777818,
      "mean_q_best_prior": 0.08715384093550438,
      "mean_q_best_q": -0.16606765559185485,
      "mean_q_best_u": 0.02816352239241187,
      "mean_q_margin": 0.038030270429851826,
      "mean_visit_best_prior": 0.5689051678118697,
      "mean_visit_best_q": -0.20409792602170665,
      "mean_visit_best_u": 0.0617417404509847,
      "mean_visit_margin": 0.3874505778588808,
      "n": 274
    }
  },
  "D768": {
    "1200": {
      "mean_prior_ratio_visit_over_q": 73.27166161124764,
      "mean_q_best_prior": 0.09431546155279978,
      "mean_q_best_q": -0.19684880360562665,
      "mean_q_best_u": 0.01923012459393534,
      "mean_q_margin": 0.03045031309224096,
      "mean_visit_best_prior": 0.5469319972802292,
      "mean_visit_best_q": -0.22729911669786756,
      "mean_visit_best_u": 0.04352286817568508,
      "mean_visit_margin": 0.3499842171717172,
      "n": 264
    },
    "768": {
      "mean_prior_ratio_visit_over_q": 73.27166161124764,
      "mean_q_best_prior": 0.09431546155279978,
      "mean_q_best_q": -0.19684880360562665,
      "mean_q_best_u": 0.01923012459393534,
      "mean_q_margin": 0.03045031309224096,
      "mean_visit_best_prior": 0.5469319972802292,
      "mean_visit_best_q": -0.22729911669786756,
      "mean_visit_best_u": 0.04352286817568508,
      "mean_visit_margin": 0.3499842171717172,
      "n": 264
    }
  }
}
```

## Search-Budget Trend

```json
{
  "D1200": {
    "causal_1200": {
      "n": 233,
      "normalized_store_margin_delta": {
        "lower": -0.02575554363376252,
        "mean": 0.0055436337625178805,
        "median": 0.0,
        "n": 233,
        "upper": 0.039341917024320466
      },
      "outcome_delta": {
        "lower": -0.09871244635193133,
        "mean": 0.012875536480686695,
        "median": 0.0,
        "n": 233,
        "upper": 0.12446351931330472
      },
      "q_better_fraction": 0.38197424892703863,
      "store_margin_delta": {
        "lower": -1.2362660944206008,
        "mean": 0.26609442060085836,
        "median": 0.0,
        "n": 233,
        "upper": 1.888412017167382
      },
      "tied_fraction": 0.1759656652360515,
      "visit_better_fraction": 0.44206008583690987
    },
    "disagreement_fraction": 0.3033854166666667,
    "paired_1200": {
      "q_minus_visit_concordance": {
        "lower": -0.03045257729828042,
        "mean": -0.004877645502645504,
        "median": 0.0,
        "n": 768,
        "upper": 0.020488384589947083
      },
      "visit_minus_q_regret": {
        "lower": -0.008192274305555556,
        "mean": 0.0016818576388888888,
        "median": 0.0,
        "n": 768,
        "upper": 0.011338975694444446
      }
    }
  },
  "D384": {
    "causal_1200": {
      "n": 274,
      "normalized_store_margin_delta": {
        "lower": -0.030721563260340635,
        "mean": 0.0007603406326034072,
        "median": 0.0,
        "n": 274,
        "upper": 0.03254257907542578
      },
      "outcome_delta": {
        "lower": -0.11313868613138686,
        "mean": 0.0036496350364963502,
        "median": 0.0,
        "n": 274,
        "upper": 0.12043795620437957
      },
      "q_better_fraction": 0.40875912408759124,
      "store_margin_delta": {
        "lower": -1.47463503649635,
        "mean": 0.0364963503649635,
        "median": 0.0,
        "n": 274,
        "upper": 1.562043795620438
      },
      "tied_fraction": 0.14233576642335766,
      "visit_better_fraction": 0.4489051094890511
    },
    "disagreement_fraction": 0.3567708333333333,
    "paired_1200": {
      "q_minus_visit_concordance": {
        "lower": -0.025531167328042326,
        "mean": 0.0028955853174603154,
        "median": 0.0,
        "n": 768,
        "upper": 0.030826926256613752
      },
      "visit_minus_q_regret": {
        "lower": -0.010904947916666666,
        "mean": 0.0004340277777777771,
        "median": 0.0,
        "n": 768,
        "upper": 0.011881510416666666
      }
    }
  },
  "D768": {
    "causal_1200": {
      "n": 264,
      "normalized_store_margin_delta": {
        "lower": -0.019258996212121213,
        "mean": 0.011994949494949492,
        "median": 0.0,
        "n": 264,
        "upper": 0.043876262626262624
      },
      "outcome_delta": {
        "lower": -0.030303030303030304,
        "mean": 0.08333333333333333,
        "median": 0.0,
        "n": 264,
        "upper": 0.19696969696969696
      },
      "q_better_fraction": 0.42803030303030304,
      "store_margin_delta": {
        "lower": -0.924431818181818,
        "mean": 0.5757575757575758,
        "median": 0.0,
        "n": 264,
        "upper": 2.106060606060606
      },
      "tied_fraction": 0.16287878787878787,
      "visit_better_fraction": 0.4090909090909091
    },
    "disagreement_fraction": 0.34375,
    "paired_1200": {
      "q_minus_visit_concordance": {
        "lower": -0.021933593749999994,
        "mean": 0.005309606481481482,
        "median": 0.0,
        "n": 768,
        "upper": 0.031524884259259256
      },
      "visit_minus_q_regret": {
        "lower": -0.006998697916666668,
        "mean": 0.004014756944444444,
        "median": 0.0,
        "n": 768,
        "upper": 0.014811197916666666
      }
    }
  }
}
```

## Phase, Player, Entropy, And Domain Slices

```json
{
  "D1200": {
    "1200": {
      "phase": {
        "late": {
          "n": 44,
          "normalized_store_margin_delta": {
            "lower": -0.02178030303030303,
            "mean": -0.0037878787878787845,
            "median": 0.0,
            "n": 44,
            "upper": 0.014204545454545463
          },
          "outcome_delta": {
            "lower": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "n": 44,
            "upper": 0.0
          },
          "q_better_fraction": 0.22727272727272727,
          "store_margin_delta": {
            "lower": -1.0454545454545454,
            "mean": -0.18181818181818182,
            "median": 0.0,
            "n": 44,
            "upper": 0.6818181818181818
          },
          "tied_fraction": 0.4772727272727273,
          "visit_better_fraction": 0.29545454545454547
        },
        "midgame": {
          "n": 54,
          "normalized_store_margin_delta": {
            "lower": -0.11342592592592592,
            "mean": -0.06558641975308642,
            "median": -0.04166666666666663,
            "n": 54,
            "upper": -0.02160493827160493
          },
          "outcome_delta": {
            "lower": -0.09259259259259259,
            "mean": -0.018518518518518517,
            "median": 0.0,
            "n": 54,
            "upper": 0.037037037037037035
          },
          "q_better_fraction": 0.25925925925925924,
          "store_margin_delta": {
            "lower": -5.444444444444445,
            "mean": -3.1481481481481484,
            "median": -2.0,
            "n": 54,
            "upper": -1.037037037037037
          },
          "tied_fraction": 0.2222222222222222,
          "visit_better_fraction": 0.5185185185185185
        },
        "opening": {
          "n": 135,
          "normalized_store_margin_delta": {
            "lower": -0.015748456790123454,
            "mean": 0.03703703703703703,
            "median": 0.0,
            "n": 135,
            "upper": 0.08919753086419753
          },
          "outcome_delta": {
            "lower": -0.17037037037037037,
            "mean": 0.02962962962962963,
            "median": 0.0,
            "n": 135,
            "upper": 0.2222222222222222
          },
          "q_better_fraction": 0.48148148148148145,
          "store_margin_delta": {
            "lower": -0.7559259259259256,
            "mean": 1.7777777777777777,
            "median": 0.0,
            "n": 135,
            "upper": 4.281481481481482
          },
          "tied_fraction": 0.05925925925925926,
          "visit_better_fraction": 0.45925925925925926
        }
      },
      "player": {
        "0": {
          "n": 123,
          "normalized_store_margin_delta": {
            "lower": -0.04268292682926828,
            "mean": 0.0003387533875338773,
            "median": 0.0,
            "n": 123,
            "upper": 0.043360433604336036
          },
          "outcome_delta": {
            "lower": -0.14634146341463414,
            "mean": 0.016260162601626018,
            "median": 0.0,
            "n": 123,
            "upper": 0.17073170731707318
          },
          "q_better_fraction": 0.4146341463414634,
          "store_margin_delta": {
            "lower": -2.048780487804878,
            "mean": 0.016260162601626018,
            "median": 0.0,
            "n": 123,
            "upper": 2.0813008130081303
          },
          "tied_fraction": 0.17886178861788618,
          "visit_better_fraction": 0.4065040650406504
        },
        "1": {
          "n": 110,
          "normalized_store_margin_delta": {
            "lower": -0.03787878787878788,
            "mean": 0.011363636363636366,
            "median": 0.0,
            "n": 110,
            "upper": 0.06287878787878787
          },
          "outcome_delta": {
            "lower": -0.15454545454545454,
            "mean": 0.00909090909090909,
            "median": 0.0,
            "n": 110,
            "upper": 0.17272727272727273
          },
          "q_better_fraction": 0.34545454545454546,
          "store_margin_delta": {
            "lower": -1.8181818181818181,
            "mean": 0.5454545454545454,
            "median": 0.0,
            "n": 110,
            "upper": 3.018181818181818
          },
          "tied_fraction": 0.17272727272727273,
          "visit_better_fraction": 0.4818181818181818
        }
      },
      "policy_entropy_quartile": {
        "1": {
          "n": 55,
          "normalized_store_margin_delta": {
            "lower": -0.07803030303030303,
            "mean": -0.021212121212121227,
            "median": -0.04166666666666663,
            "n": 55,
            "upper": 0.034848484848484844
          },
          "outcome_delta": {
            "lower": -0.36363636363636365,
            "mean": -0.10909090909090909,
            "median": 0.0,
            "n": 55,
            "upper": 0.14545454545454545
          },
          "q_better_fraction": 0.34545454545454546,
          "store_margin_delta": {
            "lower": -3.7454545454545456,
            "mean": -1.018181818181818,
            "median": -2.0,
            "n": 55,
            "upper": 1.6727272727272726
          },
          "tied_fraction": 0.09090909090909091,
          "visit_better_fraction": 0.5636363636363636
        },
        "2": {
          "n": 68,
          "normalized_store_margin_delta": {
            "lower": -0.035539215686274515,
            "mean": 0.02205882352941177,
            "median": 0.0,
            "n": 68,
            "upper": 0.08026960784313725
          },
          "outcome_delta": {
            "lower": -0.058823529411764705,
            "mean": 0.16176470588235295,
            "median": 0.0,
            "n": 68,
            "upper": 0.38235294117647056
          },
          "q_better_fraction": 0.45588235294117646,
          "store_margin_delta": {
            "lower": -1.7058823529411764,
            "mean": 1.0588235294117647,
            "median": 0.0,
            "n": 68,
            "upper": 3.8529411764705883
          },
          "tied_fraction": 0.17647058823529413,
          "visit_better_fraction": 0.36764705882352944
        },
        "3": {
          "n": 60,
          "normalized_store_margin_delta": {
            "lower": -0.04027777777777778,
            "mean": 0.03055555555555555,
            "median": 0.0,
            "n": 60,
            "upper": 0.10833333333333334
          },
          "outcome_delta": {
            "lower": -0.06666666666666667,
            "mean": 0.11666666666666667,
            "median": 0.0,
            "n": 60,
            "upper": 0.31666666666666665
          },
          "q_better_fraction": 0.38333333333333336,
          "store_margin_delta": {
            "lower": -1.9333333333333333,
            "mean": 1.4666666666666666,
            "median": 0.0,
            "n": 60,
            "upper": 5.2
          },
          "tied_fraction": 0.2,
          "visit_better_fraction": 0.4166666666666667
        },
        "4": {
          "n": 50,
          "normalized_store_margin_delta": {
            "lower": -0.085,
            "mean": -0.017499999999999998,
            "median": 0.0,
            "n": 50,
            "upper": 0.051666666666666666
          },
          "outcome_delta": {
            "lower": -0.4,
            "mean": -0.18,
            "median": 0.0,
            "n": 50,
            "upper": 0.04
          },
          "q_better_fraction": 0.32,
          "store_margin_delta": {
            "lower": -4.08,
            "mean": -0.84,
            "median": 0.0,
            "n": 50,
            "upper": 2.48
          },
          "tied_fraction": 0.24,
          "visit_better_fraction": 0.44
        }
      },
      "source_domain": {
        "additional_standard_start_selfplay": {
          "n": 66,
          "normalized_store_margin_delta": {
            "lower": -0.08207070707070706,
            "mean": -0.03661616161616161,
            "median": 0.0,
            "n": 66,
            "upper": 0.008222853535353312
          },
          "outcome_delta": {
            "lower": -0.19696969696969696,
            "mean": -0.07575757575757576,
            "median": 0.0,
            "n": 66,
            "upper": 0.030303030303030304
          },
          "q_better_fraction": 0.2878787878787879,
          "store_margin_delta": {
            "lower": -3.9393939393939394,
            "mean": -1.7575757575757576,
            "median": 0.0,
            "n": 66,
            "upper": 0.39469696969695867
          },
          "tied_fraction": 0.24242424242424243,
          "visit_better_fraction": 0.4696969696969697
        },
        "independent_opening_suite_diagnostic": {
          "n": 98,
          "normalized_store_margin_delta": {
            "lower": -0.017857142857142856,
            "mean": 0.04549319727891156,
            "median": 0.04166666666666666,
            "n": 98,
            "upper": 0.11268069727891142
          },
          "outcome_delta": {
            "lower": -0.22448979591836735,
            "mean": 0.01020408163265306,
            "median": 0.0,
            "n": 98,
            "upper": 0.24489795918367346
          },
          "q_better_fraction": 0.5102040816326531,
          "store_margin_delta": {
            "lower": -0.8571428571428571,
            "mean": 2.183673469387755,
            "median": 2.0,
            "n": 98,
            "upper": 5.408673469387748
          },
          "tied_fraction": 0.061224489795918366,
          "visit_better_fraction": 0.42857142857142855
        },
        "pr176_standard_start_pilot": {
          "n": 69,
          "normalized_store_margin_delta": {
            "lower": -0.048324275362318835,
            "mean": -0.0108695652173913,
            "median": 0.0,
            "n": 69,
            "upper": 0.026570048309178744
          },
          "outcome_delta": {
            "lower": -0.043478260869565216,
            "mean": 0.10144927536231885,
            "median": 0.0,
            "n": 69,
            "upper": 0.2608695652173913
          },
          "q_better_fraction": 0.2898550724637681,
          "store_margin_delta": {
            "lower": -2.319565217391304,
            "mean": -0.5217391304347826,
            "median": 0.0,
            "n": 69,
            "upper": 1.2753623188405796
          },
          "tied_fraction": 0.2753623188405797,
          "visit_better_fraction": 0.43478260869565216
        }
      }
    },
    "768": {
      "phase": {
        "late": {
          "n": 44,
          "normalized_store_margin_delta": {
            "lower": -0.021780303030303018,
            "mean": -0.001893939393939389,
            "median": 0.0,
            "n": 44,
            "upper": 0.01704545454545455
          },
          "outcome_delta": {
            "lower": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "n": 44,
            "upper": 0.0
          },
          "q_better_fraction": 0.29545454545454547,
          "store_margin_delta": {
            "lower": -1.0454545454545454,
            "mean": -0.09090909090909091,
            "median": 0.0,
            "n": 44,
            "upper": 0.8181818181818182
          },
          "tied_fraction": 0.4318181818181818,
          "visit_better_fraction": 0.2727272727272727
        },
        "midgame": {
          "n": 54,
          "normalized_store_margin_delta": {
            "lower": -0.12499999999999999,
            "mean": -0.08101851851851853,
            "median": -0.041666666666666685,
            "n": 54,
            "upper": -0.03935185185185186
          },
          "outcome_delta": {
            "lower": -0.16666666666666666,
            "mean": -0.05555555555555555,
            "median": 0.0,
            "n": 54,
            "upper": 0.037037037037037035
          },
          "q_better_fraction": 0.18518518518518517,
          "store_margin_delta": {
            "lower": -6.0,
            "mean": -3.888888888888889,
            "median": -2.0,
            "n": 54,
            "upper": -1.8888888888888888
          },
          "tied_fraction": 0.24074074074074073,
          "visit_better_fraction": 0.5740740740740741
        },
        "opening": {
          "n": 135,
          "normalized_store_margin_delta": {
            "lower": -0.012345679012345678,
            "mean": 0.037654320987654324,
            "median": 0.0,
            "n": 135,
            "upper": 0.08858024691358024
          },
          "outcome_delta": {
            "lower": -0.1925925925925926,
            "mean": 0.007407407407407408,
            "median": 0.0,
            "n": 135,
            "upper": 0.2074074074074074
          },
          "q_better_fraction": 0.4888888888888889,
          "store_margin_delta": {
            "lower": -0.5925925925925926,
            "mean": 1.8074074074074074,
            "median": 0.0,
            "n": 135,
            "upper": 4.2518518518518515
          },
          "tied_fraction": 0.07407407407407407,
          "visit_better_fraction": 0.43703703703703706
        }
      },
      "player": {
        "0": {
          "n": 123,
          "normalized_store_margin_delta": {
            "lower": -0.05793529810298103,
            "mean": -0.017615176151761506,
            "median": 0.0,
            "n": 123,
            "upper": 0.023712737127371264
          },
          "outcome_delta": {
            "lower": -0.13008130081300814,
            "mean": 0.016260162601626018,
            "median": 0.0,
            "n": 123,
            "upper": 0.17073170731707318
          },
          "q_better_fraction": 0.34146341463414637,
          "store_margin_delta": {
            "lower": -2.780894308943089,
            "mean": -0.8455284552845529,
            "median": 0.0,
            "n": 123,
            "upper": 1.1382113821138211
          },
          "tied_fraction": 0.22764227642276422,
          "visit_better_fraction": 0.43089430894308944
        },
        "1": {
          "n": 110,
          "normalized_store_margin_delta": {
            "lower": -0.021969696969696962,
            "mean": 0.025378787878787883,
            "median": 0.0,
            "n": 110,
            "upper": 0.07310606060606062
          },
          "outcome_delta": {
            "lower": -0.21818181818181817,
            "mean": -0.03636363636363636,
            "median": 0.0,
            "n": 110,
            "upper": 0.14545454545454545
          },
          "q_better_fraction": 0.42727272727272725,
          "store_margin_delta": {
            "lower": -1.0545454545454545,
            "mean": 1.2181818181818183,
            "median": 0.0,
            "n": 110,
            "upper": 3.5090909090909093
          },
          "tied_fraction": 0.12727272727272726,
          "visit_better_fraction": 0.44545454545454544
        }
      },
      "policy_entropy_quartile": {
        "1": {
          "n": 55,
          "normalized_store_margin_delta": {
            "lower": -0.06515151515151515,
            "mean": -0.006818181818181822,
            "median": -0.04166666666666663,
            "n": 55,
            "upper": 0.05303030303030304
          },
          "outcome_delta": {
            "lower": -0.34545454545454546,
            "mean": -0.10909090909090909,
            "median": 0.0,
            "n": 55,
            "upper": 0.12727272727272726
          },
          "q_better_fraction": 0.34545454545454546,
          "store_margin_delta": {
            "lower": -3.1272727272727274,
            "mean": -0.32727272727272727,
            "median": -2.0,
            "n": 55,
            "upper": 2.5454545454545454
          },
          "tied_fraction": 0.10909090909090909,
          "visit_better_fraction": 0.5454545454545454
        },
        "2": {
          "n": 68,
          "normalized_store_margin_delta": {
            "lower": -0.05882352941176469,
            "mean": -0.003676470588235286,
            "median": 0.0,
            "n": 68,
            "upper": 0.050245098039215695
          },
          "outcome_delta": {
            "lower": -0.23529411764705882,
            "mean": -0.029411764705882353,
            "median": 0.0,
            "n": 68,
            "upper": 0.17647058823529413
          },
          "q_better_fraction": 0.4264705882352941,
          "store_margin_delta": {
            "lower": -2.823529411764706,
            "mean": -0.17647058823529413,
            "median": 0.0,
            "n": 68,
            "upper": 2.411764705882353
          },
          "tied_fraction": 0.17647058823529413,
          "visit_better_fraction": 0.39705882352941174
        },
        "3": {
          "n": 60,
          "normalized_store_margin_delta": {
            "lower": -0.0736111111111111,
            "mean": -0.00833333333333333,
            "median": 0.0,
            "n": 60,
            "upper": 0.059722222222222225
          },
          "outcome_delta": {
            "lower": -0.25,
            "mean": -0.03333333333333333,
            "median": 0.0,
            "n": 60,
            "upper": 0.18374999999999392
          },
          "q_better_fraction": 0.31666666666666665,
          "store_margin_delta": {
            "lower": -3.533333333333333,
            "mean": -0.4,
            "median": 0.0,
            "n": 60,
            "upper": 2.8666666666666667
          },
          "tied_fraction": 0.23333333333333334,
          "visit_better_fraction": 0.45
        },
        "4": {
          "n": 50,
          "normalized_store_margin_delta": {
            "lower": -0.032499999999999994,
            "mean": 0.03500000000000001,
            "median": 0.0,
            "n": 50,
            "upper": 0.10416666666666669
          },
          "outcome_delta": {
            "lower": -0.12,
            "mean": 0.16,
            "median": 0.0,
            "n": 50,
            "upper": 0.44
          },
          "q_better_fraction": 0.44,
          "store_margin_delta": {
            "lower": -1.56,
            "mean": 1.68,
            "median": 0.0,
            "n": 50,
            "upper": 5.0
          },
          "tied_fraction": 0.2,
          "visit_better_fraction": 0.36
        }
      },
      "source_domain": {
        "additional_standard_start_selfplay": {
          "n": 66,
          "normalized_store_margin_delta": {
            "lower": -0.07828282828282826,
            "mean": -0.029671717171717172,
            "median": 0.0,
            "n": 66,
            "upper": 0.01957070707070708
          },
          "outcome_delta": {
            "lower": -0.21212121212121213,
            "mean": -0.07575757575757576,
            "median": 0.0,
            "n": 66,
            "upper": 0.045454545454545456
          },
          "q_better_fraction": 0.3333333333333333,
          "store_margin_delta": {
            "lower": -3.757575757575758,
            "mean": -1.4242424242424243,
            "median": 0.0,
            "n": 66,
            "upper": 0.9393939393939394
          },
          "tied_fraction": 0.22727272727272727,
          "visit_better_fraction": 0.4393939393939394
        },
        "independent_opening_suite_diagnostic": {
          "n": 98,
          "normalized_store_margin_delta": {
            "lower": -0.013180272108843533,
            "mean": 0.04719387755102042,
            "median": 0.04166666666666666,
            "n": 98,
            "upper": 0.10756802721088436
          },
          "outcome_delta": {
            "lower": -0.15306122448979592,
            "mean": 0.09183673469387756,
            "median": 0.0,
            "n": 98,
            "upper": 0.336734693877551
          },
          "q_better_fraction": 0.5102040816326531,
          "store_margin_delta": {
            "lower": -0.6326530612244898,
            "mean": 2.2653061224489797,
            "median": 2.0,
            "n": 98,
            "upper": 5.163265306122449
          },
          "tied_fraction": 0.061224489795918366,
          "visit_better_fraction": 0.42857142857142855
        },
        "pr176_standard_start_pilot": {
          "n": 69,
          "normalized_store_margin_delta": {
            "lower": -0.06280193236714977,
            "mean": -0.029589371980676328,
            "median": 0.0,
            "n": 69,
            "upper": 0.002415458937198067
          },
          "outcome_delta": {
            "lower": -0.21739130434782608,
            "mean": -0.08695652173913043,
            "median": 0.0,
            "n": 69,
            "upper": 0.028985507246376812
          },
          "q_better_fraction": 0.2463768115942029,
          "store_margin_delta": {
            "lower": -3.0144927536231885,
            "mean": -1.4202898550724639,
            "median": 0.0,
            "n": 69,
            "upper": 0.11594202898550725
          },
          "tied_fraction": 0.30434782608695654,
          "visit_better_fraction": 0.4492753623188406
        }
      }
    }
  },
  "D384": {
    "1200": {
      "phase": {
        "late": {
          "n": 43,
          "normalized_store_margin_delta": {
            "lower": -0.017441860465116275,
            "mean": 0.0009689922480620198,
            "median": 0.0,
            "n": 43,
            "upper": 0.019379844961240313
          },
          "outcome_delta": {
            "lower": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "n": 43,
            "upper": 0.0
          },
          "q_better_fraction": 0.23255813953488372,
          "store_margin_delta": {
            "lower": -0.8372093023255814,
            "mean": 0.046511627906976744,
            "median": 0.0,
            "n": 43,
            "upper": 0.9302325581395349
          },
          "tied_fraction": 0.46511627906976744,
          "visit_better_fraction": 0.3023255813953488
        },
        "midgame": {
          "n": 50,
          "normalized_store_margin_delta": {
            "lower": -0.08,
            "mean": -0.03166666666666666,
            "median": -0.04166666666666663,
            "n": 50,
            "upper": 0.016666666666666666
          },
          "outcome_delta": {
            "lower": -0.14,
            "mean": -0.02,
            "median": 0.0,
            "n": 50,
            "upper": 0.1
          },
          "q_better_fraction": 0.32,
          "store_margin_delta": {
            "lower": -3.84,
            "mean": -1.52,
            "median": -2.0,
            "n": 50,
            "upper": 0.8
          },
          "tied_fraction": 0.16,
          "visit_better_fraction": 0.52
        },
        "opening": {
          "n": 181,
          "normalized_store_margin_delta": {
            "lower": -0.036141804788213626,
            "mean": 0.00966850828729281,
            "median": 0.0,
            "n": 181,
            "upper": 0.055248618784530384
          },
          "outcome_delta": {
            "lower": -0.16022099447513813,
            "mean": 0.011049723756906077,
            "median": 0.0,
            "n": 181,
            "upper": 0.18232044198895028
          },
          "q_better_fraction": 0.47513812154696133,
          "store_margin_delta": {
            "lower": -1.7348066298342542,
            "mean": 0.46408839779005523,
            "median": 0.0,
            "n": 181,
            "upper": 2.6519337016574585
          },
          "tied_fraction": 0.06077348066298342,
          "visit_better_fraction": 0.46408839779005523
        }
      },
      "player": {
        "0": {
          "n": 149,
          "normalized_store_margin_delta": {
            "lower": -0.04809843400447428,
            "mean": -0.00727069351230425,
            "median": 0.0,
            "n": 149,
            "upper": 0.0341233221476509
          },
          "outcome_delta": {
            "lower": -0.20134228187919462,
            "mean": -0.04697986577181208,
            "median": 0.0,
            "n": 149,
            "upper": 0.10738255033557047
          },
          "q_better_fraction": 0.40268456375838924,
          "store_margin_delta": {
            "lower": -2.3087248322147653,
            "mean": -0.348993288590604,
            "median": 0.0,
            "n": 149,
            "upper": 1.6379194630872436
          },
          "tied_fraction": 0.1476510067114094,
          "visit_better_fraction": 0.44966442953020136
        },
        "1": {
          "n": 125,
          "normalized_store_margin_delta": {
            "lower": -0.03899999999999999,
            "mean": 0.010333333333333332,
            "median": 0.0,
            "n": 125,
            "upper": 0.06199999999999999
          },
          "outcome_delta": {
            "lower": -0.112,
            "mean": 0.064,
            "median": 0.0,
            "n": 125,
            "upper": 0.24
          },
          "q_better_fraction": 0.416,
          "store_margin_delta": {
            "lower": -1.872,
            "mean": 0.496,
            "median": 0.0,
            "n": 125,
            "upper": 2.976
          },
          "tied_fraction": 0.136,
          "visit_better_fraction": 0.448
        }
      },
      "policy_entropy_quartile": {
        "1": {
          "n": 77,
          "normalized_store_margin_delta": {
            "lower": -0.0817099567099567,
            "mean": -0.02543290043290043,
            "median": -0.04166666666666663,
            "n": 77,
            "upper": 0.03409090909090909
          },
          "outcome_delta": {
            "lower": -0.24675324675324675,
            "mean": -0.025974025974025976,
            "median": 0.0,
            "n": 77,
            "upper": 0.22077922077922077
          },
          "q_better_fraction": 0.4025974025974026,
          "store_margin_delta": {
            "lower": -3.9220779220779223,
            "mean": -1.2207792207792207,
            "median": -2.0,
            "n": 77,
            "upper": 1.6363636363636365
          },
          "tied_fraction": 0.06493506493506493,
          "visit_better_fraction": 0.5324675324675324
        },
        "2": {
          "n": 77,
          "normalized_store_margin_delta": {
            "lower": -0.07142857142857142,
            "mean": -0.011904761904761906,
            "median": 0.0,
            "n": 77,
            "upper": 0.046536796536796536
          },
          "outcome_delta": {
            "lower": -0.16883116883116883,
            "mean": 0.06493506493506493,
            "median": 0.0,
            "n": 77,
            "upper": 0.2987012987012987
          },
          "q_better_fraction": 0.4025974025974026,
          "store_margin_delta": {
            "lower": -3.4285714285714284,
            "mean": -0.5714285714285714,
            "median": 0.0,
            "n": 77,
            "upper": 2.2337662337662336
          },
          "tied_fraction": 0.16883116883116883,
          "visit_better_fraction": 0.42857142857142855
        },
        "3": {
          "n": 66,
          "normalized_store_margin_delta": {
            "lower": -0.026515151515151523,
            "mean": 0.044191919191919185,
            "median": 0.0,
            "n": 66,
            "upper": 0.11931818181818181
          },
          "outcome_delta": {
            "lower": -0.13636363636363635,
            "mean": 0.09090909090909091,
            "median": 0.0,
            "n": 66,
            "upper": 0.3181818181818182
          },
          "q_better_fraction": 0.42424242424242425,
          "store_margin_delta": {
            "lower": -1.2727272727272727,
            "mean": 2.121212121212121,
            "median": 0.0,
            "n": 66,
            "upper": 5.7272727272727275
          },
          "tied_fraction": 0.18181818181818182,
          "visit_better_fraction": 0.3939393939393939
        },
        "4": {
          "n": 54,
          "normalized_store_margin_delta": {
            "lower": -0.0625,
            "mean": 0.00308641975308642,
            "median": 0.0,
            "n": 54,
            "upper": 0.06790123456790124
          },
          "outcome_delta": {
            "lower": -0.4074074074074074,
            "mean": -0.14814814814814814,
            "median": 0.0,
            "n": 54,
            "upper": 0.09259259259259259
          },
          "q_better_fraction": 0.4074074074074074,
          "store_margin_delta": {
            "lower": -3.0,
            "mean": 0.14814814814814814,
            "median": 0.0,
            "n": 54,
            "upper": 3.259259259259259
          },
          "tied_fraction": 0.16666666666666666,
          "visit_better_fraction": 0.42592592592592593
        }
      },
      "source_domain": {
        "additional_standard_start_selfplay": {
          "n": 70,
          "normalized_store_margin_delta": {
            "lower": -0.05892857142857143,
            "mean": -0.006547619047619043,
            "median": 0.0,
            "n": 70,
            "upper": 0.04583333333333334
          },
          "outcome_delta": {
            "lower": -0.12857142857142856,
            "mean": 0.0,
            "median": 0.0,
            "n": 70,
            "upper": 0.14285714285714285
          },
          "q_better_fraction": 0.3142857142857143,
          "store_margin_delta": {
            "lower": -2.8285714285714287,
            "mean": -0.3142857142857143,
            "median": 0.0,
            "n": 70,
            "upper": 2.2
          },
          "tied_fraction": 0.24285714285714285,
          "visit_better_fraction": 0.44285714285714284
        },
        "independent_opening_suite_diagnostic": {
          "n": 126,
          "normalized_store_margin_delta": {
            "lower": -0.04431216931216932,
            "mean": 0.01289682539682539,
            "median": 0.02083333333333333,
            "n": 126,
            "upper": 0.07208994708994708
          },
          "outcome_delta": {
            "lower": -0.2222222222222222,
            "mean": 0.0,
            "median": 0.0,
            "n": 126,
            "upper": 0.22242063492063202
          },
          "q_better_fraction": 0.5,
          "store_margin_delta": {
            "lower": -2.126984126984127,
            "mean": 0.6190476190476191,
            "median": 1.0,
            "n": 126,
            "upper": 3.4603174603174605
          },
          "tied_fraction": 0.047619047619047616,
          "visit_better_fraction": 0.4523809523809524
        },
        "pr176_standard_start_pilot": {
          "n": 78,
          "normalized_store_margin_delta": {
            "lower": -0.04914529914529914,
            "mean": -0.012286324786324788,
            "median": 0.0,
            "n": 78,
            "upper": 0.02458600427350409
          },
          "outcome_delta": {
            "lower": -0.14102564102564102,
            "mean": 0.01282051282051282,
            "median": 0.0,
            "n": 78,
            "upper": 0.16666666666666666
          },
          "q_better_fraction": 0.34615384615384615,
          "store_margin_delta": {
            "lower": -2.358974358974359,
            "mean": -0.5897435897435898,
            "median": 0.0,
            "n": 78,
            "upper": 1.1801282051281958
          },
          "tied_fraction": 0.20512820512820512,
          "visit_better_fraction": 0.44871794871794873
        }
      }
    },
    "768": {
      "phase": {
        "late": {
          "n": 43,
          "normalized_store_margin_delta": {
            "lower": -0.016472868217054258,
            "mean": 0.002906976744186053,
            "median": 0.0,
            "n": 43,
            "upper": 0.02131782945736435
          },
          "outcome_delta": {
            "lower": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "n": 43,
            "upper": 0.0
          },
          "q_better_fraction": 0.3023255813953488,
          "store_margin_delta": {
            "lower": -0.7906976744186046,
            "mean": 0.13953488372093023,
            "median": 0.0,
            "n": 43,
            "upper": 1.0232558139534884
          },
          "tied_fraction": 0.4418604651162791,
          "visit_better_fraction": 0.2558139534883721
        },
        "midgame": {
          "n": 50,
          "normalized_store_margin_delta": {
            "lower": -0.08416666666666667,
            "mean": -0.03916666666666666,
            "median": -0.041666666666666644,
            "n": 50,
            "upper": 0.005
          },
          "outcome_delta": {
            "lower": -0.2,
            "mean": -0.06,
            "median": 0.0,
            "n": 50,
            "upper": 0.08
          },
          "q_better_fraction": 0.28,
          "store_margin_delta": {
            "lower": -4.04,
            "mean": -1.88,
            "median": -2.0,
            "n": 50,
            "upper": 0.24
          },
          "tied_fraction": 0.18,
          "visit_better_fraction": 0.54
        },
        "opening": {
          "n": 181,
          "normalized_store_margin_delta": {
            "lower": -0.042587476979742175,
            "mean": 0.0036832412523020255,
            "median": 0.0,
            "n": 181,
            "upper": 0.04995395948434622
          },
          "outcome_delta": {
            "lower": -0.22665745856353578,
            "mean": -0.06077348066298342,
            "median": 0.0,
            "n": 181,
            "upper": 0.11049723756906077
          },
          "q_better_fraction": 0.43646408839779005,
          "store_margin_delta": {
            "lower": -2.044198895027624,
            "mean": 0.17679558011049723,
            "median": 0.0,
            "n": 181,
            "upper": 2.3977900552486187
          },
          "tied_fraction": 0.09392265193370165,
          "visit_better_fraction": 0.4696132596685083
        }
      },
      "player": {
        "0": {
          "n": 149,
          "normalized_store_margin_delta": {
            "lower": -0.05536912751677853,
            "mean": -0.012304250559284113,
            "median": 0.0,
            "n": 149,
            "upper": 0.032158836689038024
          },
          "outcome_delta": {
            "lower": -0.21476510067114093,
            "mean": -0.06040268456375839,
            "median": 0.0,
            "n": 149,
            "upper": 0.10067114093959731
          },
          "q_better_fraction": 0.3691275167785235,
          "store_margin_delta": {
            "lower": -2.6577181208053693,
            "mean": -0.5906040268456376,
            "median": 0.0,
            "n": 149,
            "upper": 1.5436241610738255
          },
          "tied_fraction": 0.19463087248322147,
          "visit_better_fraction": 0.436241610738255
        },
        "1": {
          "n": 125,
          "normalized_store_margin_delta": {
            "lower": -0.04066666666666666,
            "mean": 0.005333333333333333,
            "median": 0.0,
            "n": 125,
            "upper": 0.05333333333333334
          },
          "outcome_delta": {
            "lower": -0.216,
            "mean": -0.04,
            "median": 0.0,
            "n": 125,
            "upper": 0.128
          },
          "q_better_fraction": 0.408,
          "store_margin_delta": {
            "lower": -1.952,
            "mean": 0.256,
            "median": 0.0,
            "n": 125,
            "upper": 2.56
          },
          "tied_fraction": 0.128,
          "visit_better_fraction": 0.464
        }
      },
      "policy_entropy_quartile": {
        "1": {
          "n": 77,
          "normalized_store_margin_delta": {
            "lower": -0.08333333333333334,
            "mean": -0.025432900432900432,
            "median": -0.04166666666666663,
            "n": 77,
            "upper": 0.031939935064934866
          },
          "outcome_delta": {
            "lower": -0.36363636363636365,
            "mean": -0.14285714285714285,
            "median": 0.0,
            "n": 77,
            "upper": 0.07792207792207792
          },
          "q_better_fraction": 0.37662337662337664,
          "store_margin_delta": {
            "lower": -4.0,
            "mean": -1.2207792207792207,
            "median": -2.0,
            "n": 77,
            "upper": 1.5331168831168738
          },
          "tied_fraction": 0.09090909090909091,
          "visit_better_fraction": 0.5324675324675324
        },
        "2": {
          "n": 77,
          "normalized_store_margin_delta": {
            "lower": -0.08982683982683982,
            "mean": -0.02976190476190476,
            "median": 0.0,
            "n": 77,
            "upper": 0.030844155844155844
          },
          "outcome_delta": {
            "lower": -0.37662337662337664,
            "mean": -0.15584415584415584,
            "median": 0.0,
            "n": 77,
            "upper": 0.05194805194805195
          },
          "q_better_fraction": 0.38961038961038963,
          "store_margin_delta": {
            "lower": -4.311688311688312,
            "mean": -1.4285714285714286,
            "median": 0.0,
            "n": 77,
            "upper": 1.4805194805194806
          },
          "tied_fraction": 0.19480519480519481,
          "visit_better_fraction": 0.4155844155844156
        },
        "3": {
          "n": 66,
          "normalized_store_margin_delta": {
            "lower": -0.05618686868686869,
            "mean": 0.00883838383838384,
            "median": 0.0,
            "n": 66,
            "upper": 0.07828282828282829
          },
          "outcome_delta": {
            "lower": -0.19696969696969696,
            "mean": 0.045454545454545456,
            "median": 0.0,
            "n": 66,
            "upper": 0.30303030303030304
          },
          "q_better_fraction": 0.36363636363636365,
          "store_margin_delta": {
            "lower": -2.696969696969697,
            "mean": 0.42424242424242425,
            "median": 0.0,
            "n": 66,
            "upper": 3.757575757575758
          },
          "tied_fraction": 0.21212121212121213,
          "visit_better_fraction": 0.42424242424242425
        },
        "4": {
          "n": 54,
          "normalized_store_margin_delta": {
            "lower": -0.0308641975308642,
            "mean": 0.046296296296296294,
            "median": 0.0,
            "n": 54,
            "upper": 0.12268518518518519
          },
          "outcome_delta": {
            "lower": -0.14814814814814814,
            "mean": 0.1111111111111111,
            "median": 0.0,
            "n": 54,
            "upper": 0.37037037037037035
          },
          "q_better_fraction": 0.42592592592592593,
          "store_margin_delta": {
            "lower": -1.4814814814814814,
            "mean": 2.2222222222222223,
            "median": 0.0,
            "n": 54,
            "upper": 5.888888888888889
          },
          "tied_fraction": 0.16666666666666666,
          "visit_better_fraction": 0.4074074074074074
        }
      },
      "source_domain": {
        "additional_standard_start_selfplay": {
          "n": 70,
          "normalized_store_margin_delta": {
            "lower": -0.061309523809523814,
            "mean": -0.010119047619047618,
            "median": 0.0,
            "n": 70,
            "upper": 0.04166666666666666
          },
          "outcome_delta": {
            "lower": -0.24285714285714285,
            "mean": -0.07142857142857142,
            "median": 0.0,
            "n": 70,
            "upper": 0.08571428571428572
          },
          "q_better_fraction": 0.34285714285714286,
          "store_margin_delta": {
            "lower": -2.942857142857143,
            "mean": -0.4857142857142857,
            "median": 0.0,
            "n": 70,
            "upper": 2.0
          },
          "tied_fraction": 0.22857142857142856,
          "visit_better_fraction": 0.42857142857142855
        },
        "independent_opening_suite_diagnostic": {
          "n": 126,
          "normalized_store_margin_delta": {
            "lower": -0.04828042328042328,
            "mean": 0.011574074074074077,
            "median": 0.0,
            "n": 126,
            "upper": 0.07176752645502635
          },
          "outcome_delta": {
            "lower": -0.23015873015873015,
            "mean": -0.007936507936507936,
            "median": 0.0,
            "n": 126,
            "upper": 0.2222222222222222
          },
          "q_better_fraction": 0.4523809523809524,
          "store_margin_delta": {
            "lower": -2.3174603174603177,
            "mean": 0.5555555555555556,
            "median": 0.0,
            "n": 126,
            "upper": 3.4448412698412643
          },
          "tied_fraction": 0.0873015873015873,
          "visit_better_fraction": 0.4603174603174603
        },
        "pr176_standard_start_pilot": {
          "n": 78,
          "normalized_store_margin_delta": {
            "lower": -0.0641025641025641,
            "mean": -0.024572649572649572,
            "median": 0.0,
            "n": 78,
            "upper": 0.013888888888888893
          },
          "outcome_delta": {
            "lower": -0.23076923076923078,
            "mean": -0.10256410256410256,
            "median": 0.0,
            "n": 78,
            "upper": 0.01282051282051282
          },
          "q_better_fraction": 0.32051282051282054,
          "store_margin_delta": {
            "lower": -3.076923076923077,
            "mean": -1.1794871794871795,
            "median": 0.0,
            "n": 78,
            "upper": 0.6666666666666666
          },
          "tied_fraction": 0.23076923076923078,
          "visit_better_fraction": 0.44871794871794873
        }
      }
    }
  },
  "D768": {
    "1200": {
      "phase": {
        "late": {
          "n": 44,
          "normalized_store_margin_delta": {
            "lower": -0.02083333333333333,
            "mean": -0.0028409090909090867,
            "median": 0.0,
            "n": 44,
            "upper": 0.015151515151515154
          },
          "outcome_delta": {
            "lower": -0.06818181818181818,
            "mean": -0.022727272727272728,
            "median": 0.0,
            "n": 44,
            "upper": 0.0
          },
          "q_better_fraction": 0.22727272727272727,
          "store_margin_delta": {
            "lower": -1.0,
            "mean": -0.13636363636363635,
            "median": 0.0,
            "n": 44,
            "upper": 0.7272727272727273
          },
          "tied_fraction": 0.4772727272727273,
          "visit_better_fraction": 0.29545454545454547
        },
        "midgame": {
          "n": 56,
          "normalized_store_margin_delta": {
            "lower": -0.08928571428571429,
            "mean": -0.04389880952380953,
            "median": -0.020833333333333315,
            "n": 56,
            "upper": -1.9329775875177348e-18
          },
          "outcome_delta": {
            "lower": -0.03571428571428571,
            "mean": 0.08928571428571429,
            "median": 0.0,
            "n": 56,
            "upper": 0.23214285714285715
          },
          "q_better_fraction": 0.32142857142857145,
          "store_margin_delta": {
            "lower": -4.285714285714286,
            "mean": -2.107142857142857,
            "median": -1.0,
            "n": 56,
            "upper": 0.0
          },
          "tied_fraction": 0.17857142857142858,
          "visit_better_fraction": 0.5
        },
        "opening": {
          "n": 164,
          "normalized_store_margin_delta": {
            "lower": -0.013719512195121951,
            "mean": 0.035060975609756094,
            "median": 0.04166666666666666,
            "n": 164,
            "upper": 0.08257113821138211
          },
          "outcome_delta": {
            "lower": -0.06707317073170732,
            "mean": 0.10975609756097561,
            "median": 0.0,
            "n": 164,
            "upper": 0.2865853658536585
          },
          "q_better_fraction": 0.5182926829268293,
          "store_margin_delta": {
            "lower": -0.6585365853658537,
            "mean": 1.6829268292682926,
            "median": 2.0,
            "n": 164,
            "upper": 3.9634146341463414
          },
          "tied_fraction": 0.07317073170731707,
          "visit_better_fraction": 0.40853658536585363
        }
      },
      "player": {
        "0": {
          "n": 149,
          "normalized_store_margin_delta": {
            "lower": -0.03970917225950783,
            "mean": 0.0002796420581655456,
            "median": 0.0,
            "n": 149,
            "upper": 0.039709172259507826
          },
          "outcome_delta": {
            "lower": -0.06711409395973154,
            "mean": 0.0738255033557047,
            "median": 0.0,
            "n": 149,
            "upper": 0.2214765100671141
          },
          "q_better_fraction": 0.42953020134228187,
          "store_margin_delta": {
            "lower": -1.9060402684563758,
            "mean": 0.013422818791946308,
            "median": 0.0,
            "n": 149,
            "upper": 1.9060402684563758
          },
          "tied_fraction": 0.15436241610738255,
          "visit_better_fraction": 0.4161073825503356
        },
        "1": {
          "n": 115,
          "normalized_store_margin_delta": {
            "lower": -0.02391304347826087,
            "mean": 0.027173913043478257,
            "median": 0.0,
            "n": 115,
            "upper": 0.08007246376811593
          },
          "outcome_delta": {
            "lower": -0.09565217391304348,
            "mean": 0.09565217391304348,
            "median": 0.0,
            "n": 115,
            "upper": 0.2782608695652174
          },
          "q_better_fraction": 0.4260869565217391,
          "store_margin_delta": {
            "lower": -1.1478260869565218,
            "mean": 1.3043478260869565,
            "median": 0.0,
            "n": 115,
            "upper": 3.8434782608695652
          },
          "tied_fraction": 0.17391304347826086,
          "visit_better_fraction": 0.4
        }
      },
      "policy_entropy_quartile": {
        "1": {
          "n": 64,
          "normalized_store_margin_delta": {
            "lower": -0.08854166666666667,
            "mean": -0.03125000000000001,
            "median": -0.020833333333333315,
            "n": 64,
            "upper": 0.02473958333333333
          },
          "outcome_delta": {
            "lower": -0.3125,
            "mean": -0.078125,
            "median": 0.0,
            "n": 64,
            "upper": 0.15625
          },
          "q_better_fraction": 0.421875,
          "store_margin_delta": {
            "lower": -4.25,
            "mean": -1.5,
            "median": -1.0,
            "n": 64,
            "upper": 1.1875
          },
          "tied_fraction": 0.078125,
          "visit_better_fraction": 0.5
        },
        "2": {
          "n": 77,
          "normalized_store_margin_delta": {
            "lower": -0.06439393939393939,
            "mean": -0.0021645021645021628,
            "median": 0.0,
            "n": 77,
            "upper": 0.05952380952380951
          },
          "outcome_delta": {
            "lower": -0.07792207792207792,
            "mean": 0.12987012987012986,
            "median": 0.0,
            "n": 77,
            "upper": 0.33766233766233766
          },
          "q_better_fraction": 0.38961038961038963,
          "store_margin_delta": {
            "lower": -3.090909090909091,
            "mean": -0.1038961038961039,
            "median": 0.0,
            "n": 77,
            "upper": 2.857142857142857
          },
          "tied_fraction": 0.19480519480519481,
          "visit_better_fraction": 0.4155844155844156
        },
        "3": {
          "n": 71,
          "normalized_store_margin_delta": {
            "lower": -0.00999119718309859,
            "mean": 0.05457746478873239,
            "median": 0.0,
            "n": 71,
            "upper": 0.12265258215962442
          },
          "outcome_delta": {
            "lower": -0.014084507042253521,
            "mean": 0.18309859154929578,
            "median": 0.0,
            "n": 71,
            "upper": 0.39436619718309857
          },
          "q_better_fraction": 0.43661971830985913,
          "store_margin_delta": {
            "lower": -0.4795774647887317,
            "mean": 2.619718309859155,
            "median": 0.0,
            "n": 71,
            "upper": 5.887323943661972
          },
          "tied_fraction": 0.19718309859154928,
          "visit_better_fraction": 0.36619718309859156
        },
        "4": {
          "n": 52,
          "normalized_store_margin_delta": {
            "lower": -0.039262820512820526,
            "mean": 0.0280448717948718,
            "median": 0.0,
            "n": 52,
            "upper": 0.09294871794871794
          },
          "outcome_delta": {
            "lower": -0.19230769230769232,
            "mean": 0.07692307692307693,
            "median": 0.0,
            "n": 52,
            "upper": 0.34615384615384615
          },
          "q_better_fraction": 0.4807692307692308,
          "store_margin_delta": {
            "lower": -1.8846153846153846,
            "mean": 1.3461538461538463,
            "median": 0.0,
            "n": 52,
            "upper": 4.461538461538462
          },
          "tied_fraction": 0.17307692307692307,
          "visit_better_fraction": 0.34615384615384615
        }
      },
      "source_domain": {
        "additional_standard_start_selfplay": {
          "n": 66,
          "normalized_store_margin_delta": {
            "lower": -0.10227272727272728,
            "mean": -0.054924242424242424,
            "median": 0.0,
            "n": 66,
            "upper": -0.008838383838383838
          },
          "outcome_delta": {
            "lower": -0.19696969696969696,
            "mean": -0.06060606060606061,
            "median": 0.0,
            "n": 66,
            "upper": 0.07575757575757576
          },
          "q_better_fraction": 0.25757575757575757,
          "store_margin_delta": {
            "lower": -4.909090909090909,
            "mean": -2.6363636363636362,
            "median": 0.0,
            "n": 66,
            "upper": -0.42424242424242425
          },
          "tied_fraction": 0.2727272727272727,
          "visit_better_fraction": 0.4696969696969697
        },
        "independent_opening_suite_diagnostic": {
          "n": 119,
          "normalized_store_margin_delta": {
            "lower": -0.011204481792717085,
            "mean": 0.04796918767507002,
            "median": 0.04166666666666667,
            "n": 119,
            "upper": 0.10854341736694675
          },
          "outcome_delta": {
            "lower": -0.06722689075630252,
            "mean": 0.15126050420168066,
            "median": 0.0,
            "n": 119,
            "upper": 0.3697478991596639
          },
          "q_better_fraction": 0.5462184873949579,
          "store_margin_delta": {
            "lower": -0.5378151260504201,
            "mean": 2.302521008403361,
            "median": 2.0,
            "n": 119,
            "upper": 5.2100840336134455
          },
          "tied_fraction": 0.058823529411764705,
          "visit_better_fraction": 0.3949579831932773
        },
        "pr176_standard_start_pilot": {
          "n": 79,
          "normalized_store_margin_delta": {
            "lower": -0.020569620253164556,
            "mean": 0.013713080168776374,
            "median": 0.0,
            "n": 79,
            "upper": 0.0479957805907173
          },
          "outcome_delta": {
            "lower": -0.02531645569620253,
            "mean": 0.10126582278481013,
            "median": 0.0,
            "n": 79,
            "upper": 0.24050632911392406
          },
          "q_better_fraction": 0.3924050632911392,
          "store_margin_delta": {
            "lower": -0.9873417721518988,
            "mean": 0.6582278481012658,
            "median": 0.0,
            "n": 79,
            "upper": 2.3037974683544302
          },
          "tied_fraction": 0.22784810126582278,
          "visit_better_fraction": 0.379746835443038
        }
      }
    },
    "768": {
      "phase": {
        "late": {
          "n": 44,
          "normalized_store_margin_delta": {
            "lower": -0.019886363636363636,
            "mean": -0.0009469696969696914,
            "median": 0.0,
            "n": 44,
            "upper": 0.017992424242424247
          },
          "outcome_delta": {
            "lower": -0.06818181818181818,
            "mean": -0.022727272727272728,
            "median": 0.0,
            "n": 44,
            "upper": 0.0
          },
          "q_better_fraction": 0.29545454545454547,
          "store_margin_delta": {
            "lower": -0.9545454545454546,
            "mean": -0.045454545454545456,
            "median": 0.0,
            "n": 44,
            "upper": 0.8636363636363636
          },
          "tied_fraction": 0.4318181818181818,
          "visit_better_fraction": 0.2727272727272727
        },
        "midgame": {
          "n": 56,
          "normalized_store_margin_delta": {
            "lower": -0.109375,
            "mean": -0.06547619047619048,
            "median": -0.020833333333333315,
            "n": 56,
            "upper": -0.023809523809523808
          },
          "outcome_delta": {
            "lower": -0.14285714285714285,
            "mean": 0.017857142857142856,
            "median": 0.0,
            "n": 56,
            "upper": 0.16071428571428573
          },
          "q_better_fraction": 0.25,
          "store_margin_delta": {
            "lower": -5.25,
            "mean": -3.142857142857143,
            "median": -1.0,
            "n": 56,
            "upper": -1.1428571428571428
          },
          "tied_fraction": 0.25,
          "visit_better_fraction": 0.5
        },
        "opening": {
          "n": 164,
          "normalized_store_margin_delta": {
            "lower": -0.027184959349593498,
            "mean": 0.019817073170731708,
            "median": 0.0,
            "n": 164,
            "upper": 0.06859756097560975
          },
          "outcome_delta": {
            "lower": -0.1951219512195122,
            "mean": -0.018292682926829267,
            "median": 0.0,
            "n": 164,
            "upper": 0.16463414634146342
          },
          "q_better_fraction": 0.4573170731707317,
          "store_margin_delta": {
            "lower": -1.3048780487804879,
            "mean": 0.9512195121951219,
            "median": 0.0,
            "n": 164,
            "upper": 3.292682926829268
          },
          "tied_fraction": 0.08536585365853659,
          "visit_better_fraction": 0.4573170731707317
        }
      },
      "player": {
        "0": {
          "n": 149,
          "normalized_store_margin_delta": {
            "lower": -0.06404502237136465,
            "mean": -0.020693512304250556,
            "median": 0.0,
            "n": 149,
            "upper": 0.0220917225950783
          },
          "outcome_delta": {
            "lower": -0.16778523489932887,
            "mean": -0.006711409395973154,
            "median": 0.0,
            "n": 149,
            "upper": 0.15436241610738255
          },
          "q_better_fraction": 0.348993288590604,
          "store_margin_delta": {
            "lower": -3.0741610738255027,
            "mean": -0.9932885906040269,
            "median": 0.0,
            "n": 149,
            "upper": 1.0604026845637584
          },
          "tied_fraction": 0.20134228187919462,
          "visit_better_fraction": 0.44966442953020136
        },
        "1": {
          "n": 115,
          "normalized_store_margin_delta": {
            "lower": -0.024275362318840577,
            "mean": 0.022826086956521736,
            "median": 0.0,
            "n": 115,
            "upper": 0.07101449275362319
          },
          "outcome_delta": {
            "lower": -0.19130434782608696,
            "mean": -0.017391304347826087,
            "median": 0.0,
            "n": 115,
            "upper": 0.1565217391304348
          },
          "q_better_fraction": 0.43478260869565216,
          "store_margin_delta": {
            "lower": -1.1652173913043478,
            "mean": 1.0956521739130434,
            "median": 0.0,
            "n": 115,
            "upper": 3.408695652173913
          },
          "tied_fraction": 0.14782608695652175,
          "visit_better_fraction": 0.41739130434782606
        }
      },
      "policy_entropy_quartile": {
        "1": {
          "n": 64,
          "normalized_store_margin_delta": {
            "lower": -0.077490234375,
            "mean": -0.016276041666666678,
            "median": -0.04166666666666663,
            "n": 64,
            "upper": 0.04687499999999999
          },
          "outcome_delta": {
            "lower": -0.359375,
            "mean": -0.140625,
            "median": 0.0,
            "n": 64,
            "upper": 0.078125
          },
          "q_better_fraction": 0.375,
          "store_margin_delta": {
            "lower": -3.7195312499999993,
            "mean": -0.78125,
            "median": -2.0,
            "n": 64,
            "upper": 2.25
          },
          "tied_fraction": 0.109375,
          "visit_better_fraction": 0.515625
        },
        "2": {
          "n": 77,
          "normalized_store_margin_delta": {
            "lower": -0.09523809523809522,
            "mean": -0.0367965367965368,
            "median": 0.0,
            "n": 77,
            "upper": 0.022186147186147188
          },
          "outcome_delta": {
            "lower": -0.2987012987012987,
            "mean": -0.1038961038961039,
            "median": 0.0,
            "n": 77,
            "upper": 0.09090909090909091
          },
          "q_better_fraction": 0.37662337662337664,
          "store_margin_delta": {
            "lower": -4.571428571428571,
            "mean": -1.7662337662337662,
            "median": 0.0,
            "n": 77,
            "upper": 1.0649350649350648
          },
          "tied_fraction": 0.15584415584415584,
          "visit_better_fraction": 0.4675324675324675
        },
        "3": {
          "n": 71,
          "normalized_store_margin_delta": {
            "lower": -0.03755868544600938,
            "mean": 0.022300469483568067,
            "median": 0.0,
            "n": 71,
            "upper": 0.08509389671361503
          },
          "outcome_delta": {
            "lower": -0.14084507042253522,
            "mean": 0.08450704225352113,
            "median": 0.0,
            "n": 71,
            "upper": 0.30985915492957744
          },
          "q_better_fraction": 0.36619718309859156,
          "store_margin_delta": {
            "lower": -1.8028169014084507,
            "mean": 1.0704225352112675,
            "median": 0.0,
            "n": 71,
            "upper": 4.084507042253521
          },
          "tied_fraction": 0.2535211267605634,
          "visit_better_fraction": 0.38028169014084506
        },
        "4": {
          "n": 52,
          "normalized_store_margin_delta": {
            "lower": -0.03685897435897435,
            "mean": 0.035256410256410256,
            "median": 0.0,
            "n": 52,
            "upper": 0.10657051282051283
          },
          "outcome_delta": {
            "lower": -0.15384615384615385,
            "mean": 0.15384615384615385,
            "median": 0.0,
            "n": 52,
            "upper": 0.46153846153846156
          },
          "q_better_fraction": 0.4423076923076923,
          "store_margin_delta": {
            "lower": -1.7692307692307692,
            "mean": 1.6923076923076923,
            "median": 0.0,
            "n": 52,
            "upper": 5.115384615384615
          },
          "tied_fraction": 0.19230769230769232,
          "visit_better_fraction": 0.36538461538461536
        }
      },
      "source_domain": {
        "additional_standard_start_selfplay": {
          "n": 66,
          "normalized_store_margin_delta": {
            "lower": -0.09722222222222222,
            "mean": -0.047979797979797977,
            "median": 0.0,
            "n": 66,
            "upper": 0.0012626262626262606
          },
          "outcome_delta": {
            "lower": -0.2727272727272727,
            "mean": -0.12121212121212122,
            "median": 0.0,
            "n": 66,
            "upper": 0.030303030303030304
          },
          "q_better_fraction": 0.2878787878787879,
          "store_margin_delta": {
            "lower": -4.666666666666667,
            "mean": -2.303030303030303,
            "median": 0.0,
            "n": 66,
            "upper": 0.06060606060606061
          },
          "tied_fraction": 0.25757575757575757,
          "visit_better_fraction": 0.45454545454545453
        },
        "independent_opening_suite_diagnostic": {
          "n": 119,
          "normalized_store_margin_delta": {
            "lower": -0.03571428571428572,
            "mean": 0.024159663865546212,
            "median": 0.0,
            "n": 119,
            "upper": 0.08403361344537816
          },
          "outcome_delta": {
            "lower": -0.17647058823529413,
            "mean": 0.05042016806722689,
            "median": 0.0,
            "n": 119,
            "upper": 0.2773109243697479
          },
          "q_better_fraction": 0.46218487394957986,
          "store_margin_delta": {
            "lower": -1.7142857142857142,
            "mean": 1.1596638655462186,
            "median": 0.0,
            "n": 119,
            "upper": 4.033613445378151
          },
          "tied_fraction": 0.08403361344537816,
          "visit_better_fraction": 0.453781512605042
        },
        "pr176_standard_start_pilot": {
          "n": 79,
          "normalized_store_margin_delta": {
            "lower": -0.03691983122362868,
            "mean": -0.002109704641350209,
            "median": 0.0,
            "n": 79,
            "upper": 0.03217299578059073
          },
          "outcome_delta": {
            "lower": -0.1518987341772152,
            "mean": -0.012658227848101266,
            "median": 0.0,
            "n": 79,
            "upper": 0.12658227848101267
          },
          "q_better_fraction": 0.35443037974683544,
          "store_margin_delta": {
            "lower": -1.7721518987341771,
            "mean": -0.10126582278481013,
            "median": 0.0,
            "n": 79,
            "upper": 1.5443037974683544
          },
          "tied_fraction": 0.25316455696202533,
          "visit_better_fraction": 0.3924050632911392
        }
      }
    }
  },
  "minimum_interpretable_disagreement_states": 32
}
```

## Exact Classification Criteria

D768 requires >=64 disagreements, positive direct Q-minus-visit margin with lower 95% CI >=0 at both continuations, and positive paired concordance and regret-difference lower CIs.

Per-state traces remain in the workdir. No model training, replay generation, or runtime tuning was performed.
