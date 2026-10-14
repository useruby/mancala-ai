# Seed479 full-population versus minibatch normalized-gradient ablation

Exploratory only: the existing evaluation cohort and seed478 findings informed the protocol. This is not fresh confirmation.

M divides each minibatch fresh-policy numerator by that minibatch's fresh policy mass; F divides the full fresh training-policy numerator by the complete fresh policy mass.

Fresh training occurrences: 60015; denominator: 60015.

## Endpoint metrics

```json
{
  "endpoints": {
    "F": {
      "fresh": {
        "ce": {
          "equal_input": 1.2148886635736442,
          "exposure": 1.2122353367302963
        },
        "mse": {
          "equal_input": 0.47434775345576374,
          "exposure": 0.4740148275760148
        }
      },
      "pooled": {
        "ce": {
          "equal_input": 1.1913837333352568,
          "exposure": 1.1633279055925365
        },
        "mse": {
          "equal_input": 0.43550887254116477,
          "exposure": 0.3806659104008082
        }
      }
    },
    "M": {
      "fresh": {
        "ce": {
          "equal_input": 1.2252092014560414,
          "exposure": 1.2228101046003839
        },
        "mse": {
          "equal_input": 0.4817279819895144,
          "exposure": 0.48150311569539267
        }
      },
      "pooled": {
        "ce": {
          "equal_input": 1.1950317793763114,
          "exposure": 1.1602901679766802
        },
        "mse": {
          "equal_input": 0.4388648627926346,
          "exposure": 0.3801242145944374
        }
      }
    },
    "initializer": {
      "fresh": {
        "ce": {
          "equal_input": 1.2230564645173767,
          "exposure": 1.2211020997932704
        },
        "mse": {
          "equal_input": 0.4797530627560058,
          "exposure": 0.4799264142267028
        }
      },
      "pooled": {
        "ce": {
          "equal_input": 1.1981604508755597,
          "exposure": 1.170699160830002
        },
        "mse": {
          "equal_input": 0.4403286732584311,
          "exposure": 0.3857784073757292
        }
      }
    }
  },
  "initializer": {
    "fresh": {
      "ce": {
        "equal_input": 1.2230564645173767,
        "exposure": 1.2211020997932704
      },
      "mse": {
        "equal_input": 0.4797530627560058,
        "exposure": 0.4799264142267028
      }
    },
    "pooled": {
      "ce": {
        "equal_input": 1.1981604508755597,
        "exposure": 1.170699160830002
      },
      "mse": {
        "equal_input": 0.4403286732584311,
        "exposure": 0.3857784073757292
      }
    }
  },
  "training_policy_ce": {
    "F": 0.9492128445556364,
    "M": 0.9665001512339924,
    "initializer": 0.9895806899605396
  }
}
```

## Gates

- PASS — F_fresh_CE_improves_initializer_exposure_by_0.001
- PASS — F_fresh_CE_improves_M_exposure_by_0.001
- PASS — F_pooled_CE_regression_exposure_at_most_0.005
- PASS — F_pooled_MSE_regression_exposure_at_most_0.01
- PASS — F_fresh_CE_improves_initializer_equal_input_by_0.001
- PASS — F_fresh_CE_improves_M_equal_input_by_0.001
- PASS — F_pooled_CE_regression_equal_input_at_most_0.005
- PASS — F_pooled_MSE_regression_equal_input_at_most_0.01
- PASS — F_full_training_policy_CE_decreases

Decision: `advance_to_fresh_confirmation`.

No search, game, runtime export, promotion, or strength claim is made. Passing authorizes only a separately designed fresh confirmation.
