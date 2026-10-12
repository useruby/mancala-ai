# Seed464 dependency-closure audit

**Classification:** `invalid_attempt_dependency_closure_incomplete`
**Registered decision:** none. The original endpoint gate arithmetic is preserved as descriptive evidence only.

## Requested dependencies

| Dependency path | Registered source path / SHA256 | Frozen expected input SHA256 | Actual SHA256 | Enforced by executed loader |
|---|---|---|---|---|
| `docs/data/seed416-policy-target-softening/registration-v3.json` | `ml/alphazero_lite/seed435_adam_direction.py` / `c153c859f430c1eb857e2176de9df3dd5d99590c3838324bd55eef7fb4df3d81` | `4839ed63d6a48945ea11ea4995570f11118adbd65c027f49c3c5f04066bbd066` | `4839ed63d6a48945ea11ea4995570f11118adbd65c027f49c3c5f04066bbd066` | `seed435_adam_direction.py:98-103` |
| `docs/data/seed426-canonical-overlap/seed427-evaluation-verification-receipt.json` | `ml/alphazero_lite/seed435_adam_direction.py` / `c153c859f430c1eb857e2176de9df3dd5d99590c3838324bd55eef7fb4df3d81` | `fbcaff1e183b2247673dd7f301c410e11524d13bbe47e067079ee8947aa60ee1` | `fbcaff1e183b2247673dd7f301c410e11524d13bbe47e067079ee8947aa60ee1` | `seed435_adam_direction.py:33-40,109-111` |
| `docs/data/seed426-canonical-overlap/seed427-supplemental-verification-receipt.json` | `ml/alphazero_lite/seed435_adam_direction.py` / `c153c859f430c1eb857e2176de9df3dd5d99590c3838324bd55eef7fb4df3d81` | `1dd0f8e2442a32654b29c8969c1300cbbfecb83f84809d2a478290e0bda9209e` | `1dd0f8e2442a32654b29c8969c1300cbbfecb83f84809d2a478290e0bda9209e` | `seed435_adam_direction.py:41-45,109-111` |

The named registration and receipt files are transitively bound: the frozen seed435 loader contains their expected SHA256 values, its exact source SHA256 is registered by seed464, and `_inputs()` checks each file. Their actual hashes match. They are absent from seed464's direct input inventory, which explains the original relocated-bundle failure but is not itself an identity failure.

## Other consumed replay dependencies

| Derivative path | Frozen expected SHA256 (seed416 registration) | Actual SHA256 | Direct seed464 input binding | Loader digest check |
|---|---|---|---|---|
| `.tmp/seed416-policy-target-softening/derivatives/A/fresh.jsonl` | `5c089a5c8f2ba1fc04529631eb15d5a8ef9b49eb273fefbc0444ca3fd85a7049` | `5c089a5c8f2ba1fc04529631eb15d5a8ef9b49eb273fefbc0444ca3fd85a7049` | `None` | none; consumed by `train.load_jsonl_replay` via seed435 `_inputs()` |
| `.tmp/seed416-policy-target-softening/derivatives/A/generic_bootstrap.jsonl` | `6996657767cd75801967a67bdc0574012467743d31bccb1fff8f039dbe7f3da7` | `6996657767cd75801967a67bdc0574012467743d31bccb1fff8f039dbe7f3da7` | `None` | none; consumed by `train.load_jsonl_replay` via seed435 `_inputs()` |
| `.tmp/seed416-policy-target-softening/derivatives/A/random_teacher.jsonl` | `1172e2ad6da263990de3546cfc7f04338440f82a02aedd49b92a58e68faeea7d` | `1172e2ad6da263990de3546cfc7f04338440f82a02aedd49b92a58e68faeea7d` | `None` | none; consumed by `train.load_jsonl_replay` via seed435 `_inputs()` |
| `.tmp/seed416-policy-target-softening/derivatives/A/opening_disagreement.jsonl` | `0218563175aeb94315c7086510557c7490152af8a69c6f5fe9e7a04217f10055` | `0218563175aeb94315c7086510557c7490152af8a69c6f5fe9e7a04217f10055` | `None` | none; consumed by `train.load_jsonl_replay` via seed435 `_inputs()` |
| `.tmp/seed416-policy-target-softening/derivatives/A/stability.jsonl` | `81ae668eb6a9a5b34f073ca544df15d8e6d2f86659c2df3f203791d99c7a93c6` | `81ae668eb6a9a5b34f073ca544df15d8e6d2f86659c2df3f203791d99c7a93c6` | `None` | none; consumed by `train.load_jsonl_replay` via seed435 `_inputs()` |

All five current derivative files match the digest recorded in seed416 registration-v3. However, seed435 `_inputs()` does not enforce those derivative digests before passing the absolute derivative paths to `train.load_jsonl_replay`. The five canonical source JSONL.gz hashes in seed464's input inventory do not bind the distinct transformed derivative bytes actually loaded.

## Executable helper source closure

| Imported helper source | Prospective expected SHA256 chain | Actual SHA256 | seed464 source hash | Runtime comparison |
|---|---|---|---|---|
| `ml/alphazero_lite/verify_seed434_census_source.py` | `4b632366f7b1ad920b77c692c0b9993de5420b723deba71bc9d081522789ec51` via seed462 registration.json, itself directly input-hash-bound by seed464 | `4b632366f7b1ad920b77c692c0b9993de5420b723deba71bc9d081522789ec51` | `None` | none |
| `ml/alphazero_lite/seed429_policy_normalization.py` | `b2743afc7bc4acdadff9e25fc456f8dc3757dfd0c98c9bc214aee805d70011d6` via seed462 registration.json, itself directly input-hash-bound by seed464 | `b2743afc7bc4acdadff9e25fc456f8dc3757dfd0c98c9bc214aee805d70011d6` | `None` | none |
| `ml/alphazero_lite/seed432_policy_target_census.py` | `None` via None | `7b8dc48bdc1cf53d8dcfc53446e37edb2f84fdd583009d0693cc43a8e97204eb` | `None` | none |
| `ml/alphazero_lite/seed433_census_correction.py` | `None` via None | `3b4b7cf7387ade4fc5598f1360059a2b19418bbecd46e7020a1f5808746567eb` | `None` | none |
| `ml/alphazero_lite/exact_root_policy_targets.py` | `None` via None | `2aca4c1e2e9e9e199fb1d951b143c506785c23121531e26628a59c9afaf79898` | `None` | none |
| `ml/alphazero_lite/build_opening_suite.py` | `5c15d541574905ff9d7c75bc8daf8813311e22f22f220563e18cb9b85fb8bd87` via seed416 registration-v3 source_hashes, whose file SHA is enforced by registered seed435 source | `5c15d541574905ff9d7c75bc8daf8813311e22f22f220563e18cb9b85fb8bd87` | `None` | none |
| `ml/alphazero_lite/kalah_rules.py` | `6e5338ef9a85b690f51b6d0d184c8bc0f54c2b28ac084a4a17a88f9f552ad036` via seed416 registration-v3 source_hashes, whose file SHA is enforced by registered seed435 source | `6e5338ef9a85b690f51b6d0d184c8bc0f54c2b28ac084a4a17a88f9f552ad036` | `None` | none |

The census helper imports additional target/census helpers. `seed432_policy_target_census.py`, `seed433_census_correction.py`, and `exact_root_policy_targets.py` have no prospective source hash in seed464, seed462, or seed416 inventories. seed464 also does not check the nested seed462/seed416 source commitments against live imported helper files. This leaves the executed source closure incomplete.

## Descriptive outcomes (not registered decisions)

```json
{
  "arms": {
    "C": {
      "fresh": {
        "equal_input": {
          "policy_ce": 1.24463951587677,
          "value_mse": 0.7304377555847168
        },
        "exposure": {
          "policy_ce": 1.2439298629760742,
          "value_mse": 0.7251620292663574
        },
        "occurrence_count": 859,
        "unique_identities": 851
      },
      "pooled": {
        "equal_input": {
          "policy_ce": 1.2397260665893555,
          "value_mse": 0.6266827583312988
        },
        "exposure": {
          "policy_ce": 1.2343672513961792,
          "value_mse": 0.48848193883895874
        },
        "occurrence_count": 2607,
        "unique_identities": 1242
      },
      "training_objective": {
        "policy_denominator": 60015.0,
        "policy_numerator": 56411.449911238626,
        "total": 1.003192162028282,
        "value_denominator": 134502,
        "value_numerator": 28351.371320998584
      }
    },
    "T": {
      "fresh": {
        "equal_input": {
          "policy_ce": 1.2438956499099731,
          "value_mse": 0.7095920443534851
        },
        "exposure": {
          "policy_ce": 1.2430698871612549,
          "value_mse": 0.7037367820739746
        },
        "occurrence_count": 859,
        "unique_identities": 851
      },
      "pooled": {
        "equal_input": {
          "policy_ce": 1.2393710613250732,
          "value_mse": 0.6213906407356262
        },
        "exposure": {
          "policy_ce": 1.2344416379928589,
          "value_mse": 0.503898024559021
        },
        "occurrence_count": 2607,
        "unique_identities": 1242
      },
      "training_objective": {
        "policy_denominator": 60015.0,
        "policy_numerator": 56468.628125594,
        "total": 0.9701944542289608,
        "value_denominator": 134502,
        "value_numerator": 13130.031116155304
      }
    },
    "initializer": {
      "fresh": {
        "equal_input": {
          "policy_ce": 1.223056435585022,
          "value_mse": 0.7242156863212585
        },
        "exposure": {
          "policy_ce": 1.2211021184921265,
          "value_mse": 0.7197046875953674
        },
        "occurrence_count": 859,
        "unique_identities": 851
      },
      "pooled": {
        "equal_input": {
          "policy_ce": 1.1981604099273682,
          "value_mse": 0.6169980764389038
        },
        "exposure": {
          "policy_ce": 1.170699119567871,
          "value_mse": 0.4764106571674347
        },
        "occurrence_count": 2607,
        "unique_identities": 1242
      },
      "training_objective": {
        "policy_denominator": 60015.0,
        "policy_numerator": 59389.6851071585,
        "total": 1.052280466113173,
        "value_denominator": 134502,
        "value_numerator": 28110.817646422067
      }
    }
  },
  "effects_T_minus_C": {
    "fresh": {
      "equal_input": {
        "policy_ce": -0.000743865966796875,
        "value_mse": -0.02084571123123169
      },
      "exposure": {
        "policy_ce": -0.0008599758148193359,
        "value_mse": -0.021425247192382812
      }
    },
    "pooled": {
      "equal_input": {
        "policy_ce": -0.00035500526428222656,
        "value_mse": -0.005292117595672607
      },
      "exposure": {
        "policy_ce": 7.43865966796875e-05,
        "value_mse": 0.015416085720062256
      }
    }
  },
  "gates_as_recorded_descriptively": {
    "exact_control": true,
    "fresh_mse_equal_input": true,
    "fresh_mse_exposure": true,
    "fresh_vs_C_equal_input": false,
    "fresh_vs_C_exposure": false,
    "fresh_vs_initializer_equal_input": false,
    "fresh_vs_initializer_exposure": false,
    "pooled_ce_equal_input": false,
    "pooled_ce_exposure": false,
    "pooled_mse_equal_input": false,
    "pooled_mse_exposure": false,
    "training_objective_decreases": true
  }
}
```

## Required before a separately authorized run

- Bind every derivative JSONL byte hash actually consumed, or reconstruct derivatives into the registered bundle and verify exact bytes before loading.
- Freeze and hash the complete recursive executable source closure, including verify_seed434_census_source.py and its imported policy/census helpers; check those hashes at runtime.
- Include seed416 registration-v3.json and both seed427 receipts in the explicit input inventory and physically relocated bundle, even though seed435 currently transitively commits and verifies their identities.
- Re-run the complete registration-to-publication workflow tests before creating a new immutable registration; do not alter this attempt's freeze or regenerate its outputs.
