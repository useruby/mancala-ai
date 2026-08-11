# AlphaZero-Lite Distribution-Aligned Self-Play Results

- Classification: `distribution_alignment_preflight_failed`
- Stop reasons: `nearest_distance_median_improvement_below_20pct`
- Frozen corpus SHA256: `8922c6cbda4e23184b81164c36880e5837b6a1693cb27168355d6f87b97d709a`
- `ply2_unavailable_due_frozen_eval_exhaustion`: `true`

## Structural Depth Availability

| Ply | Raw prefixes | Unique states | Duplicate prefixes | Excluded state | Excluded canonical | Excluded alternate | Eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 35 | 35 | 0 | 35 | 35 | 35 | 0 |
| 4 | 942 | 942 | 0 | 675 | 675 | 675 | 267 |
| 6 | 23233 | 23115 | 118 | 1984 | 1984 | 7 | 21131 |

## Compact Record

```json
{
  "classification": "distribution_alignment_preflight_failed",
  "current_weights_sha256": "8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a",
  "opening_depth_feasibility": {
    "depths": {
      "2": {
        "duplicate_transposed_prefixes": 0,
        "eligible_after_all_exclusions": 0,
        "excluded_by_alternate_prefix": 35,
        "excluded_by_canonical_prefix": 35,
        "excluded_by_exact_evaluation_state": 35,
        "raw_legal_prefixes": 35,
        "unique_resulting_states": 35
      },
      "4": {
        "duplicate_transposed_prefixes": 0,
        "eligible_after_all_exclusions": 267,
        "excluded_by_alternate_prefix": 675,
        "excluded_by_canonical_prefix": 675,
        "excluded_by_exact_evaluation_state": 675,
        "raw_legal_prefixes": 942,
        "unique_resulting_states": 942
      },
      "6": {
        "duplicate_transposed_prefixes": 118,
        "eligible_after_all_exclusions": 21131,
        "excluded_by_alternate_prefix": 7,
        "excluded_by_canonical_prefix": 1984,
        "excluded_by_exact_evaluation_state": 1984,
        "raw_legal_prefixes": 23233,
        "unique_resulting_states": 23115
      }
    },
    "exclusion_suite_hashes": {
      "/tmp/azlite_budget_tactical_confirmation/suites/heldout_seed49_large.jsonl": "9d45df32f023e5e9a7ba12d72f3f467a2ce49399b38357e17f0bbc68e008111b",
      "/tmp/azlite_budget_tactical_confirmation/suites/heldout_seed50_large.jsonl": "7f58f98df175a707a00ca42b21ef7625669f95b3b788cfedcd428c91b0edd857",
      "/tmp/azlite_budget_tactical_confirmation/suites/heldout_seed51_large.jsonl": "33fdd0f619bef908968eb16711a8d5b448da5b8b5ea7e7166be0293dab8ef943",
      "/tmp/azlite_budget_tactical_confirmation/suites/heldout_seed52_large.jsonl": "161e7f440d879ff095ea717ce66e4bdb3d3d88a48c0ff61131a1979ce610fbb1",
      "/tmp/azlite_budget_tactical_confirmation/suites/heldout_seed53_large.jsonl": "978154ee6f5f9fd47c3ebf199d61f665505a528efc33d7f375541b3dac9e65b2",
      "/tmp/azlite_budget_tactical_confirmation/suites/heldout_seed54_large.jsonl": "968a2b89093f824c8309e1227f7ad725c5d8a7e5c1ee373d4e137c8261895f9b",
      "/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed43_large.jsonl": "5e0ed96ba56f99318c32a139692309759659da16a0a98c590e35f7e2496cade9",
      "/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed44_large.jsonl": "323f7abec32fb00d3b7b6b153ddd5692435755a3bc5340d8aa5aec32ca639620",
      "/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed45_large.jsonl": "ca72c8b7fe1adf7229f81183321475033f8548dcdeb1c71fd16c85c35ce89cda",
      "/tmp/azlite_opening_suite/large_eval.jsonl": "ebc86f053a8a4c12f3e937b5acfe37df7b0d2f1f0002969dcac077a9f99d65a4",
      "/tmp/azlite_opening_suite/medium_eval.jsonl": "57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04",
      "/tmp/azlite_pr123_weighted_candidate_preflight/suites/heldout_seed46_large.jsonl": "ae675e137845a516e83deb46198932761b3743fe5c7771adcd2d5e6e23717c5c",
      "/tmp/azlite_pr123_weighted_candidate_preflight/suites/heldout_seed47_large.jsonl": "f135bb057dd04d3cffe144c80fb3f338c5477c2704ef908e7f44982d71e4e561",
      "/tmp/azlite_pr123_weighted_candidate_preflight/suites/heldout_seed48_large.jsonl": "1a37452d5ed5e27879a5434cfb6291cba6f9aa4071ddf602b3ddcc370bf5bd5c"
    },
    "ply2_unavailable_due_frozen_eval_exhaustion": true,
    "requested_depths": [
      2,
      4,
      6
    ],
    "required_feasible_depths": [
      4,
      6
    ],
    "schema": "azlite_opening_depth_feasibility_v1",
    "structurally_unavailable_depths": [
      2
    ]
  },
  "orthogonal_classifications": [
    "opening_depth_structural_exhaustion_confirmed"
  ],
  "pilot_distribution_audit": {
    "opening_seeded": {
      "current_model_value_distribution": {
        "mean": 0.03382341597008392,
        "p50": 0.07247756615356152
      },
      "legal_move_count_distribution": {
        "1": 421,
        "2": 682,
        "3": 984,
        "4": 1309,
        "5": 1295,
        "6": 629
      },
      "nearest_board_l1": {
        "by_evaluation_family": {
          "fixed_large": {
            "median": 14.0,
            "p90": 18.0
          },
          "heldout_43_48": {
            "median": 13.0,
            "p90": 17.0
          },
          "heldout_49_54": {
            "median": 13.0,
            "p90": 17.0
          },
          "medium": {
            "median": 13.0,
            "p90": 17.299999999999997
          },
          "pooled_evaluation_corpus": {
            "median": 13.0,
            "p90": 17.0
          }
        },
        "by_suite": {
          "heldout_seed43_large": {
            "median": 12.5,
            "p90": 17.0
          },
          "heldout_seed44_large": {
            "median": 13.0,
            "p90": 17.0
          },
          "heldout_seed45_large": {
            "median": 13.0,
            "p90": 17.0
          },
          "heldout_seed46_large": {
            "median": 13.0,
            "p90": 17.0
          },
          "heldout_seed47_large": {
            "median": 13.0,
            "p90": 17.0
          },
          "heldout_seed48_large": {
            "median": 13.0,
            "p90": 17.0
          },
          "heldout_seed49_large": {
            "median": 13.0,
            "p90": 17.0
          },
          "heldout_seed50_large": {
            "median": 13.0,
            "p90": 17.0
          },
          "heldout_seed51_large": {
            "median": 13.0,
            "p90": 17.69999999999999
          },
          "heldout_seed52_large": {
            "median": 13.0,
            "p90": 17.0
          },
          "heldout_seed53_large": {
            "median": 13.0,
            "p90": 17.0
          },
          "heldout_seed54_large": {
            "median": 13.0,
            "p90": 17.0
          },
          "large_eval": {
            "median": 14.0,
            "p90": 18.0
          },
          "medium_eval": {
            "median": 13.0,
            "p90": 17.299999999999997
          }
        },
        "median": 13.0,
        "p90": 17.0
      },
      "phase_distribution": {
        "late": 1419,
        "midgame": 1536,
        "opening": 2365
      },
      "player_distribution": {
        "0": 2762,
        "1": 2558
      },
      "policy_entropy": {
        "mean": 0.30707810227005355,
        "p50": 0.0
      },
      "puct_visit_entropy": {
        "mean": 1.1434777544535204,
        "p50": 1.203439106516261
      },
      "replay_trajectory_state_overlap": {
        "by_evaluation_opening_depth": {
          "5": 4,
          "6": 2
        },
        "by_phase": {
          "opening": 6
        },
        "fraction_replay_rows": 0.0011278195488721805,
        "fraction_unique_replay_states": 0.001129305477131564,
        "overlapping_replay_rows": 6,
        "unique_overlapping_replay_states": 6
      },
      "seeded_prefix_overlap": 0,
      "seeded_start_state_overlap": 0,
      "store_difference_distribution": {
        "-1": 374,
        "-10": 123,
        "-11": 104,
        "-12": 81,
        "-13": 49,
        "-14": 43,
        "-15": 28,
        "-16": 44,
        "-17": 40,
        "-18": 38,
        "-19": 33,
        "-2": 274,
        "-20": 12,
        "-21": 5,
        "-22": 8,
        "-23": 5,
        "-24": 2,
        "-26": 1,
        "-3": 170,
        "-4": 137,
        "-5": 158,
        "-6": 139,
        "-7": 127,
        "-8": 146,
        "-9": 116,
        "0": 390,
        "1": 386,
        "10": 107,
        "11": 108,
        "12": 103,
        "13": 70,
        "14": 59,
        "15": 60,
        "16": 49,
        "17": 42,
        "18": 31,
        "19": 26,
        "2": 312,
        "20": 24,
        "21": 11,
        "22": 11,
        "23": 14,
        "24": 13,
        "25": 4,
        "26": 2,
        "27": 3,
        "3": 204,
        "4": 211,
        "5": 176,
        "6": 165,
        "7": 191,
        "8": 170,
        "9": 121
      }
    },
    "standard_start_control": {
      "current_model_value_distribution": {
        "mean": 0.026725310576590104,
        "p50": 0.057654891827346545
      },
      "legal_move_count_distribution": {
        "1": 403,
        "2": 619,
        "3": 903,
        "4": 1212,
        "5": 1445,
        "6": 739
      },
      "nearest_board_l1": {
        "by_evaluation_family": {
          "fixed_large": {
            "median": 13.0,
            "p90": 18.0
          },
          "heldout_43_48": {
            "median": 13.0,
            "p90": 19.0
          },
          "heldout_49_54": {
            "median": 14.0,
            "p90": 20.0
          },
          "medium": {
            "median": 14.0,
            "p90": 18.299999999999997
          },
          "pooled_evaluation_corpus": {
            "median": 13.0,
            "p90": 19.0
          }
        },
        "by_suite": {
          "heldout_seed43_large": {
            "median": 13.0,
            "p90": 18.0
          },
          "heldout_seed44_large": {
            "median": 13.0,
            "p90": 19.69999999999999
          },
          "heldout_seed45_large": {
            "median": 13.0,
            "p90": 19.0
          },
          "heldout_seed46_large": {
            "median": 12.5,
            "p90": 19.0
          },
          "heldout_seed47_large": {
            "median": 13.0,
            "p90": 18.0
          },
          "heldout_seed48_large": {
            "median": 13.0,
            "p90": 20.0
          },
          "heldout_seed49_large": {
            "median": 14.0,
            "p90": 19.0
          },
          "heldout_seed50_large": {
            "median": 14.0,
            "p90": 19.0
          },
          "heldout_seed51_large": {
            "median": 14.0,
            "p90": 20.0
          },
          "heldout_seed52_large": {
            "median": 14.0,
            "p90": 19.0
          },
          "heldout_seed53_large": {
            "median": 14.0,
            "p90": 20.0
          },
          "heldout_seed54_large": {
            "median": 14.0,
            "p90": 19.69999999999999
          },
          "large_eval": {
            "median": 13.0,
            "p90": 18.0
          },
          "medium_eval": {
            "median": 14.0,
            "p90": 18.299999999999997
          }
        },
        "median": 13.0,
        "p90": 19.0
      },
      "phase_distribution": {
        "late": 1372,
        "midgame": 1325,
        "opening": 2624
      },
      "player_distribution": {
        "0": 2822,
        "1": 2499
      },
      "policy_entropy": {
        "mean": 0.2633636763281775,
        "p50": 0.0
      },
      "puct_visit_entropy": {
        "mean": 1.1568215558943302,
        "p50": 1.2294715085793455
      },
      "replay_trajectory_state_overlap": {
        "by_evaluation_opening_depth": {
          "1": 140,
          "2": 140,
          "3": 136,
          "4": 105,
          "5": 53,
          "6": 12
        },
        "by_phase": {
          "opening": 586
        },
        "fraction_replay_rows": 0.11012967487314415,
        "fraction_unique_replay_states": 0.03528083545018346,
        "overlapping_replay_rows": 586,
        "unique_overlapping_replay_states": 125
      },
      "seeded_prefix_overlap": 0,
      "seeded_start_state_overlap": 0,
      "store_difference_distribution": {
        "-1": 269,
        "-10": 78,
        "-11": 48,
        "-12": 60,
        "-13": 37,
        "-14": 24,
        "-15": 21,
        "-16": 11,
        "-17": 13,
        "-18": 21,
        "-19": 1,
        "-2": 134,
        "-20": 1,
        "-21": 4,
        "-22": 1,
        "-23": 3,
        "-24": 4,
        "-25": 1,
        "-3": 120,
        "-4": 136,
        "-5": 166,
        "-6": 214,
        "-7": 177,
        "-8": 156,
        "-9": 110,
        "0": 570,
        "1": 569,
        "10": 107,
        "11": 110,
        "12": 101,
        "13": 79,
        "14": 59,
        "15": 47,
        "16": 30,
        "17": 17,
        "18": 17,
        "19": 9,
        "2": 370,
        "20": 5,
        "21": 12,
        "22": 5,
        "23": 4,
        "24": 11,
        "25": 10,
        "26": 6,
        "27": 6,
        "28": 3,
        "3": 168,
        "30": 3,
        "31": 12,
        "4": 156,
        "5": 194,
        "6": 209,
        "7": 206,
        "8": 232,
        "9": 184
      }
    }
  },
  "pilot_replays": {
    "opening_seeded": {
      "game_count": 157,
      "lane": "opening_seeded",
      "phase_distribution": {
        "late": 1419,
        "midgame": 1536,
        "opening": 2365
      },
      "player_distribution": {
        "0": 2762,
        "1": 2558
      },
      "policy_entropy": {
        "mean": 0.30707810227005355
      },
      "random_seeds": {
        "base_seed": 42
      },
      "replay_sha256": "0da1541857bed660fd367e2cf5ccdce349171b26df07f9aebf6b32703aae3a93",
      "row_count": 5320,
      "start_state_hashes": [
        "00a000f7f0af3fdb67f321e440784f75d229d03ee9b4f437df9c2f1aa1c92041",
        "01f25e517f6d982a5e1a028080bbadd1b43121852c5a0570ec282b57e87cffa1",
        "056897e3cd9aa065b2426faf65d118e44e8b9e16eda84974d6e1b09dc1d9cbde",
        "0aea48d850c4ec14120060a8d8c73138d41690f7376c289f82c9a1925c50a83f",
        "0e5fc97f6aab10e39ae1f0747689b92a77ce64bb19a26f9af892fcf2e5754bdb",
        "11860d6d3f12c91c648aae657ec921bc0c24ffa7f7fdb2cdf34dc0a14ac2c6fc",
        "1309e7062a6ad1a305cb21f993807fa41f6dd5220af8afaf6b12dd6c1dd1eb4c",
        "162736058a04f8cf5c28684d8c90d003606a57f31ba8bb730d82fd3ae011c58c",
        "191961b2f78a23de3a9fb1e2efadece2fd42f57516989332fe905950e91010ce",
        "1a194eb711d46097f4a029f919a2d939ec84d4543579706e5c1936c26efe6877",
        "1a7e75899249e569d53f7e8beadb763d9d3f9f3973bc1e660fd1b48861bd2966",
        "1a94515bc0178c12bd42ddb5a0f6d396931fa71c05c4b20d3f3e33c9df472f10",
        "1b65855fdb2ffcd619a6177e40e44320c26ef43c90adb02cffe3399d8b531b67",
        "1cf6e33478e3dc3b3944c3ea290ec1e5a8bae56ea35ad0eea4f763aa1492bcee",
        "1d0409f34d7561933ce27f17f56b2890eb731db145d180e3b1bcd621d4f8525e",
        "1efe39050cf30d00b09aee158b489bb75a9c71d36c1db9c6e60056ce1f661b0d",
        "20c5118a4978781da727ab939e97072e9cc4cb1db8a2a901d8d39d6a9fa8d483",
        "21465dc1d1c90f6a86329d7da5fa569b16591c9b5c7ab614746fdbb9e4c86809",
        "225de0563e0db617c3ce5ac7cf311e74dd57071f0ce5eb6d8f6697f163a05869",
        "23740d15299e3426b0f492a1431c3cde547b4a502e2c9c293a0693ea581c5141",
        "23812bfea4a3d873d929ab6d6ee5a782cd87366c3bcadb8155741ab51dfbbb38",
        "245b188e3701ca161aecdfa94ed6b4c44f3a97c5ce69f40a2f1beaf2170fd3a9",
        "264d43cace3f2ad760444c6986a2ba4cb39771bbe198ed6ceb833583f29268ae",
        "287d9b70710f5ebedcc6ace83b4abb0908fd5b4abad4883fe38c9bb2e6860eb0",
        "2a386e77f22436e6a3299a2856daa01c2bb57ce6a77a7b930803478f033883e9",
        "2f5636843ec2c1578f9128fd8206a35ec6e44c44fea8da2c46be494c96599b1a",
        "30bc6214e51fbca59f362e4e256235148eb85d4e0fae6b81e1e26dda568212cf",
        "3184c7c8ab9b0f8ad0a6644df44008862be050b42961dc338941d412c100f6fc",
        "336da54bcf0f155eb4e1de713a23132c578d1a98816144c6ba5c739f4cc94367",
        "3e64aa80f751b383cda4229ad754e2721759ea82de92bcb9b345983bf9e559c4",
        "431bcffb44084bd78bf1f1bce51f7c753c0679c4b8cd5ad7de305d2434c68a2c",
        "44e8cd24afc7e6c93d0c22bc1c81fc82f7e410f6a19c5c8f619fc2d889fabf53",
        "4504d24a727896d639f3ba711211be8abdf93e7249597e33fe3b003605f3bc62",
        "49e3d0975436a0b0c55b2ef986fa8718f5444dad9a53790bc4a9738d343eddbe",
        "4bfd288a7b88ab50beb60e8b93ad264b6403be16f7885f8fc3c3a320f48bc482",
        "4c2968a42d1611129390590be60954df0e87a74823810e140a91eae6ffe4776f",
        "4ca5de459f04b06160238d7e36e262c9632fa7556140035554258f887aaa78a2",
        "4ca6cd04cc3ce0cb20e8fa0329517e70e575089de8cdcbbbee12ddce1e073bf3",
        "4ce51af140850fccb30188094737cc5056dbb9be272e1524b542a6c23279cea7",
        "4e8ec304fa6bfacd509c984126704fce0c8a1e14dedf349bc1095e69c691177f",
        "4fa3a1ae5c52cc2112926b7bb9b5ac4c01b5530f7a7162df56f8606d1b769736",
        "51120ed4f64732326ed00d257bcaa785cb24310a49f5458dcc385db97968ca20",
        "529f29da480bdfa6a7d9a90f9c2ca9699398f0bd63ee4ea312410405f14dafe5",
        "54120457a548658fd599e4f9664191fdef95bdecee9f966e6560c2883747423d",
        "5660d07bfb548dd3c42b92daf657067b5cab1d01be828c0ae9f59b93c44e794e",
        "56de8dc761d859e704534d8ca423aafde90635640a42f340ce341d081d00fff6",
        "56ef61def3a32c3441374e89cf706898bf881c2e93a20a27adf01c2b48e52ead",
        "579d3d08eafdcbf9e42e631460129e8aa6c55c75ac6d448b4f864e53dbbc7914",
        "581a50ed6bde0f1b0f2b8cec3ef2fa05120f4d0e3150f5e556ba5c3cd48c575d",
        "5d02e9e000393d4732eddbc2b1a99dd93a2fd8e2d02caf2d9341911a1db9aa7a",
        "5d32492b449c3916f2a7217952fbf37b2945e3dfa698bcb6bee9de218480b172",
        "5e727450a34cf2c936b46f39734f2354d644c7d0cde8c3d63e53055d85fab1fb",
        "5ffc1a0e4616a672da51f03c79f71a7418247f10a7e87f805b5074bd84cfb432",
        "60c07caed749ae9a257eaecc084c2ef827111a93ba83a58cdaf81c29b0418c26",
        "616c000a117036635c27803231e0d3c0631f374fe39c1de09a2e38315e9a77e3",
        "61acfae4b588129bc224556f022a8bbc51f0edc0c342bdfab59b878a59acbbae",
        "66a6f06401a23bfe867867db725b2227b5316a25bb6bfee8b9ebc621e3a8232e",
        "68e768d2e0311dfd8fc7b2f8779b58945ee79d89a12e71365a35019d41042e85",
        "6a05af0b69245019f068d8054ead20cdc179b08ae009c05c69eb614829048861",
        "6a5dcd78f62e64effe01801f78910e8bb268d4474ec3ac6cecd34c762356ecb5",
        "6ba789ff597bb1796a301c5e91035df3db3c7265f60f5c483ff8da4f1f0a6db3",
        "6c47f1609210a8aabbd7407f511e59dd24ad77618671d45e0d4f8e845cc2e7f6",
        "6f41d1fc88a6293119decc0931e5046c3cd11de86c0da3fb30831320bee3fbe5",
        "72674d195c56812d4a0ad23c60c5688c99cb786eb2268d470f570e897717e6f7",
        "74612e112735033911cc0cceff0f32495199f8c7e69c2ec9a8391bb3d3ffcc66",
        "76124e6baa1411ab131e791550edc59190fc114ce6c41e22a2b59d18f4187bc1",
        "77373d1c94dfd8bb8fcdfa8aefb8838230fa9650dd903ed30fc0251b582716cd",
        "773b2c10eba08ab2e799396cea0c7ed4460494683218801551914a4d07c05edb",
        "77b88064c6ec050835dc5e3e5995cceeb68f51b4dcbe597446ad9d427da5dfbb",
        "7b5123c11161bd0a2a4036ccb3b887d68dbf312fae0ab17756f76b6507c6c758",
        "7bc27efe7a82e97dc5b85c8b17c748a546c07163d8384c2df5903a5e13da7b45",
        "7cabfd1d8387f00cd4a24e39227371a3a11a9cd0b31ffd6110d16cec0931bd76",
        "7ec679e30347df68d438aa61c3af7cea8176d362581d67de1267f5a2fb87fe45",
        "8004f45aa7a7af3423fa3d69c21ef3cc8a7db0bd69f27112099b015719188ad7",
        "809db8bd8eba9aab7cc3eadd6b7d7bec0f1526d184f466a5572bca31da64e3d9",
        "82b1d2ffcdd63175a96a05b6b3a2656a0df4b738b7fb091442628b913667d85a",
        "866c477c78f46d471cb1d2d9ed836a3b3c6581c761bb4ac203a8b4fd9865bbc0",
        "8734f4dd00f1db70becfca02d3793beab4ebd26d1684c7f58db87ca696a60a0c",
        "8787bf84ae1700a6c581f22cb36f1f3a2e59b138db250cfa53f8aa3a20c1765c",
        "87df38181f96b7f1d45fb88d69b1786cb1e5ee44af623ac6e920555a673bbe80",
        "8846df943638d8bcc6732e8d13e8f47efc9f71b2a46073b7f3dd51e0d390c5eb",
        "89053714ac0e1f8b89a475d74b79173347d2180f0ab6173f5b781d315fd8fea0",
        "8946aa683d7722f3f328a12250f0bb5ae1d83a1342d0ff85f285abefd6271080",
        "9021db5c7b386294a42699cc385da69330de5395b75c86a8cb9671e6f40f3de6",
        "90575faf2b3850eca13eeb1b1de94a503dff2943d067196d447b7702c9e0a6de",
        "928ee49d37ecf0eb157e6e9fa79386e7428c5b6dbf448244fac011b3dc5b5725",
        "92ba3caa8849ac428dd76b5e75ddac1040dafc69b0598ef76dde8a333c15579a",
        "956136776f7ee246744f76226b09f25733f6f1a8b49dd2cd634bc8ab0cbff7d2",
        "9795bf1e46a8731d50756c50e593cdca3125c6ab21473f5b41610ec2cebb2865",
        "98ccbd5206aa088f65aaf527762ae7091069eab32a3c4e2f193771504352bf10",
        "990758acc5c5560e36fbaa0f804c5d3d7cff504f4015e0a437db267a1cef0661",
        "9a5b610bce99245a95187d76c2dba47c89e64fd9dfcaf12a91e22c6f41852078",
        "9d54fa71086a5066c3a00c72a93f9dbdb1175a629315c6b45888120b1a80b12e",
        "9e07095e737e4689619bcaa9ea6b4bf582f3d38c4dea42071920f6b53a0d84d7",
        "a4ad63268a485a05e1e667848a0ae34fd4419fc8bcd0fd698b1e00b7a2fe1bc5",
        "a54955472c168d1f9007ff5ed676dfcd5e8f4cd103a06d6a97e3372261ca4715",
        "a55067a026a5a2dba73c54231b2282e8ee7f97163f68c7179efbfe04143326cd",
        "a79656df0cafb92e3968f69042ee90869b6f2730759038406944343a3eed3dbc",
        "a842fb0dddc44821c9fc059b326f97825fde83b2010dfaf4f4aababed83319ec",
        "a87bf1829e126a6c2a189a2ce98ba73ed9a04097b083b3f174f4e342e7673623",
        "a913dcfa2259dabb92ac3acc3e20b05412ed85b388057d4691840c15a9df4bc6",
        "a9c31b52a09a9a27cc142a6bf66a85e1040dc2b4bea8c40894d307b51ff1516d",
        "ab8f84885c4ed1172dbfff089a1263694aff63870f2528153a72f5e53ddbd636",
        "ad0e48f8ed88ec4cfd5110d02e37377a776c561a88d83cf30f334d0809f18257",
        "b06db11278cef9ca13a0a3490655053082ebc3f68cee56ffc6aac0f433e80838",
        "b3786ac73c16a83a32ae3ba94bbb860dde690b651365045eac53aec717f61924",
        "b3c53469de6795485547ce5d644875f94020889970e48d7ea1ad240648bbeb98",
        "b532487a4f3761fba366deba56adb4c4744589a3d32f6e24ef23feaef2770a53",
        "b8b0944872506faf421713e4e05f24c6a7b967e93253f76afa115f305080ff2b",
        "b99a47dd1f5360d5233f98b4b15255806123d5598c0d37419bd3655d72bf8e6e",
        "ba19f9f1ba57473a2237cbcec4b298c3815c97184f74da173fd95f9795e692ba",
        "bd6fb660ad475090f8ae4214a6045bbe1bd8e6e5ae270b7cc0320fd055294173",
        "c069c0a155f1b8a27a2a311133eefdfe01c63bbec7d6e7472cc16d758d2174cc",
        "c24e490903726f4b9dd3eed910b7df79d59459833efac4121ef2b92495e5a54e",
        "c36552df177dd97ac2a19eefdb2b53f9ef4a8b9ef0de747d7bea5a3c405dd179",
        "c618092af485ac518e9931a69efb771d6fec6341b02cbbe2ec15974816e74c35",
        "c715fe9aa9ed4352d5f6502e58a6000d975f90479e07f721f0719bbd67b204f9",
        "c74e11ecd2528ea525a1c7a84677cb566efb2f15b22c02127001964dcafbafbe",
        "c8fda6c611bab26ba44878efeb35f6bde5cd77ef1db982ff91e7ab9a6df16321",
        "c99d3db0d73bb575c767b8bfcf048c290aec766f22b3465969185713580fbd0e",
        "ca8396f8ca7c78e3508a1bc8f427ac5ccaf6f1094ddcb5c1ea980a6011984ee5",
        "cb880de7e33b8ef17f93165be494430b5d07de9af83e06339ad8d4cb5412ace6",
        "ce168787833a6b5254421e2b2af3c69f54d4cef0bcc01e8526fe6a6c16b84277",
        "ce39912d221c69a4d5ce5c0c88d7bcc0ee9cd0c76adb11b2557ca6dc464aff04",
        "cf069432d5639a99de58083c0450db50380a6a0ca93bd66ab6c36e535a3c46ee",
        "cff3b7274b5ae504853011cf80c344fa2bccb990abd720a6a4a628fa2be59065",
        "d161c9be01aa716c78332a66ac8f2714b26964d69fa93597f182d31f7c9613d1",
        "d2a32dab0eb11045d23a152e3baf6dcc37267d700f741e8e2d89a00d8dffb27a",
        "d524e78292326c8e9e4afaa3a6a5fb3d8ec559b65fb8342c447d47fd6e1d3e4c",
        "d7e52bc9abc6c7db0c70662df6d0fda521285616c83f9c78921d7f0970cdeb95",
        "d806b4e1b03a468e4ea78abb758aca561d6964e356740c8328a9481b920d882b",
        "d842ad6cbcdf7ed20ea67277e4cc77119e689ac38aa564562b1ab45fb65824dd",
        "d8f8968e5a3abf97a560dceccd6a0ef9b4d8bcd83af4433628f1679cf784886e",
        "d94277b5087fb1ce7517d59c89f7413d5b4c8105cf9d3375475026ef9b27089e",
        "db4a6ae7414d59a495c1bb5ec8009ff04ebb996010dd8749bc1c4d8e07971c40",
        "dd59b039da381912e0aad7ddf61d70396f7f1f2317143ba2a96cedaaf7ea66e7",
        "de42d0e59c592c4bdfcc7c85563bc0c8e995f3a263126350f05fd14ec1e67348",
        "df76d18d58563d87a9112f08c7998a9fbabbaba2957db95c9550062cacb99bad",
        "e2ea672316fc7b90101850a9ac226114c45c6147360087269266a4143cbfb504",
        "e7ece390f373d5ff365269c0e61d238f7863efa1528f3c4beff249b5c18ebe51",
        "e80efaa22627ad1c35a93ec747efd9a8f109f1308e007dadab64a0913c30a081",
        "eb6e6cb6bdcd2fd7c679812036110eb78f4e27b67bb8d6553556b1c98008cb79",
        "eb93f12828af2c7c7320ddfbdc2a4fb5bc2cc8b8fc01ebfcb810919fd836a01c",
        "ebb4604ea89c619b8024ca8137b36439dfe28e607d1845df29ae24303913de96",
        "ec9e4799a7a95c952a043778b901a3e5c05d4f65518c242090262355e4ecbb80",
        "ed3ec21a3cdb2df72deb0aabbb5f3d7c0860c67f1c560913cafb0a177ce4fb31",
        "ef653033c3dd89b6b2da2e1ef3ae4af5acfd21077cc8e41a91c00eba2157af54",
        "f13a464204392250d1895ad52e48e691b7dafea620c2d57dcbaca3a6e131a645",
        "f1c36434754e10eb8594ed931c4b0fd51f3ad94f877012b7e514db8f92020267",
        "f2b6a044380c93f93c9259da538e995538e32ca1b958e27cb162e8feebce0fde",
        "f3af54ab7a7c9ef68a62c7d611d55e6b1955032aef4322ebc2bb1d90d2e33a2d",
        "f79767fbb965a153b1969d8fc4c52c731ba50c5ba22769c2579a39923a3c4201",
        "f9ff92715f5dfb5db72db6d29ecf730d296a963bcb618da0431a2d9951ebbf1a",
        "fd790ab5280e6f8dad0013fea6cf36e6419b33d20e2e606d6872607a011d73f7",
        "ff332447c0e1df9b315d7a7daa74a7b7821581069f9ad2a855c03a0d66cd7706",
        "ffb70b6db26f4ca86b16939e67c0015508cc1884aff56d27c487775f0677e2da",
        "ffd20cbaf0dec179f45554d925cf1774920e486bf2162fb38c70600ecc257a60"
      ],
      "start_state_source": "training_opening_corpus",
      "terminal_target_distribution": {
        "-1.0": 2201,
        "0.0": 623,
        "1.0": 2496
      },
      "unique_state_count": 5313
    },
    "standard_start_control": {
      "game_count": 140,
      "lane": "standard_start_control",
      "phase_distribution": {
        "late": 1372,
        "midgame": 1325,
        "opening": 2624
      },
      "player_distribution": {
        "0": 2822,
        "1": 2499
      },
      "policy_entropy": {
        "mean": 0.2633636763281775
      },
      "random_seeds": {
        "base_seed": 42
      },
      "replay_sha256": "485ea65c8ca1e9a3416bbccbf97f7e212a9f6082da759fd4c5cc8452c612385b",
      "row_count": 5321,
      "start_state_hashes": [
        "423b71c9047da16c7f1e26d205b9f43fc6649ff9e56f9e334f9eca6a6653e1c3"
      ],
      "start_state_source": "standard_initial_state",
      "terminal_target_distribution": {
        "-1.0": 2277,
        "0.0": 526,
        "1.0": 2518
      },
      "unique_state_count": 3543
    }
  },
  "runtime_profile": {
    "c_puct_schedule": {
      "768:768": 0.9
    },
    "default_c_puct": 1.25,
    "tactical_root_bias": 0.0
  },
  "schema": "azlite_distribution_aligned_selfplay_v1",
  "stop_reasons": [
    "nearest_distance_median_improvement_below_20pct"
  ],
  "training_opening_corpus": {
    "corpus_sha256": "8922c6cbda4e23184b81164c36880e5837b6a1693cb27168355d6f87b97d709a",
    "duplicate_counts": {
      "corpus_states": 0,
      "enumeration": 118
    },
    "exact_overlap_counts": {
      "alternate_prefixes": 0,
      "prefixes": 0,
      "states": 0
    },
    "exclusion_suite_hashes": {
      "/tmp/azlite_budget_tactical_confirmation/suites/heldout_seed49_large.jsonl": "9d45df32f023e5e9a7ba12d72f3f467a2ce49399b38357e17f0bbc68e008111b",
      "/tmp/azlite_budget_tactical_confirmation/suites/heldout_seed50_large.jsonl": "7f58f98df175a707a00ca42b21ef7625669f95b3b788cfedcd428c91b0edd857",
      "/tmp/azlite_budget_tactical_confirmation/suites/heldout_seed51_large.jsonl": "33fdd0f619bef908968eb16711a8d5b448da5b8b5ea7e7166be0293dab8ef943",
      "/tmp/azlite_budget_tactical_confirmation/suites/heldout_seed52_large.jsonl": "161e7f440d879ff095ea717ce66e4bdb3d3d88a48c0ff61131a1979ce610fbb1",
      "/tmp/azlite_budget_tactical_confirmation/suites/heldout_seed53_large.jsonl": "978154ee6f5f9fd47c3ebf199d61f665505a528efc33d7f375541b3dac9e65b2",
      "/tmp/azlite_budget_tactical_confirmation/suites/heldout_seed54_large.jsonl": "968a2b89093f824c8309e1227f7ad725c5d8a7e5c1ee373d4e137c8261895f9b",
      "/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed43_large.jsonl": "5e0ed96ba56f99318c32a139692309759659da16a0a98c590e35f7e2496cade9",
      "/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed44_large.jsonl": "323f7abec32fb00d3b7b6b153ddd5692435755a3bc5340d8aa5aec32ca639620",
      "/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed45_large.jsonl": "ca72c8b7fe1adf7229f81183321475033f8548dcdeb1c71fd16c85c35ce89cda",
      "/tmp/azlite_opening_suite/large_eval.jsonl": "ebc86f053a8a4c12f3e937b5acfe37df7b0d2f1f0002969dcac077a9f99d65a4",
      "/tmp/azlite_opening_suite/medium_eval.jsonl": "57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04",
      "/tmp/azlite_pr123_weighted_candidate_preflight/suites/heldout_seed46_large.jsonl": "ae675e137845a516e83deb46198932761b3743fe5c7771adcd2d5e6e23717c5c",
      "/tmp/azlite_pr123_weighted_candidate_preflight/suites/heldout_seed47_large.jsonl": "f135bb057dd04d3cffe144c80fb3f338c5477c2704ef908e7f44982d71e4e561",
      "/tmp/azlite_pr123_weighted_candidate_preflight/suites/heldout_seed48_large.jsonl": "1a37452d5ed5e27879a5434cfb6291cba6f9aa4071ddf602b3ddcc370bf5bd5c"
    },
    "generator_seed": 42,
    "generator_version": "build_opening_suite_v1",
    "phase_distribution": {
      "early": 2048
    },
    "player_to_move_distribution": {
      "0": 1073,
      "1": 975
    },
    "ply2_unavailable_due_frozen_eval_exhaustion": true,
    "prefix_depth_distribution": {
      "4": 195,
      "6": 1853
    },
    "required_feasible_depths": [
      4,
      6
    ],
    "schema": "training_opening_corpus_v1",
    "state_count": 2048,
    "structurally_unavailable_depths": [
      2
    ]
  }
}
```
