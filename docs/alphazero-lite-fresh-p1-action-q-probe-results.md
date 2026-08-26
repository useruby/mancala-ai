# Action-Q Probe

**Classification:** `static_action_q_not_learnable`

**Recommended follow-up:** test a search-conditioned Q correction model using state plus root visit/Q statistics rather than expanding the policy network again.

```json
{
  "classification": "static_action_q_not_learnable",
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
  "frozen_offline": {
    "amplified": {
      "hashes": [
        "013807a2efe746bf62ed46e16f05b65680b500b149bbb039c1db9c912f4688e6",
        "02ed9872133bcf81ccedc7752b4e45da4eed3e7ddb19b8bd2af2eb1b3a5b4c8e",
        "15254d26866a519e19b05ecb7d074efca290aee7abeada5aad969411560f1cb6",
        "188c503236f7619ab9b08523afef1335dc2a110079e95a6d45866601a67b10cd",
        "2a9f8e2a5823944f71283c4afc3b75295b470f794cb00d86bf43307dfeb12da5",
        "2ea19f706ad5e1a5ce7313af831803e1a54b282b3446d2edbe88262a33fe43f0",
        "2f05808aa02b8d8cfff5427b0e1412404e9679022172b74e4c9c3035a592a5e3",
        "2f9f8c9a2de6867a3732c41a2c67c896a902dff9bdd621cff95b0257dc62e864",
        "362958a9d30519f98e27a71256c89f1acba0a61e3d93e609adb59609bff61674",
        "37cd52ab179163de0784b3bc5e242c12e84d7226c58499baa8e01d6050c9356b",
        "4502ce0c8cb224305db955d0b6a7ac0a3b8a757e2d3ac5eeb1b15f24a7e38697",
        "6805f4349f1762bbcd7372428cae1c1687bae8f6a13e1c00354b37d39aeef7a1",
        "6a2bcb7f4757ba4582ab22e17d2d29e11a6502e40c3326308c4d40f4027dbc1e",
        "6d97c9b583700cb29c9156fe1e0d303e7f884272f22a826b04734dc7a4e65014",
        "700da2fd2f248840eeb53f501fac491889cf188e44737429c8b228969c253014",
        "70126c0e1da58b32ee60ff96ce9ca8286c1912b22fa3c7c232bb8a847e599ae2",
        "7f9548bfa2270253e9e030f378b1c70a2f03d63429f3fb0f8752f433ca9f01e5",
        "8f1d0a37da6fa80727a7d2fd261bf1a54366c273a47019b9a7577f8dce1b6494",
        "934f7962a8d86e237fa09895ba3bca4bb7a0c46902c952262be945914c08b267",
        "9377e1f332ea635ddf69a42542e8752ee65ff25da4af989589e0eec759d242f4",
        "951a3042f34f40919ac5a5eb66a7072029979b6f604f11e37c733fd28471258b",
        "966a611e92bed9435bb1c67a58eeaea3d87df8959291b481aca509e9f023b8e4",
        "a15239440233e6fa063c7a2dfd1c140fdd2df2f89c72efaf35ee409cf8eb146b",
        "a482b39e3d6943f9a7f1c360fa8dd1177ff89dbede16049136ab46032640ed01",
        "a6a0b90f828281f037127400b801dfd1f92611669b29fafeabecd5dac29a5ef7",
        "afe40174ccfd7dc7a5de76bf3753da9bc6ebaf8ce75f25b2f7241c450ffe1d61",
        "b94cfa81d691473293345c4e31ba97e3fe017e2fb0818ed5b3bab17c4a4835d7",
        "bd815c9c00d4e0eac19b49257928c4820d6e54fa21f62f62fa99bff31209cc41",
        "c77c01ee1fffa63b9ae1b1df4d1dea07485efa23da81b76221e7aed7781184b4",
        "cd86d5db65feb065b02f2c696b2598934922f81874e632ae14946eaa0f522055",
        "ce1a40b6c89e5b666369ea705346f426e430647c0c39c1c4488412d0b410aefc",
        "d4a9f78ae368a2e5d0edbd2f241d359dca4f03efb7b3891cf895d94d155a8d16",
        "d89fa6df131c62b8415bff3bd444badbd6867f5be484f45dda5973b282373e90",
        "e3c5f232ebd455665d7c89a94884c33c61e3421104dfb8bfd79418968ff91438",
        "e6f397b4ede2144309468114e995d1843c15770773058e45ae644576222cb40e",
        "ed1b01e0f0e8c9d1f7a92485669235b6bc4369fe793996c6a6840e319b0260bf",
        "ee50f0a3b7cbf15433705ee08e3eb6bde775b453374eb66e16e30c1323c73ab5",
        "f4955ba6a6e86f529ea6398a94d0f739510de1778cead33cf3302e54b4a48958",
        "fdec50d74542bcb4dd393e97be2ec07ef25ef6e12444c753cc4d3abf0f9f0709",
        "fe844ca60fd0c8df245ad384a4041488a5be4f08c48d1d06613b553b60cb0812"
      ],
      "one_step_value_q_metrics": {
        "best_q_action_agreement": 0.35,
        "cosine": 0.300093372764359,
        "l1": 0.1345559024471796,
        "mse": 0.03136854007278636,
        "pairwise_rank_agreement": 0.61,
        "top_two_ordering_agreement": 0.1,
        "visit_bin_centered_q_l1": {
          "1": 0.05389990657567978,
          "2_7": 0.1883643240109086,
          "32_plus": 0.12691295415190204,
          "8_31": 0.09709134422096044
        }
      },
      "probe_q_metrics": {
        "best_q_action_agreement": 0.3,
        "cosine": 0.22243213901550388,
        "l1": 0.07304565587217804,
        "mse": 0.014455725862248369,
        "pairwise_rank_agreement": 0.5833333333333334,
        "top_two_ordering_agreement": 0.2,
        "visit_bin_centered_q_l1": {
          "1": 0.013644196093082428,
          "2_7": 0.21964579779654742,
          "32_plus": 0.06215536468241933,
          "8_31": 0.13100378640142646
        }
      },
      "selection_consequence": {
        "exact_flip_action_recall": 0.24109547279101673,
        "exact_flip_detection_rate": 0.8339007164270108,
        "exact_parent_flip_count": 19681,
        "exact_parent_score_regret_captured": 0.29494620270336375,
        "false_flip_rate": 0.8421907553232811,
        "nonflip_preservation_rate": 0.1578092446767188,
        "overall_exact_parent_action_agreement": 0.19195833333333334
      }
    },
    "washed": {
      "hashes": [
        "0347d2b31667eb751e1dbb9d42097fd29b3e4507a447da7b0cd735fada78de23",
        "046d95f1e70f67e3fe754292f6f1c8a92019b8a9a4bf10e47f36440e6c89db09",
        "07c2452705d43679b3f862dfc321aeb16e2e7f8d20893e94e4f6a37dc1ddda40",
        "096898d2d88363e213b36771398db724afd6a5a3571346826046bc32476354d4",
        "0c948430b56bdef88f601bc0fb879f5bdef18b285a03ecbb2918a607ee6c8143",
        "0f9140d59b5339aded929c5f14ddef7755a16c5b900cd7abd449f59ab7df3b18",
        "19a46f0773e2f100996ff7cd22af1fc4cb1ca6378ae5b66a1428dabbcfdad3d0",
        "2033250a9f6439989144ec8a14e1bf9a70cce5a08806b6e99b02112f485b7382",
        "3330531f636169691927faaaabc4b2e68f04450c91a88439f29023420562c269",
        "336acfa7ded673514ad4ef1c0e56dfd6c4d45b8953c3a269147e3fd091b70e4f",
        "342ef91973a3fff82f58e981eb2e6e1b2a26f4c74175b5ad6c20c70efde3705d",
        "46b9b6e5a1ed0325d5c31630bfb9b26917551b3878f9fd67b89371b473b8f38a",
        "47f2361ec0e33ec141c5585a0d5cf40351333f7ad92de1b1d0c9bb1a22ab0acd",
        "497819c496849da0cac8ee7143ce915b4df1962631bb8e2acf80b11786b58b15",
        "4b9792cd7cae924c397ba85ca5e912597884c14f6284e78516245b69eeac9af3",
        "54dc994804cd7e6093e85de20fdd885993105a182f5350733d1fb57db2085c5a",
        "553b183d8ca98e641a07b256798fe7c48b34b0718b2a34e9eb92bb68703cf789",
        "6f9e1e6fefd6176f3782ac3d5b3ca087a70914589cb10bbb5d6a53baa8d37742",
        "7239f8b2c6dd4e4c0bd1dc6a73b9d0ef39a5ca6b39337aa0e1419fda4e0f7cba",
        "7795a0427a6b447971f77427d756aa74e6eea5c0718b9de113eb28661970ba2e",
        "8676fc0442593252a55925d4689a22c324229cd19f0f8632437d1b1a9311642d",
        "8bbf0083453efbccfb3f4f32e431374f8c68ed33ea4db66e0d92a7d3a98826ca",
        "920419c02b6da607f91fd3e3d14bfbb3493958d149390fe6f4bd719985776ff9",
        "a755d4c554746f6ea147d78961417e9d557c6f5da12fe0f1502a780c971a94df",
        "aaca3ee760cf8f67c9ab04a62d3013d974f8d0cc211bd2b8035510cd7cfd693f",
        "b9e27e926dc565d8de119af9a35fab79f98d4b360f18dd74b61bb9dec51e8849",
        "c2d6250b2fa4c3a1b08b68f36eabd6bc05a368a506800f8c6cd4660228bcf966",
        "cc3617e8863481a8089c4e6675a747d8c85e56496f19c4f4e2072c2b25d9897e",
        "cfd31c47ba43f707c03d1bda3fef92bea24a36237b9d73c81caf330686c45b93",
        "de27e90f400c912ca445b9b03130a3615a243a7e2b03d1ff0d2ff8876553b0c5",
        "e3ee4b4a73a9f15e1abb8cbaa47a240e590621e8c4157957513b513e331197a9",
        "ead506a1149a0aa405bdc1513e54f6c7a0dc0d603d5bb4b77761e226b0e43089",
        "ed67a8db689cf12a42705036853e917fc3a79084f329f3218f57b0504e4f09b7",
        "ef9a24395dc93c13c0e07065c36eba08e140414e3fbcb4094af83ed34557184e",
        "f21473a7346ae37144c86322779edb8aefa55d827e51b30a238303e6704ed2f1",
        "f89ef146c48e3bbb02c68084b52330c6e2759b1d8527c2555cf0c1073a099fe2",
        "f9ed552d9613cc1cdfc3fb4094b0cbd0f901dc59a9f1a91afe4d3f7782a548da",
        "fb1a08a6157b3eb59ad341976df7be2ea75e69cc713c98b707444493dbd76d24",
        "fc5bf26ecc9c3c6475d08f6fa46e43693db3148154dfd3b4ea6a641130312c17",
        "fd51a92ff90680c9a8761731bb182ccee58b84e06abad382a11bdabbe5951a5c"
      ],
      "one_step_value_q_metrics": {
        "best_q_action_agreement": 0.425,
        "cosine": 0.2666154845688806,
        "l1": 0.1286818398747968,
        "mse": 0.0297586314573266,
        "pairwise_rank_agreement": 0.5875,
        "top_two_ordering_agreement": 0.25,
        "visit_bin_centered_q_l1": {
          "1": 0.10069429806123177,
          "2_7": 0.1818788829548457,
          "32_plus": 0.1335047202557189,
          "8_31": 0.1270620077225455
        }
      },
      "probe_q_metrics": {
        "best_q_action_agreement": 0.3,
        "cosine": 0.2453615935824917,
        "l1": 0.09811730619654474,
        "mse": 0.021156292465889077,
        "pairwise_rank_agreement": 0.5691666666666666,
        "top_two_ordering_agreement": 0.15,
        "visit_bin_centered_q_l1": {
          "1": 0.22406496951977412,
          "2_7": 0.12719289320133007,
          "32_plus": 0.09073459787296993,
          "8_31": 0.09273832832098106
        }
      },
      "selection_consequence": {
        "exact_flip_action_recall": 0.25660944743766445,
        "exact_flip_detection_rate": 0.8072923192582383,
        "exact_parent_flip_count": 7981,
        "exact_parent_score_regret_captured": 0.30386724324545744,
        "false_flip_rate": 0.8728104150528498,
        "nonflip_preservation_rate": 0.1271895849471501,
        "overall_exact_parent_action_agreement": 0.14870833333333333
      }
    }
  },
  "guardrails": {
    "arena_run": false,
    "c_puct": 1.25,
    "fpu_mode": "zero",
    "live_qprobe_search_skipped": true,
    "policy_value_trunk_frozen": true,
    "root_noise": false,
    "simulations": 1200
  },
  "hard_root": {
    "eligible_actions": [
      0,
      1,
      2,
      3,
      4,
      5
    ],
    "live_qprobe_search": "skipped_validation_gate_failed",
    "p1_q": [
      -0.4860794246196747,
      -0.5124726891517639,
      -0.5100886821746826,
      -0.523773729801178,
      -0.49783089756965637,
      -0.47887831926345825
    ],
    "p1_selected_move": 3,
    "probe_best_q_action": 2,
    "probe_q": [
      0.007318049669265747,
      0.019861415028572083,
      0.14320267736911774,
      0.028324756771326065,
      0.026895467191934586,
      0.025796394795179367
    ],
    "state_hash": "362958a9d30519f98e27a71256c89f1acba0a61e3d93e609adb59609bff61674"
  },
  "hashes": {
    "a16_weights": "74e160734554dba9ecf4ecf3ac13b3e44526c4f6e7e5e52d9fe8bde0054eb789",
    "frozen_target_cache": "48c9dce6f3efdce005590044c8349033784ec25b1cae526815c1d9e836b2f5f6",
    "p1_weights": "77969733ece5ced92d3a143a0fe9d82863ca3ec4faa477470ff5826ac22e4e12",
    "probe_checkpoint": "e152e3d1cd7c947895a1780f3eb55acd1d69f93a1074f66780aeebbb912bb9e4",
    "replay": "892827d8ee67a66e6324a2aaec7011df1a21625fc3f6bcd87cab39ce655d2a88",
    "split_manifest": "c26c4bc0a8d34ab9673e993d88d2153b7f59d6033a8a136d0626822eebdd56ad",
    "target_cache": "0a869446bd122693031b3f0ee239bc94788e8383825bdc2a37a8b6433b3f0bb6"
  },
  "metrics": {
    "train": {
      "one_step_value_q": {
        "best_q_action_agreement": 0.43679950186799504,
        "cosine": 0.27729505952496414,
        "l1": 0.125227245323118,
        "mse": 0.030995435936261258,
        "pairwise_rank_agreement": 0.6175695309256952,
        "top_two_ordering_agreement": 0.286425902864259,
        "visit_bin_centered_q_l1": {
          "1": 0.09458219914566102,
          "2_7": 0.1360984779786733,
          "32_plus": 0.12575968023819328,
          "8_31": 0.1363990776788685
        }
      },
      "probe": {
        "best_q_action_agreement": 0.4897260273972603,
        "cosine": 0.3501892154669644,
        "l1": 0.08013797391567162,
        "mse": 0.016200512593423855,
        "pairwise_rank_agreement": 0.6419468659194687,
        "top_two_ordering_agreement": 0.3147571606475716,
        "visit_bin_centered_q_l1": {
          "1": 0.12638735533710876,
          "2_7": 0.1116591981319964,
          "32_plus": 0.0738146324278803,
          "8_31": 0.0961446991883575
        }
      }
    },
    "validation": {
      "one_step_value_q": {
        "best_q_action_agreement": 0.45080946450809467,
        "cosine": 0.2783228751872869,
        "l1": 0.12171751700656613,
        "mse": 0.029436311851875466,
        "pairwise_rank_agreement": 0.6091324200913243,
        "top_two_ordering_agreement": 0.2777085927770859,
        "visit_bin_centered_q_l1": {
          "1": 0.0852807168093932,
          "2_7": 0.11824004040239137,
          "32_plus": 0.12242557890247302,
          "8_31": 0.13461328803685646
        }
      },
      "probe": {
        "best_q_action_agreement": 0.47073474470734744,
        "cosine": 0.2959737780622161,
        "l1": 0.0868906668251844,
        "mse": 0.018964963399474382,
        "pairwise_rank_agreement": 0.6209215442092155,
        "top_two_ordering_agreement": 0.298879202988792,
        "visit_bin_centered_q_l1": {
          "1": 0.1328626522612467,
          "2_7": 0.10705964289141516,
          "32_plus": 0.07970368217515728,
          "8_31": 0.09989809769807953
        }
      }
    }
  },
  "optimization": {
    "batch_size": 256,
    "best_validation_centered_q_mse": 0.018224112689495087,
    "lr": 0.001,
    "seed": 237,
    "steps": 2000,
    "validation_history": [
      {
        "step": 50,
        "validation_centered_q_mse": 0.020121697336435318
      },
      {
        "step": 100,
        "validation_centered_q_mse": 0.01961655169725418
      },
      {
        "step": 150,
        "validation_centered_q_mse": 0.019323915243148804
      },
      {
        "step": 200,
        "validation_centered_q_mse": 0.019004330039024353
      },
      {
        "step": 250,
        "validation_centered_q_mse": 0.019238105043768883
      },
      {
        "step": 300,
        "validation_centered_q_mse": 0.01887768693268299
      },
      {
        "step": 350,
        "validation_centered_q_mse": 0.018826929852366447
      },
      {
        "step": 400,
        "validation_centered_q_mse": 0.018600188195705414
      },
      {
        "step": 450,
        "validation_centered_q_mse": 0.018868338316679
      },
      {
        "step": 500,
        "validation_centered_q_mse": 0.018504925072193146
      },
      {
        "step": 550,
        "validation_centered_q_mse": 0.01838100701570511
      },
      {
        "step": 600,
        "validation_centered_q_mse": 0.018300993368029594
      },
      {
        "step": 650,
        "validation_centered_q_mse": 0.018408484756946564
      },
      {
        "step": 700,
        "validation_centered_q_mse": 0.018566040322184563
      },
      {
        "step": 750,
        "validation_centered_q_mse": 0.018344134092330933
      },
      {
        "step": 800,
        "validation_centered_q_mse": 0.018279608339071274
      },
      {
        "step": 850,
        "validation_centered_q_mse": 0.018224112689495087
      },
      {
        "step": 900,
        "validation_centered_q_mse": 0.018409430980682373
      },
      {
        "step": 950,
        "validation_centered_q_mse": 0.01823573186993599
      },
      {
        "step": 1000,
        "validation_centered_q_mse": 0.01857089065015316
      },
      {
        "step": 1050,
        "validation_centered_q_mse": 0.018329709768295288
      },
      {
        "step": 1100,
        "validation_centered_q_mse": 0.01829400658607483
      },
      {
        "step": 1150,
        "validation_centered_q_mse": 0.018623050302267075
      },
      {
        "step": 1200,
        "validation_centered_q_mse": 0.018380509689450264
      },
      {
        "step": 1250,
        "validation_centered_q_mse": 0.018568040803074837
      },
      {
        "step": 1300,
        "validation_centered_q_mse": 0.018420657142996788
      },
      {
        "step": 1350,
        "validation_centered_q_mse": 0.01867390237748623
      },
      {
        "step": 1400,
        "validation_centered_q_mse": 0.01863108202815056
      },
      {
        "step": 1450,
        "validation_centered_q_mse": 0.018264368176460266
      },
      {
        "step": 1500,
        "validation_centered_q_mse": 0.018470298498868942
      },
      {
        "step": 1550,
        "validation_centered_q_mse": 0.01852767914533615
      },
      {
        "step": 1600,
        "validation_centered_q_mse": 0.01847161166369915
      },
      {
        "step": 1650,
        "validation_centered_q_mse": 0.019172506406903267
      },
      {
        "step": 1700,
        "validation_centered_q_mse": 0.01864495500922203
      },
      {
        "step": 1750,
        "validation_centered_q_mse": 0.018862251192331314
      },
      {
        "step": 1800,
        "validation_centered_q_mse": 0.018876904621720314
      },
      {
        "step": 1850,
        "validation_centered_q_mse": 0.01894381269812584
      },
      {
        "step": 1900,
        "validation_centered_q_mse": 0.019102543592453003
      },
      {
        "step": 1950,
        "validation_centered_q_mse": 0.019023632630705833
      },
      {
        "step": 2000,
        "validation_centered_q_mse": 0.018919019028544426
      }
    ],
    "weight_decay": 0
  },
  "recommended_follow_up": "test a search-conditioned Q correction model using state plus root visit/Q statistics rather than expanding the policy network again.",
  "schema": "azlite_action_q_probe_v1",
  "validation_selection_consequence": {
    "gate_passed": false,
    "learned_probe": {
      "exact_flip_action_recall": 0.2861724606328923,
      "exact_flip_detection_rate": 0.7729159329878754,
      "exact_parent_flip_count": 159016,
      "exact_parent_score_regret_captured": 0.3543568077040725,
      "false_flip_rate": 0.7451179740089289,
      "nonflip_preservation_rate": 0.2548820259910712,
      "overall_exact_parent_action_agreement": 0.2600456621004566
    },
    "one_step_value_q": {
      "exact_flip_action_recall": 0.29827816068823265,
      "exact_flip_detection_rate": 0.7494591739196056,
      "exact_parent_flip_count": 159016,
      "exact_parent_score_regret_captured": 0.34169629557102715,
      "false_flip_rate": 0.7037599554552415,
      "nonflip_preservation_rate": 0.29624004454475855,
      "overall_exact_parent_action_agreement": 0.2965763802407638
    }
  }
}
```
