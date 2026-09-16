# R61 A0 Unit Geometry Audit

Inherited #322 classification: `early_divergence_local_decomposition_mixed`.

## Snapshot And Full-A0 Parity

{
  "snapshots": {
    "T61": "50736a6063cd0558329c433ad7a6c60b741c83ceee75d36e21e703749af8c9de",
    "T63": "bab60465a9bc93ac569ed28b356ef2abde2c1cbb0987a4a474e26117f9a57324"
  },
  "full_a0_patch": {
    "82": {
      "forward": {
        "cluster_margin_delta": 0.21002688705921174,
        "anchor_margin_delta": 0.6088429689407349,
        "cluster_optimal_mass_delta": 0.017028812551870942,
        "control_margin_delta": 0.28401845600456,
        "cluster_top1_repair_count": 0,
        "control_degradation_count": 0,
        "state_margin_deltas": {
          "cluster_anchor-244dadc26015ebbc": 0.6088429689407349,
          "matched_control-11f57287d52ab7f2": 0.28401845600456,
          "one_ply_child-0c052588b0daa7dc": 0.25275564193725586,
          "one_ply_child-487e68a18d1f94c0": -0.1907210499048233,
          "one_ply_child-a10cdabeaeea0081": 0.2190195918083191,
          "one_ply_child-e95450b6d1ff9a43": 0.3754577338695526,
          "structural_cluster-0fb60203365b0571": 0.3936225175857544
        }
      },
      "reverse": {
        "cluster_margin_delta": -0.2140550509095192,
        "anchor_margin_delta": -0.5154966115951538,
        "cluster_optimal_mass_delta": -0.01681597838178277,
        "control_margin_delta": -0.16924428939819336,
        "cluster_top1_repair_count": 0,
        "control_degradation_count": 0,
        "state_margin_deltas": {
          "cluster_anchor-244dadc26015ebbc": -0.5154966115951538,
          "matched_control-11f57287d52ab7f2": -0.16924428939819336,
          "one_ply_child-0c052588b0daa7dc": -0.19083940982818604,
          "one_ply_child-487e68a18d1f94c0": 0.2558854818344116,
          "one_ply_child-a10cdabeaeea0081": -0.10254359990358353,
          "one_ply_child-e95450b6d1ff9a43": -0.28186535835266113,
          "structural_cluster-0fb60203365b0571": -0.7509123682975769
        }
      }
    },
    "108": {
      "forward": {
        "cluster_margin_delta": -0.038921861350536345,
        "anchor_margin_delta": 0.405875526368618,
        "cluster_optimal_mass_delta": -0.008081628940999508,
        "control_margin_delta": -0.01015445590019226,
        "cluster_top1_repair_count": 0,
        "control_degradation_count": 0,
        "state_margin_deltas": {
          "cluster_anchor-244dadc26015ebbc": 0.405875526368618,
          "matched_control-11f57287d52ab7f2": -0.01015445590019226,
          "one_ply_child-0c052588b0daa7dc": 0.02704453468322754,
          "one_ply_child-487e68a18d1f94c0": -0.1248871386051178,
          "one_ply_child-a10cdabeaeea0081": -0.33849066495895386,
          "one_ply_child-e95450b6d1ff9a43": 0.2895718291401863,
          "structural_cluster-0fb60203365b0571": -0.047847867012023926
        }
      },
      "reverse": {
        "cluster_margin_delta": -0.002208012342453003,
        "anchor_margin_delta": -0.9698241949081421,
        "cluster_optimal_mass_delta": 0.00467930780723691,
        "control_margin_delta": -0.0013703852891921997,
        "cluster_top1_repair_count": 0,
        "control_degradation_count": 0,
        "state_margin_deltas": {
          "cluster_anchor-244dadc26015ebbc": -0.9698241949081421,
          "matched_control-11f57287d52ab7f2": -0.0013703852891921997,
          "one_ply_child-0c052588b0daa7dc": -0.04958963394165039,
          "one_ply_child-487e68a18d1f94c0": 0.13486766815185547,
          "one_ply_child-a10cdabeaeea0081": 0.2773096561431885,
          "one_ply_child-e95450b6d1ff9a43": -0.44773682951927185,
          "structural_cluster-0fb60203365b0571": 0.07410907745361328
        }
      }
    },
    "final": {
      "forward": {
        "cluster_margin_delta": -0.3423907458782196,
        "anchor_margin_delta": 1.072024941444397,
        "cluster_optimal_mass_delta": -0.016034961049444973,
        "control_margin_delta": -0.04018676280975342,
        "cluster_top1_repair_count": 0,
        "control_degradation_count": 0,
        "state_margin_deltas": {
          "cluster_anchor-244dadc26015ebbc": 1.072024941444397,
          "matched_control-11f57287d52ab7f2": -0.04018676280975342,
          "one_ply_child-0c052588b0daa7dc": -0.27467775344848633,
          "one_ply_child-487e68a18d1f94c0": 0.02506965398788452,
          "one_ply_child-a10cdabeaeea0081": 0.11307498812675476,
          "one_ply_child-e95450b6d1ff9a43": -0.10755383968353271,
          "structural_cluster-0fb60203365b0571": -1.4678667783737183
        }
      },
      "reverse": {
        "cluster_margin_delta": -0.02258724868297577,
        "anchor_margin_delta": -1.0284014344215393,
        "cluster_optimal_mass_delta": -0.0019115905743092299,
        "control_margin_delta": 0.07224540412425995,
        "cluster_top1_repair_count": 0,
        "control_degradation_count": 0,
        "state_margin_deltas": {
          "cluster_anchor-244dadc26015ebbc": -1.0284014344215393,
          "matched_control-11f57287d52ab7f2": 0.07224540412425995,
          "one_ply_child-0c052588b0daa7dc": 0.2448890209197998,
          "one_ply_child-487e68a18d1f94c0": 0.07890704274177551,
          "one_ply_child-a10cdabeaeea0081": -0.15531378984451294,
          "one_ply_child-e95450b6d1ff9a43": -0.29105134308338165,
          "structural_cluster-0fb60203365b0571": 0.00963282585144043
        }
      }
    }
  }
}

## Unitwise Trajectories And Sensitivity

All 96 units at G0, pre-80, 80, 81, 82, 108, and final are in the result artifact; every single-unit forward/reverse patch is in the patch matrix.

## Causal Ranking And Prefixes

{
  "ranking": [
    {
      "unit": 78,
      "82": {
        "forward": {
          "cluster_margin_delta": 0.007367908954620361,
          "anchor_margin_delta": -0.027119755744934082,
          "cluster_optimal_mass_delta": 0.0007421038113534451,
          "control_margin_delta": 0.008155882358551025,
          "cluster_top1_repair_count": 0,
          "control_degradation_count": 0,
          "state_margin_deltas": {
            "cluster_anchor-244dadc26015ebbc": -0.027119755744934082,
            "matched_control-11f57287d52ab7f2": 0.008155882358551025,
            "one_ply_child-0c052588b0daa7dc": 0.0015944242477416992,
            "one_ply_child-487e68a18d1f94c0": -0.0022255778312683105,
            "one_ply_child-a10cdabeaeea0081": 0.011745661497116089,
            "one_ply_child-e95450b6d1ff9a43": -0.005541890859603882,
            "structural_cluster-0fb60203365b0571": 0.03126692771911621
          }
        },
        "reverse": {
          "cluster_margin_delta": -0.0006557226181030274,
          "anchor_margin_delta": 0.02634298801422119,
          "cluster_optimal_mass_delta": -0.0002269226126372814,
          "control_margin_delta": -0.0069463253021240234,
          "cluster_top1_repair_count": 0,
          "control_degradation_count": 0,
          "state_margin_deltas": {
            "cluster_anchor-244dadc26015ebbc": 0.02634298801422119,
            "matched_control-11f57287d52ab7f2": -0.0069463253021240234,
            "one_ply_child-0c052588b0daa7dc": 0.001950383186340332,
            "one_ply_child-487e68a18d1f94c0": -0.0015037059783935547,
            "one_ply_child-a10cdabeaeea0081": -0.012291014194488525,
            "one_ply_child-e95450b6d1ff9a43": -0.0006402730941772461,
            "structural_cluster-0fb60203365b0571": 0.009205996990203857
          }
        },
        "same_direction": true,
        "bidirectional_unit_support": true,
        "causal_score": 0.0006557226181030274
      },
      "108": {
        "forward": {
          "cluster_margin_delta": 0.021512776613235474,
          "anchor_margin_delta": -0.0431673526763916,
          "cluster_optimal_mass_delta": 0.00467052087187767,
          "control_margin_delta": 0.012216717004776001,
          "cluster_top1_repair_count": 0,
          "control_degradation_count": 0,
          "state_margin_deltas": {
            "cluster_anchor-244dadc26015ebbc": -0.0431673526763916,
            "matched_control-11f57287d52ab7f2": 0.012216717004776001,
            "one_ply_child-0c052588b0daa7dc": 0.010427236557006836,
            "one_ply_child-487e68a18d1f94c0": 0.005786418914794922,
            "one_ply_child-a10cdabeaeea0081": 0.06267908215522766,
            "one_ply_child-e95450b6d1ff9a43": 0.04591667652130127,
            "structural_cluster-0fb60203365b0571": -0.01724553108215332
          }
        },
        "reverse": {
          "cluster_margin_delta": -0.03953472375869751,
          "anchor_margin_delta": 0.14783191680908203,
          "cluster_optimal_mass_delta": -0.006273433472961188,
          "control_margin_delta": -0.00792744755744934,
          "cluster_top1_repair_count": 0,
          "control_degradation_count": 0,
          "state_margin_deltas": {
            "cluster_anchor-244dadc26015ebbc": 0.14783191680908203,
            "matched_control-11f57287d52ab7f2": -0.00792744755744934,
            "one_ply_child-0c052588b0daa7dc": -0.003836393356323242,
            "one_ply_child-487e68a18d1f94c0": -0.014026045799255371,
            "one_ply_child-a10cdabeaeea0081": -0.052982330322265625,
            "one_ply_child-e95450b6d1ff9a43": -0.062443673610687256,
            "structural_cluster-0fb60203365b0571": -0.06438517570495605
          }
        },
        "same_direction": true,
        "bidirectional_unit_support": true,
        "causal_score": 0.019296059608459475
      },
      "final": {
        "forward": {
          "cluster_margin_delta": 0.01803573966026306,
          "anchor_margin_delta": -0.2420242577791214,
          "cluster_optimal_mass_delta": 0.0006433369591832161,
          "control_margin_delta": 0.014216959476470947,
          "cluster_top1_repair_count": 0,
          "control_degradation_count": 0,
          "state_margin_deltas": {
            "cluster_anchor-244dadc26015ebbc": -0.2420242577791214,
            "matched_control-11f57287d52ab7f2": 0.014216959476470947,
            "one_ply_child-0c052588b0daa7dc": 0.007628917694091797,
            "one_ply_child-487e68a18d1f94c0": 0.038762450218200684,
            "one_ply_child-a10cdabeaeea0081": 0.06862881779670715,
            "one_ply_child-e95450b6d1ff9a43": 0.019481897354125977,
            "structural_cluster-0fb60203365b0571": -0.0443233847618103
          }
        },
        "reverse": {
          "cluster_margin_delta": -0.047711601667106154,
          "anchor_margin_delta": 0.16222292184829712,
          "cluster_optimal_mass_delta": -0.00559000838547945,
          "control_margin_delta": -0.026358231902122498,
          "cluster_top1_repair_count": 0,
          "control_degradation_count": 0,
          "state_margin_deltas": {
            "cluster_anchor-244dadc26015ebbc": 0.16222292184829712,
            "matched_control-11f57287d52ab7f2": -0.026358231902122498,
            "one_ply_child-0c052588b0daa7dc": 0.009575843811035156,
            "one_ply_child-487e68a18d1f94c0": 0.010683299042284489,
            "one_ply_child-a10cdabeaeea0081": -0.05017787218093872,
            "one_ply_child-e95450b6d1ff9a43": -0.05197586119174957,
            "structural_cluster-0fb60203365b0571": -0.1566634178161621
          }
        },
        "same_direction": true,
        "bidirectional_unit_support": true,
        "causal_score": 0.013818780183792114
      },
      "eligible": true
    },
    {
      "unit": 41,
      "82": {
        "forward": {
          "cluster_margin_delta": 0.009339892864227295,
          "anchor_margin_delta": 0.000563502311706543,
          "cluster_optimal_mass_delta": 0.001465876353904605,
          "control_margin_delta": 0.0003097653388977051,
          "cluster_top1_repair_count": 0,
          "control_degradation_count": 0,
          "state_margin_deltas": {
            "cluster_anchor-244dadc26015ebbc": 0.000563502311706543,
            "matched_control-11f57287d52ab7f2": 0.0003097653388977051,
            "one_ply_child-0c052588b0daa7dc": 0.004977703094482422,
            "one_ply_child-487e68a18d1f94c0": 0.009330332279205322,
            "one_ply_child-a10cdabeaeea0081": 0.0067296624183654785,
            "one_ply_child-e95450b6d1ff9a43": 0.017628371715545654,
            "structural_cluster-0fb60203365b0571": 0.008033394813537598
          }
        },
        "reverse": {
          "cluster_margin_delta": -0.008182120323181153,
          "anchor_margin_delta": 0.0014998912811279297,
          "cluster_optimal_mass_delta": -0.0015579200349748134,
          "control_margin_delta": 0.0009889006614685059,
          "cluster_top1_repair_count": 0,
          "control_degradation_count": 0,
          "state_margin_deltas": {
            "cluster_anchor-244dadc26015ebbc": 0.0014998912811279297,
            "matched_control-11f57287d52ab7f2": 0.0009889006614685059,
            "one_ply_child-0c052588b0daa7dc": -0.0066879987716674805,
            "one_ply_child-487e68a18d1f94c0": -0.007694840431213379,
            "one_ply_child-a10cdabeaeea0081": -0.006347358226776123,
            "one_ply_child-e95450b6d1ff9a43": 0.00014269351959228516,
            "structural_cluster-0fb60203365b0571": -0.020323097705841064
          }
        },
        "same_direction": true,
        "bidirectional_unit_support": true,
        "causal_score": 0.008182120323181153
      },
      "108": {
        "forward": {
          "cluster_margin_delta": 0.002650648355484009,
          "anchor_margin_delta": 0.0005314350128173828,
          "cluster_optimal_mass_delta": 0.00034924130886793137,
          "control_margin_delta": -0.002388298511505127,
          "cluster_top1_repair_count": 0,
          "control_degradation_count": 0,
          "state_margin_deltas": {
            "cluster_anchor-244dadc26015ebbc": 0.0005314350128173828,
            "matched_control-11f57287d52ab7f2": -0.002388298511505127,
            "one_ply_child-0c052588b0daa7dc": 0.00048041343688964844,
            "one_ply_child-487e68a18d1f94c0": 0.0007648468017578125,
            "one_ply_child-a10cdabeaeea0081": -0.0033794939517974854,
            "one_ply_child-e95450b6d1ff9a43": 0.006461203098297119,
            "structural_cluster-0fb60203365b0571": 0.00892627239227295
          }
        },
        "reverse": {
          "cluster_margin_delta": -0.0031337201595306396,
          "anchor_margin_delta": -0.0007376670837402344,
          "cluster_optimal_mass_delta": -0.0007027973420917988,
          "control_margin_delta": 0.003037974238395691,
          "cluster_top1_repair_count": 0,
          "control_degradation_count": 0,
          "state_margin_deltas": {
            "cluster_anchor-244dadc26015ebbc": -0.0007376670837402344,
            "matched_control-11f57287d52ab7f2": 0.003037974238395691,
            "one_ply_child-0c052588b0daa7dc": -0.0003578662872314453,
            "one_ply_child-487e68a18d1f94c0": 0.0001335740089416504,
            "one_ply_child-a10cdabeaeea0081": 0.0013141632080078125,
            "one_ply_child-e95450b6d1ff9a43": -0.003939896821975708,
            "structural_cluster-0fb60203365b0571": -0.012818574905395508
          }
        },
        "same_direction": true,
        "bidirectional_unit_support": true,
        "causal_score": 0.002650648355484009
      },
      "final": {
        "forward": {
          "cluster_margin_delta": 0.009131979942321778,
          "anchor_margin_delta": 0.000963747501373291,
          "cluster_optimal_mass_delta": 0.0007944578770548105,
          "control_margin_delta": 0.0022083520889282227,
          "cluster_top1_repair_count": 0,
          "control_degradation_count": 0,
          "state_margin_deltas": {
            "cluster_anchor-244dadc26015ebbc": 0.000963747501373291,
            "matched_control-11f57287d52ab7f2": 0.0022083520889282227,
            "one_ply_child-0c052588b0daa7dc": 0.007704734802246094,
            "one_ply_child-487e68a18d1f94c0": 0.003180861473083496,
            "one_ply_child-a10cdabeaeea0081": 0.007439255714416504,
            "one_ply_child-e95450b6d1ff9a43": 0.009308338165283203,
            "structural_cluster-0fb60203365b0571": 0.01802670955657959
          }
        },
        "reverse": {
          "cluster_margin_delta": -0.004587525129318237,
          "anchor_margin_delta": -0.003211081027984619,
          "cluster_optimal_mass_delta": -0.0006430446170270443,
          "control_margin_delta": -0.0011954307556152344,
          "cluster_top1_repair_count": 0,
          "control_degradation_count": 0,
          "state_margin_deltas": {
            "cluster_anchor-244dadc26015ebbc": -0.003211081027984619,
            "matched_control-11f57287d52ab7f2": -0.0011954307556152344,
            "one_ply_child-0c052588b0daa7dc": -0.01032257080078125,
            "one_ply_child-487e68a18d1f94c0": -0.005383670330047607,
            "one_ply_child-a10cdabeaeea0081": -0.004042327404022217,
            "one_ply_child-e95450b6d1ff9a43": -0.0015818774700164795,
            "structural_cluster-0fb60203365b0571": -0.0016071796417236328
          }
        },
        "same_direction": true,
        "bidirectional_unit_support": true,
        "causal_score": 0.004587525129318237
      },
      "eligible": true
    },
    {
      "unit": 75,
      "82": {
        "forward": {
          "cluster_margin_delta": 0.016424494981765746,
          "anchor_margin_delta": -0.021927952766418457,
          "cluster_optimal_mass_delta": 0.0021485963836312292,
          "control_margin_delta": -0.00801241397857666,
          "cluster_top1_repair_count": 0,
          "control_degradation_count": 0,
          "state_margin_deltas": {
            "cluster_anchor-244dadc26015ebbc": -0.021927952766418457,
            "matched_control-11f57287d52ab7f2": -0.00801241397857666,
            "one_ply_child-0c052588b0daa7dc": -0.014278531074523926,
            "one_ply_child-487e68a18d1f94c0": -0.03576478362083435,
            "one_ply_child-a10cdabeaeea0081": -0.005355656147003174,
            "one_ply_child-e95450b6d1ff9a43": -0.00034743547439575195,
            "structural_cluster-0fb60203365b0571": 0.13786888122558594
          }
        },
        "reverse": {
          "cluster_margin_delta": -0.04701954126358032,
          "anchor_margin_delta": -0.008866548538208008,
          "cluster_optimal_mass_delta": -0.002692074794322252,
          "control_margin_delta": -0.006725013256072998,
          "cluster_top1_repair_count": 0,
          "control_degradation_count": 0,
          "state_margin_deltas": {
            "cluster_anchor-244dadc26015ebbc": -0.008866548538208008,
            "matched_control-11f57287d52ab7f2": -0.006725013256072998,
            "one_ply_child-0c052588b0daa7dc": 0.01368415355682373,
            "one_ply_child-487e68a18d1f94c0": 0.029733657836914062,
            "one_ply_child-a10cdabeaeea0081": 0.01437234878540039,
            "one_ply_child-e95450b6d1ff9a43": 0.0075179338455200195,
            "structural_cluster-0fb60203365b0571": -0.3004058003425598
          }
        },
        "same_direction": true,
        "bidirectional_unit_support": true,
        "causal_score": 0.016424494981765746
      },
      "108": {
        "forward": {
          "cluster_margin_delta": 0.003924453258514404,
          "anchor_margin_delta": -0.02773338556289673,
          "cluster_optimal_mass_delta": 0.0011655472218990326,
          "control_margin_delta": -0.0019434690475463867,
          "cluster_top1_repair_count": 0,
          "control_degradation_count": 0,
          "state_margin_deltas": {
            "cluster_anchor-244dadc26015ebbc": -0.02773338556289673,
            "matched_control-11f57287d52ab7f2": -0.0019434690475463867,
            "one_ply_child-0c052588b0daa7dc": -0.011732816696166992,
            "one_ply_child-487e68a18d1f94c0": -0.012689411640167236,
            "one_ply_child-a10cdabeaeea0081": -0.014505952596664429,
            "one_ply_child-e95450b6d1ff9a43": 0.003044337034225464,
            "structural_cluster-0fb60203365b0571": 0.055506110191345215
          }
        },
        "reverse": {
          "cluster_margin_delta": -0.0006188631057739257,
          "anchor_margin_delta": 0.018252849578857422,
          "cluster_optimal_mass_delta": -0.0008923714980483055,
          "control_margin_delta": 0.0005876719951629639,
          "cluster_top1_repair_count": 0,
          "control_degradation_count": 0,
          "state_margin_deltas": {
            "cluster_anchor-244dadc26015ebbc": 0.018252849578857422,
            "matched_control-11f57287d52ab7f2": 0.0005876719951629639,
            "one_ply_child-0c052588b0daa7dc": 0.007453441619873047,
            "one_ply_child-487e68a18d1f94c0": 0.013680577278137207,
            "one_ply_child-a10cdabeaeea0081": 0.008435726165771484,
            "one_ply_child-e95450b6d1ff9a43": -0.0020492076873779297,
            "structural_cluster-0fb60203365b0571": -0.030614852905273438
          }
        },
        "same_direction": true,
        "bidirectional_unit_support": true,
        "causal_score": 0.0006188631057739257
      },
      "final": {
        "forward": {
          "cluster_margin_delta": 0.01171296238899231,
          "anchor_margin_delta": -0.03575718402862549,
          "cluster_optimal_mass_delta": -0.0008161022793501616,
          "control_margin_delta": -0.002204716205596924,
          "cluster_top1_repair_count": 0,
          "control_degradation_count": 0,
          "state_margin_deltas": {
            "cluster_anchor-244dadc26015ebbc": -0.03575718402862549,
            "matched_control-11f57287d52ab7f2": -0.002204716205596924,
            "one_ply_child-0c052588b0daa7dc": -0.009881019592285156,
            "one_ply_child-487e68a18d1f94c0": -0.03215622901916504,
            "one_ply_child-a10cdabeaeea0081": -0.0029653608798980713,
            "one_ply_child-e95450b6d1ff9a43": 0.015096068382263184,
            "structural_cluster-0fb60203365b0571": 0.08847135305404663
          }
        },
        "reverse": {
          "cluster_margin_delta": -0.003682047128677368,
          "anchor_margin_delta": 0.008348464965820312,
          "cluster_optimal_mass_delta": -0.00025559356436133385,
          "control_margin_delta": 0.00989106297492981,
          "cluster_top1_repair_count": 0,
          "control_degradation_count": 0,
          "state_margin_deltas": {
            "cluster_anchor-244dadc26015ebbc": 0.008348464965820312,
            "matched_control-11f57287d52ab7f2": 0.00989106297492981,
            "one_ply_child-0c052588b0daa7dc": 0.005915641784667969,
            "one_ply_child-487e68a18d1f94c0": 0.034478187561035156,
            "one_ply_child-a10cdabeaeea0081": -0.00611191987991333,
            "one_ply_child-e95450b6d1ff9a43": -0.01946362853050232,
            "structural_cluster-0fb60203365b0571": -0.033228516578674316
          }
        },
        "same_direction": true,
        "bidirectional_unit_support": true,
        "causal_score": 0.003682047128677368
      },
      "eligible": true
    },
    {
      "unit": 1,
      "82": {
        "forward": {
          "cluster_margin_delta": 0.0033894777297973633,
          "anchor_margin_delta": -0.004362583160400391,
          "cluster_optimal_mass_delta": 0.001425861893221736,
          "control_margin_delta": -0.0006170868873596191,
          "cluster_top1_repair_count": 0,
          "control_degradation_count": 0,
          "state_margin_deltas": {
            "cluster_anchor-244dadc26015ebbc": -0.004362583160400391,
            "matched_control-11f57287d52ab7f2": -0.0006170868873596191,
            "one_ply_child-0c052588b0daa7dc": -0.001299738883972168,
            "one_ply_child-487e68a18d1f94c0": -0.013563692569732666,
            "one_ply_child-a10cdabeaeea0081": 0.01734936237335205,
            "one_ply_child-e95450b6d1ff9a43": 0.009429991245269775,
            "structural_cluster-0fb60203365b0571": 0.005031466484069824
          }
        },
        "reverse": {
          "cluster_margin_delta": -0.006744563579559326,
          "anchor_margin_delta": -0.016139745712280273,
          "cluster_optimal_mass_delta": -0.0003677042201161385,
          "control_margin_delta": -0.0002837181091308594,
          "cluster_top1_repair_count": 0,
          "control_degradation_count": 0,
          "state_margin_deltas": {
            "cluster_anchor-244dadc26015ebbc": -0.016139745712280273,
            "matched_control-11f57287d52ab7f2": -0.0002837181091308594,
            "one_ply_child-0c052588b0daa7dc": 0.0021973848342895508,
            "one_ply_child-487e68a18d1f94c0": 0.028556108474731445,
            "one_ply_child-a10cdabeaeea0081": -0.013100385665893555,
            "one_ply_child-e95450b6d1ff9a43": -0.011263608932495117,
            "structural_cluster-0fb60203365b0571": -0.040112316608428955
          }
        },
        "same_direction": true,
        "bidirectional_unit_support": true,
        "causal_score": 0.0033894777297973633
      },
      "108": {
        "forward": {
          "cluster_margin_delta": 0.00855497270822525,
          "anchor_margin_delta": -0.02022320032119751,
          "cluster_optimal_mass_delta": 0.0013906117528676987,
          "control_margin_delta": -0.006286144256591797,
          "cluster_top1_repair_count": 0,
          "control_degradation_count": 0,
          "state_margin_deltas": {
            "cluster_anchor-244dadc26015ebbc": -0.02022320032119751,
            "matched_control-11f57287d52ab7f2": -0.006286144256591797,
            "one_ply_child-0c052588b0daa7dc": 0.011761903762817383,
            "one_ply_child-487e68a18d1f94c0": 0.009722352027893066,
            "one_ply_child-a10cdabeaeea0081": 0.01762804388999939,
            "one_ply_child-e95450b6d1ff9a43": -0.008655808866024017,
            "structural_cluster-0fb60203365b0571": 0.01231837272644043
          }
        },
        "reverse": {
          "cluster_margin_delta": -0.008070027828216553,
          "anchor_margin_delta": -0.009955167770385742,
          "cluster_optimal_mass_delta": -0.0016775100491940975,
          "control_margin_delta": 0.008102044463157654,
          "cluster_top1_repair_count": 0,
          "control_degradation_count": 0,
          "state_margin_deltas": {
            "cluster_anchor-244dadc26015ebbc": -0.009955167770385742,
            "matched_control-11f57287d52ab7f2": 0.008102044463157654,
            "one_ply_child-0c052588b0daa7dc": -0.0029380321502685547,
            "one_ply_child-487e68a18d1f94c0": -0.009486138820648193,
            "one_ply_child-a10cdabeaeea0081": -0.02345883846282959,
            "one_ply_child-e95450b6d1ff9a43": -0.002850174903869629,
            "structural_cluster-0fb60203365b0571": -0.0016169548034667969
          }
        },
        "same_direction": true,
        "bidirectional_unit_support": true,
        "causal_score": 0.008070027828216553
      },
      "final": {
        "forward": {
          "cluster_margin_delta": 0.0030005395412445067,
          "anchor_margin_delta": -0.059806644916534424,
          "cluster_optimal_mass_delta": 0.0006031367462128401,
          "control_margin_delta": -0.0072547197341918945,
          "cluster_top1_repair_count": 0,
          "control_degradation_count": 0,
          "state_margin_deltas": {
            "cluster_anchor-244dadc26015ebbc": -0.059806644916534424,
            "matched_control-11f57287d52ab7f2": -0.0072547197341918945,
            "one_ply_child-0c052588b0daa7dc": -0.002055644989013672,
            "one_ply_child-487e68a18d1f94c0": -0.04556924104690552,
            "one_ply_child-a10cdabeaeea0081": 0.019196540117263794,
            "one_ply_child-e95450b6d1ff9a43": 0.013206422328948975,
            "structural_cluster-0fb60203365b0571": 0.030224621295928955
          }
        },
        "reverse": {
          "cluster_margin_delta": -0.003070181608200073,
          "anchor_margin_delta": -0.04130399227142334,
          "cluster_optimal_mass_delta": -0.0015296026133000851,
          "control_margin_delta": -0.005209475755691528,
          "cluster_top1_repair_count": 0,
          "control_degradation_count": 0,
          "state_margin_deltas": {
            "cluster_anchor-244dadc26015ebbc": -0.04130399227142334,
            "matched_control-11f57287d52ab7f2": -0.005209475755691528,
            "one_ply_child-0c052588b0daa7dc": -0.0017821788787841797,
            "one_ply_child-487e68a18d1f94c0": 0.007736086845397949,
            "one_ply_child-a10cdabeaeea0081": -0.012131929397583008,
            "one_ply_child-e95450b6d1ff9a43": -0.01040598750114441,
            "structural_cluster-0fb60203365b0571": 0.0012331008911132812
          }
        },
        "same_direction": true,
        "bidirectional_unit_support": true,
        "causal_score": 0.0030005395412445067
      },
      "eligible": true
    }
  ],
  "prefixes": [
    {
      "size": 1,
      "units": [
        78
      ],
      "forward": {
        "cluster_margin_delta": 0.01803573966026306,
        "anchor_margin_delta": -0.2420242577791214,
        "cluster_optimal_mass_delta": 0.0006433369591832161,
        "control_margin_delta": 0.014216959476470947,
        "cluster_top1_repair_count": 0,
        "control_degradation_count": 0,
        "state_margin_deltas": {
          "cluster_anchor-244dadc26015ebbc": -0.2420242577791214,
          "matched_control-11f57287d52ab7f2": 0.014216959476470947,
          "one_ply_child-0c052588b0daa7dc": 0.007628917694091797,
          "one_ply_child-487e68a18d1f94c0": 0.038762450218200684,
          "one_ply_child-a10cdabeaeea0081": 0.06862881779670715,
          "one_ply_child-e95450b6d1ff9a43": 0.019481897354125977,
          "structural_cluster-0fb60203365b0571": -0.0443233847618103
        }
      },
      "reverse": {
        "cluster_margin_delta": -0.047711601667106154,
        "anchor_margin_delta": 0.16222292184829712,
        "cluster_optimal_mass_delta": -0.00559000838547945,
        "control_margin_delta": -0.026358231902122498,
        "cluster_top1_repair_count": 0,
        "control_degradation_count": 0,
        "state_margin_deltas": {
          "cluster_anchor-244dadc26015ebbc": 0.16222292184829712,
          "matched_control-11f57287d52ab7f2": -0.026358231902122498,
          "one_ply_child-0c052588b0daa7dc": 0.009575843811035156,
          "one_ply_child-487e68a18d1f94c0": 0.010683299042284489,
          "one_ply_child-a10cdabeaeea0081": -0.05017787218093872,
          "one_ply_child-e95450b6d1ff9a43": -0.05197586119174957,
          "structural_cluster-0fb60203365b0571": -0.1566634178161621
        }
      },
      "forward_transfer_fraction": -0.028985406318611212,
      "reverse_transfer_fraction": -0.07667776240279678,
      "forward_correct_direction_fraction": 0.0,
      "control_degradation_fraction": 0.0,
      "step108_directional": true
    },
    {
      "size": 2,
      "units": [
        78,
        41
      ],
      "forward": {
        "cluster_margin_delta": 0.025317490100860596,
        "anchor_margin_delta": -0.24096538126468658,
        "cluster_optimal_mass_delta": 0.0013390085892751812,
        "control_margin_delta": 0.01636683940887451,
        "cluster_top1_repair_count": 0,
        "control_degradation_count": 0,
        "state_margin_deltas": {
          "cluster_anchor-244dadc26015ebbc": -0.24096538126468658,
          "matched_control-11f57287d52ab7f2": 0.01636683940887451,
          "one_ply_child-0c052588b0daa7dc": 0.01778101921081543,
          "one_ply_child-487e68a18d1f94c0": 0.04010826349258423,
          "one_ply_child-a10cdabeaeea0081": 0.07619184255599976,
          "one_ply_child-e95450b6d1ff9a43": 0.019191980361938477,
          "structural_cluster-0fb60203365b0571": -0.026685655117034912
        }
      },
      "reverse": {
        "cluster_margin_delta": -0.051603153347969055,
        "anchor_margin_delta": 0.1588262915611267,
        "cluster_optimal_mass_delta": -0.006212253402918577,
        "control_margin_delta": -0.02736780047416687,
        "cluster_top1_repair_count": 0,
        "control_degradation_count": 0,
        "state_margin_deltas": {
          "cluster_anchor-244dadc26015ebbc": 0.1588262915611267,
          "matched_control-11f57287d52ab7f2": -0.02736780047416687,
          "one_ply_child-0c052588b0daa7dc": 0.0008418560028076172,
          "one_ply_child-487e68a18d1f94c0": 0.004687845706939697,
          "one_ply_child-a10cdabeaeea0081": -0.05409055948257446,
          "one_ply_child-e95450b6d1ff9a43": -0.050779715180397034,
          "structural_cluster-0fb60203365b0571": -0.1586751937866211
        }
      },
      "forward_transfer_fraction": -0.040687975728418674,
      "reverse_transfer_fraction": -0.08293191159790016,
      "forward_correct_direction_fraction": 0.0,
      "control_degradation_fraction": 0.0,
      "step108_directional": true
    },
    {
      "size": 4,
      "units": [
        78,
        41,
        75,
        1
      ],
      "forward": {
        "cluster_margin_delta": 0.03698117136955261,
        "anchor_margin_delta": -0.3399323374032974,
        "cluster_optimal_mass_delta": 0.0010654536308720708,
        "control_margin_delta": 0.006997108459472656,
        "cluster_top1_repair_count": 0,
        "control_degradation_count": 0,
        "state_margin_deltas": {
          "cluster_anchor-244dadc26015ebbc": -0.3399323374032974,
          "matched_control-11f57287d52ab7f2": 0.006997108459472656,
          "one_ply_child-0c052588b0daa7dc": 0.008625268936157227,
          "one_ply_child-487e68a18d1f94c0": -0.03981703519821167,
          "one_ply_child-a10cdabeaeea0081": 0.09204784035682678,
          "one_ply_child-e95450b6d1ff9a43": 0.02377110719680786,
          "structural_cluster-0fb60203365b0571": 0.10027867555618286
        }
      },
      "reverse": {
        "cluster_margin_delta": -0.058131042122840884,
        "anchor_margin_delta": 0.12349390983581543,
        "cluster_optimal_mass_delta": -0.008207035064697266,
        "control_margin_delta": -0.022524595260620117,
        "cluster_top1_repair_count": 0,
        "control_degradation_count": 0,
        "state_margin_deltas": {
          "cluster_anchor-244dadc26015ebbc": 0.12349390983581543,
          "matched_control-11f57287d52ab7f2": -0.022524595260620117,
          "one_ply_child-0c052588b0daa7dc": 0.004797458648681641,
          "one_ply_child-487e68a18d1f94c0": 0.05000251531600952,
          "one_ply_child-a10cdabeaeea0081": -0.07200819253921509,
          "one_ply_child-e95450b6d1ff9a43": -0.08380158245563507,
          "structural_cluster-0fb60203365b0571": -0.1896454095840454
        }
      },
      "forward_transfer_fraction": -0.05943278725886435,
      "reverse_transfer_fraction": -0.09342294285616506,
      "forward_correct_direction_fraction": 0.0,
      "control_degradation_fraction": 0.0,
      "step108_directional": true
    }
  ],
  "minimal": null,
  "leave_one_out": []
}

## Classification

`a0_unit_geometry_state_specific`

Exactly one next experiment: expand the frozen exact subcluster before modifying training.
