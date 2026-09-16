# R61 Step-82 Parameter vs Adam Decomposition

Inherited #321 classification: `earliest_divergence_mixed`.

Step 82 was selected as the largest absolute replay x recipient interaction; this audit fixes replay and decomposes recipient state over steps 80-84.

## Snapshot Verification

```json
{
  "T61": true,
  "T63": true
}
```

## Five-Step Factorial Trajectories

```json
[
  {
    "step": 80,
    "batch_indexes_sha256": "94329fe8f038d2fec82f1fe5424403ec3caa37069833d2ed7faf7afb49f52e35",
    "same_t61_batch_in_all_cells": true,
    "cells": {
      "PP61_A61": {
        "step": 80,
        "cluster_a0_effect": -0.00020587444305419922,
        "control_a0_effect": 0.01614093780517578,
        "cluster_specific_a0_effect": -0.01634681224822998,
        "anchor_a0_only_margin_effect": -0.1264655590057373,
        "cluster_mean_margin": 1.4188529759645463,
        "anchor_margin": -1.9199742376804352,
        "input_layer_update_norm": 0.009778978303074837,
        "a0_representation_movement": 0.006776216812431813,
        "support_flips": 1
      },
      "P61_A63": {
        "step": 80,
        "cluster_a0_effect": 0.00039611458778381347,
        "control_a0_effect": -0.02700662612915039,
        "cluster_specific_a0_effect": 0.027402740716934205,
        "anchor_a0_only_margin_effect": 0.1226157546043396,
        "cluster_mean_margin": 1.4275958478450774,
        "anchor_margin": -1.153254747390747,
        "input_layer_update_norm": 0.009089012630283833,
        "a0_representation_movement": 0.004878586158156395,
        "support_flips": 0
      },
      "P63_A61": {
        "step": 80,
        "cluster_a0_effect": -0.035168445110321044,
        "control_a0_effect": 0.03353524208068848,
        "cluster_specific_a0_effect": -0.06870368719100953,
        "anchor_a0_only_margin_effect": -0.18300235271453857,
        "cluster_mean_margin": 1.9686266243457795,
        "anchor_margin": -0.7998379468917847,
        "input_layer_update_norm": 0.01019580103456974,
        "a0_representation_movement": 0.007245735544711351,
        "support_flips": 1
      },
      "P63_A63": {
        "step": 80,
        "cluster_a0_effect": -0.01597756743431091,
        "control_a0_effect": -0.030446946620941162,
        "cluster_specific_a0_effect": 0.01446937918663025,
        "anchor_a0_only_margin_effect": -0.0022003650665283203,
        "cluster_mean_margin": 2.092642068862915,
        "anchor_margin": 0.16256272792816162,
        "input_layer_update_norm": 0.00830575730651617,
        "a0_representation_movement": 0.0041747540701180695,
        "support_flips": 0
      }
    },
    "factorial": {
      "parameter_main_effect": -0.03264511823654175,
      "adam_main_effect": 0.06346130967140198,
      "interaction": 0.03942351341247559,
      "abs_parameter_main_effect": 0.03264511823654175,
      "abs_adam_main_effect": 0.06346130967140198,
      "abs_interaction": 0.03942351341247559
    }
  },
  {
    "step": 81,
    "batch_indexes_sha256": "afcb9f2ef7f01f391f883796be91fed5fa2eb25d52eb69a3842b37e61ccf8127",
    "same_t61_batch_in_all_cells": true,
    "cells": {
      "PP61_A61": {
        "step": 81,
        "cluster_a0_effect": 0.038923409581184384,
        "control_a0_effect": 0.08652713894844055,
        "cluster_specific_a0_effect": -0.04760372936725617,
        "anchor_a0_only_margin_effect": -0.18712452054023743,
        "cluster_mean_margin": 1.6232985138893128,
        "anchor_margin": -2.561057686805725,
        "input_layer_update_norm": 0.009913242422044277,
        "a0_representation_movement": 0.00647963285446167,
        "support_flips": 0
      },
      "P61_A63": {
        "step": 81,
        "cluster_a0_effect": 0.05549914240837097,
        "control_a0_effect": 0.029791444540023804,
        "cluster_specific_a0_effect": 0.025707697868347167,
        "anchor_a0_only_margin_effect": -0.03111809492111206,
        "cluster_mean_margin": 1.6390474632382392,
        "anchor_margin": -1.295503169298172,
        "input_layer_update_norm": 0.00904294103384018,
        "a0_representation_movement": 0.005510369781404734,
        "support_flips": 0
      },
      "P63_A61": {
        "step": 81,
        "cluster_a0_effect": 0.041444826126098636,
        "control_a0_effect": 0.04524177312850952,
        "cluster_specific_a0_effect": -0.003796947002410886,
        "anchor_a0_only_margin_effect": -0.17029714584350586,
        "cluster_mean_margin": 2.2102381378412246,
        "anchor_margin": -1.5734665244817734,
        "input_layer_update_norm": 0.009775366634130478,
        "a0_representation_movement": 0.006337158102542162,
        "support_flips": 1
      },
      "P63_A63": {
        "step": 81,
        "cluster_a0_effect": 0.02973095178604126,
        "control_a0_effect": 0.003652215003967285,
        "cluster_specific_a0_effect": 0.026078736782073973,
        "anchor_a0_only_margin_effect": -0.06965231895446777,
        "cluster_mean_margin": 2.194653847813606,
        "anchor_margin": -0.21083390712738037,
        "input_layer_update_norm": 0.007732158061116934,
        "a0_representation_movement": 0.004451016196981072,
        "support_flips": 1
      }
    },
    "factorial": {
      "parameter_main_effect": 0.022088910639286044,
      "adam_main_effect": 0.05159355551004409,
      "interaction": -0.04343574345111847,
      "abs_parameter_main_effect": 0.022088910639286044,
      "abs_adam_main_effect": 0.05159355551004409,
      "abs_interaction": 0.04343574345111847
    }
  },
  {
    "step": 82,
    "batch_indexes_sha256": "eaad8e3daaa7fa72b46cc3b58053aad661f9da96f4cd8c1708c83912846c2691",
    "same_t61_batch_in_all_cells": true,
    "cells": {
      "PP61_A61": {
        "step": 82,
        "cluster_a0_effect": 0.046933434903621674,
        "control_a0_effect": 0.03934133052825928,
        "cluster_specific_a0_effect": 0.007592104375362396,
        "anchor_a0_only_margin_effect": -0.127821683883667,
        "cluster_mean_margin": 1.8938814848661423,
        "anchor_margin": -3.00339138507843,
        "input_layer_update_norm": 0.009299010969698429,
        "a0_representation_movement": 0.0056932344101369384,
        "support_flips": 0
      },
      "P61_A63": {
        "step": 82,
        "cluster_a0_effect": 0.08296914547681808,
        "control_a0_effect": 0.056334882974624634,
        "cluster_specific_a0_effect": 0.026634262502193445,
        "anchor_a0_only_margin_effect": -0.05537053942680359,
        "cluster_mean_margin": 1.9108728080987931,
        "anchor_margin": -1.8349714875221252,
        "input_layer_update_norm": 0.00959273986518383,
        "a0_representation_movement": 0.005903661623597145,
        "support_flips": 1
      },
      "P63_A61": {
        "step": 82,
        "cluster_a0_effect": 0.04905886054039001,
        "control_a0_effect": 0.028003022074699402,
        "cluster_specific_a0_effect": 0.02105583846569061,
        "anchor_a0_only_margin_effect": -0.13136811554431915,
        "cluster_mean_margin": 2.348748852312565,
        "anchor_margin": -2.4514615535736084,
        "input_layer_update_norm": 0.009141209535300732,
        "a0_representation_movement": 0.005446651391685009,
        "support_flips": 0
      },
      "P63_A63": {
        "step": 82,
        "cluster_a0_effect": -0.008556133508682251,
        "control_a0_effect": 0.027030229568481445,
        "cluster_specific_a0_effect": -0.035586363077163695,
        "anchor_a0_only_margin_effect": -0.13662594556808472,
        "cluster_mean_margin": 2.1986606985330583,
        "anchor_margin": -1.0975831151008606,
        "input_layer_update_norm": 0.008095436729490757,
        "a0_representation_movement": 0.004922516644001007,
        "support_flips": 0
      }
    },
    "factorial": {
      "parameter_main_effect": -0.024378445744514463,
      "adam_main_effect": -0.01880002170801163,
      "interaction": -0.07568435966968536,
      "abs_parameter_main_effect": 0.024378445744514463,
      "abs_adam_main_effect": 0.01880002170801163,
      "abs_interaction": 0.07568435966968536
    }
  },
  {
    "step": 83,
    "batch_indexes_sha256": "2128c622b62dbae05d961fe13de74eee0a67897ae24d4bd1d24804df23d5f841",
    "same_t61_batch_in_all_cells": true,
    "cells": {
      "PP61_A61": {
        "step": 83,
        "cluster_a0_effect": -0.008513963222503662,
        "control_a0_effect": 0.0008664131164550781,
        "cluster_specific_a0_effect": -0.00938037633895874,
        "anchor_a0_only_margin_effect": 0.0264737606048584,
        "cluster_mean_margin": 1.8922761142253877,
        "anchor_margin": -2.7507951259613037,
        "input_layer_update_norm": 0.007879670709371567,
        "a0_representation_movement": 0.004160594940185547,
        "support_flips": 0
      },
      "P61_A63": {
        "step": 83,
        "cluster_a0_effect": 0.019718974828720093,
        "control_a0_effect": 0.03348561376333237,
        "cluster_specific_a0_effect": -0.013766638934612274,
        "anchor_a0_only_margin_effect": 0.058131396770477295,
        "cluster_mean_margin": 1.9760246083140374,
        "anchor_margin": -1.634220153093338,
        "input_layer_update_norm": 0.008290139958262444,
        "a0_representation_movement": 0.004111745301634074,
        "support_flips": 0
      },
      "P63_A61": {
        "step": 83,
        "cluster_a0_effect": -0.008766549825668334,
        "control_a0_effect": 0.0025818049907684326,
        "cluster_specific_a0_effect": -0.011348354816436767,
        "anchor_a0_only_margin_effect": -0.0693432092666626,
        "cluster_mean_margin": 2.3172420255839823,
        "anchor_margin": -2.6283681392669678,
        "input_layer_update_norm": 0.008245090954005718,
        "a0_representation_movement": 0.004216341068968177,
        "support_flips": 0
      },
      "P63_A63": {
        "step": 83,
        "cluster_a0_effect": -0.013295924663543702,
        "control_a0_effect": 0.026236891746520996,
        "cluster_specific_a0_effect": -0.039532816410064696,
        "anchor_a0_only_margin_effect": -0.07040849328041077,
        "cluster_mean_margin": 2.1419708490371705,
        "anchor_margin": -1.4495239332318306,
        "input_layer_update_norm": 0.007931909523904324,
        "a0_representation_movement": 0.004077074211090803,
        "support_flips": 2
      }
    },
    "factorial": {
      "parameter_main_effect": -0.013867077976465223,
      "adam_main_effect": -0.01628536209464073,
      "interaction": -0.023798198997974397,
      "abs_parameter_main_effect": 0.013867077976465223,
      "abs_adam_main_effect": 0.01628536209464073,
      "abs_interaction": 0.023798198997974397
    }
  },
  {
    "step": 84,
    "batch_indexes_sha256": "deede724cc9eeceab60070e731c74e166897a99799e8fd1f57c1cfc114369742",
    "same_t61_batch_in_all_cells": true,
    "cells": {
      "PP61_A61": {
        "step": 84,
        "cluster_a0_effect": -0.02441921830177307,
        "control_a0_effect": 0.036763012409210205,
        "cluster_specific_a0_effect": -0.061182230710983276,
        "anchor_a0_only_margin_effect": 0.08714187145233154,
        "cluster_mean_margin": 1.8298848003149033,
        "anchor_margin": -2.1341867446899414,
        "input_layer_update_norm": 0.008055374957621098,
        "a0_representation_movement": 0.004541590344160795,
        "support_flips": 0
      },
      "P61_A63": {
        "step": 84,
        "cluster_a0_effect": 0.012656885385513305,
        "control_a0_effect": 0.0058640241622924805,
        "cluster_specific_a0_effect": 0.0067928612232208245,
        "anchor_a0_only_margin_effect": 0.10757377743721008,
        "cluster_mean_margin": 2.060662168264389,
        "anchor_margin": -1.174962729215622,
        "input_layer_update_norm": 0.008234506472945213,
        "a0_representation_movement": 0.004463617037981748,
        "support_flips": 1
      },
      "P63_A61": {
        "step": 84,
        "cluster_a0_effect": -0.006340539455413819,
        "control_a0_effect": 0.020794928073883057,
        "cluster_specific_a0_effect": -0.027135467529296874,
        "anchor_a0_only_margin_effect": 0.035169124603271484,
        "cluster_mean_margin": 2.262285406887531,
        "anchor_margin": -2.2296138405799866,
        "input_layer_update_norm": 0.00818853173404932,
        "a0_representation_movement": 0.005198307614773512,
        "support_flips": 0
      },
      "P63_A63": {
        "step": 84,
        "cluster_a0_effect": -0.014663901925086976,
        "control_a0_effect": -0.0039692223072052,
        "cluster_specific_a0_effect": -0.010694679617881776,
        "anchor_a0_only_margin_effect": 0.02296692132949829,
        "cluster_mean_margin": 2.142369283735752,
        "anchor_margin": -1.3486423790454865,
        "input_layer_update_norm": 0.008155221119523048,
        "a0_representation_movement": 0.005114121548831463,
        "support_flips": 1
      }
    },
    "factorial": {
      "parameter_main_effect": 0.008279611170291901,
      "adam_main_effect": 0.0422079399228096,
      "interaction": -0.05153430402278901,
      "abs_parameter_main_effect": 0.008279611170291901,
      "abs_adam_main_effect": 0.0422079399228096,
      "abs_interaction": 0.05153430402278901
    }
  }
]
```

## Cumulative Effects

```json
{
  "a0_factorial": {
    "parameter_main_effect": -0.0405221201479435,
    "adam_main_effect": 0.12217742130160332,
    "interaction": -0.15502909272909163,
    "abs_parameter_main_effect": 0.0405221201479435,
    "abs_adam_main_effect": 0.12217742130160332,
    "abs_interaction": 0.15502909272909163
  },
  "margin_factorial": {
    "parameter_main_effect": 0.25705386102199557,
    "adam_main_effect": 0.055430622398853324,
    "interaction": -0.35069349110126447,
    "abs_parameter_main_effect": 0.25705386102199557,
    "abs_adam_main_effect": 0.055430622398853324,
    "abs_interaction": 0.35069349110126447
  }
}
```

## Input-Layer Parameter and Adam Localization

```json
{
  "geometry": {
    "parameters": {
      "l2_distance": 0.22216324508190155,
      "drift_from_g0_cosine": 0.7426173686981201
    },
    "adam_first_moments": {
      "l2_distance": 0.10470180213451385,
      "cosine": 0.21747533977031708
    },
    "adam_second_moments": {
      "l2_distance": 0.00019846897339448333,
      "relative_scale_ratio": 1.0954397916793823
    },
    "a0_units": [
      {
        "unit": 0,
        "row_weight_distance": 0.03439332917332649,
        "bias_difference": -0.0015346705913543701
      },
      {
        "unit": 1,
        "row_weight_distance": 0.019450919702649117,
        "bias_difference": -0.0010215453803539276
      },
      {
        "unit": 2,
        "row_weight_distance": 0.0278486255556345,
        "bias_difference": -0.0008779317140579224
      },
      {
        "unit": 3,
        "row_weight_distance": 0.02169298194348812,
        "bias_difference": 0.00042492151260375977
      },
      {
        "unit": 4,
        "row_weight_distance": 0.01817554607987404,
        "bias_difference": -2.2843480110168457e-05
      },
      {
        "unit": 5,
        "row_weight_distance": 0.01646595261991024,
        "bias_difference": 0.00097694993019104
      },
      {
        "unit": 6,
        "row_weight_distance": 0.01584654487669468,
        "bias_difference": -0.00032372772693634033
      },
      {
        "unit": 7,
        "row_weight_distance": 0.019463233649730682,
        "bias_difference": -0.001261334866285324
      },
      {
        "unit": 8,
        "row_weight_distance": 0.04858752712607384,
        "bias_difference": 0.0019841939210891724
      },
      {
        "unit": 9,
        "row_weight_distance": 0.02731710486114025,
        "bias_difference": -0.002213411033153534
      },
      {
        "unit": 10,
        "row_weight_distance": 0.02862287499010563,
        "bias_difference": -0.00030300021171569824
      },
      {
        "unit": 11,
        "row_weight_distance": 0.01994389109313488,
        "bias_difference": -0.0003864467144012451
      },
      {
        "unit": 12,
        "row_weight_distance": 0.04154663532972336,
        "bias_difference": 0.009393565356731415
      },
      {
        "unit": 13,
        "row_weight_distance": 0.01324405800551176,
        "bias_difference": 7.43865966796875e-05
      },
      {
        "unit": 14,
        "row_weight_distance": 0.009549693204462528,
        "bias_difference": 0.00044162943959236145
      },
      {
        "unit": 15,
        "row_weight_distance": 0.012852806597948074,
        "bias_difference": 0.0018263757228851318
      },
      {
        "unit": 16,
        "row_weight_distance": 0.019072342664003372,
        "bias_difference": 0.000883445143699646
      },
      {
        "unit": 17,
        "row_weight_distance": 0.030258143320679665,
        "bias_difference": -0.004445221275091171
      },
      {
        "unit": 18,
        "row_weight_distance": 0.012478470802307129,
        "bias_difference": 0.0015794187784194946
      },
      {
        "unit": 19,
        "row_weight_distance": 0.017530253157019615,
        "bias_difference": 0.0017146971076726913
      },
      {
        "unit": 20,
        "row_weight_distance": 0.02186640538275242,
        "bias_difference": 0.0031355544924736023
      },
      {
        "unit": 21,
        "row_weight_distance": 0.020731423050165176,
        "bias_difference": -0.0002811998128890991
      },
      {
        "unit": 22,
        "row_weight_distance": 0.02451879344880581,
        "bias_difference": -0.0016859844326972961
      },
      {
        "unit": 23,
        "row_weight_distance": 0.026660015806555748,
        "bias_difference": -0.0010436177253723145
      },
      {
        "unit": 24,
        "row_weight_distance": 0.01856147311627865,
        "bias_difference": 0.004755154252052307
      },
      {
        "unit": 25,
        "row_weight_distance": 0.014134379103779793,
        "bias_difference": 0.0015296563506126404
      },
      {
        "unit": 26,
        "row_weight_distance": 0.02011726424098015,
        "bias_difference": 0.0009799282997846603
      },
      {
        "unit": 27,
        "row_weight_distance": 0.023292342200875282,
        "bias_difference": -0.00036810338497161865
      },
      {
        "unit": 28,
        "row_weight_distance": 0.014723491854965687,
        "bias_difference": -0.0005565676838159561
      },
      {
        "unit": 29,
        "row_weight_distance": 0.03677836433053017,
        "bias_difference": 0.0016554370522499084
      },
      {
        "unit": 30,
        "row_weight_distance": 0.0013610431924462318,
        "bias_difference": 0.00034683942794799805
      },
      {
        "unit": 31,
        "row_weight_distance": 0.025840621441602707,
        "bias_difference": 0.0004100389778614044
      },
      {
        "unit": 32,
        "row_weight_distance": 0.017195874825119972,
        "bias_difference": 0.0003065839409828186
      },
      {
        "unit": 33,
        "row_weight_distance": 0.013493030332028866,
        "bias_difference": 0.000643312931060791
      },
      {
        "unit": 34,
        "row_weight_distance": 0.0,
        "bias_difference": 0.0
      },
      {
        "unit": 35,
        "row_weight_distance": 0.01156275998800993,
        "bias_difference": -0.0003095269203186035
      },
      {
        "unit": 36,
        "row_weight_distance": 0.014832135289907455,
        "bias_difference": 0.00024412572383880615
      },
      {
        "unit": 37,
        "row_weight_distance": 0.0,
        "bias_difference": 0.0
      },
      {
        "unit": 38,
        "row_weight_distance": 0.02565760351717472,
        "bias_difference": 0.0009962096810340881
      },
      {
        "unit": 39,
        "row_weight_distance": 0.019482843577861786,
        "bias_difference": -0.0002079010009765625
      },
      {
        "unit": 40,
        "row_weight_distance": 0.018899284303188324,
        "bias_difference": -2.6501715183258057e-05
      },
      {
        "unit": 41,
        "row_weight_distance": 0.02011485956609249,
        "bias_difference": -0.0008704811334609985
      },
      {
        "unit": 42,
        "row_weight_distance": 0.01828613132238388,
        "bias_difference": 0.002386830747127533
      },
      {
        "unit": 43,
        "row_weight_distance": 0.0219014510512352,
        "bias_difference": 0.001648150384426117
      },
      {
        "unit": 44,
        "row_weight_distance": 0.018315350636839867,
        "bias_difference": 0.00021085143089294434
      },
      {
        "unit": 45,
        "row_weight_distance": 0.026853060349822044,
        "bias_difference": -0.002283763140439987
      },
      {
        "unit": 46,
        "row_weight_distance": 0.025388462468981743,
        "bias_difference": -0.0017804724629968405
      },
      {
        "unit": 47,
        "row_weight_distance": 0.022717207670211792,
        "bias_difference": 0.0009156316518783569
      },
      {
        "unit": 48,
        "row_weight_distance": 0.024782825261354446,
        "bias_difference": -0.004659004509449005
      },
      {
        "unit": 49,
        "row_weight_distance": 0.011517544277012348,
        "bias_difference": 0.0020858291536569595
      },
      {
        "unit": 50,
        "row_weight_distance": 0.019336221739649773,
        "bias_difference": -0.002591542899608612
      },
      {
        "unit": 51,
        "row_weight_distance": 0.014463105238974094,
        "bias_difference": -0.0005926266312599182
      },
      {
        "unit": 52,
        "row_weight_distance": 0.02669774554669857,
        "bias_difference": -0.0022113248705863953
      },
      {
        "unit": 53,
        "row_weight_distance": 0.01175715122371912,
        "bias_difference": -0.002873547375202179
      },
      {
        "unit": 54,
        "row_weight_distance": 0.023451434448361397,
        "bias_difference": -0.000357896089553833
      },
      {
        "unit": 55,
        "row_weight_distance": 0.013426448218524456,
        "bias_difference": 0.0032211244106292725
      },
      {
        "unit": 56,
        "row_weight_distance": 0.015952887013554573,
        "bias_difference": -0.00045828521251678467
      },
      {
        "unit": 57,
        "row_weight_distance": 0.017676912248134613,
        "bias_difference": 0.0026070624589920044
      },
      {
        "unit": 58,
        "row_weight_distance": 0.01549601275473833,
        "bias_difference": -0.0007992833852767944
      },
      {
        "unit": 59,
        "row_weight_distance": 0.019623015075922012,
        "bias_difference": 0.00017309188842773438
      },
      {
        "unit": 60,
        "row_weight_distance": 0.019121132791042328,
        "bias_difference": 0.0006368234753608704
      },
      {
        "unit": 61,
        "row_weight_distance": 0.01785542443394661,
        "bias_difference": -0.0010056346654891968
      },
      {
        "unit": 62,
        "row_weight_distance": 0.02936105616390705,
        "bias_difference": -0.001008160412311554
      },
      {
        "unit": 63,
        "row_weight_distance": 0.026134375482797623,
        "bias_difference": 0.000976845622062683
      },
      {
        "unit": 64,
        "row_weight_distance": 0.016884204000234604,
        "bias_difference": 0.0017905160784721375
      },
      {
        "unit": 65,
        "row_weight_distance": 0.027747200801968575,
        "bias_difference": 0.0002011433243751526
      },
      {
        "unit": 66,
        "row_weight_distance": 0.018125638365745544,
        "bias_difference": 0.003942783921957016
      },
      {
        "unit": 67,
        "row_weight_distance": 0.029133176431059837,
        "bias_difference": -0.001346580684185028
      },
      {
        "unit": 68,
        "row_weight_distance": 0.032069917768239975,
        "bias_difference": 0.0016582459211349487
      },
      {
        "unit": 69,
        "row_weight_distance": 0.01347687840461731,
        "bias_difference": 0.0004385039210319519
      },
      {
        "unit": 70,
        "row_weight_distance": 0.019522394984960556,
        "bias_difference": 0.0008291378617286682
      },
      {
        "unit": 71,
        "row_weight_distance": 0.02799876034259796,
        "bias_difference": 0.0011054202914237976
      },
      {
        "unit": 72,
        "row_weight_distance": 0.03447885438799858,
        "bias_difference": 0.0022659003734588623
      },
      {
        "unit": 73,
        "row_weight_distance": 0.0,
        "bias_difference": 0.0
      },
      {
        "unit": 74,
        "row_weight_distance": 0.0,
        "bias_difference": 0.0
      },
      {
        "unit": 75,
        "row_weight_distance": 0.015434193424880505,
        "bias_difference": 0.002368524670600891
      },
      {
        "unit": 76,
        "row_weight_distance": 0.02042410708963871,
        "bias_difference": -0.0006737727671861649
      },
      {
        "unit": 77,
        "row_weight_distance": 0.0181940495967865,
        "bias_difference": -0.002240613102912903
      },
      {
        "unit": 78,
        "row_weight_distance": 0.013402451761066914,
        "bias_difference": -0.0010070893913507462
      },
      {
        "unit": 79,
        "row_weight_distance": 0.020325930789113045,
        "bias_difference": 0.0004039853811264038
      },
      {
        "unit": 80,
        "row_weight_distance": 0.0,
        "bias_difference": 0.0
      },
      {
        "unit": 81,
        "row_weight_distance": 0.05000792816281319,
        "bias_difference": -0.005605459213256836
      },
      {
        "unit": 82,
        "row_weight_distance": 0.05160628631711006,
        "bias_difference": 4.172325134277344e-05
      },
      {
        "unit": 83,
        "row_weight_distance": 0.029032452031970024,
        "bias_difference": 0.002514413557946682
      },
      {
        "unit": 84,
        "row_weight_distance": 0.01847110688686371,
        "bias_difference": 0.0014310255646705627
      },
      {
        "unit": 85,
        "row_weight_distance": 0.040762558579444885,
        "bias_difference": 0.008024975657463074
      },
      {
        "unit": 86,
        "row_weight_distance": 0.017756987363100052,
        "bias_difference": 0.0009722858667373657
      },
      {
        "unit": 87,
        "row_weight_distance": 0.01831440068781376,
        "bias_difference": 0.00032639503479003906
      },
      {
        "unit": 88,
        "row_weight_distance": 0.022426342591643333,
        "bias_difference": -0.001682877540588379
      },
      {
        "unit": 89,
        "row_weight_distance": 0.015187347307801247,
        "bias_difference": -0.00011646747589111328
      },
      {
        "unit": 90,
        "row_weight_distance": 0.017496280372142792,
        "bias_difference": 5.260109901428223e-06
      },
      {
        "unit": 91,
        "row_weight_distance": 0.017812885344028473,
        "bias_difference": -0.0002837330102920532
      },
      {
        "unit": 92,
        "row_weight_distance": 0.01890474557876587,
        "bias_difference": 0.0019621923565864563
      },
      {
        "unit": 93,
        "row_weight_distance": 0.012977606616914272,
        "bias_difference": -4.947185516357422e-05
      },
      {
        "unit": 94,
        "row_weight_distance": 0.021649744361639023,
        "bias_difference": 0.0006600767374038696
      },
      {
        "unit": 95,
        "row_weight_distance": 0.018352627754211426,
        "bias_difference": -0.0018481345614418387
      }
    ]
  },
  "top_a0_units": [
    {
      "unit": 52,
      "cluster_mean_absolute_delta": 0.014829003810882568,
      "matched_control_delta": 0.0,
      "relu_support_flip_frequency": 0.0,
      "anchor_margin_first_order_contribution": -0.026745691895484924
    },
    {
      "unit": 79,
      "cluster_mean_absolute_delta": 0.010987761616706847,
      "matched_control_delta": 0.0,
      "relu_support_flip_frequency": 0.0,
      "anchor_margin_first_order_contribution": 0.04476227983832359
    },
    {
      "unit": 57,
      "cluster_mean_absolute_delta": 0.010444647073745728,
      "matched_control_delta": 0.013653546571731567,
      "relu_support_flip_frequency": 0.2,
      "anchor_margin_first_order_contribution": 0.0
    },
    {
      "unit": 55,
      "cluster_mean_absolute_delta": 0.010203735530376434,
      "matched_control_delta": 0.0,
      "relu_support_flip_frequency": 0.0,
      "anchor_margin_first_order_contribution": -0.14454244077205658
    },
    {
      "unit": 48,
      "cluster_mean_absolute_delta": 0.01017295867204666,
      "matched_control_delta": 0.0,
      "relu_support_flip_frequency": 0.4,
      "anchor_margin_first_order_contribution": 0.0
    },
    {
      "unit": 1,
      "cluster_mean_absolute_delta": 0.008378402888774871,
      "matched_control_delta": 0.0027489718049764633,
      "relu_support_flip_frequency": 0.0,
      "anchor_margin_first_order_contribution": -0.007526183035224676
    },
    {
      "unit": 36,
      "cluster_mean_absolute_delta": 0.008307746052742005,
      "matched_control_delta": 0.0042448341846466064,
      "relu_support_flip_frequency": 0.0,
      "anchor_margin_first_order_contribution": -0.006732847075909376
    },
    {
      "unit": 18,
      "cluster_mean_absolute_delta": 0.008101576566696167,
      "matched_control_delta": 0.0071408748626708984,
      "relu_support_flip_frequency": 0.0,
      "anchor_margin_first_order_contribution": -0.05650515481829643
    },
    {
      "unit": 75,
      "cluster_mean_absolute_delta": 0.0075570732355117794,
      "matched_control_delta": 0.00807216763496399,
      "relu_support_flip_frequency": 0.0,
      "anchor_margin_first_order_contribution": -0.013741446658968925
    },
    {
      "unit": 0,
      "cluster_mean_absolute_delta": 0.007488480210304261,
      "matched_control_delta": 0.0,
      "relu_support_flip_frequency": 0.0,
      "anchor_margin_first_order_contribution": 0.0
    }
  ],
  "input_parameter_transplants": {
    "T61_with_T63_input": {
      "cumulative_cluster_specific_a0_movement": -0.1862959861755371,
      "final_cluster_mean_margin": 2.0172528265044094,
      "final_anchor_margin": -1.6605295538902283,
      "final_a0_distance_from_own_initial_state": 0.02186397910118103
    },
    "T63_with_T61_input": {
      "cumulative_cluster_specific_a0_movement": 0.3513225391507149,
      "final_cluster_mean_margin": 2.0462872684001923,
      "final_anchor_margin": -1.6175909340381622,
      "final_a0_distance_from_own_initial_state": 0.015194962173700333
    }
  },
  "input_layer_fraction": 0.4333204517788644,
  "input_adam_transplants": {
    "T61_input_adam_T63": {
      "cumulative_cluster_specific_a0_movement": 0.5077751874923706,
      "final_cluster_mean_margin": 1.9142198853194714,
      "final_anchor_margin": -1.5554412603378296,
      "final_a0_distance_from_own_initial_state": 0.02369164675474167
    },
    "T63_input_adam_T61": {
      "cumulative_cluster_specific_a0_movement": -0.6155146658420563,
      "final_cluster_mean_margin": 2.2355191841721536,
      "final_anchor_margin": -1.5618027448654175,
      "final_a0_distance_from_own_initial_state": 0.034289709851145746
    }
  },
  "downstream_fixed_context": {
    "T61_downstream_T63_input": {
      "cluster_mean_margin_delta": 0.3344240814447403,
      "anchor_margin_delta": 0.33083850145339966
    },
    "T63_downstream_T61_input": {
      "cluster_mean_margin_delta": -0.3492376208305359,
      "anchor_margin_delta": -0.21155500411987305
    }
  }
}
```

## T63-Batch Robustness

```json
{
  "steps": [
    {
      "step": 80,
      "responses": {
        "PP61_A61": 0.001822063326835632,
        "P61_A63": 0.047627052664756774,
        "P63_A61": -0.03514017462730408,
        "P63_A63": 0.0484390377998352
      },
      "factorial": {
        "parameter_main_effect": -0.018075126409530642,
        "adam_main_effect": 0.06469210088253022,
        "interaction": 0.03777422308921814,
        "abs_parameter_main_effect": 0.018075126409530642,
        "abs_adam_main_effect": 0.06469210088253022,
        "abs_interaction": 0.03777422308921814
      }
    },
    {
      "step": 81,
      "responses": {
        "PP61_A61": -0.012916284799575808,
        "P61_A63": 0.04223931450396776,
        "P63_A61": -0.0377910315990448,
        "P63_A63": 0.00999491810798645
      },
      "factorial": {
        "parameter_main_effect": -0.028559571597725153,
        "adam_main_effect": 0.05147077450528741,
        "interaction": -0.007369649596512318,
        "abs_parameter_main_effect": 0.028559571597725153,
        "abs_adam_main_effect": 0.05147077450528741,
        "abs_interaction": 0.007369649596512318
      }
    },
    {
      "step": 82,
      "responses": {
        "PP61_A61": 0.0038225114345550523,
        "P61_A63": 0.02287845462560653,
        "P63_A61": -0.02683899998664856,
        "P63_A63": -0.02303563356399536
      },
      "factorial": {
        "parameter_main_effect": -0.038287799805402756,
        "adam_main_effect": 0.01142965480685234,
        "interaction": -0.01525257676839828,
        "abs_parameter_main_effect": 0.038287799805402756,
        "abs_adam_main_effect": 0.01142965480685234,
        "abs_interaction": 0.01525257676839828
      }
    },
    {
      "step": 83,
      "responses": {
        "PP61_A61": 0.0036797165870666518,
        "P61_A63": 0.017867392301559447,
        "P63_A61": -0.015739911794662477,
        "P63_A63": -0.02125008702278137
      },
      "factorial": {
        "parameter_main_effect": -0.029268553853034972,
        "adam_main_effect": 0.00433875024318695,
        "interaction": -0.01969785094261169,
        "abs_parameter_main_effect": 0.029268553853034972,
        "abs_adam_main_effect": 0.00433875024318695,
        "abs_interaction": 0.01969785094261169
      }
    },
    {
      "step": 84,
      "responses": {
        "PP61_A61": -0.043277549743652347,
        "P61_A63": 0.055847978591918944,
        "P63_A61": -0.03916067481040955,
        "P63_A63": 0.00990276038646698
      },
      "factorial": {
        "parameter_main_effect": -0.020914171636104584,
        "adam_main_effect": 0.07409448176622391,
        "interaction": -0.05006209313869475,
        "abs_parameter_main_effect": 0.020914171636104584,
        "abs_adam_main_effect": 0.07409448176622391,
        "abs_interaction": 0.05006209313869475
      }
    }
  ],
  "final_a0_factorial": {
    "parameter_main_effect": -0.13510522330179808,
    "adam_main_effect": 0.2060257622040808,
    "interaction": -0.05460794735699889,
    "abs_parameter_main_effect": 0.13510522330179808,
    "abs_adam_main_effect": 0.2060257622040808,
    "abs_interaction": 0.05460794735699889
  }
}
```

## Classification

`early_divergence_local_decomposition_mixed`

Exactly one next experiment: move to the input-layer/A0 unit-level representation geometry at steps 80–82 rather than doing another optimizer-state sweep.
