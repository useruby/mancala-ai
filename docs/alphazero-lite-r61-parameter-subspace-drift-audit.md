# R61 Parameter Subspace Drift Audit

Inherited PR #311 classification: `grad_clip_relaxation_unstable`.

## Result

Hard classification: `anchor_drift_shared_trunk_primary`

Exactly one next experiment: run one frozen-R61 existing trainable-scope ablation chosen from this audit to limit destructive trunk movement while allowing policy learning.

## Baseline Reproduction

| seed | expected canonical delta | observed checkpoint delta | final top |
| --- | ---: | ---: | ---: |
| T61 | -0.1262 | -0.1262 | 3 |
| T63 | 0.1548 | 0.1548 | 0 |

## Parameter Groups

`trunk_input=input_layer.*`; `trunk_residual=residual_layers.*`; `policy_hidden=policy_hidden_layer.*`; `policy_readout=policy_head.*`; `value_hidden=value_hidden_layer.*`; `value_readout=value_head.*`. The runner rejects any unclassified or overlapping trainable tensor. Rolled-up groups are `shared_trunk`, `policy_path`, and `value_path`.

## Hybrid Matrix

| hybrid | anchor top | anchor mass | margin | cluster mass | control mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| T61_native | 3 | 0.2737 | -0.2410 | 0.6980 | 0.8100 |
| T63_native | 0 | 0.5548 | 0.9733 | 0.7137 | 0.8352 |
| T61_heads_T63_shared_trunk | 0 | 0.5400 | 0.8377 | 0.7020 | 0.8162 |
| T63_heads_T61_shared_trunk | 3 | 0.2656 | -0.1657 | 0.7140 | 0.8264 |
| T61_trunk_value_T63_policy_path | 3 | 0.2656 | -0.1657 | 0.7140 | 0.8264 |
| T63_trunk_value_T61_policy_path | 0 | 0.5400 | 0.8377 | 0.7020 | 0.8162 |
| T61_T63_policy_hidden | 3 | 0.2908 | -0.1060 | 0.7087 | 0.8147 |
| T61_T63_policy_readout | 3 | 0.2493 | -0.3056 | 0.7037 | 0.8222 |
| T63_T61_policy_hidden | 0 | 0.5247 | 0.8231 | 0.7102 | 0.8263 |
| T63_T61_policy_readout | 0 | 0.5703 | 0.9952 | 0.7052 | 0.8256 |
| T61_T63_value_path | 3 | 0.2737 | -0.2410 | 0.6980 | 0.8100 |
| T63_T61_value_path | 0 | 0.5548 | 0.9733 | 0.7137 | 0.8352 |

All per-step scalar telemetry, matched-progress distances, anchor probes, feature/logit decompositions, and reverse transplants are retained in the JSON artifact. Value-path swaps are negative controls; they cannot directly enter residual_v3 policy logits.
