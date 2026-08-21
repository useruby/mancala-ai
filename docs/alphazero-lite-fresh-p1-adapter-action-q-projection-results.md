# PR #218 Action-Level Robust-Q Projection

**Classification:** `supervision_magnitude_explains_rescue`

**Recommended next experiment:** Do not build an expensive Q-gated target pipeline.

## Pre-Training Audit

Original CE opportunity retained: 0.9519

| Metric | P50 | P75 | P90 | P95 | P99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| legal_actions_supported_fraction | 0.6667 | 0.8000 | 1.0000 | 1.0000 | 1.0000 |
| positive_delta_actions_supported_fraction | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| negative_delta_actions_supported_fraction | 0.6667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| original_teacher_l1 | 0.8252 | 1.3352 | 1.6585 | 1.7779 | 1.9105 |
| projected_teacher_l1 | 0.4873 | 0.9125 | 1.2696 | 1.4459 | 1.7385 |
| retained_delta_mass | 0.2437 | 0.4563 | 0.6348 | 0.7229 | 0.8692 |
| discarded_delta_mass | 0.0981 | 0.2486 | 0.4156 | 0.5309 | 0.7612 |

## Expected-Q Change

| Teacher | Q budget | Mean | P50 | P90 |
| --- | ---: | ---: | ---: | ---: |
| stored384 | 384 | +0.068898 | +0.029227 | +0.183746 |
| stored384 | 1200 | +0.082662 | +0.030908 | +0.228495 |
| action_q_projected | 384 | +0.062205 | +0.025424 | +0.161546 |
| action_q_projected | 1200 | +0.070655 | +0.023583 | +0.194735 |
| magnitude_matched | 384 | +0.055266 | +0.018176 | +0.145647 |
| magnitude_matched | 1200 | +0.063751 | +0.017642 | +0.177270 |

## Gradient Direction

```json
{
  "cosines": {
    "action_q_projected_vs_magnitude_matched": 0.9871602058410645,
    "baseline_stored384_vs_action_q_projected": 0.8690527677536011,
    "baseline_stored384_vs_magnitude_matched": 0.9132032990455627
  },
  "norms": {
    "action_q_projected": 0.005827634129673243,
    "baseline_stored384": 0.008602460846304893,
    "magnitude_matched": 0.005674439948052168
  }
}
```

## Training

| Lane | Step | Original fit | Lane fit | Mean L1 | Q-supported movement |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline_stored384 | 1 | 0.0182 | 0.0182 | 0.000120 | 0.6868 |
| baseline_stored384 | 4 | 0.0864 | 0.0864 | 0.000500 | 0.6874 |
| baseline_stored384 | 16 | 0.3260 | 0.3260 | 0.001754 | 0.6876 |
| baseline_stored384 | 46 | 0.7557 | 0.7557 | 0.003667 | 0.6888 |
| action_q_projected | 1 | 0.0236 | 0.0257 | 0.000133 | 0.6847 |
| action_q_projected | 4 | 0.0965 | 0.1021 | 0.000565 | 0.6876 |
| action_q_projected | 16 | 0.3252 | 0.3612 | 0.001838 | 0.6857 |
| action_q_projected | 46 | 0.7124 | 0.7744 | 0.003589 | 0.6867 |
| magnitude_matched | 1 | 0.0216 | 0.0231 | 0.000128 | 0.6825 |
| magnitude_matched | 4 | 0.0927 | 0.0971 | 0.000551 | 0.6865 |
| magnitude_matched | 16 | 0.3142 | 0.3369 | 0.001745 | 0.6855 |
| magnitude_matched | 46 | 0.6679 | 0.6986 | 0.003236 | 0.6872 |

## Post-Training Value Support

| Lane | Step | Q budget | Mean | Positive fraction | Positive mass | Negative mass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_stored384 | 1 | 384 | +0.000002 | 0.4954 | 0.000017 | -0.000016 |
| baseline_stored384 | 1 | 1200 | +0.000002 | 0.4934 | 0.000019 | -0.000017 |
| baseline_stored384 | 4 | 384 | +0.000008 | 0.5292 | 0.000072 | -0.000064 |
| baseline_stored384 | 4 | 1200 | +0.000009 | 0.5122 | 0.000080 | -0.000071 |
| baseline_stored384 | 16 | 384 | +0.000028 | 0.5408 | 0.000250 | -0.000222 |
| baseline_stored384 | 16 | 1200 | +0.000029 | 0.5224 | 0.000275 | -0.000246 |
| baseline_stored384 | 46 | 384 | +0.000062 | 0.5437 | 0.000533 | -0.000470 |
| baseline_stored384 | 46 | 1200 | +0.000065 | 0.5251 | 0.000586 | -0.000521 |
| action_q_projected | 1 | 384 | +0.000002 | 0.5363 | 0.000019 | -0.000017 |
| action_q_projected | 1 | 1200 | +0.000002 | 0.5124 | 0.000020 | -0.000019 |
| action_q_projected | 4 | 384 | +0.000009 | 0.5419 | 0.000080 | -0.000071 |
| action_q_projected | 4 | 1200 | +0.000009 | 0.5228 | 0.000088 | -0.000078 |
| action_q_projected | 16 | 384 | +0.000028 | 0.5386 | 0.000261 | -0.000233 |
| action_q_projected | 16 | 1200 | +0.000028 | 0.5160 | 0.000286 | -0.000258 |
| action_q_projected | 46 | 384 | +0.000058 | 0.5401 | 0.000519 | -0.000461 |
| action_q_projected | 46 | 1200 | +0.000059 | 0.5191 | 0.000568 | -0.000510 |
| magnitude_matched | 1 | 384 | +0.000002 | 0.5222 | 0.000018 | -0.000016 |
| magnitude_matched | 1 | 1200 | +0.000002 | 0.5049 | 0.000020 | -0.000018 |
| magnitude_matched | 4 | 384 | +0.000009 | 0.5376 | 0.000078 | -0.000069 |
| magnitude_matched | 4 | 1200 | +0.000009 | 0.5166 | 0.000085 | -0.000076 |
| magnitude_matched | 16 | 384 | +0.000027 | 0.5347 | 0.000248 | -0.000221 |
| magnitude_matched | 16 | 1200 | +0.000027 | 0.5119 | 0.000272 | -0.000245 |
| magnitude_matched | 46 | 384 | +0.000054 | 0.5488 | 0.000472 | -0.000418 |
| magnitude_matched | 46 | 1200 | +0.000056 | 0.5275 | 0.000517 | -0.000461 |

## Arena

| Lane | Step | Context | Effect | 95% CI | Seat P0 | Seat P1 |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| baseline_stored384 | 46 | 384:256 | -0.0195 | [-0.0391, -0.0039] | -0.0391 | +0.0000 |
| baseline_stored384 | 46 | 1200:1200 | -0.0234 | [-0.0430, -0.0078] | -0.0469 | +0.0000 |
| action_q_projected | 16 | 384:256 | -0.0176 | [-0.0293, -0.0078] | -0.0195 | -0.0156 |
| action_q_projected | 16 | 1200:1200 | -0.0234 | [-0.0430, -0.0078] | -0.0469 | +0.0000 |
| action_q_projected | 46 | 384:256 | -0.0098 | [-0.0195, -0.0020] | -0.0195 | +0.0000 |
| action_q_projected | 46 | 1200:1200 | -0.0195 | [-0.0391, -0.0039] | -0.0391 | +0.0000 |
| magnitude_matched | 16 | 384:256 | -0.0176 | [-0.0293, -0.0078] | -0.0195 | -0.0156 |
| magnitude_matched | 16 | 1200:1200 | -0.0234 | [-0.0430, -0.0078] | -0.0469 | +0.0000 |
| magnitude_matched | 46 | 384:256 | -0.0098 | [-0.0195, -0.0020] | -0.0195 | +0.0000 |
| magnitude_matched | 46 | 1200:1200 | -0.0234 | [-0.0430, -0.0078] | -0.0469 | +0.0000 |

## Contracts

```json
{
  "guardrails": {
    "batch_size": 512,
    "beta": 0.95,
    "fresh_self_play_generated": false,
    "grad_clip": 1.0,
    "lr": 1e-05,
    "optimizer": "Adam",
    "promotion": false,
    "steps": 46,
    "trainable_parameters": [
      "policy_adapter.weight",
      "policy_adapter.bias"
    ],
    "weight_decay": 0.0
  },
  "hashes": {
    "arena_suite": "ff21c42946ed32f525c5ab95b71c4d90a4cfe2ccc351406dd29cae52da4b1837",
    "batch_manifest": "4f476379b1ae4a5687591a31d6726f943643d6e3b2c36d72fa233d37613892c3",
    "p1_checkpoint": "e4a9c95302c17b63405107d26b4cd4f90755d2b21980846ac6e9f0baabf7efe9",
    "replay": "892827d8ee67a66e6324a2aaec7011df1a21625fc3f6bcd87cab39ce655d2a88",
    "root_q_cache": "57963336d66bd7fe9b2fc83252f2a1f5fd621578a5788cdb44ec0da8cb94231f"
  },
  "invariants": {
    "a16_state_hash": true,
    "all_passed": true,
    "baseline_reproduced": true,
    "filtered_actions_unchanged": true,
    "full_replay_state_round_trip": true,
    "magnitude_matched": true,
    "non_adapter_parameters_bit_identical": true,
    "p1_checkpoint_hash": true,
    "pr214_batch_plan": true,
    "pr217_batch_manifest_hash": true,
    "projection_normalization": true,
    "replay_hash": true,
    "root_q_cache_hash": true,
    "root_q_visit_contract": true,
    "same_initial_state": true
  }
}
```
