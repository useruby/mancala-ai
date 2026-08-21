# PR #214 Root-Q Advantage-Gated Adapter Retrain

**Primary classification:** `root_q_sign_not_predictive`

**Recommended next experiment:** Move from state-level expected advantage to action-level/search-trajectory sensitivity rather than another teacher-budget change.

## Root-Q Audit

Child Q is backed up in the root player-to-move perspective. Every legal child is cached; an unvisited child records the existing `zero_fpu` Q=0 contract explicitly.

| Group | Unique fraction | Replay-row fraction | Stored CE opportunity |
| --- | ---: | ---: | ---: |
| robust_positive | 0.7798 | 0.7721 | 0.9027 |
| budget_conflicted | 0.0643 | 0.0641 | 0.0597 |
| robust_nonpositive | 0.1560 | 0.1638 | 0.0377 |

| Delta-Q budget | Mean | Median | P10 | P25 | P75 | P90 | P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 384 | 0.077860 | 0.038006 | -0.001233 | 0.003967 | 0.103183 | 0.200267 | 0.298199 |
| 1200 | 0.093618 | 0.041359 | -0.000085 | 0.004719 | 0.117618 | 0.254355 | 0.382630 |

## Training

| Lane | Step | CE(stored384) | CE(P1) | CE(lane target) | Fit fraction | Mean L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_stored384 | 1 | 1.100826 | 0.947235 | 0.954914 | 0.0182 | 0.000120 |
| baseline_stored384 | 4 | 1.100762 | 0.947235 | 0.954911 | 0.0864 | 0.000500 |
| baseline_stored384 | 16 | 1.100536 | 0.947237 | 0.954902 | 0.3260 | 0.001754 |
| baseline_stored384 | 46 | 1.100130 | 0.947246 | 0.954891 | 0.7557 | 0.003667 |
| robust_advantage | 1 | 1.100825 | 0.947235 | 0.954733 | 0.0193 | 0.000144 |
| robust_advantage | 4 | 1.100760 | 0.947235 | 0.954730 | 0.0885 | 0.000540 |
| robust_advantage | 16 | 1.100546 | 0.947237 | 0.954721 | 0.3152 | 0.001701 |
| robust_advantage | 46 | 1.100145 | 0.947246 | 0.954710 | 0.7392 | 0.003615 |
| matched_random | 1 | 1.100828 | 0.947235 | 0.953190 | 0.0168 | 0.000138 |
| matched_random | 4 | 1.100763 | 0.947235 | 0.953188 | 0.0850 | 0.000523 |
| matched_random | 16 | 1.100552 | 0.947237 | 0.953182 | 0.3081 | 0.001676 |
| matched_random | 46 | 1.100206 | 0.947244 | 0.953174 | 0.6753 | 0.003226 |

| Lane | Step | Robust-positive fit | Budget-conflicted fit | Robust-nonpositive fit |
| --- | ---: | ---: | ---: | ---: |
| baseline_stored384 | 1 | 0.000020 | 0.000001 | 0.000008 |
| baseline_stored384 | 4 | 0.000098 | -0.000000 | 0.000038 |
| baseline_stored384 | 16 | 0.000379 | 0.000002 | 0.000091 |
| baseline_stored384 | 46 | 0.000875 | 0.000058 | 0.000206 |
| robust_advantage | 1 | 0.000023 | -0.000002 | 0.000004 |
| robust_advantage | 4 | 0.000103 | -0.000006 | 0.000025 |
| robust_advantage | 16 | 0.000383 | -0.000012 | 0.000017 |
| robust_advantage | 46 | 0.000890 | 0.000027 | 0.000053 |
| matched_random | 1 | 0.000020 | 0.000001 | 0.000001 |
| matched_random | 4 | 0.000100 | 0.000006 | 0.000015 |
| matched_random | 16 | 0.000370 | 0.000010 | 0.000029 |
| matched_random | 46 | 0.000798 | 0.000063 | 0.000104 |

## Value-Support Diagnostics

| Lane | Step | Q budget | Mean expected-Q change | Positive fraction |
| --- | ---: | --- | ---: | ---: |
| baseline_stored384 | 1 | 384 | +0.000002 | 0.5215 |
| baseline_stored384 | 1 | 1200 | +0.000002 | 0.5247 |
| baseline_stored384 | 4 | 384 | +0.000010 | 0.5468 |
| baseline_stored384 | 4 | 1200 | +0.000011 | 0.5405 |
| baseline_stored384 | 16 | 384 | +0.000034 | 0.5591 |
| baseline_stored384 | 16 | 1200 | +0.000039 | 0.5525 |
| baseline_stored384 | 46 | 384 | +0.000074 | 0.5628 |
| baseline_stored384 | 46 | 1200 | +0.000085 | 0.5557 |
| robust_advantage | 1 | 384 | +0.000002 | 0.5217 |
| robust_advantage | 1 | 1200 | +0.000003 | 0.5279 |
| robust_advantage | 4 | 384 | +0.000010 | 0.5413 |
| robust_advantage | 4 | 1200 | +0.000012 | 0.5389 |
| robust_advantage | 16 | 384 | +0.000031 | 0.5594 |
| robust_advantage | 16 | 1200 | +0.000035 | 0.5541 |
| robust_advantage | 46 | 384 | +0.000068 | 0.5629 |
| robust_advantage | 46 | 1200 | +0.000077 | 0.5549 |
| matched_random | 1 | 384 | +0.000002 | 0.5190 |
| matched_random | 1 | 1200 | +0.000002 | 0.5262 |
| matched_random | 4 | 384 | +0.000010 | 0.5491 |
| matched_random | 4 | 1200 | +0.000012 | 0.5448 |
| matched_random | 16 | 384 | +0.000031 | 0.5637 |
| matched_random | 16 | 1200 | +0.000036 | 0.5554 |
| matched_random | 46 | 384 | +0.000064 | 0.5643 |
| matched_random | 46 | 1200 | +0.000073 | 0.5570 |

## Arena

| Lane | Step | Context | Effect | 95% CI | Seat P0 | Seat P1 |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| baseline_stored384 | 46 | 384:256 | -0.0195 | [-0.0391, -0.0039] | -0.0391 | +0.0000 |
| baseline_stored384 | 46 | 1200:1200 | -0.0234 | [-0.0430, -0.0078] | -0.0469 | +0.0000 |
| robust_advantage | 16 | 384:256 | -0.0176 | [-0.0293, -0.0078] | -0.0195 | -0.0156 |
| robust_advantage | 46 | 384:256 | -0.0195 | [-0.0391, -0.0039] | -0.0391 | +0.0000 |
| matched_random | 16 | 384:256 | -0.0176 | [-0.0293, -0.0078] | -0.0195 | -0.0156 |
| matched_random | 46 | 384:256 | -0.0195 | [-0.0391, -0.0039] | -0.0391 | +0.0000 |

## Contracts

```json
{
  "hashes": {
    "arena_suite": "ff21c42946ed32f525c5ab95b71c4d90a4cfe2ccc351406dd29cae52da4b1837",
    "batch_manifest": "4f476379b1ae4a5687591a31d6726f943643d6e3b2c36d72fa233d37613892c3",
    "clean_target_cache": "2e8f9168db67297505b6ad8b0059d4aa521214d32adcc50191f40c7f8112fd16",
    "p1_checkpoint": "e4a9c95302c17b63405107d26b4cd4f90755d2b21980846ac6e9f0baabf7efe9",
    "replay": "892827d8ee67a66e6324a2aaec7011df1a21625fc3f6bcd87cab39ce655d2a88",
    "root_q_cache": "57963336d66bd7fe9b2fc83252f2a1f5fd621578a5788cdb44ec0da8cb94231f"
  },
  "invariants": {
    "a16_state_hash": true,
    "all_passed": true,
    "baseline_reproduced": true,
    "full_replay_state_round_trip": true,
    "matched_random_unique_count": true,
    "non_adapter_parameters_bit_identical": true,
    "p1_checkpoint_hash": true,
    "pr214_batch_plan": true,
    "pr217_batch_manifest_hash": true,
    "pr217_clean_target_cache_hash": true,
    "replay_hash": true,
    "root_q_perspective": true,
    "same_initial_state": true
  },
  "matched_random": {
    "matched_random_unique_states": 18180,
    "random_mask_sha256": "aada4c0f2c75d567813282d49b7ae453fdb3b42db014b270a9ec685dc2bf6474",
    "robust_mask_sha256": "99edca0421620f917133f494a1a492db4cc6fe68f5172008a9a66e0dc9fa8b11",
    "robust_positive_unique_states": 18180,
    "seed": 217
  },
  "search_contract": {
    "c_puct": 1.25,
    "fpu_mode": "zero",
    "q_perspective": "root player to move; higher is better",
    "root_noise": false,
    "seed": "sha256(pr214-teacher-audit:encoded-state-hash)",
    "simulations": [
      384,
      1200
    ],
    "unvisited_q_rule": "zero_fpu: q_value=0.0, recorded explicitly"
  }
}
```
