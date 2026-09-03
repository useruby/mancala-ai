# PR #265 Unique-Data Scale Results

**Classification:** `scale_degrades_strength`.

## Frozen Contract

- Five 3,500-game A16 replays (seeds 68-72), ordinary reused-tree 384-simulation PUCT.
- Pure raw noisy visit-count targets, terminal-outcome values, and full 73,741-parameter Adam updates.
- `repeat20_matched` and `unique100_once` have identical presentation, step, and batch-size sequences.

## Frozen Evidence

- Candidate aggregate SHA-256: `60ad77daddb1fa01645068867892ec5525381b0b304d85235a989631c3c340ce`.
- Suite manifest SHA-256: `076b8d031b01ffe5dc710ab2ae271c2d018a9e8174ee1ea95989aeff40fb85f0`; preflight SHA-256: `15f6c5749050b2de178c8762bf9a906a904c27a6b93ffcd4824fed07dc80c249`.
- A16 hashes: `{"a16_artifact_model": "8eaa5765786468df9fade11727449af2263a5d0d6da0fa2138040ef2b8629d34", "a16_artifact_weights": "74e160734554dba9ecf4ecf3ac13b3e44526c4f6e7e5e52d9fe8bde0054eb789", "a16_checkpoint": "8eaa5765786468df9fade11727449af2263a5d0d6da0fa2138040ef2b8629d34", "a16_snapshot": "f38a3a9db15a095870c0294150fe81e4719cf77102c4b3de316fcae5e1b9c3ff", "p1_checkpoint": "e4a9c95302c17b63405107d26b4cd4f90755d2b21980846ac6e9f0baabf7efe9"}`.

## Results

### Primary 1200:1200

| Lane | Mean | Median | SD | Range | Positive seeds | Hierarchical CI95 |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| repeat20_matched | -0.098437 | -0.088542 | 0.022932 | [-0.123047, -0.075521] | 0/5 | [-0.126953, -0.070312] |
| unique100_once | -0.107943 | -0.123047 | 0.026857 | [-0.129557, -0.075521] | 0/5 | [-0.138281, -0.076953] |

Paired unique-minus-repeat: -0.009505; CI95 [-0.040625, +0.021875].

Per-seed effects (AN/AO/AP):
- Seed 68: unique -0.123047 (AN/AO/AP {'AN': -0.11328125, 'AO': -0.14453125, 'AP': -0.111328125}), repeat -0.075521 (AN/AO/AP {'AN': -0.060546875, 'AO': -0.099609375, 'AP': -0.06640625}), contrast -0.047526 (AN/AO/AP {'AN': -0.052734375, 'AO': -0.044921875, 'AP': -0.044921875}).
- Seed 69: unique -0.082031 (AN/AO/AP {'AN': -0.06640625, 'AO': -0.109375, 'AP': -0.0703125}), repeat -0.123047 (AN/AO/AP {'AN': -0.11328125, 'AO': -0.14453125, 'AP': -0.111328125}), contrast +0.041016 (AN/AO/AP {'AN': 0.046875, 'AO': 0.03515625, 'AP': 0.041015625}).
- Seed 70: unique -0.129557 (AN/AO/AP {'AN': -0.119140625, 'AO': -0.154296875, 'AP': -0.115234375}), repeat -0.123047 (AN/AO/AP {'AN': -0.11328125, 'AO': -0.14453125, 'AP': -0.111328125}), contrast -0.006510 (AN/AO/AP {'AN': -0.005859375, 'AO': -0.009765625, 'AP': -0.00390625}).
- Seed 71: unique -0.129557 (AN/AO/AP {'AN': -0.119140625, 'AO': -0.154296875, 'AP': -0.115234375}), repeat -0.082031 (AN/AO/AP {'AN': -0.06640625, 'AO': -0.109375, 'AP': -0.0703125}), contrast -0.047526 (AN/AO/AP {'AN': -0.052734375, 'AO': -0.044921875, 'AP': -0.044921875}).
- Seed 72: unique -0.075521 (AN/AO/AP {'AN': -0.060546875, 'AO': -0.099609375, 'AP': -0.06640625}), repeat -0.088542 (AN/AO/AP {'AN': -0.080078125, 'AO': -0.111328125, 'AP': -0.07421875}), contrast +0.013021 (AN/AO/AP {'AN': 0.01953125, 'AO': 0.01171875, 'AP': 0.0078125}).

Replay completion, exclusions, unique exposure, presentations, and optimizer steps are preserved in the compact JSON summary.

### Secondary 384:384

| Lane | Mean | Median | SD | Range | Positive seeds | Hierarchical CI95 |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| repeat20_matched | -0.012891 | -0.035807 | 0.049058 | [-0.040365, +0.074219] | 1/5 | [-0.053516, +0.035547] |
| unique100_once | -0.013411 | -0.024740 | 0.039718 | [-0.040365, +0.055990] | 1/5 | [-0.050400, +0.026562] |

Paired unique-minus-repeat: -0.000521; CI95 [-0.013672, +0.013281].

Per-seed effects (AN/AO/AP):
- Seed 68: unique -0.024740 (AN/AO/AP {'AN': -0.01171875, 'AO': -0.046875, 'AP': -0.015625}), repeat -0.035807 (AN/AO/AP {'AN': -0.025390625, 'AO': -0.0546875, 'AP': -0.02734375}), contrast +0.011068 (AN/AO/AP {'AN': 0.013671875, 'AO': 0.0078125, 'AP': 0.01171875}).
- Seed 69: unique -0.037760 (AN/AO/AP {'AN': -0.029296875, 'AO': -0.056640625, 'AP': -0.02734375}), repeat -0.024740 (AN/AO/AP {'AN': -0.01171875, 'AO': -0.046875, 'AP': -0.015625}), contrast -0.013021 (AN/AO/AP {'AN': -0.017578125, 'AO': -0.009765625, 'AP': -0.01171875}).
- Seed 70: unique +0.055990 (AN/AO/AP {'AN': 0.0546875, 'AO': 0.05078125, 'AP': 0.0625}), repeat +0.074219 (AN/AO/AP {'AN': 0.076171875, 'AO': 0.064453125, 'AP': 0.08203125}), contrast -0.018229 (AN/AO/AP {'AN': -0.021484375, 'AO': -0.013671875, 'AP': -0.01953125}).
- Seed 71: unique -0.040365 (AN/AO/AP {'AN': -0.03125, 'AO': -0.05859375, 'AP': -0.03125}), repeat -0.037760 (AN/AO/AP {'AN': -0.029296875, 'AO': -0.056640625, 'AP': -0.02734375}), contrast -0.002604 (AN/AO/AP {'AN': -0.001953125, 'AO': -0.001953125, 'AP': -0.00390625}).
- Seed 72: unique -0.020182 (AN/AO/AP {'AN': -0.005859375, 'AO': -0.04296875, 'AP': -0.01171875}), repeat -0.040365 (AN/AO/AP {'AN': -0.03125, 'AO': -0.05859375, 'AP': -0.03125}), contrast +0.020182 (AN/AO/AP {'AN': 0.025390625, 'AO': 0.015625, 'AP': 0.01953125}).

Replay completion, exclusions, unique exposure, presentations, and optimizer steps are preserved in the compact JSON summary.

## Next Action

Close incremental A16 fitting; do not propose more epochs, target mixing, or isolated scopes.
