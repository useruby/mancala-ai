# Fresh P1 Checkpoint-Selection Execution

**Classification:** `no_safe_learning_checkpoint`

**Next experiment:** investigate why the existing policy-head update is not compositionally safe

## Lineage And Contract

- p0_weights_sha256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- p1_weights_sha256: `77969733ece5ced92d3a143a0fe9d82863ca3ec4faa477470ff5826ac22e4e12`
- p1_checkpoint_sha256: `e4a9c95302c17b63405107d26b4cd4f90755d2b21980846ac6e9f0baabf7efe9`
- p1_state_hash: `a86acb54b97c860289530fcb7ca64194724d43667580a52e2948359bfa3ebdf4`
- generation_seed: `44`
- fresh_replay_sha256: `892827d8ee67a66e6324a2aaec7011df1a21625fc3f6bcd87cab39ce655d2a88`
- fresh_game_count: `700`
- fresh_position_count: `27642`
- selfplay_config: `{'games_requested': 700, 'games_generated': 700, 'positions_generated': 27642, 'seed': 44, 'simulations': 384, 'c_puct': 1.25, 'player_mode': 'puct', 'input_encoding': 'kalah_v3', 'policy_target_mode': 'default', 'value_target_mode': 'default', 'policy_target_noise_mode': 'noisy', 'tree_reuse_enabled': True, 'replay_path': '/tmp/azlite_fresh_p1_checkpoint_selection/fresh_p1_self_play.jsonl', 'replay_sha256': '892827d8ee67a66e6324a2aaec7011df1a21625fc3f6bcd87cab39ce655d2a88', 'checkpoint_npz_sha256': 'e4a9c95302c17b63405107d26b4cd4f90755d2b21980846ac6e9f0baabf7efe9'}`

## Training Eligibility

| Step | CE(search) | CE(P1) | CE(mixed) | Search improvement | Fit fraction | L1 mean/p99/max | Top-1 | Trunk/value/policy drift | Eligible |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: |
| 1 | 1.087671 | 0.917013 | 0.925546 | +0.000106 | 0.0225 | 0.000613/0.001589/0.001862 | 0.0000 | 0.00e+00/0.00e+00/0.000084 | False |
| 4 | 1.087365 | 0.917016 | 0.925534 | +0.000412 | 0.0874 | 0.002029/0.005172/0.005866 | 0.0020 | 0.00e+00/0.00e+00/0.000238 | False |
| 16 | 1.086442 | 0.917030 | 0.925501 | +0.001335 | 0.2830 | 0.004400/0.010867/0.013056 | 0.0039 | 0.00e+00/0.00e+00/0.000707 | True |
| 46 | 1.085398 | 0.917041 | 0.925459 | +0.002379 | 0.5044 | 0.005438/0.014529/0.016301 | 0.0039 | 0.00e+00/0.00e+00/0.001661 | True |

## P1 Vs P0 Reference Control

| Budget | Effect | 95% CI | Seat A | Seat B | W/D/L |
| --- | ---: | --- | ---: | ---: | --- |
| 384:256 | -0.0137 | [-0.0293, +0.0039] | -0.0156 | -0.0117 | 392/38/82 |
| 1200:1200 | +0.0234 | [+0.0078, +0.0430] | +0.0273 | +0.0195 | 204/128/180 |

## Arena Matrix

| Step | Match | Budget | Effect | 95% CI | Seat A | Seat B | W/D/L | Safe |
| ---: | --- | --- | ---: | --- | ---: | ---: | --- | ---: |
| 16 | candidate_vs_p1 | 384:256 | -0.0742 | [-0.1055, -0.0430] | -0.1484 | +0.0000 | 362/38/112 | False |
| 16 | candidate_vs_p1 | 1200:1200 | -0.0234 | [-0.0430, -0.0078] | -0.0391 | -0.0078 | 180/128/204 | False |
| 16 | candidate_vs_p0 | 384:256 | -0.0879 | [-0.1230, -0.0547] | -0.1641 | -0.0117 | 354/38/120 | False |
| 16 | candidate_vs_p0 | 1200:1200 | +0.0039 | [+0.0000, +0.0098] | +0.0078 | +0.0000 | 184/148/180 | True |
| 46 | candidate_vs_p1 | 384:256 | -0.0820 | [-0.1133, -0.0508] | -0.1641 | +0.0000 | 354/46/112 | False |
| 46 | candidate_vs_p1 | 1200:1200 | -0.0234 | [-0.0430, -0.0078] | -0.0391 | -0.0078 | 180/128/204 | False |
| 46 | candidate_vs_p0 | 384:256 | -0.0801 | [-0.1133, -0.0469] | -0.1484 | -0.0117 | 354/46/112 | False |
| 46 | candidate_vs_p0 | 1200:1200 | +0.0039 | [+0.0000, +0.0098] | +0.0078 | +0.0000 | 184/148/180 | True |

## Selection And Cumulative Gate

- Selected checkpoint: `None`
- Cumulative lineage gate: `{"passed": false, "failure_reasons": ["cumulative_lineage_context_missing"], "missing_contexts": ["384:256", "1200:1200"]}`

## Search Diagnostics

| Step | Reference | Budget | Move changes | Visit JS | Q-rank changes | Root-value delta | Visit-margin delta |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | vs_p1 | 384:256 | 0.0078 | 0.000237 | -0.0273 | -0.000032 | +0.5117 |
| 16 | vs_p1 | 1200:1200 | 0.0117 | 0.000360 | +0.0078 | +0.000220 | -4.6328 |
| 16 | vs_p0 | 384:256 | 0.0234 | 0.001577 | +0.0078 | +0.000143 | -0.6992 |
| 16 | vs_p0 | 1200:1200 | 0.0156 | 0.002657 | +0.0078 | +0.000577 | -4.2578 |
| 46 | vs_p1 | 384:256 | 0.0156 | 0.000180 | -0.0312 | -0.000117 | +0.1211 |
| 46 | vs_p1 | 1200:1200 | 0.0117 | 0.000507 | -0.0039 | -0.000044 | -4.4609 |
| 46 | vs_p0 | 384:256 | 0.0273 | 0.001594 | +0.0039 | +0.000058 | -1.0898 |
| 46 | vs_p0 | 1200:1200 | 0.0156 | 0.002832 | -0.0039 | +0.000313 | -4.0859 |

## Replay Comparison

Diagnostic-only comparison against the PR #204 replay is retained in the JSON summary, including entropy, L1 tails, top-1 disagreement, game length, and outcome distribution.
