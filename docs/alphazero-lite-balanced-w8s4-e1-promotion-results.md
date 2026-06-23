# AlphaZero-Lite Balanced w8s4 e1 Promotion Results

**Date**: 2026-06-23

**Classification**: `promoted_balanced_w8s4_e1_current`

## Summary

- Previous current weights SHA256: `6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece`
- Promoted current weights SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- Manifest candidate hash: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- Exact files changed under `model-artifact/current`: `metadata.json`, `weights.json`
- Training run: not run
- Replay generation: not run
- Self-play generation: not run

## Artifact Integrity

- Previous current weights SHA256 matched expected parent: `True`
- Candidate artifact verified at `/tmp/azlite_balanced_opening_puct_replay/balanced_w8s4_policy_head/artifact_balanced_w8s4_policy_head_e1`
- Candidate checkpoint SHA256: `18dacb5ef38602a77abf3927dc8748ac167c2ca91237385ec010ab27a123adc9`
- Promoted `model-artifact/current/weights.json` SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- Metadata JSON parse: `True`
- Runtime loader: `True`
- Runtime test: `True`

## Metadata Provenance

- version: `azlite-balanced-w8s4-policy-head-e1`
- parent_version: `azlite-control-ep2-puct-policy-head-e1`
- parent_weights_sha256: `6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece`
- source_experiment: `docs/alphazero-lite-balanced-opening-puct-replay-results.md`
- source_runner: `ml/alphazero_lite/run_balanced_opening_puct_replay.py`
- selected_lane: `balanced_w8s4_policy_head_e1`
- trainable_scope: `policy_head`
- replay_sources: `generic_bootstrap,random_teacher,opening_puct_disagreement_replay,equal_budget_stability_replay`
- replay_weights: `4,1,8,4`
- architecture: `residual_v3 / kalah_v3 / hidden_sizes 96,3`
- architecture_change: `none`

## Fixed Large-Suite Before/After

| Candidate | 384:256 DS | 768:256 DS | 768:768 DS | 1200:1200 DS | 1200:256 DS | 256:768 DS |
|---|---|---|---|---|---|---|
| previous_current_ref | -0.3932 | -0.3984 | -0.1589 | -0.4167 | -0.1706 | -0.3984 |
| promoted_current | -0.2201 | -0.3854 | -0.2266 | -0.3620 | -0.0859 | -0.2826 |

## Held-Out Spot Check

| Suite | Previous 384:256 DS | Promoted 384:256 DS | Delta |
|---|---|---|---|
| heldout_seed43_large | -0.3932 | -0.2044 | +0.1888 |
| heldout_seed44_large | -0.4089 | -0.2096 | +0.1992 |
| heldout_seed45_large | -0.4349 | -0.2370 | +0.1979 |
| heldout_seed46_large | -0.3854 | -0.2044 | +0.1810 |
| heldout_seed47_large | -0.3802 | -0.1784 | +0.2018 |
| heldout_seed48_large | -0.3932 | -0.2057 | +0.1875 |

## Gate Classification Table

| Candidate | Classification | 384:256 DS | 1200:1200 DS | 1200:256 DS | 256:768 DS |
|---|---|---|---|---|---|
| previous_current_ref | high_search_breakthrough | +0.0000 | +1.0000 | +1.0000 | +0.0000 |
| promoted_current | high_search_breakthrough | +0.0000 | +1.0000 | +1.0000 | +0.0000 |

## Acceptance Checks

- `384:256` improvement >= `0.15` DS: `True`
- `768:768` regression >= `-0.10` DS: `True`
- `1200:1200` regression >= `-0.03` DS: `True`
- `1200:256` regression >= `-0.03` DS: `True`
- Gate keeps high-search signature: `True`

## Artifacts

- Promotion workdir: `/tmp/azlite_balanced_w8s4_e1_promotion`
- Previous current backup: `/tmp/azlite_balanced_w8s4_e1_promotion/previous_current_artifact`
- Fixed large-suite report: `/tmp/azlite_balanced_w8s4_e1_promotion/fixed_large_suite/temperature_benchmark_report.json`
- Held-out report dir: `/tmp/azlite_balanced_w8s4_e1_promotion/heldout_spot_check`
- Gate report dir: `/tmp/azlite_balanced_w8s4_e1_promotion/gate`
- Summary metrics: `/tmp/azlite_balanced_w8s4_e1_promotion/summary_metrics.json`
