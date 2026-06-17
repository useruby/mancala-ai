# AlphaZero-Lite Control EP2 PUCT e1 Promotion Results

**Date**: 2026-06-17

**Classification**: `promoted_e1_current`

## Summary

- Previous current weights SHA256: `6946aaffb2e32916c1b7ed57437f6c86b94778833954c546af0e78097d9d2781`
- Promoted current weights SHA256: `6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece`
- Expected e1 weights SHA256: `6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece`
- Metadata version: `azlite-control-ep2-puct-policy-head-e1`
- Changed files under `model-artifact/current`: `metadata.json`, `weights.json`
- Training run: not run
- Self-play generation: not run

## Artifact Integrity

- Candidate artifact verified at `/tmp/azlite_control_ep2_puct_smoke/puct_policy_head_e1/artifact_puct_policy_head_e1`
- Candidate checkpoint SHA256: `a793f32565b0c706c4228e4de3bc00aea5c471089ec940c4fe85e726fe4f9357`
- Candidate weights SHA256: `6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece`
- Checked-in `model-artifact/current/weights.json` SHA256 after replacement: `6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece`
- Metadata JSON parse: `True`
- Runtime loader: `True`

## Gate Results

- Classification: `high_search_breakthrough`
- Standard disadvantaged-seat score (`384:256`): `+0.0000`
- Equal-high disadvantaged-seat score (`1200:1200`): `+1.0000`
- Challenger-high disadvantaged-seat score (`1200:256`): `+0.0000`
- Current-high disadvantaged-seat score (`256:768`): `+0.0000`

## Fixed Large-Suite Before/After

| Candidate | 384:256 DS | 768:256 DS | 768:768 DS | 1200:1200 DS | 1200:256 DS | 256:768 DS |
|---|---:|---:|---:|---:|---:|---:|
| previous_current_ref | -0.1523 | -0.5547 | -0.6458 | -0.2630 | -0.1029 | -0.5547 |
| promoted_current | -0.0208 | +0.0573 | -0.2161 | +0.0703 | -0.1237 | -0.1458 |

## Acceptance Checks

- `384:256` exact deterministic reproduction of PR #119 e1 value `-0.0208`: `True`
- `384:256` promoted current stronger than previous current: `True`
- `1200:1200` regression within accepted tolerance (`<= 0.03`): `True`
- No training or self-play was run during promotion: `true`

## Runtime Probes

- `opening_start` legal moves `[0, 1, 2, 3, 4, 5]` selected `2`
- `sparse_legal_mask` legal moves `[1, 5]` selected `1`

## Artifacts

- Promotion workdir: `/tmp/azlite_control_ep2_puct_e1_promotion`
- Gate report: `/tmp/azlite_control_ep2_puct_e1_promotion/gate_report.json`
- Opening-suite report: `/tmp/azlite_control_ep2_puct_e1_promotion/opening_suite_eval/temperature_benchmark_report.json`

