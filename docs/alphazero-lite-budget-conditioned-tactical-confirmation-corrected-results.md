# Budget-Conditioned Tactical Confirmation

Classification: `budget_conditioned_tactical_profile_rejected_confirmed`.

Decision contract SHA256: `749125a3f6a540cc1649a227a773b5003547388e808d31a488dcb25f05f9b0ce`.

| Budget | Mean DS | Median | Worst suite | Hierarchical 95% CI |
|---|---:|---:|---:|---|
| 768:256 | +0.0339 | +0.0334 | +0.0312 | [+0.0287, +0.0390] |
| 1200:1200 | +0.0104 | +0.0107 | +0.0039 | [-0.0012, +0.0216] |
| 1200:256 | +0.0461 | +0.0467 | +0.0420 | [+0.0408, +0.0514] |
| 256:768 | +0.0339 | +0.0334 | +0.0312 | [+0.0287, +0.0390] |

Suite independence audit: `passed`.
Expected-null errors: `0`.

## Correction Manifest

Measurement correction of decision contract: `749125a3f6a540cc1649a227a773b5003547388e808d31a488dcb25f05f9b0ce`.
Diagnosis: `stale_suite_cache`.
Code base commit: `b5b7148631e146f22c909c5f296eb5c2e8ef74e4`.

## Player-Seat Effects

| Budget | Player seat | Mean E-D | Worst suite | Hierarchical 95% CI |
|---|---|---:|---:|---|
| 768:256 | E_minus_D_player0 | +0.1081 | +0.1016 | [+0.0890, +0.1272] |
| 768:256 | E_minus_D_player1 | +0.0273 | +0.0208 | [+0.0200, +0.0356] |
| 1200:1200 | E_minus_D_player0 | +0.0208 | +0.0078 | [-0.0024, +0.0432] |
| 1200:1200 | E_minus_D_player1 | +0.0208 | +0.0078 | [-0.0024, +0.0432] |
| 1200:256 | E_minus_D_player0 | +0.2422 | +0.2331 | [+0.2263, +0.2585] |
| 1200:256 | E_minus_D_player1 | -0.0579 | -0.0651 | [-0.0690, -0.0467] |
| 256:768 | E_minus_D_player0 | +0.0273 | +0.0208 | [+0.0200, +0.0356] |
| 256:768 | E_minus_D_player1 | +0.1081 | +0.1016 | [+0.0890, +0.1272] |

This measurement-correction rerun uses a clean work directory and retains only cache entries with matching per-seat provenance manifests. Existing arena files without manifests are reported as `legacy_cache_without_manifest` and rerun.
E-D effects are calculated independently for `challenger_starts_0` and `challenger_starts_1` before pooling, controlling player seat across orientations.

## Frozen Suites

| Suite | SHA256 |
|---|---|
| `/tmp/azlite_budget_tactical_confirmation/suites/heldout_seed49_large.jsonl` | `9d45df32f023e5e9a7ba12d72f3f467a2ce49399b38357e17f0bbc68e008111b` |
| `/tmp/azlite_budget_tactical_confirmation/suites/heldout_seed50_large.jsonl` | `7f58f98df175a707a00ca42b21ef7625669f95b3b788cfedcd428c91b0edd857` |
| `/tmp/azlite_budget_tactical_confirmation/suites/heldout_seed51_large.jsonl` | `33fdd0f619bef908968eb16711a8d5b448da5b8b5ea7e7166be0293dab8ef943` |
| `/tmp/azlite_budget_tactical_confirmation/suites/heldout_seed52_large.jsonl` | `161e7f440d879ff095ea717ce66e4bdb3d3d88a48c0ff61131a1979ce610fbb1` |
| `/tmp/azlite_budget_tactical_confirmation/suites/heldout_seed53_large.jsonl` | `978154ee6f5f9fd47c3ebf199d61f665505a528efc33d7f375541b3dac9e65b2` |
| `/tmp/azlite_budget_tactical_confirmation/suites/heldout_seed54_large.jsonl` | `968a2b89093f824c8309e1227f7ad725c5d8a7e5c1ee373d4e137c8261895f9b` |
