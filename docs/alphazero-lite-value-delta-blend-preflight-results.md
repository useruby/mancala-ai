# AlphaZero-Lite Value-Delta Blend Preflight Results

- classification: `outcome_value_learning_blocked_by_search`
- audit: `/tmp/azlite_value_delta_blend_preflight_smoke/value_delta_audit.json`

## Lanes

| lane | alpha | probe pass | abort reason |
|---|---:|---|---|
| blend_alpha_000 | 0.0 | False | search-aware probe gate failed |
| blend_alpha_010 | 0.1 | False | search-aware probe gate failed |
| blend_alpha_100 | 1.0 | False | search-aware probe gate failed |
| global025 | budget_conditioned | False | search-aware probe gate failed |

## Value Metrics

| alpha | MAE | sign accuracy | correlation |
|---:|---:|---:|---:|
| 0.0 | 0.756740 | 0.667969 | 0.532381 |
| 0.1 | 0.752791 | 0.675781 | 0.539377 |
| 1.0 | 0.718643 | 0.681152 | 0.570857 |

## Search Probes

### blend_alpha_000

| budget | changed move rate | visit KL | mean abs root-value delta | mean abs child-Q delta |
|---|---:|---:|---:|---:|
| 384:256 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 768:256 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 768:768 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1200:1200 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1200:256 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 256:768 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

### blend_alpha_010

| budget | changed move rate | visit KL | mean abs root-value delta | mean abs child-Q delta |
|---|---:|---:|---:|---:|
| 384:256 | 0.000000 | 0.001453 | 0.001323 | 0.003794 |
| 768:256 | 0.000000 | 0.000040 | 0.000907 | 0.001616 |
| 768:768 | 0.000000 | 0.000046 | 0.000781 | 0.002150 |
| 1200:1200 | 0.000000 | 0.000388 | 0.000570 | 0.003213 |
| 1200:256 | 0.000000 | 0.000388 | 0.000570 | 0.003213 |
| 256:768 | 0.000000 | 0.000482 | 0.002134 | 0.004202 |

### blend_alpha_100

| budget | changed move rate | visit KL | mean abs root-value delta | mean abs child-Q delta |
|---|---:|---:|---:|---:|
| 384:256 | 0.500000 | 0.005790 | 0.011235 | 0.015446 |
| 768:256 | 0.500000 | 0.003328 | 0.006810 | 0.010669 |
| 768:768 | 0.500000 | 0.004075 | 0.006520 | 0.007395 |
| 1200:1200 | 0.000000 | 0.010676 | 0.008033 | 0.020758 |
| 1200:256 | 0.000000 | 0.010676 | 0.008033 | 0.020758 |
| 256:768 | 0.000000 | 0.002868 | 0.011901 | 0.014864 |

### global025

| budget | changed move rate | visit KL | mean abs root-value delta | mean abs child-Q delta |
|---|---:|---:|---:|---:|
| 384:256 | 0.000000 | 0.000164 | 0.004356 | 0.005703 |
| 768:256 | 0.000000 | 0.000289 | 0.002658 | 0.004772 |
| 768:768 | 0.000000 | 0.000175 | 0.001617 | 0.002563 |
| 1200:1200 | 0.000000 | 0.002887 | 0.000746 | 0.007062 |
| 1200:256 | 0.000000 | 0.002887 | 0.000746 | 0.007062 |
| 256:768 | 0.000000 | 0.001108 | 0.003900 | 0.010716 |
