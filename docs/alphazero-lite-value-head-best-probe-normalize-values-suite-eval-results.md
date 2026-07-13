# AlphaZero-Lite Value-Head Best-Probe Normalize-Values Suite Eval

- source summary: `/tmp/azlite_terminal_outcome_selfplay_iteration`
- candidate: `value_head_only_best_probe`
- candidate artifact: `/tmp/azlite_terminal_outcome_selfplay_iteration/value_head_only_e2/artifact_value_head_only_e2`
- budget pairs: `384:256,768:768,1200:1200,1200:256,256:768`

## Budget Comparison

| Suite | Budget | Baseline DS | Normalize DS | Normalize-Baseline |
|---|---|---|---|---|
| fixed_large | 1200:1200 | +0.2904 | +0.1536 | -0.1367 |
| fixed_large | 1200:256 | -0.3464 | +0.1458 | +0.4922 |
| fixed_large | 256:768 | -0.3151 | +0.0156 | +0.3307 |
| fixed_large | 384:256 | -0.4219 | +0.0091 | +0.4310 |
| fixed_large | 768:768 | +0.0260 | -0.1497 | -0.1758 |
| heldout_seed43_large | 1200:1200 | +0.3177 | +0.1693 | -0.1484 |
| heldout_seed43_large | 1200:256 | -0.3451 | +0.1367 | +0.4818 |
| heldout_seed43_large | 256:768 | -0.2982 | +0.0065 | +0.3047 |
| heldout_seed43_large | 384:256 | -0.4102 | -0.0013 | +0.4089 |
| heldout_seed43_large | 768:768 | +0.0521 | -0.1445 | -0.1966 |
| heldout_seed44_large | 1200:1200 | +0.2982 | +0.1654 | -0.1328 |
| heldout_seed44_large | 1200:256 | -0.3411 | +0.1484 | +0.4896 |
| heldout_seed44_large | 256:768 | -0.3099 | -0.0013 | +0.3086 |
| heldout_seed44_large | 384:256 | -0.4141 | +0.0052 | +0.4193 |
| heldout_seed44_large | 768:768 | +0.0378 | -0.1536 | -0.1914 |
| heldout_seed45_large | 1200:1200 | +0.2760 | +0.1523 | -0.1237 |
| heldout_seed45_large | 1200:256 | -0.3451 | +0.1393 | +0.4844 |
| heldout_seed45_large | 256:768 | -0.3164 | -0.0208 | +0.2956 |
| heldout_seed45_large | 384:256 | -0.4115 | +0.0039 | +0.4154 |
| heldout_seed45_large | 768:768 | +0.0273 | -0.1706 | -0.1979 |
| heldout_seed46_large | 1200:1200 | +0.2760 | +0.1576 | -0.1185 |
| heldout_seed46_large | 1200:256 | -0.3529 | +0.1276 | +0.4805 |
| heldout_seed46_large | 256:768 | -0.3008 | -0.0078 | +0.2930 |
| heldout_seed46_large | 384:256 | -0.4049 | +0.0117 | +0.4167 |
| heldout_seed46_large | 768:768 | +0.0312 | -0.1497 | -0.1810 |
| heldout_seed47_large | 1200:1200 | +0.2917 | +0.1445 | -0.1471 |
| heldout_seed47_large | 1200:256 | -0.3503 | +0.1380 | +0.4883 |
| heldout_seed47_large | 256:768 | -0.3411 | +0.0169 | +0.3581 |
| heldout_seed47_large | 384:256 | -0.4310 | -0.0039 | +0.4271 |
| heldout_seed47_large | 768:768 | +0.0104 | -0.1602 | -0.1706 |
| heldout_seed48_large | 1200:1200 | +0.2695 | +0.1393 | -0.1302 |
| heldout_seed48_large | 1200:256 | -0.3438 | +0.1458 | +0.4896 |
| heldout_seed48_large | 256:768 | -0.3464 | +0.0169 | +0.3633 |
| heldout_seed48_large | 384:256 | -0.4401 | -0.0065 | +0.4336 |
| heldout_seed48_large | 768:768 | -0.0091 | -0.1602 | -0.1510 |

